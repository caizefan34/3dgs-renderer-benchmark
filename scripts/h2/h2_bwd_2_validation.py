#!/usr/bin/env python3
"""H2-BWD-2: SCALAR_ADJOINT Production Validation.

Validates the first production-source Trainable HiGS backward candidate
(SCALAR_ADJOINT) that crossed the kernel performance gate, using the exact
Codex production-source build (binary SHA256 da53009841c5f9a6...).

Does NOT use the old load_inline H2-BWD-1 microkernel. Loads the frozen
production extension from /tmp/higs_h2_bwd_cf (TORCH_EXTENSIONS_DIR points
at the existing cache so the exact binary is reused, never rebuilt).

Produces, under --out-dir:
  resource_validation.json   - ptxas-derived registers/spills/shared + theoretical occupancy
  correctness_by_tensor.csv  - per-tensor baseline vs scalar_adjoint gradient metrics
  kernel_timing.csv          - interleaved backward-only kernel timing (5 reps x 100)
  backward_fb_timing.csv     - T_forward / T_backward / T_F+B through the autograd function
  occupancy.json             - theoretical + NCU attempted achieved occupancy
  analysis.json              - gate evaluation + mechanism falsification + classification
  provenance.json            - run identity
  run.log                    - (written by launcher)

Run this in a fresh Python process on the A100 (mx).
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import runpy
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

VARIANTS = ("baseline", "scalar_adjoint")
K_SH = 16
SH_DEGREE = 3

# Frozen experimental identity (mission)
B2_BASE_COMMIT = "77ab983ffe43420b2131669cb35776b883ca4c3c"
B2_PATCH_SHA256 = "74e5d8b3b6273b9446ec0551ce91409783e2aa935c8d8e354b4099341390c84c"
H2_PATCH_SHA256 = "d001eac2d6908126103ead7c060f896cfc569989ac90d264236a97edc68137ff"
BINARY_SHA256_FROZEN = "da53009841c5f9a6145dfb01e5ab84de5286f52710c9a7011bcb4b85eb18842c"

# A100 SM80 limits
SM80_REGS_PER_SM = 65536
SM80_MAX_THREADS_PER_SM = 2048
SM80_MAX_BLOCKS_PER_SM = 32
SM80_SHARED_PER_SM_DEFAULT = 100 * 1024  # 100 KB default per-block opt-in ceiling
BLOCK_THREADS = 128  # grid (1, tile_h, tile_w), block (16,8,1) = 128


def bootstrap(source, core_so):
    """Load the frozen production extension. TORCH_EXTENSIONS_DIR must point at
    the existing CF cache so the exact binary (da5300...) is reused."""
    sys.path.insert(0, source)
    spec = importlib.util.spec_from_file_location("gsplat_cuda", core_so)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    sys.modules["gsplat.csrc"] = core
    backend = runpy.run_path(
        str(Path(source) / "gsplat/experimental/render/kernels/cuda/build.py")
    )["build_and_load_experimental_gaussian_render_inference_scene"]()
    return backend


def load_fixture(ply_path, cameras_path, max_long_side, device, cam_idx=0):
    v = PlyData.read(ply_path)["vertex"]
    means = torch.tensor(np.column_stack([v["x"], v["y"], v["z"]]), device=device, dtype=torch.float32)
    quats = torch.tensor(np.column_stack([v[f"rot_{i}"] for i in range(4)]), device=device, dtype=torch.float32)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scales = torch.exp(torch.tensor(np.column_stack([v[f"scale_{i}"] for i in range(3)]), device=device, dtype=torch.float32))
    opacities = torch.sigmoid(torch.tensor(v["opacity"], device=device, dtype=torch.float32))
    sh = torch.zeros((len(v), K_SH, 3), device=device, dtype=torch.float32)
    sh[:, 0] = torch.tensor(np.column_stack([v[f"f_dc_{i}"] for i in range(3)]), device=device, dtype=torch.float32)
    rest = torch.stack([torch.tensor(v[f"f_rest_{i}"], device=device, dtype=torch.float32) for i in range(45)], 1)
    sh[:, 1:] = rest.reshape(len(v), 3, 15).permute(0, 2, 1)
    cams = json.loads(Path(cameras_path).read_text())
    c = cams[cam_idx]
    native_w, native_h = int(c["width"]), int(c["height"])
    scale = min(1.0, max_long_side / max(native_w, native_h))
    width, height = int(round(native_w * scale)), int(round(native_h * scale))
    R = np.asarray(c["rotation"], dtype=np.float32).T
    p = np.asarray(c["position"], dtype=np.float32)
    vm = np.eye(4, dtype=np.float32); vm[:3, :3] = R; vm[:3, 3] = -R @ p
    K = np.array([[float(c["fx"]) * width / native_w, 0, (width - 1) / 2],
                  [0, float(c["fy"]) * width / native_w, (height - 1) / 2], [0, 0, 1]], dtype=np.float32)
    return (means, quats, scales, opacities, sh), torch.tensor(vm, device=device)[None, None], torch.tensor(K, device=device)[None, None], width, height


def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def metrics(reference, value):
    a = reference.detach().float().reshape(-1)
    b = value.detach().float().reshape(-1)
    finite = torch.isfinite(a) & torch.isfinite(b)
    af, bf = a[finite], b[finite]
    d = bf - af
    denom = af.norm().clamp_min(1e-30)
    bnorm = bf.norm().clamp_min(1e-30)
    cos = float(torch.dot(af, bf) / (af.norm().clamp_min(1e-30) * bnorm)) if af.numel() else float("nan")
    support = (a.abs() > 1e-10) != (b.abs() > 1e-10)
    return {
        "norm_baseline": float(af.norm()) if af.numel() else float("nan"),
        "norm_candidate": float(bf.norm()) if bf.numel() else float("nan"),
        "max_abs": float(d.abs().max()) if d.numel() else float("nan"),
        "mean_abs": float(d.abs().mean()) if d.numel() else float("nan"),
        "relative_L2": float(d.norm() / denom) if d.numel() else float("nan"),
        "cosine": cos,
        "zero_nonzero_disagreement": int(support.sum()),
        "NaN_count": int((a.isnan() | b.isnan()).sum()),
        "Inf_count": int((a.isinf() | b.isinf()).sum()),
    }


# ----------------------------- resource validation -----------------------------

def parse_ptxas_log(log_path):
    """Parse ptxas -v output (from the -Xptxas=-v build) for the CDIM=3,PX=2 bwd kernels.

    Real format (one block per instantiation):
      ptxas info : Compiling entry function '<mangled>' for 'sm_80'
      ptxas info : Function properties for <mangled>
          <S> bytes stack frame, <SS> bytes spill stores, <SL> bytes spill loads
      ptxas info : Used <R> registers, used 1 barriers, <C> bytes cmem[0]

    Mangled names carry the template params, e.g.:
      ...higs_blend_bwd_px_kernelILj3ELj2EEE...           -> baseline, CDIM=3 PX=2
      ...higs_blend_bwd_px_cf_kernelILj3ELj2ELi2EEE...    -> scalar_adjoint (cf VARIANT=2), CDIM=3 PX=2
    The cf VARIANT ints: 1=sigma_gate, 2=scalar_adjoint, 4=uv_reuse, 7=combined.
    Dynamic shared memory is a launch parameter (extern __shared__): 128*(4+12+12+12)=5120 B,
    identical for all variants; ptxas reports cmem, not dynamic shared.
    """
    text = Path(log_path).read_text(errors="replace") if Path(log_path).exists() else ""
    out = {"baseline": {}, "scalar_adjoint": {}, "cf_all": []}
    # split into per-kernel blocks
    blocks = re.split(r"ptxas info\s+: Compiling entry function '", text)
    for blk in blocks[1:]:
        mname = blk.split("'", 1)[0] if "'" in blk else blk.split("\n", 1)[0]
        # template params: ILj<CDIM>ELj<PX>E(Li<VAR>E)?E
        tp = re.search(r"ILj(\d+)ELj(\d+)E(?:Li(\d+)E)?E", mname)
        if not tp:
            continue
        cdim, px, var = int(tp.group(1)), int(tp.group(2)), (int(tp.group(3)) if tp.group(3) else None)
        if cdim != 3 or px != 2:
            continue  # only the frozen CDIM=3,PX=2 identity matters
        regs_m = re.search(r"Used\s+(\d+)\s+registers", blk)
        spill_m = re.search(r"(\d+)\s+bytes stack frame,\s*(\d+)\s+bytes spill stores,\s*(\d+)\s+bytes spill loads", blk)
        if not regs_m:
            continue
        regs = int(regs_m.group(1))
        spill_s = int(spill_m.group(2)) if spill_m else None
        spill_l = int(spill_m.group(3)) if spill_m else None
        is_cf = "higs_blend_bwd_px_cf_kernel" in mname
        rec = {"mangled": mname[:120], "kernel": "higs_blend_bwd_px_cf_kernel" if is_cf else "higs_blend_bwd_px_kernel",
               "CDIM": cdim, "PX": px, "cf_variant_int": var,
               "registers_per_thread": regs, "spill_stores_bytes": spill_s, "spill_loads_bytes": spill_l}
        if is_cf:
            out["cf_all"].append(rec)
            if var == 2:  # scalar_adjoint
                out["scalar_adjoint"] = rec
        else:
            out["baseline"] = rec
    return out


def theoretical_occupancy(regs_per_thread, dyn_shared_bytes, threads_per_block=BLOCK_THREADS):
    """A100 SM80 register-limited occupancy (default 100KB shared ceiling, no opt-in)."""
    regs_per_block = regs_per_thread * threads_per_block
    blocks_reg = SM80_REGS_PER_SM // regs_per_block if regs_per_block else 0
    blocks_thread = SM80_MAX_THREADS_PER_SM // threads_per_block
    blocks_shared = SM80_SHARED_PER_SM_DEFAULT // dyn_shared_bytes if dyn_shared_bytes > 0 else SM80_MAX_BLOCKS_PER_SM
    blocks = max(0, min(blocks_reg, blocks_thread, blocks_shared, SM80_MAX_BLOCKS_PER_SM))
    active_warps = blocks * (threads_per_block // 32)
    occ = active_warps / (SM80_MAX_THREADS_PER_SM // 32)
    return {
        "registers_per_thread": regs_per_thread,
        "threads_per_block": threads_per_block,
        "regs_per_block": regs_per_block,
        "blocks_per_sm_register_limited": blocks_reg,
        "blocks_per_sm_thread_limited": blocks_thread,
        "blocks_per_sm_shared_limited": blocks_shared,
        "blocks_per_sm_active": blocks,
        "active_warps_per_sm": active_warps,
        "max_warps_per_sm": SM80_MAX_THREADS_PER_SM // 32,
        "theoretical_occupancy": occ,
        "binding_constraint": "registers" if blocks_reg <= min(blocks_thread, blocks_shared) else ("threads" if blocks_thread <= blocks_shared else "shared"),
    }


# ----------------------------- timing helpers -----------------------------

def make_leaves(values):
    return tuple(x.detach().clone().requires_grad_(True) for x in values)


def render(values, vm, K, w, h, variant, capture_raw=False):
    from gsplat.experimental import rasterize_gaussian_higs_frozen
    from gsplat.experimental.render.functional.gaussian_inference import create_higs_renderer, _HIGS_FROZEN_TRACKER
    os.environ["HIGS_BWD_CF_VARIANT"] = variant
    if capture_raw:
        os.environ["HIGS_BWD_CF_CAPTURE_RAW"] = "1"
    else:
        os.environ.pop("HIGS_BWD_CF_CAPTURE_RAW", None)
    leaves = make_leaves(values)
    _HIGS_FROZEN_TRACKER.reset()
    handle = create_higs_renderer(*leaves, sh_degree=SH_DEGREE)
    out = rasterize_gaussian_higs_frozen(
        *leaves, backward_mode="higs_native", scene=handle, freeze_topology=True,
        viewmats=vm, Ks=K, width=w, height=h, sh_degree=SH_DEGREE,
        use_higs_culling=True, radius_clip=0.0, tile_sampling_ratio=1.0)
    return leaves, handle, out


def correctness_run(backend, values, vm, K, w, h, scene, seed):
    from gsplat.experimental.render.functional.gaussian_inference import _HIGS_FROZEN_TRACKER
    runs = {}
    forward_hashes = {}
    for v in VARIANTS:
        leaves, handle, out = render(values, vm, K, w, h, v, capture_raw=True)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        vr = torch.randn(out["frame"].shape, device="cuda", generator=gen)
        va = torch.randn(out["alpha"].shape, device="cuda", generator=gen)
        (out["frame"].float().mul(vr).sum() + out["alpha"].float().mul(va).sum()).backward()
        torch.cuda.synchronize()
        raw = backend.higs_bwd_cf_last_raw_grads()
        runs[v] = {
            "v_means2d": raw[0].detach().clone(), "v_conics": raw[1].detach().clone(),
            "v_colors": raw[2].detach().clone(), "v_opacities": raw[3].detach().clone(),
            "means": leaves[0].grad.detach().clone(), "quats": leaves[1].grad.detach().clone(),
            "scales": leaves[2].grad.detach().clone(), "opacities": leaves[3].grad.detach().clone(),
            "SH": leaves[4].grad.detach().clone(),
        }
        forward_hashes[v] = {"frame": tensor_hash(out["frame"]), "alpha": tensor_hash(out["alpha"])}
        handle.release()
    base = runs["baseline"]
    rows = []
    for v in VARIANTS[1:]:
        for name in ["v_means2d", "v_conics", "v_colors", "v_opacities", "means", "quats", "scales", "opacities", "SH"]:
            m = metrics(base[name], runs[v][name])
            rows.append({"scene": scene, "candidate": v, "tensor": name, **m})
    return rows, forward_hashes


def interleaved_timer(variants, run_fn, n_warm, n_meas, reps, seed, label):
    """run_fn(variant) -> timed ms (does its own cuda event + sync). Returns
    data dict and paired blocks."""
    rng = random.Random(seed)
    data = {v: [] for v in variants}
    blocks = []
    for rep in range(reps):
        for _ in range(n_warm):
            for v in variants:
                run_fn(v)
        torch.cuda.synchronize()
        for _ in range(n_meas):
            perm = list(variants); rng.shuffle(perm)
            block = {}
            for v in perm:
                t = run_fn(v)
                block[v] = t
                data[v].append(t)
            blocks.append(block)
        print(f"  [{label}] rep {rep+1}/{reps} done", flush=True)
    return data, blocks


def summarize_timing(data, blocks, variants, label, scene):
    rows = []
    base = np.array(data[variants[0]])
    for v in variants:
        arr = np.array(data[v])
        row = {"scene": scene, "metric": label, "variant": v,
               "median_ms": float(np.median(arr)), "mean_ms": float(np.mean(arr)),
               "p10_ms": float(np.percentile(arr, 10)), "p90_ms": float(np.percentile(arr, 90)),
               "std_ms": float(np.std(arr)), "n": len(arr)}
        if v != variants[0]:
            deltas = np.array([b[variants[0]] - b[v] for b in blocks])  # positive = candidate faster
            row["paired_median_delta_ms"] = float(np.median(deltas))
            row["paired_mean_delta_ms"] = float(np.mean(deltas))
            # bootstrap CI with replacement (fixes the 1R rng.sample bug)
            rng = random.Random(seed_for(v))
            boot = []
            for _ in range(2000):
                idx = [rng.randrange(len(deltas)) for _ in range(len(deltas))]
                boot.append(float(np.median(deltas[idx])))
            row["bootstrap_ci95_low_ms"] = float(np.percentile(boot, 2.5))
            row["bootstrap_ci95_high_ms"] = float(np.percentile(boot, 97.5))
            row["pct_speedup_vs_baseline"] = float(np.median(deltas) / np.median(base) * 100) if np.median(base) > 0 else 0.0
        rows.append(row)
    return rows


def seed_for(v):
    return 4200 + abs(hash(v)) % 10000


# ----------------------------- main -----------------------------

SCENE_CONFIGS = {
    "room": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json",
    },
    "bicycle": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/bicycle/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/bicycle/cameras.json",
    },
    "garden": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/garden/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/garden/cameras.json",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scenes", default="room,bicycle,garden")
    ap.add_argument("--cam-idx", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-long-side", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measure", type=int, default=100)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=4200)
    ap.add_argument("--source", default="/tmp/higs_h2_bwd_cf/source")
    ap.add_argument("--core-so", default="/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so")
    ap.add_argument("--ptxas-log", default="/tmp/higs_h2_bwd_cf/build.log")
    ap.add_argument("--binary", default="/tmp/higs_h2_bwd_cf/cache/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so")
    ap.add_argument("--ncu", default="")  # path to ncu binary; empty = skip
    ap.add_argument("--fb-scenes", default="room,bicycle")  # scenes for F+B timing
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = f"cuda:{args.gpu}"
    run_id = str(uuid.uuid4())[:12]
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    fb_scenes = [s.strip() for s in args.fb_scenes.split(",") if s.strip()]

    print(f"[H2-BWD-2] run_id={run_id} scenes={scenes} GPU={args.gpu}", flush=True)
    # Force PX=2 for both variants (frozen resource identity).
    os.environ["HIGS_PX_RUNTIME"] = "2"
    # Match the CF correctness protocol (torch.manual_seed before bootstrap).
    torch.manual_seed(args.seed)

    # --- verify exact binary ---
    binary_sha = "unknown"
    if Path(args.binary).exists():
        binary_sha = hashlib.sha256(Path(args.binary).read_bytes()).hexdigest()
    binary_matches = (binary_sha == BINARY_SHA256_FROZEN)
    print(f"[provenance] binary_sha256={binary_sha} matches_frozen={binary_matches}", flush=True)

    backend = bootstrap(args.source, args.core_so)
    print("[bootstrap] loaded frozen production extension", flush=True)

    # ===================== 1. Resource validation =====================
    print("\n===== 1. Resource validation =====", flush=True)
    res = parse_ptxas_log(args.ptxas_log)
    resource_validation = {
        "frozen_identity": {
            "baseline": {"registers_per_thread": 67, "spills_store_load": "0/0", "dynamic_shared_bytes": 5120},
            "scalar_adjoint": {"registers_per_thread": 56, "spills_store_load": "0/0", "dynamic_shared_bytes": 5120},
        },
        "ptxas_parsed": res,
        "binary_sha256": binary_sha,
        "binary_matches_frozen": binary_matches,
        "sm80_limits": {"regs_per_sm": SM80_REGS_PER_SM, "max_threads_per_sm": SM80_MAX_THREADS_PER_SM,
                        "max_blocks_per_sm": SM80_MAX_BLOCKS_PER_SM, "shared_per_sm_default_bytes": SM80_SHARED_PER_SM_DEFAULT,
                        "block_threads": BLOCK_THREADS},
    }
    occ_baseline = theoretical_occupancy(67, 5120)
    occ_scalar = theoretical_occupancy(56, 5120)
    resource_validation["theoretical_occupancy"] = {"baseline": occ_baseline, "scalar_adjoint": occ_scalar}
    resource_validation["register_reduction"] = {"baseline_regs": 67, "scalar_adjoint_regs": 56, "delta": -11}
    resource_validation["occupancy_change"] = {
        "baseline_blocks_per_sm": occ_baseline["blocks_per_sm_active"],
        "scalar_adjoint_blocks_per_sm": occ_scalar["blocks_per_sm_active"],
        "baseline_occupancy_pct": occ_baseline["theoretical_occupancy"] * 100,
        "scalar_adjoint_occupancy_pct": occ_scalar["theoretical_occupancy"] * 100,
        "delta_occupancy_pct": (occ_scalar["theoretical_occupancy"] - occ_baseline["theoretical_occupancy"]) * 100,
    }
    print(f"  baseline: 67 regs -> {occ_baseline['blocks_per_sm_active']} blocks/SM, {occ_baseline['theoretical_occupancy']*100:.2f}% occupancy", flush=True)
    print(f"  scalar_adjoint: 56 regs -> {occ_scalar['blocks_per_sm_active']} blocks/SM, {occ_scalar['theoretical_occupancy']*100:.2f}% occupancy", flush=True)
    (Path(args.out_dir) / "resource_validation.json").write_text(json.dumps(resource_validation, indent=2))

    # ===================== 2-4. Per-scene correctness + timing =====================
    all_correctness = []
    all_kernel_timing = []
    all_fb_timing = []
    per_scene_summary = {}
    correctness_pass_all = True
    forward_state_identical_all = True

    for scene in scenes:
        print(f"\n===== Scene {scene}/cam{args.cam_idx} =====", flush=True)
        cfg = SCENE_CONFIGS[scene]
        values, vm, K, width, height = load_fixture(cfg["ply"], cfg["cams"], args.max_long_side, device, args.cam_idx)
        print(f"  fixture: {width}x{height}, N_total={len(values[0])}", flush=True)

        # ---- 2. Correctness ----
        print(f"  -- correctness (seed={args.seed}) --", flush=True)
        crows, fhash = correctness_run(backend, values, vm, K, width, height, scene, args.seed)
        all_correctness.extend(crows)
        # verify forward state identical across variants
        fwd_same = (fhash["baseline"] == fhash["scalar_adjoint"])
        forward_state_identical_all = forward_state_identical_all and fwd_same
        print(f"  forward state identical baseline==scalar_adjoint: {fwd_same} ({fhash})", flush=True)
        # per-tensor pass gate: cosine>=0.999999, rel_L2<=1e-4, no NaN/Inf, zero support disagreement==0
        scene_pass = True
        for r in crows:
            ok = (r["cosine"] >= 0.999999 and r["relative_L2"] <= 1e-4 and r["NaN_count"] == 0 and r["Inf_count"] == 0 and r["zero_nonzero_disagreement"] == 0)
            r["pass"] = bool(ok)
            if not ok:
                scene_pass = False
                print(f"    FAIL {r['tensor']}: cos={r['cosine']:.8f} rel_L2={r['relative_L2']:.3e} NaN={r['NaN_count']} Inf={r['Inf_count']} disagree={r['zero_nonzero_disagreement']}", flush=True)
        correctness_pass_all = correctness_pass_all and scene_pass
        print(f"  correctness pass: {scene_pass}", flush=True)

        # ---- 3. Kernel timing (backward only, interleaved) ----
        print(f"  -- kernel timing (backward only, {args.reps}x{args.measure} interleaved) --", flush=True)
        def kernel_run(variant, _values=values, _vm=vm, _K=K, _w=width, _h=height):
            leaves, handle, out = render(_values, _vm, _K, _w, _h, variant, capture_raw=False)
            loss = out["frame"].float().sum() + out["alpha"].float().sum()
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); loss.backward(); e.record(); torch.cuda.synchronize()
            ms = s.elapsed_time(e); handle.release(); return ms
        kdata, kblocks = interleaved_timer(list(VARIANTS), kernel_run, args.warmup, args.measure, args.reps, args.seed, f"kernel-{scene}")
        krows = summarize_timing(kdata, kblocks, list(VARIANTS), "blend_backward_kernel", scene)
        all_kernel_timing.extend(krows)
        for r in krows:
            print(f"    {r['variant']:15s} median={r['median_ms']:.4f} mean={r['mean_ms']:.4f} std={r['std_ms']:.4f}" + (f" paired_delta={r.get('paired_median_delta_ms',0):.4f} ({r.get('pct_speedup_vs_baseline',0):.2f}%) CI95=[{r.get('bootstrap_ci95_low_ms',0):.4f},{r.get('bootstrap_ci95_high_ms',0):.4f}]" if r["variant"] != "baseline" else ""), flush=True)

        # ---- 4. F+B timing (only for fb_scenes) ----
        if scene in fb_scenes:
            print(f"  -- F+B timing (forward/backward/F+B, {args.reps}x{args.measure} interleaved) --", flush=True)
            # build a fixed target for a realistic loss (abs diff to a reference frame)
            from gsplat.experimental.render.functional.gaussian_inference import create_higs_renderer, _HIGS_FROZEN_TRACKER
            _HIGS_FROZEN_TRACKER.reset()
            ref_handle = create_higs_renderer(*[x.detach() for x in values], sh_degree=SH_DEGREE)
            with torch.no_grad():
                ref = rasterize_gaussian_higs_frozen_ref(values, vm, K, width, height, ref_handle)
            ref_handle.release()
            target = ref["frame"].detach().clone()

            # T_forward: time ONLY rasterize_gaussian_higs_frozen (the autograd forward),
            # not create_higs_renderer (topology setup is amortized in training, not per-step).
            def fwd_run_clean(variant):
                os.environ["HIGS_BWD_CF_VARIANT"] = variant
                os.environ.pop("HIGS_BWD_CF_CAPTURE_RAW", None)
                from gsplat.experimental import rasterize_gaussian_higs_frozen
                from gsplat.experimental.render.functional.gaussian_inference import create_higs_renderer, _HIGS_FROZEN_TRACKER
                leaves = make_leaves(values)
                _HIGS_FROZEN_TRACKER.reset()
                handle = create_higs_renderer(*leaves, sh_degree=SH_DEGREE)
                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record()
                out = rasterize_gaussian_higs_frozen(*leaves, backward_mode="higs_native", scene=handle, freeze_topology=True,
                    viewmats=vm, Ks=K, width=width, height=height, sh_degree=SH_DEGREE,
                    use_higs_culling=True, radius_clip=0.0, tile_sampling_ratio=1.0)
                e.record(); torch.cuda.synchronize()
                ms = s.elapsed_time(e); handle.release(); return ms

            def bwd_run(variant):
                # forward untimed, then time backward only
                leaves, handle, out = render(values, vm, K, width, height, variant, capture_raw=False)
                loss = (out["frame"] - target).abs().mean()
                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record(); loss.backward(); e.record(); torch.cuda.synchronize()
                ms = s.elapsed_time(e); handle.release(); return ms

            def fb_run(variant):
                # time forward (rasterize) + backward together through the autograd function
                os.environ["HIGS_BWD_CF_VARIANT"] = variant
                os.environ.pop("HIGS_BWD_CF_CAPTURE_RAW", None)
                from gsplat.experimental import rasterize_gaussian_higs_frozen
                from gsplat.experimental.render.functional.gaussian_inference import create_higs_renderer, _HIGS_FROZEN_TRACKER
                leaves = make_leaves(values)
                _HIGS_FROZEN_TRACKER.reset()
                handle = create_higs_renderer(*leaves, sh_degree=SH_DEGREE)
                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record()
                out = rasterize_gaussian_higs_frozen(*leaves, backward_mode="higs_native", scene=handle, freeze_topology=True,
                    viewmats=vm, Ks=K, width=width, height=height, sh_degree=SH_DEGREE,
                    use_higs_culling=True, radius_clip=0.0, tile_sampling_ratio=1.0)
                loss = (out["frame"] - target).abs().mean()
                loss.backward()
                e.record(); torch.cuda.synchronize()
                ms = s.elapsed_time(e); handle.release(); return ms

            for label, fn in [("T_forward", fwd_run_clean), ("T_backward", bwd_run), ("T_F+B", fb_run)]:
                d, b = interleaved_timer(list(VARIANTS), fn, args.warmup, args.measure, args.reps, args.seed + hash(label) % 1000, f"{label}-{scene}")
                rows = summarize_timing(d, b, list(VARIANTS), label, scene)
                all_fb_timing.extend(rows)
                for r in rows:
                    print(f"    {label:10s} {r['variant']:15s} median={r['median_ms']:.4f}" + (f" delta={r.get('paired_median_delta_ms',0):.4f} ({r.get('pct_speedup_vs_baseline',0):.2f}%)" if r["variant"] != "baseline" else ""), flush=True)

        per_scene_summary[scene] = {
            "correctness_pass": scene_pass,
            "forward_state_identical": fwd_same,
            "kernel_baseline_median_ms": next((r["median_ms"] for r in krows if r["variant"] == "baseline"), None),
            "kernel_scalar_median_ms": next((r["median_ms"] for r in krows if r["variant"] == "scalar_adjoint"), None),
            "kernel_paired_delta_ms": next((r.get("paired_median_delta_ms") for r in krows if r["variant"] == "scalar_adjoint"), None),
            "kernel_pct_speedup": next((r.get("pct_speedup_vs_baseline") for r in krows if r["variant"] == "scalar_adjoint"), None),
        }
        # free scene tensors
        del values, vm, K
        torch.cuda.empty_cache()

    # ===================== write CSVs =====================
    def write_csv(path, rows):
        if not rows:
            Path(path).write_text("")
            return
        keys = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})

    write_csv(Path(args.out_dir) / "correctness_by_tensor.csv", all_correctness)
    write_csv(Path(args.out_dir) / "kernel_timing.csv", all_kernel_timing)
    write_csv(Path(args.out_dir) / "backward_fb_timing.csv", all_fb_timing)

    # ===================== 5. Occupancy (theoretical + NCU attempt) =====================
    print("\n===== 5. Occupancy =====", flush=True)
    occupancy = {
        "theoretical": {"baseline": occ_baseline, "scalar_adjoint": occ_scalar},
        "binary_sha256": binary_sha,
        "ncu": {"attempted": bool(args.ncu), "status": "skipped" if not args.ncu else "pending"},
    }
    if args.ncu:
        # Best-effort NCU: run a one-shot backward under ncu for each variant on room.
        # This is normally done by the launcher wrapping a probe process; here we attempt
        # an in-process subprocess call which may fail on old ncu / sm_80. We record status.
        try:
            probe = Path(args.out_dir) / "ncu_probe.py"
            probe.write_text(_NCU_PROBE_SRC)
            achieved = {}
            for v in VARIANTS:
                cmd = [args.ncu, "--section", "SchedulerStats", "--csv", "--target-processes", "all",
                       "-k", "higs_blend_bwd", "--launch-skip", "0", "--launch-count", "1",
                       sys.executable, str(probe), "--variant", v, "--source", args.source,
                       "--core-so", args.core_so]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=os.environ.copy())
                achieved[v] = parse_ncu_csv(r.stdout)
                achieved[v]["returncode"] = r.returncode
                if r.returncode != 0:
                    achieved[v]["stderr_head"] = r.stderr[:500]
            occupancy["ncu"] = {"attempted": True, "status": "ran", "achieved": achieved}
        except Exception as e:
            occupancy["ncu"] = {"attempted": True, "status": "failed", "error": str(e)}
    (Path(args.out_dir) / "occupancy.json").write_text(json.dumps(occupancy, indent=2))

    # ===================== 6. analysis.json =====================
    print("\n===== 6. Analysis =====", flush=True)
    # gate evaluation
    # kernel speedup reproducible: >=8% on >=2 scenes, CI95 low > 0
    speedup_by_scene = {}
    for s in scenes:
        d = per_scene_summary.get(s, {})
        speedup_by_scene[s] = d.get("kernel_pct_speedup")
    scenes_ge_8 = [s for s in scenes if (speedup_by_scene[s] or 0) >= 8.0]
    # check CI excludes 0 for those scenes
    ci_excludes_zero = {}
    for r in all_kernel_timing:
        if r["variant"] == "scalar_adjoint":
            ci_excludes_zero[r["scene"]] = (r.get("bootstrap_ci95_low_ms", 0) > 0)
    kernel_gate = (len(scenes_ge_8) >= 2) and all(ci_excludes_zero.get(s, False) for s in scenes_ge_8)

    # F+B / backward gate: F+B >=3% OR backward >=5% (room or bicycle), correctness passing
    fb_gate = False
    bwd_gate = False
    fb_pct = {}
    bwd_pct = {}
    for r in all_fb_timing:
        if r["variant"] == "scalar_adjoint":
            if r["metric"] == "T_F+B":
                fb_pct[r["scene"]] = r.get("pct_speedup_vs_baseline")
            if r["metric"] == "T_backward":
                bwd_pct[r["scene"]] = r.get("pct_speedup_vs_baseline")
    for s in fb_pct:
        if (fb_pct[s] or 0) >= 3.0:
            fb_gate = True
    for s in bwd_pct:
        if (bwd_pct[s] or 0) >= 5.0:
            bwd_gate = True
    practical_gate = (fb_gate or bwd_gate) and correctness_pass_all

    # mechanism falsification
    occ_improves = occ_scalar["blocks_per_sm_active"] > occ_baseline["blocks_per_sm_active"]
    speedup_replicates = kernel_gate
    mechanism = {
        "hypothesis": "scalar adjoint -> smaller live state -> 67->56 registers -> higher occupancy / more latency hiding -> ~11% blend speedup",
        "register_reduction_confirmed": res.get("scalar_adjoint", {}).get("registers_per_thread") == 56 and res.get("baseline", {}).get("registers_per_thread") == 67,
        "theoretical_occupancy_improves": occ_improves,
        "theoretical_occupancy_delta_pct": (occ_scalar["theoretical_occupancy"] - occ_baseline["theoretical_occupancy"]) * 100,
        "kernel_speedup_replicates": speedup_replicates,
        "falsified": (not occ_improves) and (not speedup_replicates),
    }
    if occ_improves and speedup_replicates:
        mechanism["conclusion"] = "consistent_with_hypothesis (occupancy improves AND speedup replicates)"
    elif (not occ_improves) and speedup_replicates:
        mechanism["conclusion"] = "speedup_replicates_without_occupancy_change -> mechanism is reduced arithmetic/instruction count, not occupancy-driven latency hiding"
    elif occ_improves and (not speedup_replicates):
        mechanism["conclusion"] = "occupancy improves but speedup does NOT replicate -> occupancy benefit does not materialize; hypothesis not supported"
    else:
        mechanism["conclusion"] = "falsified (occupancy unchanged AND speedup lost)"

    # final classification
    if correctness_pass_all and kernel_gate and practical_gate:
        classification = "PROMOTE_TO_SHORT_TRAIN"
    elif not correctness_pass_all:
        classification = "KEEP_NEEDS_CORRECTNESS_REPAIR"
    elif not kernel_gate:
        classification = "KEEP_NEEDS_PERF_REPLICATION"
    else:
        classification = "DROP"

    analysis = {
        "run_id": run_id,
        "timestamp": timestamp,
        "correctness_pass_all": correctness_pass_all,
        "forward_state_identical_all": forward_state_identical_all,
        "per_scene_summary": per_scene_summary,
        "kernel_speedup_by_scene_pct": speedup_by_scene,
        "scenes_ge_8pct_speedup": scenes_ge_8,
        "ci95_excludes_zero_by_scene": ci_excludes_zero,
        "kernel_gate_pass": kernel_gate,
        "kernel_gate_definition": ">=8% reproducible blend speedup across >=2 scenes with 95% bootstrap CI excluding 0",
        "fb_pct_by_scene": fb_pct,
        "bwd_pct_by_scene": bwd_pct,
        "fb_gate_pass": fb_gate,
        "bwd_gate_pass": bwd_gate,
        "practical_gate_pass": practical_gate,
        "practical_gate_definition": "(F+B improvement >=3% OR backward improvement >=5%) AND correctness passing",
        "mechanism": mechanism,
        "classification": classification,
        "gates": {
            "correctness": correctness_pass_all,
            "kernel_speedup_reproducible": kernel_gate,
            "production_backward_fb": practical_gate,
        },
        "next_step_if_promoted": "run 800-step room training with identical seed/protocol; check loss/PSNR/N_GS/densification/NaN/walltime",
    }
    (Path(args.out_dir) / "analysis.json").write_text(json.dumps(analysis, indent=2))
    print(f"\n  CLASSIFICATION: {classification}", flush=True)
    print(f"  correctness_pass={correctness_pass_all} kernel_gate={kernel_gate} practical_gate={practical_gate}", flush=True)

    # provenance
    provenance = {
        "run_id": run_id, "timestamp": timestamp, "gpu": args.gpu,
        "gpu_name": torch.cuda.get_device_name(device),
        "scenes": scenes, "cam_idx": args.cam_idx, "max_long_side": args.max_long_side,
        "warmup": args.warmup, "measure": args.measure, "reps": args.reps, "seed": args.seed,
        "binary_sha256": binary_sha, "binary_matches_frozen": binary_matches,
        "B2_base_commit": B2_BASE_COMMIT, "B2_patch_sha256": B2_PATCH_SHA256,
        "H2_patch_sha256": H2_PATCH_SHA256, "variant_selector": "HIGS_BWD_CF_VARIANT",
        "variants": list(VARIANTS), "HIGS_PX_RUNTIME": "2",
        "source": args.source, "core_so": args.core_so,
        "torch": torch.__version__, "cuda": torch.version.cuda,
    }
    (Path(args.out_dir) / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print("\n===== DONE =====", flush=True)


# helper: reference forward (no grad) for F+B target
def rasterize_gaussian_higs_frozen_ref(values, vm, K, w, h, handle):
    from gsplat.experimental import rasterize_gaussian_higs_frozen
    return rasterize_gaussian_higs_frozen(
        *[x.detach() for x in values], backward_mode="higs_native", scene=handle,
        freeze_topology=True, viewmats=vm, Ks=K, width=w, height=h, sh_degree=SH_DEGREE,
        use_higs_culling=True, radius_clip=0.0, tile_sampling_ratio=1.0)


def parse_ncu_csv(csv_text):
    """Parse NCU csv for achieved occupancy + eligible warps. Metric names vary
    by NCU version; collect any matching sm__warps_active / eligible."""
    out = {"achieved_occupancy_pct": None, "eligible_warps_per_cycle": None, "raw_lines": csv_text.count("\n")}
    for line in csv_text.splitlines():
        low = line.lower()
        if "warps_active" in low and ("pct_of_peak" in low or "average" in low):
            try:
                parts = line.split(",")
                out["achieved_occupancy_pct"] = float(parts[-1])
            except Exception:
                pass
        if "eligible" in low and "per_cycle" in low:
            try:
                parts = line.split(",")
                out["eligible_warps_per_cycle"] = float(parts[-1])
            except Exception:
                pass
    return out


_NCU_PROBE_SRC = r'''
import argparse, os, sys
sys.path.insert(0, "/tmp/higs_h2_bwd_cf/source")
import importlib.util, runpy
import torch
ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True)
ap.add_argument("--source", default="/tmp/higs_h2_bwd_cf/source")
ap.add_argument("--core-so", default="/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so")
a = ap.parse_args()
os.environ["HIGS_PX_RUNTIME"] = "2"
os.environ["HIGS_BWD_CF_VARIANT"] = a.variant
spec = importlib.util.spec_from_file_location("gsplat_cuda", a.core_so)
core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
sys.modules["gsplat.csrc"] = core
runpy.run_path(a.source + "/gsplat/experimental/render/kernels/cuda/build.py")["build_and_load_experimental_gaussian_render_inference_scene"]()
from plyfile import PlyData
import numpy as np, json
from pathlib import Path
v = PlyData.read("/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply")["vertex"]
means = torch.tensor(np.column_stack([v["x"],v["y"],v["z"]]), device="cuda", dtype=torch.float32)
quats = torch.tensor(np.column_stack([v[f"rot_{i}"] for i in range(4)]), device="cuda", dtype=torch.float32); quats = quats/quats.norm(dim=-1,keepdim=True).clamp_min(1e-8)
scales = torch.exp(torch.tensor(np.column_stack([v[f"scale_{i}"] for i in range(3)]), device="cuda", dtype=torch.float32))
opac = torch.sigmoid(torch.tensor(v["opacity"], device="cuda", dtype=torch.float32))
sh = torch.zeros((len(v),16,3), device="cuda", dtype=torch.float32); sh[:,0]=torch.tensor(np.column_stack([v[f"f_dc_{i}"] for i in range(3)]), device="cuda", dtype=torch.float32)
rest = torch.stack([torch.tensor(v[f"f_rest_{i}"], device="cuda", dtype=torch.float32) for i in range(45)],1); sh[:,1:]=rest.reshape(len(v),3,15).permute(0,2,1)
cams = json.loads(Path("/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json").read_text()); c=cams[0]
nw,nh=int(c["width"]),int(c["height"]); sc=min(1.0,2048/max(nw,nh)); w,h=int(round(nw*sc)),int(round(nh*sc))
import numpy as np
R=np.asarray(c["rotation"],dtype=np.float32).T; p=np.asarray(c["position"],dtype=np.float32); vm=np.eye(4,dtype=np.float32); vm[:3,:3]=R; vm[:3,3]=-R@p
K=np.array([[float(c["fx"])*w/nw,0,(w-1)/2],[0,float(c["fy"])*w/nw,(h-1)/2],[0,0,1]],dtype=np.float32)
vm_t=torch.tensor(vm,device="cuda")[None,None]; K_t=torch.tensor(K,device="cuda")[None,None]
from gsplat.experimental import rasterize_gaussian_higs_frozen
from gsplat.experimental.render.functional.gaussian_inference import create_higs_renderer, _HIGS_FROZEN_TRACKER
for _ in range(3):
    leaves=tuple(x.detach().clone().requires_grad_(True) for x in (means,quats,scales,opac,sh))
    _HIGS_FROZEN_TRACKER.reset(); hd=create_higs_renderer(*leaves, sh_degree=3)
    out=rasterize_gaussian_higs_frozen(*leaves, backward_mode="higs_native", scene=hd, freeze_topology=True, viewmats=vm_t, Ks=K_t, width=w, height=h, sh_degree=3, use_higs_culling=True, radius_clip=0.0, tile_sampling_ratio=1.0)
    (out["frame"].float().sum()+out["alpha"].float().sum()).backward(); hd.release()
torch.cuda.synchronize()
'''


if __name__ == "__main__":
    main()
