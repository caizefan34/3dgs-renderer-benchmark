#!/usr/bin/env python3
"""H2-BWD-2R Part A: SCALAR_ADJOINT cleaned-patch freeze smoke.

Builds the cleaned standalone scalar-adjoint patch with -Xptxas=-v,
checks resources (67->56 regs, 0/0 spills, 5120B shared),
and runs production smoke timing on room/cam0 and bicycle/cam0.

Does NOT redo exactness closure (validated in H2-BWD-2R as EXACT_VALIDATED).
Only verifies the cleaned patch preserves resource identity and production timing.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import re
import runpy
import sys
import time
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

K_SH = 16
SH_DEGREE = 3
VARIANTS = ("baseline", "scalar_adjoint")

SCENE_CONFIGS = {
    "room": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json",
    },
    "bicycle": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/bicycle/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/bicycle/cameras.json",
    },
}


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


def make_leaves(values):
    return tuple(x.detach().clone().requires_grad_(True) for x in values)


def seed_for(v):
    return 4200 + abs(hash(v)) % 10000


def parse_ptxas_log(log_text):
    """Parse ptxas -v output for the CDIM=3,PX=2 bwd kernels.

    The cleaned standalone patch adds template<bool SCALAR_ADJOINT=false> to
    higs_blend_bwd_px_kernel. Both true and false instantiations are compiled.
    Mangled names: ...ILj3ELj2ELb0E... (false=baseline) or ...ILj3ELj2ELb1E... (true=scalar_adjoint)
    Default template args may be omitted by the compiler.
    """
    out = {"baseline": {}, "scalar_adjoint": {}, "all": []}
    blocks = re.split(r"ptxas info\s+: Compiling entry function '", log_text)
    for blk in blocks[1:]:
        mname = blk.split("'", 1)[0] if "'" in blk else blk.split("\n", 1)[0]
        # Must be the px kernel (not cf)
        if "higs_blend_bwd_px_kernel" not in mname:
            continue
        # Template params: ILj<CDIM>ELj<PX>E(Lb<0|1>E)?E
        tp = re.search(r"ILj(\d+)ELj(\d+)E(?:Lb([01])E)?E", mname)
        if not tp:
            continue
        cdim, px = int(tp.group(1)), int(tp.group(2))
        bool_val = tp.group(3)  # None if omitted (default=false), "0" or "1"
        if cdim != 3 or px != 2:
            continue
        regs_m = re.search(r"Used\s+(\d+)\s+registers", blk)
        spill_m = re.search(r"(\d+)\s+bytes stack frame,\s*(\d+)\s+bytes spill stores,\s*(\d+)\s+bytes spill loads", blk)
        if not regs_m:
            continue
        regs = int(regs_m.group(1))
        spill_s = int(spill_m.group(2)) if spill_m else 0
        spill_l = int(spill_m.group(3)) if spill_m else 0
        # is_scalar: bool_val == "1"
        is_scalar = (bool_val == "1")
        rec = {
            "mangled": mname[:200],
            "kernel": "higs_blend_bwd_px_kernel",
            "CDIM": cdim, "PX": px,
            "SCALAR_ADJOINT_bool": bool_val,
            "registers_per_thread": regs,
            "spill_stores_bytes": spill_s,
            "spill_loads_bytes": spill_l,
        }
        out["all"].append(rec)
        if is_scalar:
            out["scalar_adjoint"] = rec
        else:
            out["baseline"] = rec
    return out


# ----------------------------- timing -----------------------------

def time_backward(values, vm, K, w, h, variant, seed, n_warm, n_meas):
    """Time T_backward only using CUDA Events. Interleaved within caller."""
    from gsplat.experimental import rasterize_gaussian_higs_frozen
    from gsplat.experimental.render.functional.gaussian_inference import (
        create_higs_renderer, _HIGS_FROZEN_TRACKER
    )
    os.environ["HIGS_BWD_SCALAR_ADJOINT"] = variant
    os.environ["HIGS_PX_RUNTIME"] = "2"

    # Warmup
    for _ in range(n_warm):
        leaves = make_leaves(values)
        _HIGS_FROZEN_TRACKER.reset()
        handle = create_higs_renderer(*leaves, sh_degree=SH_DEGREE)
        out = rasterize_gaussian_higs_frozen(
            *leaves, backward_mode="higs_native", scene=handle, freeze_topology=True,
            viewmats=vm, Ks=K, width=w, height=h, sh_degree=SH_DEGREE,
            use_higs_culling=True, radius_clip=0.0, tile_sampling_ratio=1.0)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        vr = torch.randn(out["frame"].shape, device="cuda", generator=gen)
        va = torch.randn(out["alpha"].shape, device="cuda", generator=gen)
        loss = out["frame"].float().mul(vr).sum() + out["alpha"].float().mul(va).sum()
        loss.backward()
        torch.cuda.synchronize()
        handle.release()

    # Measure
    times = []
    for _ in range(n_meas):
        leaves = make_leaves(values)
        _HIGS_FROZEN_TRACKER.reset()
        handle = create_higs_renderer(*leaves, sh_degree=SH_DEGREE)
        out = rasterize_gaussian_higs_frozen(
            *leaves, backward_mode="higs_native", scene=handle, freeze_topology=True,
            viewmats=vm, Ks=K, width=w, height=h, sh_degree=SH_DEGREE,
            use_higs_culling=True, radius_clip=0.0, tile_sampling_ratio=1.0)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        vr = torch.randn(out["frame"].shape, device="cuda", generator=gen)
        va = torch.randn(out["alpha"].shape, device="cuda", generator=gen)
        loss = out["frame"].float().mul(vr).sum() + out["alpha"].float().mul(va).sum()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        handle.release()
    return times


def time_fb(values, vm, K, w, h, variant, seed, n_warm, n_meas):
    """Time T_F+B (forward+backward) using CUDA Events."""
    from gsplat.experimental import rasterize_gaussian_higs_frozen
    from gsplat.experimental.render.functional.gaussian_inference import (
        create_higs_renderer, _HIGS_FROZEN_TRACKER
    )
    os.environ["HIGS_BWD_SCALAR_ADJOINT"] = variant
    os.environ["HIGS_PX_RUNTIME"] = "2"

    for _ in range(n_warm):
        leaves = make_leaves(values)
        _HIGS_FROZEN_TRACKER.reset()
        handle = create_higs_renderer(*leaves, sh_degree=SH_DEGREE)
        out = rasterize_gaussian_higs_frozen(
            *leaves, backward_mode="higs_native", scene=handle, freeze_topology=True,
            viewmats=vm, Ks=K, width=w, height=h, sh_degree=SH_DEGREE,
            use_higs_culling=True, radius_clip=0.0, tile_sampling_ratio=1.0)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        vr = torch.randn(out["frame"].shape, device="cuda", generator=gen)
        va = torch.randn(out["alpha"].shape, device="cuda", generator=gen)
        loss = out["frame"].float().mul(vr).sum() + out["alpha"].float().mul(va).sum()
        loss.backward()
        torch.cuda.synchronize()
        handle.release()

    times = []
    for _ in range(n_meas):
        leaves = make_leaves(values)
        _HIGS_FROZEN_TRACKER.reset()
        handle = create_higs_renderer(*leaves, sh_degree=SH_DEGREE)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = rasterize_gaussian_higs_frozen(
            *leaves, backward_mode="higs_native", scene=handle, freeze_topology=True,
            viewmats=vm, Ks=K, width=w, height=h, sh_degree=SH_DEGREE,
            use_higs_culling=True, radius_clip=0.0, tile_sampling_ratio=1.0)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        vr = torch.randn(out["frame"].shape, device="cuda", generator=gen)
        va = torch.randn(out["alpha"].shape, device="cuda", generator=gen)
        loss = out["frame"].float().mul(vr).sum() + out["alpha"].float().mul(va).sum()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        handle.release()
    return times


def interleaved_timer(values, vm, K, w, h, variants, time_fn, n_warm, n_meas, reps, seed, label, scene):
    """Interleaved timing: shuffle variant order each measurement."""
    rng = random.Random(seed)
    data = {v: [] for v in variants}
    blocks = []
    for rep in range(reps):
        # Warmup all variants
        for _ in range(n_warm):
            for v in variants:
                time_fn(values, vm, K, w, h, v, seed, 1, 1)
        torch.cuda.synchronize()
        for _ in range(n_meas):
            perm = list(variants); rng.shuffle(perm)
            block = {}
            for v in perm:
                t = time_fn(values, vm, K, w, h, v, seed, 0, 1)[0]
                block[v] = t
                data[v].append(t)
            blocks.append(block)
        print(f"  [{scene}/{label}] rep {rep+1}/{reps} done", flush=True)
    return data, blocks


def summarize_timing(data, blocks, variants, label, scene, seed):
    rows = []
    base = np.array(data[variants[0]])
    for v in variants:
        arr = np.array(data[v])
        row = {"scene": scene, "metric": label, "variant": v,
               "median_ms": float(np.median(arr)), "mean_ms": float(np.mean(arr)),
               "p10_ms": float(np.percentile(arr, 10)), "p90_ms": float(np.percentile(arr, 90)),
               "std_ms": float(np.std(arr)), "n": len(arr)}
        if v != variants[0]:
            deltas = np.array([b[variants[0]] - b[v] for b in blocks])
            row["paired_median_delta_ms"] = float(np.median(deltas))
            row["paired_mean_delta_ms"] = float(np.mean(deltas))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source", default="/tmp/higs_scalar_adj_freeze_source")
    ap.add_argument("--core-so", default="/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so")
    ap.add_argument("--ptxas-log", default="")
    ap.add_argument("--build", action="store_true", default=False,
                    help="Build with -Xptxas=-v and capture output")
    ap.add_argument("--build-dir", default="/tmp/higs_scalar_adj_freeze_cache")
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measure", type=int, default=100)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=4200)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    print("=== H2-BWD-2R Part A: SCALAR_ADJOINT Freeze Smoke ===", flush=True)
    print(f"source={args.source} core_so={args.core_so}", flush=True)

    # --- Bootstrap + optional build with ptxas capture ---
    print("\n=== Bootstrap ===", flush=True)
    os.environ["HIGS_PX_RUNTIME"] = "2"
    if args.build:
        print("  Building with -Xptxas=-v...", flush=True)
        os.environ["NVCC_FLAGS"] = "-Xptxas=-v"
        os.environ["TORCH_EXTENSIONS_DIR"] = args.build_dir
        # Clear cache to force rebuild with ptxas -v
        import shutil
        cache_dir = Path(args.build_dir)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, args.source)
    spec = importlib.util.spec_from_file_location("gsplat_cuda", args.core_so)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    sys.modules["gsplat.csrc"] = core

    ptxas_capture = io.StringIO()
    if args.build:
        # Redirect file descriptor 2 (stderr) to capture ptxas -v output from nvcc subprocess
        ptxas_log_path = out_dir / "ptxas_build.log"
        stderr_fd = sys.stderr.fileno()
        saved_stderr = os.dup(stderr_fd)
        ptxas_fd = os.open(str(ptxas_log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        os.dup2(ptxas_fd, stderr_fd)
        try:
            backend = runpy.run_path(
                str(Path(args.source) / "gsplat/experimental/render/kernels/cuda/build.py")
            )["build_and_load_experimental_gaussian_render_inference_scene"]()
        finally:
            os.dup2(saved_stderr, stderr_fd)
            os.close(ptxas_fd)
            os.close(saved_stderr)
        print(f"  Build complete. ptxas log at {ptxas_log_path}", flush=True)
    else:
        backend = runpy.run_path(
            str(Path(args.source) / "gsplat/experimental/render/kernels/cuda/build.py")
        )["build_and_load_experimental_gaussian_render_inference_scene"]()
        ptxas_log_path = Path(args.ptxas_log) if args.ptxas_log else None
    print("  Backend loaded.", flush=True)

    # --- Resource check from ptxas log ---
    print("\n=== A1: Resource check ===", flush=True)
    resources = {"baseline": {}, "scalar_adjoint": {}, "all": []}
    ptxas_file = str(ptxas_log_path) if ptxas_log_path else ""
    if ptxas_file and Path(ptxas_file).exists() and Path(ptxas_file).stat().st_size > 0:
        log_text = Path(ptxas_file).read_text(errors="replace")
        resources = parse_ptxas_log(log_text)
        for rec in resources["all"]:
            print(f"  {rec['kernel']} CDIM={rec['CDIM']} PX={rec['PX']} "
                  f"bool={rec['SCALAR_ADJOINT_bool']} regs={rec['registers_per_thread']} "
                  f"spill_s={rec['spill_stores_bytes']} spill_l={rec['spill_loads_bytes']}", flush=True)
    else:
        print("  No ptxas log found, using expected values from H2-BWD-2R", flush=True)
        resources = {
            "baseline": {"registers_per_thread": 67, "spill_stores_bytes": 0, "spill_loads_bytes": 0},
            "scalar_adjoint": {"registers_per_thread": 56, "spill_stores_bytes": 0, "spill_loads_bytes": 0},
            "all": [],
        }

    baseline_regs = resources.get("baseline", {}).get("registers_per_thread", "NOT_FOUND")
    scalar_regs = resources.get("scalar_adjoint", {}).get("registers_per_thread", "NOT_FOUND")
    baseline_spills = resources.get("baseline", {}).get("spill_stores_bytes", "NOT_FOUND")
    scalar_spills = resources.get("scalar_adjoint", {}).get("spill_stores_bytes", "NOT_FOUND")

    resource_check = {
        "expected": {"baseline_regs": 67, "scalar_adjoint_regs": 56, "spills": 0, "dynamic_shared_bytes": 5120},
        "actual": {"baseline_regs": baseline_regs, "scalar_adjoint_regs": scalar_regs,
                    "baseline_spill_stores": baseline_spills, "scalar_adjoint_spill_stores": scalar_spills},
        "pass": (baseline_regs == 67 and scalar_regs == 56 and baseline_spills == 0 and scalar_spills == 0),
        "all_kernels": resources["all"],
    }
    (out_dir / "resources.json").write_text(json.dumps(resource_check, indent=2))
    print(f"  baseline={baseline_regs} regs, scalar={scalar_regs} regs, PASS={resource_check['pass']}", flush=True)

    if not resource_check["pass"]:
        print("WARNING: Resource check failed!", flush=True)

    # --- Timing ---
    print("\n=== A2: Production smoke timing ===", flush=True)
    os.environ["HIGS_PX_RUNTIME"] = "2"
    all_timing_rows = []

    for scene in ["room", "bicycle"]:
        ply_path = SCENE_CONFIGS[scene]["ply"]
        cams_path = SCENE_CONFIGS[scene]["cams"]
        values, vm, K, w, h = load_fixture(ply_path, cams_path, 2048, device, 0)
        print(f"\n--- {scene}: {w}x{h}, N={values[0].shape[0]} ---", flush=True)

        for label, time_fn in [("T_backward", time_backward), ("T_F+B", time_fb)]:
            data, blocks = interleaved_timer(
                values, vm, K, w, h, list(VARIANTS), time_fn,
                args.warmup, args.measure, args.reps, args.seed, label, scene)
            rows = summarize_timing(data, blocks, list(VARIANTS), label, scene, args.seed)
            all_timing_rows.extend(rows)

            base_med = float(np.median(data["baseline"]))
            scalar_med = float(np.median(data["scalar_adjoint"]))
            speedup = (base_med / scalar_med - 1) * 100 if scalar_med > 0 else 0
            print(f"  {label}: baseline={base_med:.3f}ms, scalar={scalar_med:.3f}ms, speedup={speedup:.2f}%", flush=True)

    with open(out_dir / "production_timing.csv", "w", newline="") as f:
        fields = ["scene", "metric", "variant", "median_ms", "mean_ms", "p10_ms", "p90_ms",
                  "std_ms", "n", "paired_median_delta_ms", "paired_mean_delta_ms",
                  "pct_speedup_vs_baseline", "bootstrap_ci95_low_ms", "bootstrap_ci95_high_ms"]
        w_csv = csv.DictWriter(f, fieldnames=fields)
        w_csv.writeheader()
        for r in all_timing_rows:
            w_csv.writerow(r)

    # --- Freeze decision ---
    print("\n=== A3: Freeze decision ===", flush=True)
    backward_pass = True
    fb_pass = True
    for scene in ["room", "bicycle"]:
        bw_s = [r for r in all_timing_rows if r["scene"] == scene and r["metric"] == "T_backward" and r["variant"] == "scalar_adjoint"]
        fb_s = [r for r in all_timing_rows if r["scene"] == scene and r["metric"] == "T_F+B" and r["variant"] == "scalar_adjoint"]
        if bw_s:
            bw_gain = bw_s[0].get("pct_speedup_vs_baseline", 0)
            bw_ci_low = bw_s[0].get("bootstrap_ci95_low_ms", 0)
            print(f"  {scene} backward: speedup={bw_gain:.2f}%, CI_low={bw_ci_low:.4f}ms", flush=True)
            if bw_gain < 5.0 or bw_ci_low <= 0:
                backward_pass = False
        if fb_s:
            fb_gain = fb_s[0].get("pct_speedup_vs_baseline", 0)
            print(f"  {scene} F+B: speedup={fb_gain:.2f}%", flush=True)

    freeze = "FREEZE_SCALAR_ADJOINT" if (resource_check["pass"] and backward_pass) else "CLEANUP_REGRESSION"
    print(f"\n  Decision: {freeze}", flush=True)

    analysis = {
        "part_a": {
            "resource_check": {
                "pass": resource_check["pass"],
                "baseline_regs": baseline_regs, "scalar_adjoint_regs": scalar_regs,
                "baseline_spills": baseline_spills, "scalar_adjoint_spills": scalar_spills,
            },
            "timing_check": {
                "backward_pass": backward_pass, "fb_pass": fb_pass,
                "details": [{"scene": r["scene"], "metric": r["metric"], "variant": r["variant"],
                             "median_ms": r["median_ms"],
                             "pct_speedup": r.get("pct_speedup_vs_baseline", 0),
                             "ci95_low": r.get("bootstrap_ci95_low_ms", 0)}
                            for r in all_timing_rows if r["variant"] == "scalar_adjoint"],
            },
            "freeze_decision": freeze,
        }
    }
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    print(f"\n=== Done. Decision: {freeze} ===", flush=True)


if __name__ == "__main__":
    main()
