#!/usr/bin/env python3
"""P18-P0-R1 trace capture: REAL contiguous 16-px subgroups from the frozen C0
macro-tile rasterizer (TILE_SIZE=16), emitted by kernel instrumentation.

Modes (separate processes: only one ext .so can be dlopened per process):
  frozen : frozen c0_ext .so (no instrumentation code at all) -> render hashes
  off    : trace .so, p18_trace_set(null)                    -> render hashes
  on     : trace .so, p18_trace_set(real buffers)            -> records + hist + hashes

All modes render the same probe cameras per scene (P18_CAMERA_SET.json) with the
ckpt_16000 diagnostic model — same scene workload as P18-P0.

Record layout (32 B, written by the rasterize kernel, one per real 16-px
row segment = (warp, gaussian, pp, row-half, row-parity)):
  [0] gaussian id                [1] macro-tile batch idx
  [2] x0 | (y0 << 16)            [3] queue idx | (mask16 << 16)
  [4] means2d.x (fp32 bits)      [5] means2d.y (fp32 bits)
  [6] conic halves 0,1 (raw)     [7] conic halves 2,3 (raw)

Subgroup pixels are x0..x0+15 (16 contiguous); x[j+1]-x[j]==1 is asserted
host-side on the materialized x_pixels array.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = "/root/3dgs-renderer-benchmark"
C0_WT = "/root/epic05_frozen/higs_c0_final30k_worktree"
SCENE_CUDA_DIR = "/root/epic05_build_cache/scene/gsplat_scene_cuda"
PHASE7 = os.path.join(REPO, "scripts", "epic05", "phase7")
SRC = os.path.join(REPO, "src")
REF_V1 = os.path.join(REPO, "baseline", "reference_v1")
CORE_SO = "/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so"
FROZEN_EXT = ("/root/epic05_build_cache/c0_ext/"
              "experimental_gaussian_render_inference_scene_cuda/"
              "experimental_gaussian_render_inference_scene_cuda.so")
TRACE_EXT = ("/root/epic05_p18_r1/build_trace/"
             "experimental_gaussian_render_inference_scene_cuda.so")
CKPT = {"room": "/root/epic05_p13x/runs/room_diag/ckpt_16000.pt",
        "bicycle": "/root/epic05_p13x/runs/bicycle_diag/ckpt_16000.pt"}
CAM_SET_PATH = "/root/epic05_p18/P18_CAMERA_SET.json"
OUT = Path(os.environ.get("P18_R1_OUT", "/root/epic05_p18_r1/artifacts"))

TILE = 16
CTA_WARPS = 10          # RASTERIZE_CTA_SIZE=320 / 32
MAX_CTAS = 65536
LOCAL_CAP = 2
IDX_GATE = 255
N_REC_SLOTS = MAX_CTAS * CTA_WARPS * LOCAL_CAP * 16
DEV = "cuda"


def log(msg):
    print(f"[p18r1 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ext_module(name, so_path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


def bootstrap(ext_so):
    os.environ["HIGS_PX_RUNTIME"] = "2"
    os.environ["HIGS_BWD_SCALAR_ADJOINT"] = "scalar_adjoint"
    os.environ["HIGS_BWD_H8_MR"] = "1"
    os.environ["HIGS_BWD_ABSGRAD"] = "1"
    os.environ.pop("HIGS_DISABLE_F9", None)
    for p in (C0_WT, SCENE_CUDA_DIR, PHASE7, SRC, REF_V1):
        if p not in sys.path:
            sys.path.insert(0, p)
    core = load_ext_module("gsplat_cuda", CORE_SO)
    sys.modules["gsplat.csrc"] = core
    exp = load_ext_module("experimental_gaussian_render_inference_scene_cuda", ext_so)
    sys.modules["gsplat.experimental.render.kernels.csrc"] = exp
    sys.modules["experimental_gaussian_render_inference_scene_cuda"] = exp
    return exp


def load_model(scene):
    from config import ReferenceV1Config
    from gaussian_model import GaussianModel
    cfg = ReferenceV1Config()
    cfg.scene = scene
    model = GaussianModel(max_sh_degree=cfg.sh_degree)
    ckpt = torch.load(CKPT[scene], map_location="cpu", weights_only=False)
    model.restore(ckpt, {
        "position_lr_init": cfg.position_lr_init,
        "position_lr_final": cfg.position_lr_final,
        "position_lr_delay_mult": cfg.position_lr_delay_mult,
        "position_lr_max_steps": cfg.position_lr_max_steps,
        "feature_lr": cfg.feature_lr,
        "opacity_lr": cfg.opacity_lr,
        "scaling_lr": cfg.scaling_lr,
        "rotation_lr": cfg.rotation_lr,
        "percent_dense": cfg.percent_dense})
    return model


def pack_scene(model, scene):
    from gsplat.scene.components.gaussian_inference_scene import GaussianInferenceScene
    with torch.no_grad():
        obj = GaussianInferenceScene.from_gaussian_tensors(
            means=model._xyz.detach(),
            quats=F.normalize(model._rotation.detach(), dim=-1),
            scales=torch.exp(model._scaling.detach()),
            opacities=torch.sigmoid(model._opacity.detach()).squeeze(-1),
            colors=model.get_features.detach(),
            sh_degree=model.get_sh_degree,
            sh_compression="none",
            id=f"p18r1_{scene}")
    return obj


def render_cam(obj, cam_rec):
    from gsplat.experimental.render.kernels.gaussian_inference_ops import (
        gaussian_render_inference_only)
    vm = torch.as_tensor(np.asarray(cam_rec["viewmatrix"], dtype=np.float32), device=DEV)
    K = torch.as_tensor(np.asarray(cam_rec["K"], dtype=np.float32), device=DEV)
    W = int(cam_rec["width"])
    H = int(cam_rec["height"])
    sh_comp = int(obj.sh_compression_mode) if obj.sh_compression_mode is not None else 0
    with torch.no_grad():
        renders, alphas = gaussian_render_inference_only(
            obj.means_planar, obj.qso_packed, obj.colors_packed,
            vm, K, W, H, obj.sh_degree, TILE,
            0.01, 1e10, 0.0, 0.3, sh_comp, None)
    return renders, alphas, W, H


def hash_tensor(t):
    arr = t.detach().to("cpu").contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest(), tuple(arr.shape), str(arr.dtype)


def extract_records(rec):
    """GPU-side compaction of non-empty records; returns (sel[n,8] int32 cpu, flat[n] int64 cpu)."""
    flags = rec[:, 0] != -1
    flat = torch.nonzero(flags).squeeze(1)
    sel = rec[flat]
    return sel.to("cpu"), flat.to("cpu")


def decode_records(sel, flat, cam_id):
    u = sel.numpy().astype(np.uint32)
    flat_np = flat.numpy().astype(np.int64)
    r = (flat_np % 16).astype(np.int32)
    grp = flat_np // 16
    cta = (grp // (CTA_WARPS * LOCAL_CAP)).astype(np.int32)
    rem = grp % (CTA_WARPS * LOCAL_CAP)
    warp = (rem // LOCAL_CAP).astype(np.int32)
    slot = (rem % LOCAL_CAP).astype(np.int32)
    out = {
        "cam": np.full(u.shape[0], cam_id, dtype=np.int32),
        "g": u[:, 0].astype(np.int64),
        "mt_batch": u[:, 1].astype(np.int32),
        "x0": (u[:, 2] & 0xFFFF).astype(np.int32),
        "y0": (u[:, 2] >> 16).astype(np.int32),
        "qidx": (u[:, 3] & 0xFFFF).astype(np.int32),
        "mask16": (u[:, 3] >> 16).astype(np.int32),
        "pos_x": u[:, 4].copy().view(np.float32),
        "pos_y": u[:, 5].copy().view(np.float32),
        "conic01_raw": u[:, 6],
        "conic23_raw": u[:, 7],
        "rec_r": r,
        "cta": cta,
        "warp": warp,
        "slot": slot,
    }
    c01 = u[:, 6].copy().view(np.uint16).reshape(-1, 2)
    c23 = u[:, 7].copy().view(np.uint16).reshape(-1, 2)
    conic_h = np.concatenate([c01, c23], axis=1)           # [n,4]: l0,l1,l2,opacity
    out["conic_h4"] = conic_h.astype(np.uint16)
    # bit-reinterpretation (NOT value conversion: astype(u16->f16) would convert values)
    cflt = conic_h.view(np.float16).astype(np.float32)
    out["c_l0"], out["c_l1"], out["c_l2"], out["c_opacity"] = (cflt[:, 0], cflt[:, 1],
                                                               cflt[:, 2], cflt[:, 3])
    return out


def assert_records(d, tag):
    """Auto-asserts for requirement 1: contiguous x runs + row-consistency."""
    n = d["x0"].shape[0]
    assert n == d["y0"].shape[0] == d["rec_r"].shape[0]
    assert np.all(d["x0"] % 16 == 0), "x0 must be a tile-column origin"
    pp = d["rec_r"] >> 2
    h = (d["rec_r"] >> 1) & 1
    ty = d["rec_r"] & 1
    row_off = ty + 4 * pp + 2 * h
    assert np.array_equal((d["y0"] % 16).astype(np.int32), row_off.astype(np.int32)), \
        "y0%16 must equal the row offset implied by the record lane"
    x_pixels = d["x0"][:, None] + np.arange(16, dtype=np.int32)[None, :]
    assert np.all(np.diff(x_pixels, axis=1) == 1), "x[j+1]-x[j] must be 1"
    assert np.all(x_pixels >= 0)
    log(f"[assert {tag}] n={n} contiguity+row-consistency PASS")
    return x_pixels


def mode_frozen_off(ext_so, mode, scenes, cams_override):
    exp = bootstrap(ext_so)
    hashes = {}
    if mode == "off":
        exp.p18_trace_set(0, 0, 0, 0, IDX_GATE)
        log("trace ctx set to null (off)")
    camset = json.load(open(CAM_SET_PATH))
    for scene in scenes:
        model = load_model(scene)
        obj = pack_scene(model, scene)
        scores = camset["scenes"][scene]
        ids = cams_override or scores["probe_ids"]
        hashes[scene] = {}
        for ci in ids:
            crec = scores["camera_records"][str(ci)]
            renders, alphas, W, H = render_cam(obj, crec)
            torch.cuda.synchronize()
            h, shape, dtype = hash_tensor(renders)
            ha, sha, da = hash_tensor(alphas)
            hashes[scene][str(ci)] = {"renders": h, "shape": list(shape), "dtype": dtype,
                                      "alphas": ha, "a_shape": list(sha), "a_dtype": da}
            log(f"[{mode}] {scene} cam{ci}: renders[{shape}] {dtype} sha={h[:16]}")
        del model, obj
        torch.cuda.empty_cache()
    outp = OUT / f"P18_R1_RENDER_HASH_{mode}.json"
    outp.write_text(json.dumps({
        "phase": "P18-P0-R1", "artifact": f"P18_R1_RENDER_HASH_{mode}", "mode": mode,
        "ext_so": ext_so, "ext_so_sha256": sha256_file(ext_so),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hashes": hashes}, indent=2))
    log(f"[{mode}] wrote {outp}")


def mode_on(scenes, cams_override, keep_all=True):
    exp = bootstrap(TRACE_EXT)
    rec = torch.full((N_REC_SLOTS, 8), -1, dtype=torch.int32, device=DEV)
    hist = torch.zeros(17, dtype=torch.int32, device=DEV)
    exp.p18_trace_set(rec.data_ptr(), hist.data_ptr(), MAX_CTAS, LOCAL_CAP, IDX_GATE)
    torch.cuda.synchronize()
    log(f"trace ctx armed: max_ctas={MAX_CTAS} local_cap={LOCAL_CAP} idx_gate={IDX_GATE} "
        f"slots={N_REC_SLOTS} ({N_REC_SLOTS * 32 / (1 << 20):.0f} MiB)")

    camset = json.load(open(CAM_SET_PATH))
    render_hashes = {}
    manifest = {
        "phase": "P18-P0-R1",
        "artifact": "P18_R1_TRACE_MANIFEST",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "primary_trace_source": (
            "real contiguous 16-px row segments emitted by macro_tile_rasterize_kernel "
            "(TILE_SIZE=16) phase-2 inner loop; one record per (warp, gaussian, pp, row-half, "
            "row-parity) with the kernel's own ok_mask ballot; NO footprint sampling"),
        "ext_so": TRACE_EXT,
        "ext_so_sha256": sha256_file(TRACE_EXT),
        "src_sha256": {f: sha256_file(f) for f in (
            "/root/epic05_p18_r1/csrc_trace/MacroTileRasterize.cu",
            "/root/epic05_p18_r1/csrc_trace/ext.cpp")},
        "record_layout": ["g", "mt_batch", "x0|y0<<16", "qidx|mask16<<16",
                          "means2d.x_bits", "means2d.y_bits", "conic_h01", "conic_h23"],
        "coordinate_units": ("x0/y0 are global pixel coordinates; subgroup pixels are "
                             "x0..x0+15 (contiguous), pixel centers (x+0.5, y+0.5); "
                             "x[j+1]-x[j]==1 asserted host-side"),
        "capture_params": {"max_ctas": MAX_CTAS, "local_cap": LOCAL_CAP, "idx_gate": IDX_GATE},
        "scenes": {},
    }

    for scene in scenes:
        model = load_model(scene)
        obj = pack_scene(model, scene)
        scores = camset["scenes"][scene]
        ids = cams_override or scores["probe_ids"]
        all_d = []
        hist_total = np.zeros(17, dtype=np.int64)
        per_cam = {}
        render_hashes[scene] = {}
        for ci in ids:
            crec = scores["camera_records"][str(ci)]
            W, H = int(crec["width"]), int(crec["height"])
            rec.fill_(-1)
            hist.zero_()
            torch.cuda.synchronize()
            renders, alphas, W, H = render_cam(obj, crec)
            torch.cuda.synchronize()
            h, shape, dtype = hash_tensor(renders)
            render_hashes[scene][str(ci)] = {"renders": h, "shape": list(shape), "dtype": dtype}
            sel, flat = extract_records(rec)
            hh = hist.to("cpu").numpy().astype(np.int64)
            hist_total += hh
            d = decode_records(sel, flat, ci)
            assert_records(d, f"{scene}/cam{ci}")
            x_pixels = d["x0"][:, None] + np.arange(16, dtype=np.int32)[None, :]
            assert np.all(x_pixels[:, 15] < W), f"x0+15 must stay inside width {W}"
            assert np.all(d["y0"] < ((H + 15) // 16) * 16), f"y0 inside padded height {H}"
            n = d["x0"].shape[0]
            per_cam[str(ci)] = {
                "n_records": int(n),
                "n_calls": int(n // 16),
                "n_ctas_used": int(np.unique(d["cta"]).size),
                "max_cta": int(d["cta"].max()) if n else -1,
                "hist17": hh.tolist(),
                "render_sha256": h,
                "shape": list(shape), "dtype": dtype,
            }
            log(f"[on] {scene} cam{ci}: records={n} calls={n // 16} "
                f"ctas={per_cam[str(ci)]['n_ctas_used']} max_cta={per_cam[str(ci)]['max_cta']} "
                f"hist={hh.tolist()} sha={h[:16]}")
            all_d.append(d)
        keys = list(all_d[0].keys())
        merged = {k: np.concatenate([d[k] for d in all_d], axis=0) for k in keys}
        n_total = merged["x0"].shape[0]
        x_pixels = merged["x0"][:, None] + np.arange(16, dtype=np.int32)[None, :]
        npz_path = OUT / f"P18_R1_TRACE_{scene}.npz"
        np.savez_compressed(npz_path, x_pixels=x_pixels, **merged)
        pop = np.array([bin(int(m)).count("1") for m in merged["mask16"]], dtype=np.int32)
        hist_pos = hist_total[1:]
        mean_pop = float((hist_pos * np.arange(1, 17)).sum() / max(hist_pos.sum(), 1))
        csum = np.cumsum(hist_pos)
        median_pop = int(np.searchsorted(csum, csum[-1] / 2.0) + 1)
        scene_entry = {
            "n_primary_records": int(n_total),
            "n_primary_calls": int(n_total // 16),
            "n_primary_records_requirement": "n_primary_records >= 8192",
            "n_primary_records_pass": bool(n_total >= 8192),
            "cams": [int(c) for c in ids],
            "per_cam": per_cam,
            "live_pop_hist17": hist_total.tolist(),
            "live_pop_mean": mean_pop,
            "live_pop_median": median_pop,
            "live_pop_hist_source": ("full-population smem histogram; every phase-2 inner-loop "
                                     "call contributes 16 real row segments (no sampling)"),
            "pop_record_check_hist17": np.bincount(pop, minlength=17)[:17].tolist(),
            "frac_pop0": float(hist_total[0] / max(hist_total.sum(), 1)),
            "frac_pop16": float(hist_total[16] / max(hist_total.sum(), 1)),
            "npz": str(npz_path),
            "npz_sha256": sha256_file(npz_path),
            "n_ctas_used": int(np.unique(merged["cta"]).size),
            "max_cta": int(merged["cta"].max()),
        }
        manifest["scenes"][scene] = scene_entry
        log(f"{scene}: n_primary_records={n_total} (calls={n_total // 16}) "
            f"hist={hist_total.tolist()} mean_pop={mean_pop:.3f} median={median_pop}")
        del model, obj, all_d, merged
        torch.cuda.empty_cache()

    manifest["n_primary_records_min"] = int(min(
        s["n_primary_records"] for s in manifest["scenes"].values()))
    manifest["gate"] = "n_primary_records >= 8192 per scene"
    outp = OUT / "P18_R1_TRACE_MANIFEST.json"
    outp.write_text(json.dumps(manifest, indent=2))
    hp = OUT / "P18_R1_RENDER_HASH_on.json"
    hp.write_text(json.dumps({
        "phase": "P18-P0-R1", "artifact": "P18_R1_RENDER_HASH_on", "mode": "on",
        "ext_so": TRACE_EXT, "ext_so_sha256": sha256_file(TRACE_EXT),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hashes": render_hashes}, indent=2))
    log(f"[on] wrote {outp} and {hp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["frozen", "off", "on"])
    ap.add_argument("--scenes", default="room,bicycle")
    ap.add_argument("--cams", default="", help="comma-separated probe cam ids (debug)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = [s for s in args.scenes.split(",") if s]
    cams = [int(c) for c in args.cams.split(",") if c] or None
    log(f"mode={args.mode} scenes={scenes} cams={cams} out={OUT}")
    if args.mode == "frozen":
        mode_frozen_off(FROZEN_EXT, "frozen", scenes, cams)
    elif args.mode == "off":
        mode_frozen_off(TRACE_EXT, "off", scenes, cams)
    else:
        mode_on(scenes, cams)


if __name__ == "__main__":
    main()
