#!/usr/bin/env python3
"""H2-1 Trainable HiGS B2 Bottleneck Decomposition Profiler.

Profiles B2 (rasterize_gaussian_higs_frozen) on room/cam0 with:
  - Direct forward timing (requires_grad=True, no backward)
  - Direct backward timing (forward setup, then time ONLY backward)
  - F+B cross-check timing
  - Forward stage decomposition (manual, calling individual gsplat functions)
  - NVTX markers for nsys kernel-level decomposition
  - Full provenance capture

Fixes H1 issues:
  1. H1's b2_fwd_bwd_fn created a new create_higs_renderer every iteration,
     inflating "backward" by ~0.22ms state_prep. This script creates the
     handle ONCE and reuses it.
  2. H1 used subtraction (fwd_bwd - forward_no_grad) for backward. This script
     times backward DIRECTLY by doing forward setup then timing only backward().
  3. H1 had no backward kernel inventory. This script uses NVTX + nsys.

Run on mx in higs-13scene-env:
  conda activate /mnt/storage_pool/liaoyuanjun/higs-13scene-env
  export CUDA_HOME=$CONDA_PREFIX
  H=<higs-tree>
  PYTHONNOUSERSITE=1 PYTHONPATH=$H:/home/liaoyuanjun/.cache/torch_extensions/py310_cu128/gsplat_scene_cuda \
    CUDA_VISIBLE_DEVICES=1 python h2_1_profile.py --out-dir <outdir> [options]

For nsys trace:
  nsys profile -t cuda,nvtx --stats=true -o <outdir>/h2_1_nsys \
    CUDA_VISIBLE_DEVICES=1 python h2_1_profile.py --out-dir <outdir> --nsys-mode
"""
import argparse, json, math, os, sys, time, subprocess, platform, traceback
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData

# ============================================================ Constants

SH_DEGREE = 3
K_SH = (SH_DEGREE + 1) ** 2  # 16 for degree 3

# ============================================================ Provenance

def capture_provenance(device_idx=0):
    prov = {}
    prov["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
    try:
        prov["hostname"] = subprocess.check_output(["hostname"], text=True).strip()
    except Exception:
        prov["hostname"] = "UNKNOWN"
    prov["python_version"] = sys.version
    prov["torch_version"] = torch.__version__
    prov["torch_cuda_version"] = torch.version.cuda
    prov["torch_cxx11_abi"] = bool(torch._C._GLIBCXX_USE_CXX11_ABI)
    prov["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET")
    prov["higs_px_runtime"] = os.environ.get("HIGS_PX_RUNTIME", "UNSET(default=2)")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(device_idx)
        gpu_info = {
            "gpu_name": p.name,
            "gpu_total_memory_mb": int(p.total_memory / 1024 / 1024),
            "gpu_compute_capability": f"{p.major}.{p.minor}",
            "gpu_multi_processor_count": int(p.multi_processor_count),
        }
        for attr in ("L2_cache_size", "l2_cache_size", "max_threads_per_multi_processor",
                     "regs_per_multiprocessor", "regs_per_thread", "shared_memory_per_block"):
            if hasattr(p, attr):
                try:
                    gpu_info[f"gpu_{attr}"] = int(getattr(p, attr))
                except Exception:
                    pass
        prov.update(gpu_info)
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version",
             "--format=csv,noheader", f"--id={device_idx}"],
            text=True,
        ).strip()
        prov["nvidia_smi"] = smi
        parts = smi.split(", ")
        if len(parts) >= 4:
            prov["gpu_index"] = parts[0].strip()
            prov["gpu_name_smi"] = parts[1].strip()
            prov["gpu_uuid_smi"] = parts[2].strip()
            prov["driver_version"] = parts[3].strip()
    except Exception as e:
        prov["nvidia_smi_error"] = str(e)
    try:
        nvcc = subprocess.check_output(["nvcc", "--version"], text=True)
        for line in nvcc.split("\n"):
            if "release" in line.lower():
                prov["nvcc_version"] = line.strip()
    except Exception:
        prov["nvcc_version"] = "UNAVAILABLE"
    # gsplat version
    try:
        import gsplat
        prov["gsplat_version"] = getattr(gsplat, "__version__", "unknown")
    except Exception:
        prov["gsplat_version"] = "UNAVAILABLE"
    return prov

# ============================================================ Data loading

def load_ply_scene(ply_path, device):
    """Load 3DGS PLY -> (means, quats, scales, opacities, sh) FP32 masters."""
    ply = PlyData.read(ply_path)
    v = ply["vertex"]
    N = len(v)
    means = torch.tensor(
        np.column_stack([v["x"], v["y"], v["z"]]),
        dtype=torch.float32, device=device,
    )
    quats = torch.tensor(
        np.column_stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]]),
        dtype=torch.float32, device=device,
    )
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scales = torch.exp(torch.tensor(
        np.column_stack([v["scale_0"], v["scale_1"], v["scale_2"]]),
        dtype=torch.float32, device=device,
    ))
    opacities = torch.sigmoid(
        torch.tensor(v["opacity"], dtype=torch.float32, device=device)
    )
    f_dc = torch.tensor(
        np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]),
        dtype=torch.float32, device=device,
    )
    n_rest = 3 * (K_SH - 1)
    f_rest_cols = [f"f_rest_{i}" for i in range(n_rest)]
    f_rest = torch.stack(
        [torch.tensor(v[c], dtype=torch.float32, device=device) for c in f_rest_cols],
        dim=1,
    )
    f_rest = f_rest.reshape(N, 3, K_SH - 1).permute(0, 2, 1)
    sh = torch.zeros(N, K_SH, 3, dtype=torch.float32, device=device)
    sh[:, 0] = f_dc
    sh[:, 1:] = f_rest
    return means, quats, scales, opacities, sh


def load_cameras(cams_path, width, height, device):
    with open(cams_path) as f:
        cams = json.load(f)
    viewmats, Ks = [], []
    for c in cams:
        R = np.asarray(c["rotation"], dtype=np.float64)
        p = np.asarray(c["position"], dtype=np.float64)
        Rw2c = R.T
        vm = np.eye(4)
        vm[:3, :3] = Rw2c
        vm[:3, 3] = -Rw2c @ p
        scale = width / float(c["width"])
        K = np.array(
            [[float(c["fx"]) * scale, 0.0, (width - 1) / 2.0],
             [0.0, float(c["fy"]) * scale, (height - 1) / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        viewmats.append(torch.tensor(vm, dtype=torch.float32, device=device))
        Ks.append(torch.tensor(K, dtype=torch.float32, device=device))
    return torch.stack(viewmats).unsqueeze(0), torch.stack(Ks).unsqueeze(0), cams

# ============================================================ Timing utils

def time_repeated(fn, warmup=20, measure=100, label=""):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(measure):
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        se.record()
        fn()
        ee.record()
        torch.cuda.synchronize()
        times.append(se.elapsed_time(ee))
    arr = np.array(times)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "n_measure": measure,
        "n_warmup": warmup,
        "all_ms": [float(x) for x in times],
    }


def time_stage(fn, warmup=5, measure=20, label=""):
    """Time a single stage with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(measure):
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        se.record()
        fn()
        ee.record()
        torch.cuda.synchronize()
        times.append(se.elapsed_time(ee))
    arr = np.array(times)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "n_measure": measure,
        "n_warmup": warmup,
    }

# ============================================================ B2 production path

def b2_production_forward(means, quats, scales, opacities, colors, viewmats, Ks,
                          width, height, handle, sh_degree=3, radius_clip=0.0):
    """B2 HiGS frozen native forward (production path)."""
    from gsplat.experimental import rasterize_gaussian_higs_frozen
    res = rasterize_gaussian_higs_frozen(
        means, quats, scales, opacities, colors,
        backward_mode="higs_native", scene=handle, freeze_topology=True,
        viewmats=viewmats, Ks=Ks, width=width, height=height,
        sh_degree=sh_degree, use_higs_culling=True, radius_clip=radius_clip,
        tile_sampling_ratio=1.0,
    )
    return res

# ============================================================ Forward stage decomposition

def decompose_forward_stages(means, quats, scales, opacities, sh, viewmats, Ks,
                             width, height, sh_degree=3, radius_clip=0.0,
                             eps2d=0.3, near_plane=0.01, far_plane=1e10,
                             tile_size=16, warmup=5, measure=20):
    """Manually replicate the B2 _native_forward_capture path to time each stage.

    This is STATIC WORKLOAD ACCOUNTING — each stage is timed individually with
    CUDA events. The sum of stages will NOT exactly equal the production forward
    total because of Python dispatch overhead between stages. The residual is
    reported as dispatch overhead.
    """
    from gsplat.cuda._wrapper import (
        fully_fused_projection, isect_tiles, isect_offset_encode,
        _make_lazy_cuda_func,
    )
    from gsplat.rendering import _maybe_evaluate_sh
    from gsplat.experimental.render.functional.gaussian_inference import (
        _cull_gaussians_batched, _gather_visible_native,
        _union_visible_mask_native,
    )

    device = means.device
    C = viewmats.shape[-3]
    N_total = means.shape[0]
    colors = sh  # SH coefficients passed as colors

    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    n_tiles = tile_width * tile_height

    stages = {}

    # ---- Stage F0: Culling projection (full N_total) ----
    def stage_cull():
        with torch.no_grad():
            vis_ids, _vm, _ratio = _cull_gaussians_batched(
                means, quats, scales, viewmats, Ks,
                width, height, eps2d=eps2d,
                near_plane=near_plane, far_plane=far_plane,
                radius_clip=radius_clip, camera_model="pinhole",
            )
        return vis_ids
    stages["F0_culling_projection"] = time_stage(stage_cull, warmup, measure)

    # Get visible_ids for subsequent stages
    visible_ids = stage_cull()
    N_visible = visible_ids.numel()
    torch.cuda.synchronize()

    # ---- Stage F1: Gather visible subset ----
    def stage_gather():
        with torch.no_grad():
            return _gather_visible_native(
                means, quats, scales, opacities, colors, visible_ids,
            )
    stages["F1_gather_visible"] = time_stage(stage_gather, warmup, measure)

    v_means, v_quats, v_scales, v_opacities, v_colors = stage_gather()
    torch.cuda.synchronize()

    # Prepare batched inputs for render projection
    v_means_b = v_means.unsqueeze(0).contiguous()
    v_quats_b = v_quats.unsqueeze(0).contiguous()
    v_scales_b = v_scales.unsqueeze(0).contiguous()
    v_opacities_b = v_opacities.unsqueeze(0).contiguous()
    v_colors_input = v_colors.unsqueeze(0) if v_colors.dim() == 2 else v_colors
    opacities_bc = torch.broadcast_to(
        v_opacities_b[..., None, :], (1, C, N_visible)
    ).contiguous()

    # ---- Stage F2: Render projection (visible subset) ----
    def stage_proj():
        with torch.no_grad():
            return fully_fused_projection(
                means=v_means_b, covars=None,
                quats=v_quats_b, scales=v_scales_b,
                viewmats=viewmats, Ks=Ks,
                width=width, height=height,
                eps2d=eps2d, near_plane=near_plane,
                far_plane=far_plane, radius_clip=radius_clip,
                packed=False, calc_compensations=False,
                camera_model="pinhole",
            )
    stages["F2_render_projection"] = time_stage(stage_proj, warmup, measure)

    radii, means2d, depths, conics, _ = stage_proj()
    torch.cuda.synchronize()

    # ---- Stage F3: SH evaluation ----
    def stage_sh():
        with torch.no_grad():
            return _maybe_evaluate_sh(
                sh_degree, v_colors_input, v_means_b, radii, viewmats,
                (1,), C, N_visible, True,
            )
    stages["F3_sh_eval"] = time_stage(stage_sh, warmup, measure)

    colors_eval = stage_sh().contiguous()
    torch.cuda.synchronize()

    # ---- Stage F4: Intersection + sort ----
    def stage_isect():
        with torch.no_grad():
            tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
                means2d, radii, depths, tile_size,
                tile_width, tile_height,
                packed=False, n_images=C, image_ids=None,
                gaussian_ids=None, conics=conics, opacities=opacities_bc,
            )
            isect_offsets = isect_offset_encode(
                isect_ids, C, tile_width, tile_height
            ).reshape((1, C, tile_height, tile_width))
            return isect_ids, flatten_ids, isect_offsets
    stages["F4_intersection_sort"] = time_stage(stage_isect, warmup, measure)

    isect_ids, flatten_ids, isect_offsets = stage_isect()
    n_isects = isect_ids.numel()
    torch.cuda.synchronize()

    # ---- Stage F5: Rasterize/blend ----
    bg_kernel = None  # no background for timing
    def stage_raster():
        with torch.no_grad():
            return _make_lazy_cuda_func("rasterize_to_pixels_3dgs")(
                means2d.contiguous(), conics.contiguous(),
                colors_eval.contiguous(), opacities_bc.contiguous(),
                bg_kernel, None, width, height, tile_size,
                isect_offsets.contiguous(), flatten_ids.contiguous(),
                False, False,
            )
    stages["F5_rasterize"] = time_stage(stage_raster, warmup, measure)

    # ---- Summary ----
    stage_medians = {}
    stage_sum = 0.0
    for k, v in stages.items():
        stage_medians[k] = v["median_ms"]
        stage_sum += v["median_ms"]

    return {
        "stages": stages,
        "stage_medians": stage_medians,
        "stage_sum_median_ms": stage_sum,
        "N_total": N_total,
        "N_visible": N_visible,
        "n_isects": int(n_isects),
        "n_tiles": int(n_tiles),
        "tile_width": tile_width,
        "tile_height": tile_height,
        "culling_ratio": 1.0 - (N_visible / max(N_total, 1)),
    }

# ============================================================ Main profiling

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scene", default="room")
    ap.add_argument("--cam-idx", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measure", type=int, default=100)
    ap.add_argument("--stage-warmup", type=int, default=5)
    ap.add_argument("--stage-measure", type=int, default=20)
    ap.add_argument("--max-long-side", type=int, default=2048)
    ap.add_argument("--radius-clip", type=float, default=0.0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--nsys-mode", action="store_true",
                    help="Run fewer iterations with NVTX markers for nsys trace")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = f"cuda:{args.gpu}"

    # ---- Provenance ----
    prov = capture_provenance(args.gpu)
    with open(os.path.join(args.out_dir, "environment.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print("[provenance]", json.dumps(prov, indent=2))

    # ---- Scene config ----
    scene_configs = {
        "room": {
            "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply",
            "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json",
            "native_w": 3114, "native_h": 2075,
        },
        "bicycle": {
            "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/bicycle/native/point_cloud/iteration_30000/point_cloud.ply",
            "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/bicycle/cameras.json",
            "native_w": 4946, "native_h": 3286,
        },
    }
    cfg = scene_configs[args.scene]
    nw, nh = cfg["native_w"], cfg["native_h"]
    if args.max_long_side > 0 and max(nw, nh) > args.max_long_side:
        scale = args.max_long_side / max(nw, nh)
        width = max(1, int(round(nw * scale)))
        height = max(1, int(round(nh * scale)))
    else:
        width, height = nw, nh
    print(f"[res] {width}x{height} (native {nw}x{nh})")

    # ---- Load scene ----
    print("[load] PLY...", cfg["ply"])
    means, quats, scales, opacities, sh = load_ply_scene(cfg["ply"], device)
    N_total = len(means)
    print(f"[load] N={N_total} Gaussians")
    viewmats_all, Ks_all, cams = load_cameras(cfg["cams"], width, height, device)
    print(f"[load] {len(cams)} cameras")

    # Select camera
    cam_idx = args.cam_idx
    vm = viewmats_all[:, [cam_idx]]  # [1,1,4,4]
    K = Ks_all[:, [cam_idx]]         # [1,1,3,3]
    print(f"[cam] idx={cam_idx} ({cams[cam_idx]['img_name']})")

    colors = sh  # SH coefficients

    # ---- Create HiGS renderer handle ONCE (not part of hot path) ----
    from gsplat.experimental import rasterize_gaussian_higs_frozen
    from gsplat.experimental.render.functional.gaussian_inference import (
        create_higs_renderer, _HIGS_FROZEN_TRACKER,
    )

    _HIGS_FROZEN_TRACKER.reset()
    handle = create_higs_renderer(means, quats, scales, opacities, sh,
                                  sh_degree=SH_DEGREE)
    torch.cuda.synchronize()
    print("[handle] created")

    result = {
        "scene": args.scene,
        "camera_idx": cam_idx,
        "camera_img_name": cams[cam_idx]["img_name"],
        "width": width, "height": height,
        "N_total": N_total,
        "warmup": args.warmup,
        "measure": args.measure,
        "nsys_mode": args.nsys_mode,
    }

    # ===== B2 metadata (single forward) =====
    with torch.no_grad():
        b2_res = b2_production_forward(
            means, quats, scales, opacities, colors, vm, K,
            width, height, handle, sh_degree=SH_DEGREE,
            radius_clip=args.radius_clip,
        )
    b2_meta = b2_res.get("metadata", {})
    b2_render = b2_res["frame"].detach()

    def _ser(v):
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                return v.item()
            elif v.numel() <= 64:
                return v.detach().cpu().flatten().tolist()
            else:
                return {"shape": list(v.shape), "dtype": str(v.dtype),
                        "numel": int(v.numel())}
        elif isinstance(v, (int, float, str, bool)):
            return v
        else:
            return str(v)

    result["B2_metadata"] = {k: _ser(v) for k, v in b2_meta.items()} if isinstance(b2_meta, dict) else {}
    print(f"[meta] n_visible={b2_meta.get('n_visible', '?')} "
          f"culling_ratio={b2_meta.get('culling_ratio', '?')} "
          f"n_isects={b2_meta.get('n_isects', '?')}")

    # ===== Production path timing =====
    if args.nsys_mode:
        # Fewer iterations for nsys trace, with NVTX markers
        n_iter = 10
        print(f"\n[nsys] Running {n_iter} iterations with NVTX markers...")
        for i in range(n_iter):
            torch.cuda.nvtx.range_push(f"H2_1_iter_{i}")
            m = means.detach().clone().requires_grad_(True)
            q = quats.detach().clone().requires_grad_(True)
            s = scales.detach().clone().requires_grad_(True)
            o = opacities.detach().clone().requires_grad_(True)
            c = sh.detach().clone().requires_grad_(True)
            torch.cuda.nvtx.range_push("B2_forward")
            res = b2_production_forward(
                m, q, s, o, c, vm, K, width, height, handle,
                sh_degree=SH_DEGREE, radius_clip=args.radius_clip,
            )
            torch.cuda.nvtx.range_pop()
            target = b2_render.clone()
            loss = (res["frame"] - target).abs().mean()
            torch.cuda.nvtx.range_push("B2_backward")
            loss.backward()
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        print("[nsys] Done. Check nsys trace for kernel-level decomposition.")
        result["nsys_note"] = "NVTX markers placed. Use nsys to extract kernel inventory."
        # Still do the main timing below if not nsys-only
        # But save and exit for nsys mode
        with open(os.path.join(args.out_dir, f"{args.scene}_cam{cam_idx}_nsys.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"[done] nsys results saved to {args.out_dir}")
        return

    # ---- B2 forward with requires_grad=True (no backward) ----
    print("\n===== B2 Forward (requires_grad=True, no backward) =====")

    def b2_fwd_grad_fn():
        m = means.detach().clone().requires_grad_(True)
        q = quats.detach().clone().requires_grad_(True)
        s = scales.detach().clone().requires_grad_(True)
        o = opacities.detach().clone().requires_grad_(True)
        c = sh.detach().clone().requires_grad_(True)
        res = b2_production_forward(
            m, q, s, o, c, vm, K, width, height, handle,
            sh_degree=SH_DEGREE, radius_clip=args.radius_clip,
        )
        return res["frame"]

    torch.cuda.nvtx.range_push("B2_forward_grad_repeated")
    result["B2_forward_grad_timing"] = time_repeated(
        b2_fwd_grad_fn, warmup=args.warmup, measure=args.measure, label="B2_fwd_grad")
    torch.cuda.nvtx.range_pop()
    print(f"[fwd_grad] median={result['B2_forward_grad_timing']['median_ms']:.3f}ms")

    # ---- B2 backward ONLY (direct timing) ----
    print("\n===== B2 Backward ONLY (direct timing) =====")

    target = b2_render.clone()

    def b2_bwd_only_fn():
        m = means.detach().clone().requires_grad_(True)
        q = quats.detach().clone().requires_grad_(True)
        s = scales.detach().clone().requires_grad_(True)
        o = opacities.detach().clone().requires_grad_(True)
        c = sh.detach().clone().requires_grad_(True)
        # Forward (not timed) to build graph
        res = b2_production_forward(
            m, q, s, o, c, vm, K, width, height, handle,
            sh_degree=SH_DEGREE, radius_clip=args.radius_clip,
        )
        loss = (res["frame"] - target).abs().mean()
        # Time ONLY the backward
        loss.backward()
        torch.cuda.synchronize()

    # For backward, we need to time only the backward() call, not the forward.
    # Strategy: do forward (untimed), start timer, call backward, stop timer.
    bwd_times = []
    for _ in range(args.warmup):
        m = means.detach().clone().requires_grad_(True)
        q = quats.detach().clone().requires_grad_(True)
        s = scales.detach().clone().requires_grad_(True)
        o = opacities.detach().clone().requires_grad_(True)
        c = sh.detach().clone().requires_grad_(True)
        res = b2_production_forward(
            m, q, s, o, c, vm, K, width, height, handle,
            sh_degree=SH_DEGREE, radius_clip=args.radius_clip,
        )
        loss = (res["frame"] - target).abs().mean()
        loss.backward()
        torch.cuda.synchronize()
    torch.cuda.synchronize()

    for _ in range(args.measure):
        m = means.detach().clone().requires_grad_(True)
        q = quats.detach().clone().requires_grad_(True)
        s = scales.detach().clone().requires_grad_(True)
        o = opacities.detach().clone().requires_grad_(True)
        c = sh.detach().clone().requires_grad_(True)
        res = b2_production_forward(
            m, q, s, o, c, vm, K, width, height, handle,
            sh_degree=SH_DEGREE, radius_clip=args.radius_clip,
        )
        loss = (res["frame"] - target).abs().mean()
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        se.record()
        loss.backward()
        ee.record()
        torch.cuda.synchronize()
        bwd_times.append(se.elapsed_time(ee))

    arr_bwd = np.array(bwd_times)
    result["B2_backward_direct_timing"] = {
        "median_ms": float(np.median(arr_bwd)),
        "mean_ms": float(np.mean(arr_bwd)),
        "std_ms": float(np.std(arr_bwd)),
        "min_ms": float(np.min(arr_bwd)),
        "max_ms": float(np.max(arr_bwd)),
        "p50_ms": float(np.percentile(arr_bwd, 50)),
        "p95_ms": float(np.percentile(arr_bwd, 95)),
        "n_measure": args.measure,
        "n_warmup": args.warmup,
        "all_ms": [float(x) for x in bwd_times],
        "method": "DIRECT (CUDA events around loss.backward() only)",
    }
    print(f"[bwd_direct] median={result['B2_backward_direct_timing']['median_ms']:.3f}ms")

    # ---- B2 F+B cross-check ----
    print("\n===== B2 F+B cross-check =====")

    def b2_fb_fn():
        m = means.detach().clone().requires_grad_(True)
        q = quats.detach().clone().requires_grad_(True)
        s = scales.detach().clone().requires_grad_(True)
        o = opacities.detach().clone().requires_grad_(True)
        c = sh.detach().clone().requires_grad_(True)
        res = b2_production_forward(
            m, q, s, o, c, vm, K, width, height, handle,
            sh_degree=SH_DEGREE, radius_clip=args.radius_clip,
        )
        loss = (res["frame"] - target).abs().mean()
        loss.backward()
        torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("B2_fb_repeated")
    result["B2_fwd_bwd_timing"] = time_repeated(
        b2_fb_fn, warmup=args.warmup, measure=args.measure, label="B2_fb")
    torch.cuda.nvtx.range_pop()
    print(f"[fb] median={result['B2_fwd_bwd_timing']['median_ms']:.3f}ms")

    # ---- Reconciliation ----
    fwd_med = result["B2_forward_grad_timing"]["median_ms"]
    bwd_med = result["B2_backward_direct_timing"]["median_ms"]
    fb_med = result["B2_fwd_bwd_timing"]["median_ms"]
    fb_check = fwd_med + bwd_med
    residual = fb_med - fb_check
    result["reconciliation"] = {
        "forward_ms": fwd_med,
        "backward_ms": bwd_med,
        "fwd_plus_bwd_ms": fb_check,
        "fb_measured_ms": fb_med,
        "residual_ms": float(residual),
        "residual_pct": float(residual / max(fb_med, 1e-6) * 100),
        "note": "residual = fb_measured - (fwd_direct + bwd_direct). "
                "Positive residual = overhead from clone/detach in fb_fn vs separate timing.",
    }
    print(f"[reconcile] fwd={fwd_med:.3f} + bwd={bwd_med:.3f} = {fb_check:.3f} "
          f"vs fb={fb_med:.3f} residual={residual:+.3f}ms "
          f"({residual/max(fb_med,1e-6)*100:+.1f}%)")

    # ===== Forward stage decomposition =====
    print("\n===== Forward Stage Decomposition =====")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    fwd_decomp = decompose_forward_stages(
        means, quats, scales, opacities, sh, vm, K,
        width, height, sh_degree=SH_DEGREE, radius_clip=args.radius_clip,
        warmup=args.stage_warmup, measure=args.stage_measure,
    )
    result["forward_stage_decomposition"] = fwd_decomp

    # Reconcile stage sum with forward total
    stage_sum = fwd_decomp["stage_sum_median_ms"]
    fwd_total = fwd_med
    stage_residual = fwd_total - stage_sum
    result["forward_stage_reconciliation"] = {
        "stage_sum_ms": stage_sum,
        "forward_total_ms": fwd_total,
        "residual_ms": float(stage_residual),
        "residual_pct": float(stage_residual / max(fwd_total, 1e-6) * 100),
        "note": "residual = forward_total - sum(stages). "
                "Includes Python dispatch overhead, tensor clone, autograd graph setup, "
                "and save_for_backward overhead not captured by individual stage timing.",
    }
    print(f"[fwd_stages] sum={stage_sum:.3f}ms vs fwd_total={fwd_total:.3f}ms "
          f"residual={stage_residual:+.3f}ms ({stage_residual/max(fwd_total,1e-6)*100:+.1f}%)")
    for k, v in fwd_decomp["stage_medians"].items():
        print(f"  {k}: {v:.3f}ms")

    # ===== Save =====
    fname = f"{args.scene}_cam{cam_idx}_h2_1.json"
    with open(os.path.join(args.out_dir, fname), "w") as f:
        def clean(o):
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()}
            if isinstance(o, list) and len(o) > 200:
                return f"[{len(o)} items]"
            if isinstance(o, torch.Tensor):
                return o.item() if o.numel() == 1 else f"tensor{list(o.shape)}"
            return o
        json.dump(clean(result), f, indent=2, default=str)
    print(f"\n[done] Results saved to {os.path.join(args.out_dir, fname)}")

    # Release handle
    try:
        handle.release()
    except Exception:
        pass


if __name__ == "__main__":
    main()
