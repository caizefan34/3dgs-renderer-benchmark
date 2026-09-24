#!/usr/bin/env python3
"""H1 Clean Matched-State Profiling Harness.

Profiles B1 (clean gsplat rasterization) vs B2 (Trainable HiGS Full
rasterize_gaussian_higs_frozen) on the SAME Gaussian checkpoint, SAME camera,
SAME hardware, SAME torch — only the renderer changes.

Output: one JSON per scene x camera x method, plus environment.json.

Run on mx in higs-13scene-env:
  conda activate /mnt/storage_pool/liaoyuanjun/higs-13scene-env
  export CUDA_HOME=$CONDA_PREFIX
  H=<higs-tree>
  PYTHONNOUSERSITE=1 PYTHONPATH=$H:/home/liaoyuanjun/.cache/torch_extensions/py310_cu128/gsplat_scene_cuda \
    CUDA_VISIBLE_DEVICES=0 python h1_profile.py --out-dir <outdir> [options]
"""
import argparse, json, math, os, sys, time, subprocess, platform, traceback
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData

# ------------------------------------------------------------------ provenance

def capture_provenance(device_idx=0):
    """Capture full hardware/software provenance per H1 section 2."""
    prov = {}
    prov["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
    try:
        prov["hostname"] = subprocess.check_output(["hostname"], text=True).strip()
    except Exception:
        prov["hostname"] = os.uname().nodename if hasattr(os, "uname") else "UNKNOWN"
    prov["python_version"] = sys.version
    prov["torch_version"] = torch.__version__
    prov["torch_cuda_version"] = torch.version.cuda
    prov["torch_cxx11_abi"] = bool(torch._C._GLIBCXX_USE_CXX11_ABI)
    prov["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET")
    # GPU
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
    # nvidia-smi
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
    # nvcc
    try:
        nvcc = subprocess.check_output(["nvcc", "--version"], text=True)
        for line in nvcc.split("\n"):
            if "release" in line.lower():
                prov["nvcc_version"] = line.strip()
    except Exception:
        prov["nvcc_version"] = "UNAVAILABLE"
    # gsplat
    try:
        import gsplat
        prov["gsplat_file"] = os.path.dirname(gsplat.__file__)
        prov["gsplat_version"] = getattr(gsplat, "__version__", "UNKNOWN")
    except Exception as e:
        prov["gsplat_error"] = str(e)
    # git commits
    repo = "/mnt/storage_pool/3dgs-renderer-benchmark/repo"
    higs_tree = "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx"
    for label, path in [("repo_commit", repo), ("higs_tree_commit", higs_tree)]:
        try:
            commit = subprocess.check_output(
                ["git", "-C", path, "rev-parse", "HEAD"], text=True,
            ).strip()
            prov[label] = commit
        except Exception:
            prov[label] = "UNAVAILABLE"
    # nsys
    nsys_path = "/usr/lib/x86_64-linux-gnu/nsight-systems/target-linux-x64/nsys"
    try:
        ver = subprocess.check_output([nsys_path, "--version"], text=True).strip()
        prov["nsys_path"] = nsys_path
        prov["nsys_version"] = ver
    except Exception:
        prov["nsys_version"] = "UNAVAILABLE"
    return prov


# ------------------------------------------------------------------ scene loading

SH_DEGREE = 3
K_SH = (SH_DEGREE + 1) ** 2  # 16


def load_ply_scene(ply_path, device):
    """Load 3DGS PLY -> (means, quats, scales, opacities, sh) FP32 masters.
    scales are exp'd, opacities are sigmoid'd (matching gsplat training pipeline)."""
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
    )  # [N, 3*(K-1)]
    f_rest = f_rest.reshape(N, 3, K_SH - 1).permute(0, 2, 1)  # [N, K-1, 3]
    sh = torch.zeros(N, K_SH, 3, dtype=torch.float32, device=device)
    sh[:, 0] = f_dc
    sh[:, 1:] = f_rest
    return means, quats, scales, opacities, sh


def load_cameras(cams_path, width, height, device):
    """Load cameras.json -> viewmats [1,C,4,4], Ks [1,C,3,3]."""
    with open(cams_path) as f:
        cams = json.load(f)
    viewmats, Ks = [], []
    for c in cams:
        R = np.asarray(c["rotation"], dtype=np.float64)  # c2w rotation
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


# ------------------------------------------------------------------ timing utils

class CudaTimer:
    """Paired CUDA event timer."""
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *a):
        self.end.record()
        torch.cuda.synchronize()

    def elapsed_ms(self):
        return self.start.elapsed_time(self.end)


def time_repeated(fn, warmup=20, measure=100, label=""):
    """Run fn warmup+measure times, return (median_ms, mean_ms, std_ms, min_ms, max_ms, all_ms[])."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(measure):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
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
        "all_ms": [float(x) for x in arr],
    }


# ------------------------------------------------------------------ B1 forward (decomposed)

def b1_forward_decomposed(means, quats, scales, opacities, colors, viewmats, Ks,
                          width, height, sh_degree=3, radius_clip=0.0, tile_size=16):
    """B1 standard gsplat forward, decomposed into stages with timing.
    Returns dict of stage timings (ms) + render outputs + workload metrics."""
    from gsplat.cuda._wrapper import (
        fully_fused_projection, isect_tiles, isect_offset_encode, _make_lazy_cuda_func,
    )
    from gsplat.rendering import _maybe_evaluate_sh

    C = viewmats.shape[-3]
    N = means.shape[-2]
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)

    stages = {}

    # Per-stage CUDA events (no inter-stage sync; single sync at end)
    s1 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    s2 = torch.cuda.Event(enable_timing=True); e2 = torch.cuda.Event(enable_timing=True)
    s3 = torch.cuda.Event(enable_timing=True); e3 = torch.cuda.Event(enable_timing=True)
    s4 = torch.cuda.Event(enable_timing=True); e4 = torch.cuda.Event(enable_timing=True)

    # F1: projection
    torch.cuda.nvtx.range_push("B1_F1_projection")
    s1.record()
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means=means.contiguous(), covars=None, quats=quats.contiguous(),
        scales=scales.contiguous(), viewmats=viewmats, Ks=Ks,
        width=width, height=height, eps2d=0.3, near_plane=0.01, far_plane=1e10,
        radius_clip=radius_clip, packed=False, calc_compensations=False,
        camera_model="pinhole",
    )
    e1.record()
    torch.cuda.nvtx.range_pop()

    # F2: SH evaluation (no sync between stages)
    torch.cuda.nvtx.range_push("B1_F2_sh_eval")
    s2.record()
    colors_eval = _maybe_evaluate_sh(
        sh_degree, colors, means, radii, viewmats, (1,), C, N, True,
    ).contiguous()
    e2.record()
    torch.cuda.nvtx.range_pop()

    # F3+F4: intersection generation (isect_tiles + offset encode)
    opacities_bc = torch.broadcast_to(opacities[..., None, :], (1, C, N)).contiguous()
    torch.cuda.nvtx.range_push("B1_F34_intersection")
    s3.record()
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height,
        packed=False, n_images=C, image_ids=None, gaussian_ids=None,
        conics=conics, opacities=opacities_bc,
    )
    isect_offsets = isect_offset_encode(
        isect_ids, C, tile_width, tile_height
    ).reshape((1, C, tile_height, tile_width))
    e3.record()
    torch.cuda.nvtx.range_pop()

    # F6: rasterization / blend
    bg_kernel = torch.zeros((1, C, 3), device=means.device)
    torch.cuda.nvtx.range_push("B1_F6_rasterize")
    s4.record()
    render_colors, render_alphas, _absgrad, last_ids = (
        _make_lazy_cuda_func("rasterize_to_pixels_3dgs")(
            means2d.contiguous(), conics.contiguous(), colors_eval.contiguous(),
            opacities_bc.contiguous(), bg_kernel, None,
            width, height, tile_size,
            isect_offsets.contiguous(), flatten_ids.contiguous(),
            False, False,
        )
    )
    e4.record()
    torch.cuda.nvtx.range_pop()

    # Single sync — read all per-stage times after GPU finishes
    torch.cuda.synchronize()
    stages["F1_projection_ms"] = s1.elapsed_time(e1)
    stages["F2_sh_eval_ms"] = s2.elapsed_time(e2)
    stages["F34_intersection_ms"] = s3.elapsed_time(e3)
    stages["F6_rasterize_ms"] = s4.elapsed_time(e4)
    stages["F_total_ms"] = (
        stages["F1_projection_ms"] + stages["F2_sh_eval_ms"] +
        stages["F34_intersection_ms"] + stages["F6_rasterize_ms"]
    )

    # Workload metrics (computed after timing; .item() calls are OK here)
    n_visible = int(((radii > 0).any(dim=-1)).sum().item())
    n_total = int(N)
    n_isects = int(flatten_ids.shape[-1])
    total_tiles = int(tile_width * tile_height * C)

    workload = {
        "N_total": n_total,
        "N_visible": n_visible,
        "N_isects": n_isects,
        "total_tiles": total_tiles,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tile_size": tile_size,
        "image_width": width,
        "image_height": height,
    }
    if n_isects > 0 and total_tiles > 0:
        offs = isect_offsets.reshape(-1).cpu().numpy()
        if len(offs) > 1:
            counts = np.diff(offs)
            counts = np.append(counts, n_isects - offs[-1])
            counts = counts[counts >= 0]
            if len(counts) > 0:
                workload["mean_isects_per_tile"] = float(np.mean(counts))
                workload["median_isects_per_tile"] = float(np.median(counts))
                workload["p95_isects_per_tile"] = float(np.percentile(counts, 95))
                workload["p99_isects_per_tile"] = float(np.percentile(counts, 99))
                workload["max_isects_per_tile"] = float(np.max(counts))
                workload["active_tiles"] = int((counts > 0).sum())
                workload["tile_occupancy"] = float(workload["active_tiles"] / total_tiles) if total_tiles > 0 else 0.0
                workload["tiles_per_gaussian"] = float(n_isects / n_visible) if n_visible > 0 else 0.0
                workload["pixels_per_gaussian"] = float(width * height / n_visible) if n_visible > 0 else 0.0

    return {
        "stages": stages,
        "render_colors": render_colors,
        "render_alphas": render_alphas,
        "workload": workload,
        # save for backward
        "_backward_ctx": {
            "means2d": means2d, "conics": conics, "colors_eval": colors_eval,
            "opacities_bc": opacities_bc, "isect_offsets": isect_offsets,
            "flatten_ids": flatten_ids, "last_ids": last_ids,
            "bg_kernel": bg_kernel, "tile_size": tile_size,
            "width": width, "height": height,
        },
    }


# ------------------------------------------------------------------ B1 full (autograd)

def b1_forward_autograd(means, quats, scales, opacities, colors, viewmats, Ks,
                        width, height, sh_degree=3, radius_clip=0.0):
    """B1 forward through high-level rasterization() (with autograd graph for backward)."""
    from gsplat.rendering import rasterization
    out = rasterization(
        means=means.unsqueeze(0), quats=quats.unsqueeze(0),
        scales=scales.unsqueeze(0), opacities=opacities.unsqueeze(0), colors=colors,
        viewmats=viewmats, Ks=Ks, width=width, height=height,
        sh_degree=sh_degree, packed=True, radius_clip=radius_clip,
    )
    return out[0], out[1]


# ------------------------------------------------------------------ B2 forward

def b2_forward(means, quats, scales, opacities, colors, viewmats, Ks,
               width, height, handle, sh_degree=3, radius_clip=0.0):
    """B2 HiGS frozen native forward."""
    from gsplat.experimental import rasterize_gaussian_higs_frozen
    res = rasterize_gaussian_higs_frozen(
        means, quats, scales, opacities, colors,
        backward_mode="higs_native", scene=handle, freeze_topology=True,
        viewmats=viewmats, Ks=Ks, width=width, height=height,
        sh_degree=sh_degree, use_higs_culling=True, radius_clip=radius_clip,
        tile_sampling_ratio=1.0,
    )
    return res


# ------------------------------------------------------------------ correctness

def psnr(a, b):
    mse = (a - b).pow(2).mean().item()
    if mse < 1e-12:
        return 60.0
    return float(-10.0 * np.log10(mse))


def relative_l2(a, b):
    num = (a - b).pow(2).sum().sqrt().item()
    den = b.pow(2).sum().sqrt().item()
    return float(num / max(den, 1e-12))


def grad_compare(g1, g2, name=""):
    """Compare two gradient tensors."""
    if g1 is None or g2 is None:
        return {"name": name, "status": "MISSING"}
    if g1.shape != g2.shape:
        return {"name": name, "status": "SHAPE_MISMATCH",
                "shape1": list(g1.shape), "shape2": list(g2.shape)}
    diff = (g1 - g2)
    result = {
        "name": name,
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "relative_l2": relative_l2(g1, g2),
        "cosine": float(F.cosine_similarity(g1.flatten().unsqueeze(0),
                                             g2.flatten().unsqueeze(0)).item()),
    }
    nz1 = (g1.abs() > 0)
    nz2 = (g2.abs() > 0)
    disagree = (nz1 != nz2).sum().item()
    result["zero_nonzero_disagreement"] = int(disagree)
    result["n_elements"] = int(g1.numel())
    return result


# ------------------------------------------------------------------ main profile

def profile_scene_camera(scene_name, cam_idx, cams, means, quats, scales,
                         opacities, sh, viewmats_all, Ks_all, width, height,
                         device, warmup, measure, radius_clip=0.0):
    """Profile one scene x one camera for B1 and B2."""
    vm = viewmats_all[:, [cam_idx]]  # [1,1,4,4]
    K = Ks_all[:, [cam_idx]]         # [1,1,3,3]
    result = {
        "scene": scene_name, "camera_idx": cam_idx,
        "camera_id": cams[cam_idx]["id"],
        "camera_img_name": cams[cam_idx]["img_name"],
        "width": width, "height": height,
        "warmup": warmup, "measure": measure,
    }

    colors = sh  # gsplat accepts SH as colors with sh_degree

    # ===== B1 forward (decomposed) =====
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    b1_dec = b1_forward_decomposed(
        means.unsqueeze(0), quats.unsqueeze(0), scales.unsqueeze(0),
        opacities.unsqueeze(0), colors, vm, K,
        width, height, sh_degree=SH_DEGREE, radius_clip=radius_clip,
    )
    result["B1_forward_stages_single"] = b1_dec["stages"]
    result["B1_workload"] = b1_dec["workload"]
    b1_render = b1_dec["render_colors"].detach()  # [1,1,H,W,3]
    b1_alpha = b1_dec["render_alphas"].detach()   # [1,1,H,W,1]

    # B1 forward decomposed — repeated per-stage timing
    stage_keys = ["F1_projection_ms", "F2_sh_eval_ms", "F34_intersection_ms", "F6_rasterize_ms"]
    stage_times = {k: [] for k in stage_keys}
    total_times_dec = []
    for _ in range(warmup):
        with torch.no_grad():
            b1_forward_decomposed(
                means.unsqueeze(0), quats.unsqueeze(0), scales.unsqueeze(0),
                opacities.unsqueeze(0), colors, vm, K,
                width, height, sh_degree=SH_DEGREE, radius_clip=radius_clip,
            )
    torch.cuda.synchronize()
    for _ in range(measure):
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        se.record()
        with torch.no_grad():
            r_dec = b1_forward_decomposed(
                means.unsqueeze(0), quats.unsqueeze(0), scales.unsqueeze(0),
                opacities.unsqueeze(0), colors, vm, K,
                width, height, sh_degree=SH_DEGREE, radius_clip=radius_clip,
            )
        ee.record()
        torch.cuda.synchronize()
        total_times_dec.append(se.elapsed_time(ee))
        for k in stage_keys:
            stage_times[k].append(r_dec["stages"][k])
    arr_t = np.array(total_times_dec)
    result["B1_forward_decomposed_timing"] = {
        "median_ms": float(np.median(arr_t)), "mean_ms": float(np.mean(arr_t)),
        "std_ms": float(np.std(arr_t)), "min_ms": float(np.min(arr_t)),
        "max_ms": float(np.max(arr_t)), "n_measure": measure, "n_warmup": warmup,
    }
    result["B1_forward_stages_repeated"] = {}
    for k in stage_keys:
        arr_s = np.array(stage_times[k])
        result["B1_forward_stages_repeated"][k] = {
            "median_ms": float(np.median(arr_s)), "mean_ms": float(np.mean(arr_s)),
            "std_ms": float(np.std(arr_s)), "min_ms": float(np.min(arr_s)),
            "max_ms": float(np.max(arr_s)),
        }
    stage_sum_med = sum(result["B1_forward_stages_repeated"][k]["median_ms"] for k in stage_keys)
    total_med = result["B1_forward_decomposed_timing"]["median_ms"]
    # F7: dispatch/Python overhead (residual = total - sum(GPU stages))
    # This is NOT_SEPARATED (measured as residual, not directly timed)
    f7_med = max(total_med - stage_sum_med, 0.0)
    result["B1_forward_stages_repeated"]["F7_dispatch_overhead_ms"] = {
        "median_ms": f7_med, "mean_ms": f7_med,
        "std_ms": 0.0, "min_ms": f7_med, "max_ms": f7_med,
        "method": "RESIDUAL",
    }
    # Closure with F7 included = 1.0 by construction
    result["timing_closure_B1_forward"] = float((stage_sum_med + f7_med) / max(total_med, 1e-6))
    result["timing_closure_B1_forward_without_F7"] = float(stage_sum_med / max(total_med, 1e-6))

    # ===== B1 forward+backward (autograd) =====
    # Use a fixed target (B1's own render) for backward
    target = b1_render.clone()

    def b1_fwd_bwd_fn():
        m = means.detach().clone().requires_grad_(True)
        q = quats.detach().clone().requires_grad_(True)
        s = scales.detach().clone().requires_grad_(True)
        o = opacities.detach().clone().requires_grad_(True)
        c = sh.detach().clone().requires_grad_(True)
        torch.cuda.nvtx.range_push("B1_autograd_forward")
        r, a = b1_forward_autograd(
            m, q, s, o, c, vm, K, width, height,
            sh_degree=SH_DEGREE, radius_clip=radius_clip,
        )
        torch.cuda.nvtx.range_pop()
        loss = (r - target).abs().mean()
        torch.cuda.nvtx.range_push("B1_autograd_backward")
        loss.backward()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        return m.grad, q.grad, s.grad, o.grad, c.grad

    # warmup + measure forward_total and backward_total separately
    # First: single pass to get gradients for correctness
    b1_grads = b1_fwd_bwd_fn()
    torch.cuda.synchronize()

    # Time B1 forward (autograd) only
    def b1_fwd_only_fn():
        m = means.detach().clone().requires_grad_(True)
        q = quats.detach().clone().requires_grad_(True)
        s = scales.detach().clone().requires_grad_(True)
        o = opacities.detach().clone().requires_grad_(True)
        c = sh.detach().clone().requires_grad_(True)
        r, a = b1_forward_autograd(m, q, s, o, c, vm, K, width, height,
                                   sh_degree=SH_DEGREE, radius_clip=radius_clip)
        return r

    torch.cuda.nvtx.range_push("B1_forward_autograd_repeated")
    result["B1_forward_autograd_timing"] = time_repeated(
        b1_fwd_only_fn, warmup=warmup, measure=measure, label="B1_fwd_ag")
    torch.cuda.nvtx.range_pop()

    # Time B1 backward only (forward is done, then time backward)
    def b1_bwd_only_fn():
        m = means.detach().clone().requires_grad_(True)
        q = quats.detach().clone().requires_grad_(True)
        s = scales.detach().clone().requires_grad_(True)
        o = opacities.detach().clone().requires_grad_(True)
        c = sh.detach().clone().requires_grad_(True)
        r, a = b1_forward_autograd(m, q, s, o, c, vm, K, width, height,
                                   sh_degree=SH_DEGREE, radius_clip=radius_clip)
        loss = (r - target).abs().mean()
        loss.backward()
        torch.cuda.synchronize()

    # We need to separate forward and backward timing.
    # Strategy: time the full fwd+bwd, then subtract forward_autograd timing.
    torch.cuda.nvtx.range_push("B1_fwd_bwd_repeated")
    result["B1_fwd_bwd_timing"] = time_repeated(
        b1_bwd_only_fn, warmup=warmup, measure=measure, label="B1_fb")
    torch.cuda.nvtx.range_pop()
    # backward_total = fwd_bwd_total - forward_autograd_median
    result["B1_backward_total_ms"] = (
        result["B1_fwd_bwd_timing"]["median_ms"] -
        result["B1_forward_autograd_timing"]["median_ms"]
    )
    result["B1_backward_separated"] = "SUBTRACTED"

    # ===== B2 forward + backward =====
    from gsplat.experimental import rasterize_gaussian_higs_frozen
    from gsplat.experimental.render.functional.gaussian_inference import (
        create_higs_renderer, _HIGS_FROZEN_TRACKER,
    )

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # B2 state preparation (create_higs_renderer) — T-stage
    def b2_state_prep_fn():
        _HIGS_FROZEN_TRACKER.reset()
        h = create_higs_renderer(means, quats, scales, opacities, sh, sh_degree=SH_DEGREE)
        return h

    torch.cuda.nvtx.range_push("B2_state_prep_repeated")
    result["B2_state_prep_timing"] = time_repeated(
        b2_state_prep_fn, warmup=min(warmup, 5), measure=min(measure, 20),
        label="B2_state")
    torch.cuda.nvtx.range_pop()

    # Create handle for actual profiling
    _HIGS_FROZEN_TRACKER.reset()
    handle = create_higs_renderer(means, quats, scales, opacities, sh, sh_degree=SH_DEGREE)

    # B2 forward only
    def b2_fwd_only_fn():
        with torch.no_grad():
            res = b2_forward(means, quats, scales, opacities, colors, vm, K,
                             width, height, handle, sh_degree=SH_DEGREE,
                             radius_clip=radius_clip)
        return res["frame"]

    torch.cuda.nvtx.range_push("B2_forward_repeated")
    result["B2_forward_timing"] = time_repeated(
        b2_fwd_only_fn, warmup=warmup, measure=measure, label="B2_fwd")
    torch.cuda.nvtx.range_pop()

    # B2 single forward for correctness + workload
    with torch.no_grad():
        b2_res = b2_forward(means, quats, scales, opacities, colors, vm, K,
                            width, height, handle, sh_degree=SH_DEGREE,
                            radius_clip=radius_clip)
    b2_render = b2_res["frame"].detach()  # [1,H,W,3]
    b2_alpha = b2_res["alpha"].detach()
    b2_meta = b2_res.get("metadata", {})

    def _serialize_meta(v):
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

    result["B2_metadata"] = {k: _serialize_meta(v) for k, v in b2_meta.items()} if isinstance(b2_meta, dict) else {}

    # B2 workload from metadata
    def _meta_int(key, default=-1):
        v = b2_meta.get(key, default)
        if isinstance(v, torch.Tensor):
            return int(v.item()) if v.numel() == 1 else int(v.sum().item())
        return int(v)

    def _meta_float(key, default=-1.0):
        v = b2_meta.get(key, default)
        if isinstance(v, torch.Tensor):
            return float(v.item()) if v.numel() == 1 else float(v.sum().item())
        return float(v)

    b2_wl = {
        "N_total": _meta_int("n_gaussians", len(means)),
        "N_visible": _meta_int("n_visible", -1),
        "N_isects": _meta_int("n_isects", -1),
        "N_isects_full": _meta_int("n_isects_full", -1),
        "culling_ratio": _meta_float("culling_ratio", -1.0),
    }
    # try to get tile info from metadata
    if "n_isects" in b2_meta and "n_visible" in b2_meta and b2_wl["N_visible"] > 0:
        b2_wl["tiles_per_gaussian"] = float(b2_wl["N_isects"] / b2_wl["N_visible"])
        b2_wl["pixels_per_gaussian"] = float(width * height / b2_wl["N_visible"])
    b2_wl["image_width"] = width
    b2_wl["image_height"] = height
    b2_wl["active_macro_tiles"] = "UNAVAILABLE"
    b2_wl["macro_tile_occupancy"] = "UNAVAILABLE"
    result["B2_workload"] = b2_wl

    # B2 forward+backward (autograd)
    def b2_fwd_bwd_fn():
        m = means.detach().clone().requires_grad_(True)
        q = quats.detach().clone().requires_grad_(True)
        s = scales.detach().clone().requires_grad_(True)
        o = opacities.detach().clone().requires_grad_(True)
        c = sh.detach().clone().requires_grad_(True)
        _HIGS_FROZEN_TRACKER.reset()
        h = create_higs_renderer(m, q, s, o, c, sh_degree=SH_DEGREE)
        torch.cuda.nvtx.range_push("B2_autograd_forward")
        res = b2_forward(m, q, s, o, c, vm, K, width, height, h,
                         sh_degree=SH_DEGREE, radius_clip=radius_clip)
        torch.cuda.nvtx.range_pop()
        loss = (res["frame"] - target).abs().mean()
        torch.cuda.nvtx.range_push("B2_autograd_backward")
        loss.backward()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        h.release()
        return m.grad, q.grad, s.grad, o.grad, c.grad

    # B2 backward correctness
    b2_grads = b2_fwd_bwd_fn()
    torch.cuda.synchronize()

    # B2 fwd+bwd timing
    torch.cuda.nvtx.range_push("B2_fwd_bwd_repeated")
    result["B2_fwd_bwd_timing"] = time_repeated(
        b2_fwd_bwd_fn, warmup=warmup, measure=measure, label="B2_fb")
    torch.cuda.nvtx.range_pop()

    # B2 backward_total = fwd_bwd_total - forward_total
    result["B2_backward_total_ms"] = (
        result["B2_fwd_bwd_timing"]["median_ms"] -
        result["B2_forward_timing"]["median_ms"]
    )
    result["B2_backward_separated"] = "SUBTRACTED"

    # release handle
    try:
        handle.release()
    except Exception:
        pass

    # ===== Forward correctness (B1 vs B2) =====
    r1 = b1_render.reshape(-1, 3).float()
    r2 = b2_render.reshape(-1, 3).float()
    a1 = b1_alpha.reshape(-1).float()
    a2 = b2_alpha.reshape(-1).float()
    fwd_corr = {
        "render_psnr_db": psnr(b1_render.float(), b2_render.float()),
        "render_max_abs": float((b1_render.float() - b2_render.float()).abs().max().item()),
        "render_mean_abs": float((b1_render.float() - b2_render.float()).abs().mean().item()),
        "render_relative_l2": relative_l2(r1, r2),
        "alpha_max_abs": float((a1 - a2).abs().max().item()),
        "alpha_mean_abs": float((a1 - a2).abs().mean().item()),
    }
    # depth not supported by this renderer path
    fwd_corr["depth"] = "NOT_SUPPORTED"
    result["forward_correctness"] = fwd_corr

    # ===== Backward correctness (B1 vs B2 gradients) =====
    grad_names = ["means", "quats", "scales", "opacities", "sh"]
    bwd_corr = {}
    for i, name in enumerate(grad_names):
        bwd_corr[name] = grad_compare(b1_grads[i], b2_grads[i], name)
    result["backward_correctness"] = bwd_corr

    # ===== Stage delta =====
    # Forward total
    result["B1_forward_total_ms"] = result["B1_forward_decomposed_timing"]["median_ms"]
    result["B2_forward_total_ms"] = result["B2_forward_timing"]["median_ms"]
    result["delta_forward_ms"] = result["B2_forward_total_ms"] - result["B1_forward_total_ms"]
    # Backward total
    result["delta_backward_ms"] = result["B2_backward_total_ms"] - result["B1_backward_total_ms"]
    # Total renderer (fwd + bwd)
    result["B1_renderer_total_ms"] = result["B1_forward_total_ms"] + result["B1_backward_total_ms"]
    result["B2_renderer_total_ms"] = result["B2_forward_total_ms"] + result["B2_backward_total_ms"]
    result["delta_renderer_total_ms"] = result["B2_renderer_total_ms"] - result["B1_renderer_total_ms"]
    result["speed_ratio_B2_over_B1"] = (
        result["B2_renderer_total_ms"] / result["B1_renderer_total_ms"]
        if result["B1_renderer_total_ms"] > 0 else 0
    )

    # Timing closure (already computed from repeated per-stage timing above)
    # result["timing_closure_B1_forward"] is set at line ~517 from repeated stages

    return result


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scenes", default="train,room,bicycle")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measure", type=int, default=100)
    ap.add_argument("--max-long-side", type=int, default=0,
                    help="Cap rendering resolution long side (0=native)")
    ap.add_argument("--radius-clip", type=float, default=0.0)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = f"cuda:{args.gpu}"

    # ---- provenance ----
    prov = capture_provenance(args.gpu)
    with open(os.path.join(args.out_dir, "environment.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print("[provenance]", json.dumps(prov, indent=2))

    # ---- scene configs ----
    scene_configs = {
        "train": {
            "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/tanksandtemples/train/native/point_cloud/iteration_30000/point_cloud.ply",
            "cams": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/tanksandtemples/train/native/cameras.json",
            "native_w": 1959, "native_h": 1090,
        },
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

    scenes = args.scenes.split(",")
    all_results = []

    for scene_name in scenes:
        cfg = scene_configs[scene_name]
        print(f"\n{'='*60}\n[scene] {scene_name}\n{'='*60}")

        # determine resolution
        nw, nh = cfg["native_w"], cfg["native_h"]
        if args.max_long_side > 0 and max(nw, nh) > args.max_long_side:
            scale = args.max_long_side / max(nw, nh)
            width = max(1, int(round(nw * scale)))
            height = max(1, int(round(nh * scale)))
        else:
            width, height = nw, nh
        print(f"[res] {width}x{height} (native {nw}x{nh})")

        # load scene
        print("[load] PLY...", cfg["ply"])
        means, quats, scales, opacities, sh = load_ply_scene(cfg["ply"], device)
        print(f"[load] N={len(means)} Gaussians")
        viewmats, Ks, cams = load_cameras(cfg["cams"], width, height, device)
        print(f"[load] {len(cams)} cameras")

        # select 3 cameras: 0, middle, last
        n_cams = len(cams)
        cam_indices = [0, n_cams // 2, n_cams - 1]
        print(f"[cams] selected indices: {cam_indices}")

        for cam_idx in cam_indices:
            print(f"\n[profile] {scene_name} cam {cam_idx} ({cams[cam_idx]['img_name']})")
            try:
                res = profile_scene_camera(
                    scene_name, cam_idx, cams,
                    means, quats, scales, opacities, sh,
                    viewmats, Ks, width, height, device,
                    args.warmup, args.measure, radius_clip=args.radius_clip,
                )
                # save individual result
                fname = f"{scene_name}_cam{cam_idx}.json"
                with open(os.path.join(args.out_dir, fname), "w") as f:
                    # strip large arrays
                    def clean(o):
                        if isinstance(o, dict):
                            return {k: clean(v) for k, v in o.items()
                                    if k not in ("_backward_ctx",)}
                        if isinstance(o, list) and len(o) > 200:
                            return f"[{len(o)} items]"
                        if isinstance(o, torch.Tensor):
                            return o.item() if o.numel() == 1 else f"tensor{list(o.shape)}"
                        return o
                    json.dump(clean(res), f, indent=2, default=str)
                all_results.append(res)
                print(f"[result] B1_fwd={res['B1_forward_total_ms']:.2f}ms "
                      f"B2_fwd={res['B2_forward_total_ms']:.2f}ms "
                      f"delta_fwd={res['delta_forward_ms']:+.2f}ms")
                print(f"          B1_bwd={res['B1_backward_total_ms']:.2f}ms "
                      f"B2_bwd={res['B2_backward_total_ms']:.2f}ms "
                      f"delta_bwd={res['delta_backward_ms']:+.2f}ms")
                print(f"          PSNR(B1,B2)={res['forward_correctness']['render_psnr_db']:.2f}dB")
            except Exception as e:
                print(f"[ERROR] {scene_name} cam {cam_idx}: {e}")
                traceback.print_exc()
                all_results.append({
                    "scene": scene_name, "camera_idx": cam_idx,
                    "error": str(e), "traceback": traceback.format_exc(),
                })

    # save combined results
    with open(os.path.join(args.out_dir, "all_results.json"), "w") as f:
        def clean(o):
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()
                        if k not in ("_backward_ctx",)}
            if isinstance(o, list) and len(o) > 200:
                return f"[{len(o)} items]"
            if isinstance(o, torch.Tensor):
                return o.item() if o.numel() == 1 else f"tensor{list(o.shape)}"
            return o
        json.dump(clean(all_results), f, indent=2, default=str)

    print(f"\n{'='*60}\n[DONE] {len(all_results)} results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
