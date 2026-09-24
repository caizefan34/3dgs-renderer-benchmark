#!/usr/bin/env python3
"""Generate B1/B2 kernel inventory CSVs using torch profiler (CUDA kernel-level)."""
import json, os, sys, csv, math
import torch
import numpy as np
from collections import defaultdict

# Add paths
H = "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx"
CACHE = os.path.expanduser("~/.cache/torch_extensions/py310_cu128")
sys.path.insert(0, H)
sys.path.insert(0, os.path.join(CACHE, "gsplat_scene_cuda"))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONNOUSERSITE"] = "1"

import gsplat
import gsplat.experimental
from plyfile import PlyData

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/storage_pool/3dgs-renderer-benchmark/repo/artifacts/h1-clean-profile"
SCENE = sys.argv[2] if len(sys.argv) > 2 else "train"
CAM_IDX = int(sys.argv[3]) if len(sys.argv) > 3 else 0

SCENES = {
    "train": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/tanksandtemples/train/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/tanksandtemples/train/native/cameras.json",
    },
    "room": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json",
    },
    "bicycle": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/bicycle/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/bicycle/cameras.json",
    },
}

def load_ply(path, device):
    ply = PlyData.read(path)
    v = ply["vertex"]
    N = len(v)
    def arr(name):
        return torch.tensor(v[name], dtype=torch.float32, device=device)
    means = torch.stack([arr("x"), arr("y"), arr("z")], dim=-1)
    sh0 = torch.stack([arr("f_dc_0"), arr("f_dc_1"), arr("f_dc_2")], dim=-1).unsqueeze(1)
    opacities = arr("opacity")
    scales = torch.stack([arr("scale_0"), arr("scale_1"), arr("scale_2")], dim=-1)
    quats = torch.stack([arr("rot_0"), arr("rot_1"), arr("rot_2"), arr("rot_3")], dim=-1)
    f_rest = []
    K_SH = 16
    for i in range(1, K_SH):
        f_rest.append(torch.stack([arr("f_rest_%d" % (3*(i-1)+j)) for j in range(3)], dim=-1))
    f_rest = torch.stack(f_rest, dim=1)
    sh = torch.zeros(N, K_SH, 3, dtype=torch.float32, device=device)
    sh[:, 0] = sh0.squeeze(1)
    sh[:, 1:] = f_rest
    return means, quats, scales, torch.sigmoid(opacities), sh

def load_cameras(path, width, height, device):
    cams = json.load(open(path))
    R = np.asarray(cams[CAM_IDX]["rotation"], dtype=np.float64)
    p = np.asarray(cams[CAM_IDX]["position"], dtype=np.float64)
    Rw2c = R.T
    vm = np.eye(4)
    vm[:3, :3] = Rw2c
    vm[:3, 3] = -Rw2c @ p
    scale = width / float(cams[CAM_IDX]["width"])
    K = np.array([[float(cams[CAM_IDX]["fx"]) * scale, 0, (width-1)/2],
                  [0, float(cams[CAM_IDX]["fy"]) * scale, (height-1)/2],
                  [0, 0, 1]], dtype=np.float64)
    vm = torch.tensor(vm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    K = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    return vm, K

device = torch.device("cuda:0")
cfg = SCENES[SCENE]
width, height = 1024, 570
if SCENE != "train":
    width, height = 2048, 1365

means, quats, scales, opacities, sh = load_ply(cfg["ply"], device)
vm, K = load_cameras(cfg["cams"], width, height, device)

from gsplat.cuda._wrapper import fully_fused_projection, isect_tiles, isect_offset_encode, _make_lazy_cuda_func
from gsplat.rendering import _maybe_evaluate_sh, rasterization
from gsplat.experimental.render.functional.gaussian_inference import create_higs_renderer, _HIGS_FROZEN_TRACKER
from gsplat.experimental import rasterize_gaussian_higs_frozen

SH_DEGREE = 3

def b1_forward():
    with torch.no_grad():
        radii, means2d, depths, conics, _ = fully_fused_projection(
            means=means.unsqueeze(0).contiguous(), covars=None,
            quats=quats.unsqueeze(0).contiguous(), scales=scales.unsqueeze(0).contiguous(),
            viewmats=vm, Ks=K, width=width, height=height,
            eps2d=0.3, packed=False, calc_compensations=False, camera_model="pinhole",
        )
        C = 1
        N = means.shape[0]
        colors_eval = _maybe_evaluate_sh(SH_DEGREE, sh, means.unsqueeze(0), radii, vm, (1,), C, N, True).contiguous()
        opac_bc = torch.broadcast_to(opacities.unsqueeze(0)[..., None, :], (1, C, N)).contiguous()
        tile_size = 16
        tw = math.ceil(width / tile_size)
        th = math.ceil(height / tile_size)
        _, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, depths, tile_size, tw, th,
            packed=False, n_images=C, conics=conics, opacities=opac_bc,
        )
        isect_offsets = isect_offset_encode(isect_ids, C, tw, th).reshape((1, C, th, tw))
        bg = torch.zeros((1, C, 3), device=device)
        render_colors, render_alphas, _, _ = _make_lazy_cuda_func("rasterize_to_pixels_3dgs")(
            means2d.contiguous(), conics.contiguous(), colors_eval.contiguous(),
            opac_bc.contiguous(), bg, None, width, height, tile_size,
            isect_offsets.contiguous(), flatten_ids.contiguous(), False, False,
        )
    return render_colors, render_alphas

def b2_forward():
    _HIGS_FROZEN_TRACKER.reset()
    handle = create_higs_renderer(means, quats, scales, opacities, sh, sh_degree=SH_DEGREE)
    with torch.no_grad():
        res = rasterize_gaussian_higs_frozen(
            means, quats, scales, opacities, sh,
            backward_mode="higs_native", scene=handle, freeze_topology=True,
            viewmats=vm, Ks=K, width=width, height=height,
            sh_degree=SH_DEGREE, use_higs_culling=True, tile_sampling_ratio=1.0,
        )
    handle.release()
    return res["frame"], res["alpha"]

def profile_kernels(fn, label, warmup=5, measure=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(measure):
            fn()
        torch.cuda.synchronize()
    # Extract CUDA kernel events
    events = prof.key_averages()
    kernels = []
    for e in events:
        if e.device_type == torch.profiler.DeviceType.CUDA:
            cuda_total = getattr(e, "self_device_time_total", 0) or getattr(e, "device_time_total", 0) or getattr(e, "cuda_time_total", 0)
            if cuda_total and cuda_total > 0:
                kernels.append({
                    "name": e.key,
                    "calls": e.count,
                    "total_us": cuda_total,
                    "mean_us": cuda_total / e.count if e.count > 0 else 0,
                })
    # Sort by total time descending
    kernels.sort(key=lambda x: x["total_us"], reverse=True)
    total_gpu_us = sum(k["total_us"] for k in kernels)
    fname = os.path.join(OUT_DIR, "kernel_inventory_%s.csv" % label)
    with open(fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel_name", "call_count", "total_us", "mean_us",
                    "pct_of_gpu_time", "calls_per_iter"])
        for k in kernels:
            pct = 100.0 * k["total_us"] / total_gpu_us if total_gpu_us > 0 else 0
            w.writerow([k["name"], k["calls"], k["total_us"], k["mean_us"],
                        "%.2f" % pct, k["calls"] / measure])
    print("Wrote %s (%d kernels, total_gpu=%.1fus per %d iters)" % (fname, len(kernels), total_gpu_us, measure))
    return kernels

print("Profiling B1 kernels (scene=%s cam=%d)..." % (SCENE, CAM_IDX))
b1_kernels = profile_kernels(b1_forward, "B1", warmup=5, measure=20)
print("Profiling B2 kernels (scene=%s cam=%d)..." % (SCENE, CAM_IDX))
b2_kernels = profile_kernels(b2_forward, "B2", warmup=5, measure=20)
print("Done.")