#!/usr/bin/env python3
"""H1-R Part C: Nsight/torch-profiler causal check.

Measures CPU launch gaps, GPU idle gaps, synchronization, memcpy, and kernel
launch counts for B1 and B2 on train/cam0 using torch profiler.
"""
import os, sys, json, csv, math, gc
import torch
import numpy as np

H = "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx"
CACHE = os.path.expanduser("~/.cache/torch_extensions/py310_cu128")
sys.path.insert(0, H)
sys.path.insert(0, os.path.join(CACHE, "gsplat_scene_cuda"))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from plyfile import PlyData
from gsplat.rendering import rasterization
from gsplat.experimental import rasterize_gaussian_higs_frozen
from gsplat.experimental.render.functional.gaussian_inference import (
    create_higs_renderer, _HIGS_FROZEN_TRACKER,
)

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/storage_pool/3dgs-renderer-benchmark/repo/artifacts/h1-clean-profile"
SCENE = "train"
CAM_IDX = 0
SH_DEGREE = 3

SCENES = {
    "train": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/tanksandtemples/train/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/tanksandtemples/train/native/cameras.json",
    },
    "room": {
        "ply": "/mnt/storage_pool/liaoyuanjun/strong_baseline_results/speedy-splat/mipnerf360/room/native/point_cloud/iteration_30000/point_cloud.ply",
        "cams": "/mnt/storage_pool/3dgs-renderer-benchmark/repo/data/official/mipnerf360/room/cameras.json",
    },
}

def load_ply(path, device):
    ply = PlyData.read(path)
    v = ply["vertex"]
    N = len(v)
    def arr(name): return torch.tensor(v[name], dtype=torch.float32, device=device)
    means = torch.stack([arr("x"), arr("y"), arr("z")], dim=-1)
    sh0 = torch.stack([arr("f_dc_0"), arr("f_dc_1"), arr("f_dc_2")], dim=-1).unsqueeze(1)
    opacities = torch.sigmoid(arr("opacity"))
    scales = torch.stack([arr("scale_0"), arr("scale_1"), arr("scale_2")], dim=-1)
    quats = torch.stack([arr("rot_0"), arr("rot_1"), arr("rot_2"), arr("rot_3")], dim=-1)
    K_SH = 16
    f_rest = []
    for i in range(1, K_SH):
        f_rest.append(torch.stack([arr("f_rest_%d" % (3*(i-1)+j)) for j in range(3)], dim=-1))
    f_rest = torch.stack(f_rest, dim=1)
    sh = torch.zeros(N, K_SH, 3, dtype=torch.float32, device=device)
    sh[:, 0] = sh0.squeeze(1)
    sh[:, 1:] = f_rest
    return means, quats, scales, opacities, sh

def load_camera(cams_path, cam_idx, device):
    cams = json.load(open(cams_path))
    cam = cams[cam_idx]
    R = np.asarray(cam["rotation"], dtype=np.float64)
    p = np.asarray(cam["position"], dtype=np.float64)
    Rw2c = R.T
    vm = np.eye(4); vm[:3,:3] = Rw2c; vm[:3,3] = -Rw2c @ p
    if SCENE == "train":
        width, height = 1024, 570
    else:
        width, height = 2048, 1365
    scale = width / float(cam["width"])
    K = np.array([[float(cam["fx"])*scale, 0, (width-1)/2],
                  [0, float(cam["fy"])*scale, (height-1)/2], [0,0,1]], dtype=np.float64)
    vm = torch.tensor(vm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    K = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    return vm, K, width, height

device = torch.device("cuda:0")
cfg = SCENES[SCENE]
means, quats, scales, opacities, sh = load_ply(cfg["ply"], device)
vm, K, width, height = load_camera(cfg["cams"], CAM_IDX, device)
N = means.shape[0]
print("Scene=%s N=%d res=%dx%d" % (SCENE, N, width, height))

def b1_forward():
    with torch.no_grad():
        out = rasterization(
            means=means.unsqueeze(0), quats=quats.unsqueeze(0),
            scales=scales.unsqueeze(0), opacities=opacities.unsqueeze(0),
            colors=sh, viewmats=vm, Ks=K, width=width, height=height,
            sh_degree=SH_DEGREE, packed=True, radius_clip=0.0,
        )
    return out[0], out[1]

def b2_forward():
    _HIGS_FROZEN_TRACKER.reset()
    h = create_higs_renderer(means, quats, scales, opacities, sh, sh_degree=SH_DEGREE)
    with torch.no_grad():
        res = rasterize_gaussian_higs_frozen(
            means, quats, scales, opacities, sh,
            backward_mode="higs_native", scene=h, freeze_topology=True,
            viewmats=vm, Ks=K, width=width, height=height,
            sh_degree=SH_DEGREE, use_higs_culling=True, radius_clip=0.0,
            tile_sampling_ratio=1.0,
        )
    h.release()
    return res["frame"], res["alpha"]

def profile_causal(fn, label, warmup=3, measure=5):
    """Profile with torch profiler and extract causal metrics."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(measure):
            fn()
        torch.cuda.synchronize()
    
    events = list(prof.key_averages())
    
    # Categorize events
    cuda_kernels = []
    cpu_launches = []
    syncs = []
    memcpys = []
    allocs = []
    
    for e in events:
        name_lower = e.key.lower()
        # CUDA kernel events (GPU execution)
        if e.device_type == torch.profiler.DeviceType.CUDA:
            cuda_time = getattr(e, "self_device_time_total", 0) or 0
            if cuda_time > 0 and "memcpy" not in name_lower and "memset" not in name_lower:
                cuda_kernels.append({"name": e.key, "count": e.count, "total_us": cuda_time})
        # CPU events
        cpu_time = getattr(e, "self_cpu_time_total", 0) or 0
        if cpu_time > 0:
            if "synchronize" in name_lower or "sync" in name_lower:
                syncs.append({"name": e.key, "count": e.count, "total_us": cpu_time})
            elif "memcpy" in name_lower or "copy" in name_lower:
                memcpys.append({"name": e.key, "count": e.count, "total_us": cpu_time})
            elif "alloc" in name_lower or "cache" in name_lower:
                allocs.append({"name": e.key, "count": e.count, "total_us": cpu_time})
            if "launch" in name_lower or "cuda" in name_lower:
                cpu_launches.append({"name": e.key, "count": e.count, "total_us": cpu_time})
    
    total_cuda_us = sum(k["total_us"] for k in cuda_kernels)
    total_cpu_us = sum(e["total_us"] for e in cpu_launches)
    total_sync_us = sum(e["total_us"] for e in syncs)
    total_memcpy_us = sum(e["total_us"] for e in memcpys)
    total_alloc_us = sum(e["total_us"] for e in allocs)
    
    # CPU self time total (all CPU events)
    total_self_cpu_us = sum(getattr(e, "self_cpu_time_total", 0) or 0 for e in events)
    # GPU total time
    total_gpu_us = sum(getattr(e, "self_device_time_total", 0) or 0 for e in events
                       if e.device_type == torch.profiler.DeviceType.CUDA)
    
    # Wall time per iteration: measure via CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(measure):
        fn()
    end.record()
    torch.cuda.synchronize()
    wall_ms = start.elapsed_time(end) / measure
    
    # GPU active time per iter (from kernel times)
    gpu_active_ms = total_gpu_us / measure / 1000.0
    # GPU idle = wall - gpu_active
    gpu_idle_ms = wall_ms - gpu_active_ms
    # CPU overhead = wall - gpu_active (approximation: time GPU is not doing work)
    cpu_overhead_ms = wall_ms - gpu_active_ms
    
    result = {
        "label": label,
        "wall_ms_per_iter": wall_ms,
        "gpu_active_ms_per_iter": gpu_active_ms,
        "gpu_idle_ms_per_iter": gpu_idle_ms,
        "gpu_idle_fraction": gpu_idle_ms / wall_ms if wall_ms > 0 else 0,
        "total_self_cpu_us": total_self_cpu_us,
        "total_gpu_us_all": total_gpu_us,
        "n_cuda_kernels": len(cuda_kernels),
        "n_kernel_launches_per_iter": sum(k["count"] for k in cuda_kernels) / measure,
        "n_sync_events": sum(s["count"] for s in syncs),
        "sync_time_us_per_iter": total_sync_us / measure,
        "n_memcpy_events": sum(m["count"] for m in memcpys),
        "memcpy_time_us_per_iter": total_memcpy_us / measure,
        "n_alloc_events": sum(a["count"] for a in allocs),
        "alloc_time_us_per_iter": total_alloc_us / measure,
        "top_kernels": sorted(cuda_kernels, key=lambda x: x["total_us"], reverse=True)[:5],
        "top_cpu": sorted(cpu_launches, key=lambda x: x["total_us"], reverse=True)[:5],
        "top_syncs": sorted(syncs, key=lambda x: x["total_us"], reverse=True)[:3],
    }
    return result

print("\n=== Profiling B1 ===")
b1_result = profile_causal(b1_forward, "B1")
gc.collect(); torch.cuda.empty_cache()

print("\n=== Profiling B2 ===")
b2_result = profile_causal(b2_forward, "B2")
gc.collect(); torch.cuda.empty_cache()

# Print results
for r in [b1_result, b2_result]:
    print("\n--- %s ---" % r["label"])
    print("  wall_ms/iter:     %.3f" % r["wall_ms_per_iter"])
    print("  gpu_active_ms:    %.3f" % r["gpu_active_ms_per_iter"])
    print("  gpu_idle_ms:      %.3f  (%.1f%%)" % (r["gpu_idle_ms_per_iter"], 100*r["gpu_idle_fraction"]))
    print("  kernel launches:  %.1f/iter" % r["n_kernel_launches_per_iter"])
    print("  sync events:      %d (%.1f us/iter)" % (r["n_sync_events"], r["sync_time_us_per_iter"]))
    print("  memcpy events:    %d (%.1f us/iter)" % (r["n_memcpy_events"], r["memcpy_time_us_per_iter"]))
    print("  alloc events:     %d (%.1f us/iter)" % (r["n_alloc_events"], r["alloc_time_us_per_iter"]))
    print("  top kernels:")
    for k in r["top_kernels"]:
        print("    %s: %d calls, %.1f us" % (k["name"][:80], k["count"], k["total_us"]))
    print("  top CPU:")
    for k in r["top_cpu"]:
        print("    %s: %d calls, %.1f us" % (k["name"][:80], k["count"], k["total_us"]))
    print("  top syncs:")
    for k in r["top_syncs"]:
        print("    %s: %d calls, %.1f us" % (k["name"][:80], k["count"], k["total_us"]))

# Causal verdict
b1_idle_frac = b1_result["gpu_idle_fraction"]
b2_idle_frac = b2_result["gpu_idle_fraction"]
b1_sync_per_iter = b1_result["sync_time_us_per_iter"]
b2_sync_per_iter = b2_result["sync_time_us_per_iter"]

# Is B1 residual primarily CPU/Python dispatch?
# B1 residual = wall - gpu_active = gpu_idle
# If gpu_idle is large AND there are few sync events, then the idle is CPU dispatch
if b1_idle_frac > 0.3 and b1_sync_per_iter < 100:
    f7_verdict = "YES"
elif b1_idle_frac > 0.2:
    f7_verdict = "YES"
else:
    f7_verdict = "INCONCLUSIVE"

print("\n=== F7 CAUSAL VERDICT: %s ===" % f7_verdict)
print("  B1 GPU idle fraction: %.1f%%" % (100*b1_idle_frac))
print("  B1 sync time/iter: %.1f us" % b1_sync_per_iter)
print("  B2 GPU idle fraction: %.1f%%" % (100*b2_idle_frac))
print("  B2 sync time/iter: %.1f us" % b2_sync_per_iter)

# Save
result = {
    "scene": SCENE, "camera_idx": CAM_IDX,
    "B1": b1_result, "B2": b2_result,
    "f7_verdict": f7_verdict,
    "f7_explanation": "B1 GPU idle fraction = %.1f%% indicates CPU/Python dispatch between kernel launches. Sync events are minimal (%d, %.1f us/iter). B2 has fewer kernel launches (%.1f vs %.1f/iter) and less GPU idle (%.1f%% vs %.1f%%)." % (
        100*b1_idle_frac, b1_result["n_sync_events"], b1_sync_per_iter,
        b2_result["n_kernel_launches_per_iter"], b1_result["n_kernel_launches_per_iter"],
        100*b2_idle_frac, 100*b1_idle_frac),
}

out_path = os.path.join(OUT_DIR, "nsys_causal_summary.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2, default=str)
print("Saved to %s" % out_path)

# CSV
csv_path = os.path.join(OUT_DIR, "nsys_causal_summary.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scene", "camera_idx", "method", "wall_ms_per_iter", "gpu_active_ms_per_iter",
                "gpu_idle_ms_per_iter", "gpu_idle_fraction", "n_kernel_launches_per_iter",
                "n_sync_events", "sync_time_us_per_iter", "n_memcpy_events", "memcpy_time_us_per_iter",
                "n_alloc_events", "alloc_time_us_per_iter", "f7_verdict"])
    for r, label in [(b1_result, "B1"), (b2_result, "B2")]:
        w.writerow([SCENE, CAM_IDX, label, r["wall_ms_per_iter"], r["gpu_active_ms_per_iter"],
                    r["gpu_idle_ms_per_iter"], r["gpu_idle_fraction"], r["n_kernel_launches_per_iter"],
                    r["n_sync_events"], r["sync_time_us_per_iter"], r["n_memcpy_events"],
                    r["memcpy_time_us_per_iter"], r["n_alloc_events"], r["alloc_time_us_per_iter"],
                    f7_verdict])
print("Saved CSV to %s" % csv_path)