"""
C19-0: PyTorch profiler for kernel-level GPU metrics.

Uses torch.profiler to capture kernel names, durations, and launch parameters
from the gsplat rasterization pipeline.
"""
import torch, gsplat, os, sys, math, json
from torch.profiler import profile, ProfilerActivity, record_function

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo, "src"))
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras

DEVICE = "cuda"
torch.set_grad_enabled(False)

scene = load_ply(os.path.join(repo, "data", "official", "mipnerf360", "room", "point_cloud.ply"), device=DEVICE)
cameras = load_cameras_from_json(os.path.join(repo, "data", "official", "mipnerf360", "room", "cameras.json"), device=DEVICE)
cameras = resize_cameras(cameras, 1920, 1080)
cam = cameras[0]
W, H = 1920, 1080

means3d = scene["xyz"].contiguous()
quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
scales = scene["scales"].exp().contiguous()
opacities = torch.sigmoid(scene["opacity"]).contiguous()
shs = scene["shs"].contiguous()
viewmats = cam.world_view_transform.unsqueeze(0).contiguous()
Ks = cam.K.unsqueeze(0).contiguous()
bg = torch.zeros(1, 3, device=DEVICE)

# Warmup
for _ in range(3):
    _ = gsplat.rasterization(means=means3d, quats=quats, scales=scales,
        opacities=opacities, colors=shs, viewmats=viewmats, Ks=Ks,
        width=W, height=H, near_plane=0.01, far_plane=1e10,
        radius_clip=0.0, eps2d=0.3, sh_degree=3, packed=False, tile_size=16,
        backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
        rasterize_mode="classic")
torch.cuda.synchronize()

print("Profiling full forward pass with torch.profiler...")

with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, with_stack=False,
             profile_memory=False) as prof:
    with record_function("full_rasterization"):
        out, alpha, meta = gsplat.rasterization(means=means3d, quats=quats, scales=scales,
            opacities=opacities, colors=shs, viewmats=viewmats, Ks=Ks,
            width=W, height=H, near_plane=0.01, far_plane=1e10,
            radius_clip=0.0, eps2d=0.3, sh_degree=3, packed=False, tile_size=16,
            backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
            rasterize_mode="classic")
torch.cuda.synchronize()

cuda_events = []
for evt in prof.events():
    if evt.device_type == torch.profiler.DeviceType.CUDA:
        cuda_events.append({
            "name": evt.name,
            "cuda_time_us": evt.cuda_time_total,
            "cuda_time_ms": round(evt.cuda_time_total / 1000, 3),
            "count": evt.count,
        })

cuda_events.sort(key=lambda x: -x["cuda_time_us"])
total_cuda = sum(e["cuda_time_us"] for e in cuda_events)

print(f"\nTotal CUDA time: {total_cuda/1000:.3f}ms")
print(f"\n{'Kernel':70s} {'Total(ms)':>10s} {'Cnt':>4s} {'Frac':>7s}")
print("="*92)
for e in cuda_events:
    frac = e["cuda_time_us"] / total_cuda * 100
    print(f"{e['name'][:70]:70s} {e['cuda_time_ms']:>10.3f} {e['count']:>4d} {frac:>6.2f}%")

out_dir = os.path.join(repo, "results", "phase-c19", "profile")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "torch_profiler_kernels.json"), "w") as f:
    json.dump({"total_cuda_us": total_cuda, "kernels": cuda_events}, f, indent=2)
print(f"\nSaved to {out_dir}/torch_profiler_kernels.json")
