"""
C19-1: Anomaly investigation - tile=16 and tile=28 timing spikes.
Run multiple times to check reproducibility.
Also measure isect_tiles with sort=False to isolate intersect vs sort cost.
"""
import torch, gsplat, json, os, sys, math, numpy as np
from datetime import datetime

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
scales_exp = scene["scales"].exp().contiguous()
opacities = torch.sigmoid(scene["opacity"]).contiguous()
shs = scene["shs"].contiguous()
viewmats = cam.world_view_transform.unsqueeze(0).contiguous()
Ks = cam.K.unsqueeze(0).contiguous()
bg = torch.zeros(1, 3, device=DEVICE)

# Get meta
out_ref, alpha_ref, meta_ref = gsplat.rasterization(
    means=means3d, quats=quats, scales=scales_exp, opacities=opacities, colors=shs,
    viewmats=viewmats, Ks=Ks, width=W, height=H, near_plane=0.01, far_plane=1e10,
    radius_clip=0.0, eps2d=0.3, sh_degree=3, packed=False, tile_size=16,
    backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
    rasterize_mode="classic")
torch.cuda.synchronize()
m2d = meta_ref["means2d"].contiguous()
radii = meta_ref["radii"].contiguous()
depths = meta_ref["depths"].contiguous()
conics = meta_ref["conics"].contiguous()
opac = meta_ref["opacities"].contiguous()
cam_center = cam.camera_center.to(DEVICE)
view_dirs = cam_center - means3d
view_dirs = view_dirs / view_dirs.norm(dim=-1, keepdim=True)
colors_sh = gsplat.spherical_harmonics(3, view_dirs, shs)

ev_s, ev_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

print("="*70)
print("C19-1 Anomaly Investigation")
print("Reproducibility study across tile sizes")
print("="*70)

N_RUNS = 30

for ts in [12, 16, 20, 24, 28, 32]:
    tw = math.ceil(W / ts)
    th = math.ceil(H / ts)
    n_tiles = tw * th

    # isect_tiles WITH sort
    tsort_list = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize(); ev_s.record()
        gsplat.isect_tiles(m2d, radii, depths, ts, tw, th, sort=True)
        ev_e.record(); torch.cuda.synchronize()
        tsort_list.append(ev_s.elapsed_time(ev_e))
    _, iid, fid = gsplat.isect_tiles(m2d, radii, depths, ts, tw, th, sort=True)

    # isect_tiles WITHOUT sort
    tnsort_list = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize(); ev_s.record()
        gsplat.isect_tiles(m2d, radii, depths, ts, tw, th, sort=False)
        ev_e.record(); torch.cuda.synchronize()
        tnsort_list.append(ev_s.elapsed_time(ev_e))
    _, iid_ns, fid_ns = gsplat.isect_tiles(m2d, radii, depths, ts, tw, th, sort=False)

    tsort_m = sum(tsort_list)/len(tsort_list)
    tnsort_m = sum(tnsort_list)/len(tnsort_list)
    sort_est = tsort_m - tnsort_m

    # Timing CV
    tsort_cv = np.std(tsort_list)/tsort_m
    tnsort_cv = np.std(tnsort_list)/tnsort_m

    n_int = iid.shape[0]
    n_int_ns = iid_ns.shape[0]

    print(f"\n  tile_size={ts:2d}:  ints={n_int:,}")
    print(f"    intersect+sort: {tsort_m:.4f}ms  CV={tsort_cv:.3f}")
    print(f"    intersect only: {tnsort_m:.4f}ms  CV={tnsort_cv:.3f}")
    print(f"    sort estimated: {sort_est:.4f}ms")
    print(f"    ints no-sort:   {n_int_ns:,}")

# Investigate if the sort kernel uses different CUB policies at different N
print("\n\n--- Deep dive: CUB sort kernel count vs N ---")
from torch.profiler import profile, ProfilerActivity

for ts in [12, 16, 20, 24, 28, 32]:
    tw = math.ceil(W / ts)
    th = math.ceil(H / ts)
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        _, iid, fid = gsplat.isect_tiles(m2d, radii, depths, ts, tw, th, sort=True)
    torch.cuda.synchronize()

    n = iid.shape[0]
    sort_kernels = 0
    sort_time_us = 0
    for evt in prof.events():
        if evt.device_type == torch.profiler.DeviceType.CUDA:
            if "RadixSort" in evt.name or "DeviceScan" in evt.name:
                sort_kernels += 1
                sort_time_us += evt.cuda_time_total if hasattr(evt, 'cuda_time_total') else (evt.device_time_total if hasattr(evt, 'device_time_total') else 0)
    print(f"  ts={ts:2d}  N={n:>8,}  sort_kernels={sort_kernels:2d}  sort_time={sort_time_us/1000:.4f}ms")

print("\nDONE")
