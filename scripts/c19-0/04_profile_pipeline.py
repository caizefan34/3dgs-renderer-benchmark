"""
C19-0: Comprehensive A100 profiling pipeline.

Performs:
  - Environment verification
  - End-to-end forward/backward timing with CUDA events
  - Per-step kernel timing
  - Tile workload statistics
  - Tile-size scaling (8, 16, 32)
  - Saves structured results to JSON
"""
import torch, gsplat, json, os, sys, math, numpy as np
from datetime import datetime
from collections import OrderedDict

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.ndarray,)): return obj.tolist()
        if isinstance(obj, torch.Tensor): return obj.detach().cpu().tolist()
        return super().default(obj)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras

DEVICE = "cuda"
torch.set_grad_enabled(True)
N_RUNS = 20

# =========================================================================
# 1. Load scene
# =========================================================================
print("=== Loading scene ===")
scene = load_ply(os.path.join(REPO_ROOT, "data", "official", "mipnerf360", "room", "point_cloud.ply"), device=DEVICE)
cameras = load_cameras_from_json(os.path.join(REPO_ROOT, "data", "official", "mipnerf360", "room", "cameras.json"), device=DEVICE)
cameras = resize_cameras(cameras, 1920, 1080)
cam = cameras[0]
W, H = 1920, 1080
N_G = scene["num_points"]
print(f"  Gaussians: {N_G}, Resolution: {W}x{H}")

means3d = scene["xyz"].contiguous()
quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
scales = scene["scales"].exp().contiguous()
opacities = torch.sigmoid(scene["opacity"]).contiguous()
shs = scene["shs"].contiguous()
viewmats = cam.world_view_transform.unsqueeze(0).contiguous()
Ks = cam.K.unsqueeze(0).contiguous()
bg = torch.zeros(1, 3, device=DEVICE)

# =========================================================================
# 2. End-to-end baseline
# =========================================================================
print("\n=== Step 2: End-to-End Baseline ===")

def fwd_pass(do_bwd=False):
    inp_m = means3d.detach().clone().requires_grad_(do_bwd)
    inp_q = quats.detach().clone().requires_grad_(do_bwd)
    inp_s = scales.detach().clone().requires_grad_(do_bwd)
    inp_o = opacities.detach().clone().requires_grad_(do_bwd)
    inp_c = shs.detach().clone().requires_grad_(do_bwd)
    out, _, _ = gsplat.rasterization(
        means=inp_m, quats=inp_q, scales=inp_s, opacities=inp_o, colors=inp_c,
        viewmats=viewmats, Ks=Ks, width=W, height=H, near_plane=0.01, far_plane=1e10,
        radius_clip=0.0, eps2d=0.3, sh_degree=3, packed=False, tile_size=16,
        backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
        rasterize_mode="classic")
    if do_bwd:
        out.sum().backward()
    return out

print("  Warmup...")
for _ in range(5): fwd_pass(False)

ev_s, ev_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

t_fwd = []
for _ in range(50):
    torch.cuda.synchronize(); ev_s.record()
    fwd_pass(False)
    ev_e.record(); torch.cuda.synchronize()
    t_fwd.append(ev_s.elapsed_time(ev_e))

t_fwd_b = []
for _ in range(50):
    torch.cuda.synchronize(); ev_s.record()
    fwd_pass(True)
    ev_e.record(); torch.cuda.synchronize()
    t_fwd_b.append(ev_s.elapsed_time(ev_e))

f_mean, f_med = sum(t_fwd)/len(t_fwd), sorted(t_fwd)[len(t_fwd)//2]
fb_mean, fb_med = sum(t_fwd_b)/len(t_fwd_b), sorted(t_fwd_b)[len(t_fwd_b)//2]
print(f"  Forward: mean={f_mean:.3f}ms median={f_med:.3f}ms")
print(f"  Forward+Bwd: mean={fb_mean:.3f}ms median={fb_med:.3f}ms")
print(f"  Backward(est): {fb_mean-f_mean:.3f}ms")

end_to_end = {"forward_mean_ms": round(f_mean,3), "forward_median_ms": round(f_med,3),
              "forward_backward_mean_ms": round(fb_mean,3), "forward_backward_median_ms": round(fb_med,3),
              "backward_estimated_ms": round(fb_mean-f_mean,3)}

# =========================================================================
# 3. Per-step kernel timing
# =========================================================================
print("\n=== Step 3: Kernel-Level Profiling ===")

# Get intermediate data from a single full call
out_rgb, out_alpha, meta = gsplat.rasterization(
    means=means3d, quats=quats, scales=scales, opacities=opacities, colors=shs,
    viewmats=viewmats, Ks=Ks, width=W, height=H, near_plane=0.01, far_plane=1e10,
    radius_clip=0.0, eps2d=0.3, sh_degree=3, packed=False, tile_size=16,
    backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
    rasterize_mode="classic")

m2d = meta["means2d"].contiguous()         # [1, N, 2]
radii_int = meta["radii"].contiguous()     # [1, N, 2]  int32
depths_m = meta["depths"].contiguous()     # [1, N]
conics_m = meta["conics"].contiguous()     # [1, N, 3]
opac_m = meta["opacities"].contiguous()    # [1, N]
tw, th = meta["tile_width"], meta["tile_height"]
# SH colors from view direction
cam_center = cam.camera_center.to(DEVICE)
view_dirs = cam_center - means3d
view_dirs = view_dirs / view_dirs.norm(dim=-1, keepdim=True)
colors_sh = gsplat.spherical_harmonics(3, view_dirs, shs)  # [N, 3]

# Compute 3a: fully_fused_projection only
print("  3a. fully_fused_projection...")
t_ffp = []
for _ in range(N_RUNS):
    torch.cuda.synchronize(); ev_s.record()
    gsplat.fully_fused_projection(means=means3d, covars=None, quats=quats, scales=scales,
        viewmats=viewmats, Ks=Ks, width=W, height=H, eps2d=0.3, near_plane=0.01,
        far_plane=1e10, radius_clip=0.0, packed=False, sparse_grad=False,
        calc_compensations=False, camera_model="pinhole", opacities=opacities)
    ev_e.record(); torch.cuda.synchronize()
    t_ffp.append(ev_s.elapsed_time(ev_e))
t_ffp_m = sum(t_ffp)/len(t_ffp)
print(f"    {t_ffp_m:.3f}ms")

# 3b: spherical_harmonics
print("  3b. spherical_harmonics...")
t_sh = []
for _ in range(N_RUNS):
    torch.cuda.synchronize(); ev_s.record()
    gsplat.spherical_harmonics(3, view_dirs, shs)
    ev_e.record(); torch.cuda.synchronize()
    t_sh.append(ev_s.elapsed_time(ev_e))
t_sh_m = sum(t_sh)/len(t_sh)
print(f"    {t_sh_m:.3f}ms")

# 3c: isect_tiles (Intersect Pass1 + CUB sort)
print("  3c. isect_tiles (intersect + sort)...")
t_isec = []
for _ in range(N_RUNS):
    torch.cuda.synchronize(); ev_s.record()
    gsplat.isect_tiles(m2d, radii_int, depths_m, 16, tw, th, sort=True)
    ev_e.record(); torch.cuda.synchronize()
    t_isec.append(ev_s.elapsed_time(ev_e))
_, iid, fid = gsplat.isect_tiles(m2d, radii_int, depths_m, 16, tw, th, sort=True)
t_isec_m = sum(t_isec)/len(t_isec)
print(f"    {t_isec_m:.3f}ms  isect_ids={tuple(iid.shape)} flatten_ids={tuple(fid.shape)}")

# 3d: isect_offset_encode
print("  3d. isect_offset_encode...")
t_off = []
for _ in range(N_RUNS):
    torch.cuda.synchronize(); ev_s.record()
    gsplat.isect_offset_encode(iid, 1, tw, th)
    ev_e.record(); torch.cuda.synchronize()
    t_off.append(ev_s.elapsed_time(ev_e))
ioff = gsplat.isect_offset_encode(iid, 1, tw, th)
t_off_m = sum(t_off)/len(t_off)
print(f"    {t_off_m:.3f}ms")

# 3e: rasterize_to_pixels
print("  3e. rasterize_to_pixels...")
t_rast = []
for _ in range(N_RUNS):
    torch.cuda.synchronize(); ev_s.record()
    gsplat.rasterize_to_pixels(m2d, conics_m, colors_sh.unsqueeze(0), opac_m, W, H, 16, ioff, fid, bg)
    ev_e.record(); torch.cuda.synchronize()
    t_rast.append(ev_s.elapsed_time(ev_e))
t_rast_m = sum(t_rast)/len(t_rast)
print(f"    {t_rast_m:.3f}ms")

# Ranking
total_sum = t_ffp_m + t_sh_m + t_isec_m + t_off_m + t_rast_m
print(f"\n  Sum of parts: {total_sum:.3f}ms  Full fwd: {f_mean:.3f}ms  Gap: {f_mean-total_sum:.3f}ms ({((f_mean-total_sum)/f_mean)*100:.1f}%)")

kernels = OrderedDict([("fully_fused_projection", t_ffp_m), ("spherical_harmonics", t_sh_m),
                       ("isect_tiles", t_isec_m), ("isect_offset_encode", t_off_m),
                       ("rasterize_to_pixels", t_rast_m)])
ranking = []
print("\n  === Kernel Ranking ===")
for name, t in sorted(kernels.items(), key=lambda x: -x[1]):
    frac = t/f_mean*100
    ranking.append({"kernel": name, "mean_ms": round(t,3), "fraction_of_forward": round(t/f_mean,4)})
    print(f"    {name:30s}: {t:8.3f}ms ({frac:5.1f}%)")
print(f"    {'unaccounted':30s}: {f_mean-total_sum:8.3f}ms ({(f_mean-total_sum)/f_mean*100:5.1f}%)")

# =========================================================================
# 6. Tile workload analysis (tile_size=16)
# =========================================================================
print("\n=== Step 6: Tile-Level Workload Analysis ===")
# isect_ids encode packed tile keys: (depth_key << 32) | (image_id * n_tiles + tile_id)
# For 1 image: tile_id = isect_id % (tile_width * tile_height)
n_tiles = tw * th
tile_ct = torch.zeros(n_tiles, dtype=torch.int64, device="cpu")
for tid in (iid % n_tiles).cpu(): tile_ct[tid.long()] += 1
tcn = tile_ct.numpy()
stcn = sorted(tcn)
def tp(p): return int(stcn[min(int(len(stcn)*p), len(stcn)-1)])
top5 = sum(sorted(tcn, reverse=True)[:max(int(len(tcn)*0.05),1)])
top1 = sum(sorted(tcn, reverse=True)[:max(int(len(tcn)*0.01),1)])
total_int = int(tcn.sum())
print(f"  Active tiles: {int((tcn>0).sum())}/{len(tcn)}")
print(f"  Mean: {tcn.mean():.1f}  P50/P90/P95/P99: {tp(0.5)}/{tp(0.9)}/{tp(0.95)}/{tp(0.99)}  Max: {int(tcn.max())}")
print(f"  Top 5%: {top5/total_int*100:.1f}%  Top 1%: {top1/total_int*100:.1f}%")
tile_stats_t16 = {"tile_size":16,"tile_width":tw,"tile_height":th,"total_tiles":len(tcn),
    "active_tiles":int((tcn>0).sum()),"total_intersections":total_int,
    "mean":round(float(tcn.mean()),2),"p50":tp(0.5),"p90":tp(0.9),"p95":tp(0.95),"p99":tp(0.99),
    "max":int(tcn.max()),"top5pct_fraction":round(top5/total_int,4),"top1pct_fraction":round(top1/total_int,4)}

# =========================================================================
# 7. Tile-size scaling
# =========================================================================
print("\n=== Step 7: Tile-Size Scaling ===")
def profile_ts(ts, nr=15):
    tw_, th_ = math.ceil(W/ts), math.ceil(H/ts)
    # isect_tiles
    tlist = []
    for _ in range(nr):
        torch.cuda.synchronize(); ev_s.record()
        gsplat.isect_tiles(m2d, radii_int, depths_m, ts, tw_, th_, sort=True)
        ev_e.record(); torch.cuda.synchronize()
        tlist.append(ev_s.elapsed_time(ev_e))
    _, iid_, fid_ = gsplat.isect_tiles(m2d, radii_int, depths_m, ts, tw_, th_, sort=True)
    tm_i = sum(tlist)/len(tlist)
    # offset
    tlist = []
    for _ in range(nr):
        torch.cuda.synchronize(); ev_s.record()
        gsplat.isect_offset_encode(iid_, 1, tw_, th_)
        ev_e.record(); torch.cuda.synchronize()
        tlist.append(ev_s.elapsed_time(ev_e))
    ioff_ = gsplat.isect_offset_encode(iid_, 1, tw_, th_)
    tm_o = sum(tlist)/len(tlist)
    # rasterize
    tlist = []
    for _ in range(nr):
        torch.cuda.synchronize(); ev_s.record()
        gsplat.rasterize_to_pixels(m2d, conics_m, colors_sh.unsqueeze(0), opac_m, W, H, ts, ioff_, fid_, bg)
        ev_e.record(); torch.cuda.synchronize()
        tlist.append(ev_s.elapsed_time(ev_e))
    tm_r = sum(tlist)/len(tlist)
    # full forward
    tlist = []
    for _ in range(nr):
        torch.cuda.synchronize(); ev_s.record()
        gsplat.rasterization(means=means3d, quats=quats, scales=scales,
            opacities=opacities, colors=shs, viewmats=viewmats, Ks=Ks,
            width=W, height=H, near_plane=0.01, far_plane=1e10,
            radius_clip=0.0, eps2d=0.3, sh_degree=3, packed=False, tile_size=ts,
            backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
            rasterize_mode="classic")
        ev_e.record(); torch.cuda.synchronize()
        tlist.append(ev_s.elapsed_time(ev_e))
    tm_f = sum(tlist)/len(tlist)
    # tile stats - isect_ids encode tile_id in lower bits
    tc = torch.zeros(tw_*th_, dtype=torch.int64, device="cpu")
    for tid in (iid_ % (tw_*th_)).cpu(): tc[tid.long()] += 1
    tcn = tc.numpy()
    stc = sorted(tcn)
    def tpp(p): return int(stc[min(int(len(stc)*p), len(stc)-1)])
    return {"tile_size":ts,"grid":f"{tw_}x{th_}","total_tiles":tw_*th_,
        "active_tiles":int((tcn>0).sum()),"total_intersections":int(tcn.sum()),
        "mean":round(float(tcn.mean()),2),"max":int(tcn.max()),
        "p50":tpp(0.5),"p90":tpp(0.9),"p95":tpp(0.95),"p99":tpp(0.99),
        "isect_tiles_ms":round(tm_i,3),"isect_offset_ms":round(tm_o,3),
        "rasterization_ms":round(tm_r,3),"full_forward_ms":round(tm_f,3)}

ts_results = {}
for ts in [8, 16, 32]:
    print(f"  tile_size={ts}:")
    r = profile_ts(ts)
    ts_results[f"tile_{ts}"] = r
    print(f"    grid={r['grid']} active={r['active_tiles']}/{r['total_tiles']}")
    print(f"    ints: mean={r['mean']} max={r['max']} P90={r['p90']}")
    print(f"    isect={r['isect_tiles_ms']}ms offset={r['isect_offset_ms']}ms raster={r['rasterization_ms']}ms fwd={r['full_forward_ms']}ms")

# =========================================================================
# Save
# =========================================================================
results = {
    "timestamp": datetime.utcnow().isoformat()+"Z",
    "gpu": "NVIDIA A100-PCIE-40GB", "scene": "room", "resolution": "1920x1080",
    "n_gaussians": N_G,
    "end_to_end": end_to_end,
    "kernel_times_ms": {k:round(v,3) for k,v in kernels.items()},
    "kernel_ranking": ranking,
    "sum_of_parts_ms": round(total_sum,3),
    "timing_gap_ms": round(f_mean-total_sum,3),
    "tile_statistics_t16": tile_stats_t16,
    "tile_scaling": ts_results,
}

out_dir = os.path.join(REPO_ROOT, "results", "phase-c19")
os.makedirs(out_dir, exist_ok=True)
p = os.path.join(out_dir, "c19-0_a100_rasterizer_profiling.json")
with open(p, "w") as f: json.dump(results, f, indent=2, cls=JsonEncoder)
print(f"\n=== Saved: {p} ===")
print("DONE")
