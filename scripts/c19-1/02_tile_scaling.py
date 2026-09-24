"""
C19-1 Goal 2: Fine-grained tile-size scaling characterization.

Tile sizes: 8, 12, 16, 20, 24, 28, 32

For each tile size:
  - total intersections
  - intersections/tile stats (mean, P50, P90, P99, max)
  - rasterizer CUDA time
  - isect_tiles CUDA time
  - full_forward CUDA time
  - intersection duplication factor

Also explores packed vs non-packed intersection count.
"""
import torch, gsplat, json, os, sys, math, numpy as np
from datetime import datetime

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.ndarray,)): return obj.tolist()
        if isinstance(obj, torch.Tensor): return obj.detach().cpu().tolist()
        return super().default(obj)

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo, "src"))
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras

DEVICE = "cuda"
torch.set_grad_enabled(False)
N_RUNS = 15

print("C19-1 Goal 2: Fine-Grained Tile-Size Scaling")
print("="*60)

scene = load_ply(os.path.join(repo, "data", "official", "mipnerf360", "room", "point_cloud.ply"), device=DEVICE)
cameras = load_cameras_from_json(os.path.join(repo, "data", "official", "mipnerf360", "room", "cameras.json"), device=DEVICE)
cameras = resize_cameras(cameras, 1920, 1080)
cam = cameras[0]
W, H = 1920, 1080
N_G = scene["num_points"]
print(f"Scene: room, {N_G} Gaussians, {W}x{H}")

means3d = scene["xyz"].contiguous()
quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
scales_exp = scene["scales"].exp().contiguous()
opacities = torch.sigmoid(scene["opacity"]).contiguous()
shs = scene["shs"].contiguous()
viewmats = cam.world_view_transform.unsqueeze(0).contiguous()
Ks = cam.K.unsqueeze(0).contiguous()
bg = torch.zeros(1, 3, device=DEVICE)

# Get meta intermediates from a full call at tile16
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

# SH colors
cam_center = cam.camera_center.to(DEVICE)
view_dirs = cam_center - means3d
view_dirs = view_dirs / view_dirs.norm(dim=-1, keepdim=True)
colors_sh = gsplat.spherical_harmonics(3, view_dirs, shs)

ev_s, ev_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

results = []

for ts in [8, 12, 16, 20, 24, 28, 32]:
    tw = math.ceil(W / ts)
    th = math.ceil(H / ts)
    n_tiles = tw * th
    
    # isect_tiles timing
    t_isec = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize(); ev_s.record()
        gsplat.isect_tiles(m2d, radii, depths, ts, tw, th, sort=True)
        ev_e.record(); torch.cuda.synchronize()
        t_isec.append(ev_s.elapsed_time(ev_e))
    _, iid, fid = gsplat.isect_tiles(m2d, radii, depths, ts, tw, th, sort=True)
    t_isec_m = sum(t_isec)/len(t_isec)
    
    # isect_offset timing
    t_off = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize(); ev_s.record()
        gsplat.isect_offset_encode(iid, 1, tw, th)
        ev_e.record(); torch.cuda.synchronize()
        t_off.append(ev_s.elapsed_time(ev_e))
    ioff = gsplat.isect_offset_encode(iid, 1, tw, th)
    t_off_m = sum(t_off)/len(t_off)
    
    # rasterize_to_pixels timing
    t_rast = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize(); ev_s.record()
        gsplat.rasterize_to_pixels(m2d, conics, colors_sh.unsqueeze(0), opac,
                                    W, H, ts, ioff, fid, bg)
        ev_e.record(); torch.cuda.synchronize()
        t_rast.append(ev_s.elapsed_time(ev_e))
    t_rast_m = sum(t_rast)/len(t_rast)
    
    # Full forward timing
    t_full = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize(); ev_s.record()
        gsplat.rasterization(means=means3d, quats=quats, scales=scales_exp,
            opacities=opacities, colors=shs, viewmats=viewmats, Ks=Ks,
            width=W, height=H, near_plane=0.01, far_plane=1e10,
            radius_clip=0.0, eps2d=0.3, sh_degree=3, packed=False, tile_size=ts,
            backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
            rasterize_mode="classic")
        ev_e.record(); torch.cuda.synchronize()
        t_full.append(ev_s.elapsed_time(ev_e))
    t_full_m = sum(t_full)/len(t_full)
    
    # Tile statistics
    n_int = iid.shape[0]
    tile_ct = torch.zeros(n_tiles, dtype=torch.int64, device="cpu")
    for tid in (iid % n_tiles).cpu(): tile_ct[tid.long()] += 1
    tcn = tile_ct.numpy()
    stcn = sorted(tcn)
    def tp(p): return int(stcn[min(int(len(stcn)*p), len(stcn)-1)])
    active = int((tcn > 0).sum())
    mean_ints = float(tcn.mean())
    max_ints = int(tcn.max())
    
    # Intersection duplication: how many (Gaussian, tile) pairs per visible Gaussian
    n_visible = int((radii.squeeze() > 0).sum().item())
    dup_factor = n_int / max(n_visible, 1)
    
    # Per-intersection costing
    cost_per_int_ns = (t_rast_m * 1e6) / max(n_int, 1)  # ns per intersection
    
    r = {
        "tile_size": ts, "grid": f"{tw}x{th}", "total_tiles": n_tiles,
        "active_tiles": active, "visible_gaussians": n_visible,
        "total_intersections": n_int,
        "intersection_duplication": round(dup_factor, 2),
        "mean_ints_per_tile": round(mean_ints, 2),
        "p50": tp(0.5), "p90": tp(0.9), "p95": tp(0.95), "p99": tp(0.99),
        "max_ints": max_ints,
        "isect_tiles_ms": round(t_isec_m, 3),
        "isect_offset_ms": round(t_off_m, 3),
        "rasterization_ms": round(t_rast_m, 3),
        "full_forward_ms": round(t_full_m, 3),
        "rasterizer_ns_per_intersection": round(cost_per_int_ns, 3),
    }
    results.append(r)
    
    print(f"\n  tile_size={ts:2d}: grid={r['grid']:8s}  active={active:5d}/{n_tiles:<5d}  ints={n_int:>8,}")
    print(f"    ints/tile: mean={mean_ints:7.1f}  P50={r['p50']:3d}  P90={r['p90']:3d}  P99={r['p99']:3d}  max={max_ints:3d}")
    print(f"    dup factor={dup_factor:.2f}x  visible={n_visible:,}")
    print(f"    isect={t_isec_m:7.3f}ms  offset={t_off_m:7.3f}ms  raster={t_rast_m:7.3f}ms  fwd={t_full_m:7.3f}ms")
    print(f"    raster ns/int={cost_per_int_ns:7.3f}")

# ====== Additional: packed mode comparison ======
print("\n\n--- Comparing packed vs non-packed (tile=16) ---")
for packed in [True, False]:
    out_p, alpha_p, meta_p = gsplat.rasterization(
        means=means3d, quats=quats, scales=scales_exp, opacities=opacities, colors=shs,
        viewmats=viewmats, Ks=Ks, width=W, height=H, near_plane=0.01, far_plane=1e10,
        radius_clip=0.0, eps2d=0.3, sh_degree=3, packed=packed, tile_size=16,
        backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
        rasterize_mode="classic")
    torch.cuda.synchronize()
    n_isect_p = meta_p["isect_ids"].shape[0]
    n_visible_p = int((meta_p["radii"].squeeze() > 0).sum().item()) if not packed else "N/A"
    print(f"  packed={packed}: isect_ids={n_isect_p:,}  flatten_ids={meta_p['flatten_ids'].shape[0]:,}")
    if packed:
        print(f"    radii shape={meta_p['radii'].shape}")

# Save
out = {
    "timestamp": datetime.utcnow().isoformat()+"Z",
    "gpu": "NVIDIA A100-PCIE-40GB", "scene": "room", "resolution": "1920x1080",
    "n_gaussians": N_G,
    "tile_scaling_fine": results,
}
os.makedirs(os.path.join(repo, "results", "phase-c19"), exist_ok=True)
p = os.path.join(repo, "results", "phase-c19", "c19-1_tile_scaling.json")
with open(p, "w") as f: json.dump(out, f, indent=2, cls=JsonEncoder)
print(f"\nSaved: {p}")
print("DONE")
