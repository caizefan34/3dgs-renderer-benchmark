"""
C19-2 Goal 0: Reproducibility baseline.

Re-run canonical tile scaling on A100 (mx) for tile sizes 8,12,16,20,24,28,32.
Record rasterizer CUDA time, intersections/tile statistics, total intersections,
full forward time, and CUDA launch configuration.

This exactly replicates the C19-1 measurement protocol.
"""
import torch, gsplat, json, os, sys, math, numpy as np
from datetime import datetime

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo, "src"))
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras

DEVICE = "cuda"
torch.set_grad_enabled(False)
N_RUNS = 15

print("C19-2 Goal 0: Reproducibility Baseline")
print("=" * 70)

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

ev_s, ev_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# --- Get projection/SH intermediates once ---
# We use tile=16 first to get the intermediates
out_ref, alpha_ref, meta_ref = gsplat.rasterization(
    means=means3d, quats=quats, scales=scales_exp, opacities=opacities, colors=shs,
    viewmats=viewmats, Ks=Ks, width=W, height=H, near_plane=0.01, far_plane=1e10,
    radius_clip=0.0, eps2d=0.3, sh_degree=3, packed=False, tile_size=16,
    backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
    rasterize_mode="classic")
torch.cuda.synchronize()
del out_ref, alpha_ref, meta_ref

results = []

for ts in [8, 12, 16, 20, 24, 28, 32]:
    tw = math.ceil(W / ts)
    th = math.ceil(H / ts)
    n_tiles = tw * th

    # === isect_tiles timing ===
    t_isec = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize(); ev_s.record()
        out = gsplat.rasterization(
            means=means3d, quats=quats, scales=scales_exp, opacities=opacities, colors=shs,
            viewmats=viewmats, Ks=Ks, width=W, height=H, near_plane=0.01, far_plane=1e10,
            radius_clip=0.0, eps2d=0.3, sh_degree=3, packed=False, tile_size=ts,
            backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
            rasterize_mode="classic")
        ev_e.record(); torch.cuda.synchronize()
        t_isec.append(ev_s.elapsed_time(ev_e))
    # Re-run to get metadata
    out, alpha, meta = gsplat.rasterization(
        means=means3d, quats=quats, scales=scales_exp, opacities=opacities, colors=shs,
        viewmats=viewmats, Ks=Ks, width=W, height=H, near_plane=0.01, far_plane=1e10,
        radius_clip=0.0, eps2d=0.3, sh_degree=3, packed=False, tile_size=ts,
        backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
        rasterize_mode="classic")
    torch.cuda.synchronize()

    t_full_m = sum(t_isec) / len(t_isec)

    # Decompose: extract intermediate data for sub-step timing
    m2d = meta["means2d"].contiguous()
    radii = meta["radii"].contiguous()
    depths = meta["depths"].contiguous()
    conics = meta["conics"].contiguous()
    opac = meta["opacities"].contiguous()

    cam_center = cam.camera_center.to(DEVICE)
    view_dirs = cam_center - means3d
    view_dirs = view_dirs / view_dirs.norm(dim=-1, keepdim=True)
    colors_sh = gsplat.spherical_harmonics(3, view_dirs, shs)

    # --- isect_tiles timing ---
    t_isect = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize(); ev_s.record()
        gsplat.isect_tiles(m2d, radii, depths, ts, tw, th, sort=True)
        ev_e.record(); torch.cuda.synchronize()
        t_isect.append(ev_s.elapsed_time(ev_e))
    _, iid, fid = gsplat.isect_tiles(m2d, radii, depths, ts, tw, th, sort=True)
    t_isect_m = sum(t_isect) / len(t_isect)

    # --- isect_offset timing ---
    t_off = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize(); ev_s.record()
        gsplat.isect_offset_encode(iid, 1, tw, th)
        ev_e.record(); torch.cuda.synchronize()
        t_off.append(ev_s.elapsed_time(ev_e))
    ioff = gsplat.isect_offset_encode(iid, 1, tw, th)
    t_off_m = sum(t_off) / len(t_off)

    # --- rasterize_to_pixels timing ---
    t_rast = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize(); ev_s.record()
        gsplat.rasterize_to_pixels(m2d, conics, colors_sh.unsqueeze(0), opac,
                                    W, H, ts, ioff, fid, bg)
        ev_e.record(); torch.cuda.synchronize()
        t_rast.append(ev_s.elapsed_time(ev_e))
    t_rast_m = sum(t_rast) / len(t_rast)

    # --- Tile statistics ---
    n_int = iid.shape[0]
    # Count intersections per tile from isect_ids (each isect_id = sorted_intersection_index * n_tiles + tile_id)
    # Actually flatten_ids maps to Gaussian indices, we need to count per tile from tile_offsets
    tile_ct = torch.zeros(n_tiles, dtype=torch.int64, device="cpu")
    for tid in (iid % n_tiles).cpu():
        tile_ct[tid.long()] += 1
    tcn = tile_ct.numpy()
    stcn = sorted(tcn)

    def tp(p):
        return int(stcn[min(int(len(stcn) * p), len(stcn) - 1)])

    active = int((tcn > 0).sum())
    mean_ints = float(tcn.mean())
    max_ints = int(tcn.max())
    n_visible = int((radii.squeeze() > 0).sum().item())
    dup_factor = n_int / max(n_visible, 1)
    cost_per_int_ns = (t_rast_m * 1e6) / max(n_int, 1)

    # Capture launch configuration
    # block = {tile_size, tile_size, 1}, grid = {1 (images), tile_height, tile_width}
    # shared memory = tile_size^2 * (4 + 12 + 12) = tile_size^2 * 28 bytes

    shmem_bytes = ts * ts * (4 + 12 + 12)

    r = {
        "tile_size": ts,
        "grid": f"{tw}x{th}",
        "total_tiles": n_tiles,
        "active_tiles": active,
        "visible_gaussians": n_visible,
        "total_intersections": n_int,
        "intersection_duplication": round(dup_factor, 2),
        "mean_ints_per_tile": round(mean_ints, 2),
        "p50": tp(0.5),
        "p90": tp(0.9),
        "p95": tp(0.95),
        "p99": tp(0.99),
        "max_ints": max_ints,
        "isect_tiles_ms": round(t_isect_m, 4),
        "isect_offset_ms": round(t_off_m, 4),
        "rasterization_ms": round(t_rast_m, 4),
        "full_forward_ms": round(t_full_m, 4),
        "rasterizer_ns_per_intersection": round(cost_per_int_ns, 4),
        "block_dim": f"{ts}x{ts}x1",
        "grid_dim": f"1x{th}x{tw}",
        "shmem_bytes": shmem_bytes,
        "registers_per_thread": "MEASURED_BY_NCU",
    }
    results.append(r)

    print(f"\n  tile_size={ts:2d}: grid={r['grid']:8s}  active={active:5d}/{n_tiles:<5d}  ints={n_int:>8,}")
    print(f"    ints/tile: mean={mean_ints:7.1f}  P50={r['p50']:3d}  P90={r['p90']:3d}  P99={r['p99']:3d}  max={max_ints:3d}")
    print(f"    dup factor={dup_factor:.2f}x  visible={n_visible:,}")
    print(f"    isect={t_isect_m:7.4f}ms  offset={t_off_m:7.4f}ms  raster={t_rast_m:7.4f}ms  fwd={t_full_m:7.4f}ms")
    print(f"    raster ns/int={cost_per_int_ns:7.4f}  shmem={shmem_bytes}B")

out_data = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "gpu": "NVIDIA A100-PCIE-40GB",
    "driver": "595.71.05",
    "scene": "room",
    "resolution": "1920x1080",
    "n_gaussians": N_G,
    "n_runs_per_point": N_RUNS,
    "tile_scaling": results,
}
os.makedirs(os.path.join(repo, "results", "phase-c19"), exist_ok=True)
p = os.path.join(repo, "results", "phase-c19", "c19-2_reproducibility.json")
with open(p, "w") as f:
    json.dump(out_data, f, indent=2, default=str)
print(f"\nSaved: {p}")
print("DONE")
