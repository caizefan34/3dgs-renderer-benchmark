"""
C19-2 Goal 2: Rasterizer Replay / Isolation.

Decouple TILE SIZE from INTERSECTIONS PER TILE.
Keep block geometry fixed (tile_size=16) while varying the number of
Gaussian intersections presented to each tile.

Target intersection counts: 100, 120, 140, 160, 180, 190, 200, 205, 210,
                             220, 240, 260, 280, 320

Method:
  1. Run the full renderer at tile=16 to get canonical sorted data
  2. Extract tile_offsets and flatten_ids
  3. For each target ints/tile: construct synthetic tile_offsets and
     flatten_ids by truncating or replicating within each tile's range
  4. Call rasterize_to_pixels() directly with these modified arrays
  5. Record timing, ns/intersection, throughput

Requires gsplat.rasterize_to_pixels() to accept custom tile_offsets/flatten_ids.
"""
import torch, gsplat, json, os, sys, math, numpy as np
from datetime import datetime

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo, "src"))
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras

DEVICE = "cuda"
torch.set_grad_enabled(False)
N_RUNS = 15

print("C19-2 Goal 2: Rasterizer Replay / Isolation")
print("=" * 70)

scene = load_ply(os.path.join(repo, "data", "official", "mipnerf360", "room", "point_cloud.ply"), device=DEVICE)
cameras = load_cameras_from_json(os.path.join(repo, "data", "official", "mipnerf360", "room", "cameras.json"), device=DEVICE)
cameras = resize_cameras(cameras, 1920, 1080)
cam = cameras[0]
W, H = 1920, 1080

means3d = scene["xyz"].contiguous()
quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
scales_exp = scene["scales"].exp().contiguous()
opacities_raw = scene["opacity"].contiguous()
shs = scene["shs"].contiguous()
viewmats = cam.world_view_transform.unsqueeze(0).contiguous()
Ks = cam.K.unsqueeze(0).contiguous()
bg = torch.zeros(1, 3, device=DEVICE)

# --- Get canonical data at tile=16 ---
TS_BASE = 16
out_ref, alpha_ref, meta_ref = gsplat.rasterization(
    means=means3d, quats=quats, scales=scales_exp, opacities=opacities_raw, colors=shs,
    viewmats=viewmats, Ks=Ks, width=W, height=H, near_plane=0.01, far_plane=1e10,
    radius_clip=0.0, eps2d=0.3, sh_degree=3, packed=False, tile_size=TS_BASE,
    backgrounds=bg, render_mode="RGB", sparse_grad=False, absgrad=False,
    rasterize_mode="classic")
torch.cuda.synchronize()
del out_ref, alpha_ref

m2d = meta_ref["means2d"].contiguous()
radii = meta_ref["radii"].contiguous()
depths = meta_ref["depths"].contiguous()
conics = meta_ref["conics"].contiguous()
opac = meta_ref["opacities"].contiguous()

cam_center = cam.camera_center.to(DEVICE)
view_dirs = cam_center - means3d
view_dirs = view_dirs / view_dirs.norm(dim=-1, keepdim=True)
colors_sh = gsplat.spherical_harmonics(3, view_dirs, shs)

# Get the canonical sorted intersection data
TW = math.ceil(W / TS_BASE)
TH = math.ceil(H / TS_BASE)
n_tiles = TW * TH

_, iid_canon, fid_canon = gsplat.isect_tiles(m2d, radii, depths, TS_BASE, TW, TH, sort=True)
ioff_canon = gsplat.isect_offset_encode(iid_canon, 1, TW, TH)

# tile_offsets[image_id * tile_height * tile_width + tile_id]
# For 1 image: tile_offsets shape is [TH, TW]
tile_offsets = ioff_canon.squeeze(0)  # [TH, TW]
# Convert to flat index for easier manipulation
tile_offsets_flat = tile_offsets.reshape(-1)  # [n_tiles]
# flatten_ids maps sorted intersection index -> Gaussian index
flatten_ids = fid_canon  # [n_isects]

n_isects_canon = flatten_ids.shape[0]
print(f"Canonical data: tile_size={TS_BASE}, grid={TW}x{TH}, n_tiles={n_tiles}")
print(f"  total intersections: {n_isects_canon:,}")

# Compute actual intersections per tile from tile_offsets
ints_per_tile_canon = []
for ti in range(n_tiles):
    lo = int(tile_offsets_flat[ti].item())
    hi = int(tile_offsets_flat[ti + 1].item()) if ti < n_tiles - 1 else n_isects_canon
    ints_per_tile_canon.append(hi - lo)
ints_per_tile_canon = np.array(ints_per_tile_canon)
print(f"  mean ints/tile: {ints_per_tile_canon.mean():.1f}")
print(f"  min: {ints_per_tile_canon.min()}, max: {ints_per_tile_canon.max()}")

ev_s, ev_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

def build_replay_data(target_ints_per_tile):
    """
    Build synthetic tile_offsets and flatten_ids where each tile has
    exactly (or approximately) target_ints_per_tile intersections.

    For tiles that have MORE ints than target: truncate.
    For tiles that have FEWER ints than target: replicate.
    """
    new_flatten_ids_list = []
    new_offsets = [0]

    for ti in range(n_tiles):
        lo = int(tile_offsets_flat[ti].item())
        hi = int(tile_offsets_flat[ti + 1].item()) if ti < n_tiles - 1 else n_isects_canon
        n_orig = hi - lo

        if n_orig == 0:
            new_offsets.append(new_offsets[-1])
            continue

        # Get the Gaussian indices for this tile
        tile_gids = flatten_ids[lo:hi]

        if n_orig >= target_ints_per_tile:
            # Truncate: just take first target_ints_per_tile
            selected = tile_gids[:target_ints_per_tile]
        else:
            # Replicate: repeat entries to reach target
            repeats = (target_ints_per_tile + n_orig - 1) // n_orig
            selected = tile_gids.repeat(repeats)[:target_ints_per_tile]

        new_flatten_ids_list.append(selected)
        new_offsets.append(new_offsets[-1] + selected.shape[0])

    new_flatten_ids = torch.cat(new_flatten_ids_list, dim=0) if new_flatten_ids_list else torch.tensor([], dtype=torch.int32, device=DEVICE)
    new_offsets_tensor = torch.tensor(new_offsets[:-1], dtype=torch.int32, device=DEVICE).reshape(TH, TW)

    return new_flatten_ids, new_offsets_tensor


TARGET_INTS = [100, 120, 140, 160, 180, 190, 200, 205, 210, 220, 240, 260, 280, 320]

replay_results = []

for target in TARGET_INTS:
    print(f"\n--- Target ints/tile: {target} ---")

    new_fid, new_toff = build_replay_data(target)
    n_total = new_fid.shape[0]
    # Get range sizes from offsets (adding sentinel)
    sentinel = torch.tensor([n_total], dtype=torch.int32, device=DEVICE)
    all_offsets = torch.cat([new_toff.reshape(-1), sentinel])
    range_sizes = (all_offsets[1:] - all_offsets[:-1]).cpu().numpy()
    active_tiles = int((range_sizes > 0).sum())
    mean_ints = float(range_sizes.mean())
    max_ints = int(range_sizes.max())

    print(f"  total ints: {n_total:,}, active: {active_tiles}/{n_tiles}")
    print(f"  mean: {mean_ints:.1f}, max: {max_ints}")

    # Warmup
    for _ in range(3):
        out_tmp, alpha_tmp = gsplat.rasterize_to_pixels(
            m2d, conics, colors_sh.unsqueeze(0), opac,
            W, H, TS_BASE, new_toff.unsqueeze(0), new_fid, bg)
    torch.cuda.synchronize()

    # Timing
    t_rast = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        ev_s.record()
        out_tmp, alpha_tmp = gsplat.rasterize_to_pixels(
            m2d, conics, colors_sh.unsqueeze(0), opac,
            W, H, TS_BASE, new_toff.unsqueeze(0), new_fid, bg)
        ev_e.record()
        torch.cuda.synchronize()
        t_rast.append(ev_s.elapsed_time(ev_e))

    t_rast_m = sum(t_rast) / len(t_rast)
    t_rast_std = np.std(t_rast)
    ns_per_int = (t_rast_m * 1e6) / max(n_total, 1)
    throughput = n_total / (t_rast_m / 1000)  # intersections/second

    r = {
        "target_ints_per_tile": target,
        "actual_total_ints": int(n_total),
        "active_tiles": int(active_tiles),
        "mean_ints_per_tile": round(float(mean_ints), 2),
        "max_ints_per_tile": int(max_ints),
        "tile_size": TS_BASE,
        "grid": f"{TW}x{TH}",
        "block_dim": f"{TS_BASE}x{TS_BASE}x1",
        "rasterize_ms_mean": round(t_rast_m, 4),
        "rasterize_ms_std": round(t_rast_std, 4),
        "rasterize_ms_cv": round(t_rast_std / t_rast_m, 4),
        "ns_per_intersection": round(ns_per_int, 4),
        "throughput_ints_per_sec": round(float(throughput)),
    }
    replay_results.append(r)

    print(f"  rasterize: {t_rast_m:.4f}ms ± {t_rast_std:.4f}ms")
    print(f"  ns/int: {ns_per_int:.4f}")
    print(f"  throughput: {throughput:,.0f} ints/s")

    # Free CUDA memory between runs
    del out_tmp, alpha_tmp
    torch.cuda.empty_cache()

# Also measure the canonical point for comparison
print(f"\n--- Canonical (unaltered) tile_size={TS_BASE} ---")
# Need to re-create the canonical isect data since we modified it
_, iid_canon2, fid_canon2 = gsplat.isect_tiles(m2d, radii, depths, TS_BASE, TW, TH, sort=True)
ioff_canon2 = gsplat.isect_offset_encode(iid_canon2, 1, TW, TH)

t_rast_canon = []
for _ in range(N_RUNS):
    torch.cuda.synchronize(); ev_s.record()
    out_tmp, alpha_tmp = gsplat.rasterize_to_pixels(
        m2d, conics, colors_sh.unsqueeze(0), opac,
        W, H, TS_BASE, ioff_canon2, fid_canon2, bg)
    ev_e.record(); torch.cuda.synchronize()
    t_rast_canon.append(ev_s.elapsed_time(ev_e))

t_rc = sum(t_rast_canon) / len(t_rast_canon)
n_total_canon = fid_canon2.shape[0]
ns_int_canon = (t_rc * 1e6) / max(n_total_canon, 1)

canon_rec = {
    "target_ints_per_tile": "canonical",
    "actual_total_ints": int(n_total_canon),
    "mean_ints_per_tile": round(float(ints_per_tile_canon.mean()), 2),
    "max_ints_per_tile": int(ints_per_tile_canon.max()),
    "tile_size": TS_BASE,
    "rasterize_ms_mean": round(t_rc, 4),
    "ns_per_intersection": round(ns_int_canon, 4),
}
replay_results.append(canon_rec)
print(f"  ints: {n_total_canon:,}, time: {t_rc:.4f}ms, ns/int: {ns_int_canon:.4f}")

del out_tmp, alpha_tmp, iid_canon2, fid_canon2, ioff_canon2

# --- Save ---
out = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "gpu": "NVIDIA A100-PCIE-40GB",
    "scene": "room",
    "resolution": "1920x1080",
    "base_tile_size": TS_BASE,
    "grid": f"{TW}x{TH}",
    "n_tiles": n_tiles,
    "n_runs_per_point": N_RUNS,
    "method": "Fixed tile_size=16, synthetic tile_offsets/flatten_ids by truncation/replication within each tile",
    "replay_data": replay_results,
}
os.makedirs(os.path.join(repo, "results", "phase-c19"), exist_ok=True)
p = os.path.join(repo, "results", "phase-c19", "c19-2_rasterizer_replay.json")
with open(p, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nSaved: {p}")
print("DONE")
