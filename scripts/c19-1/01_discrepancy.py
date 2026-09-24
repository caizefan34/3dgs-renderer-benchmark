"""C19-1 Goal 1: Investigate discrepancy between C19-0 and previous A100 baseline.

Old baseline claims: 174,446,870 intersections, forward 31.88ms, CUB sort 20.60ms
C19-0: 1,626,135 intersections, forward 3.08ms, rasterizer 1.917ms

Check: checkpoint, camera resolution, radius statistics, gsplat source version, timing scope.
"""
import torch, gsplat, json, os, sys, math, numpy as np
from collections import OrderedDict

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo, "src"))
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras

DEVICE = "cuda"
torch.set_grad_enabled(False)

# ===== Load scene at NATIVE resolution =====
print("="*70)
print("C19-1 Discrepancy Investigation")
print("="*70)

# Load scene
scene = load_ply(os.path.join(repo, "data", "official", "mipnerf360", "room", "point_cloud.ply"), device=DEVICE)
cameras_raw = load_cameras_from_json(os.path.join(repo, "data", "official", "mipnerf360", "room", "cameras.json"), device=DEVICE)

N_G = scene["num_points"]
cam = cameras_raw[0]
H_native, W_native = cam.image_height, cam.image_width
print(f"\n1. Scene: room, {N_G} Gaussians")
print(f"   Native resolution: {W_native}x{H_native}")

# Resize to 1080p
cameras_1080 = resize_cameras(cameras_raw, 1920, 1080)
cam_1080 = cameras_1080[0]
H, W = 1080, 1920

# Prepare tensors
means3d = scene["xyz"].contiguous()
quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
scales_exp = scene["scales"].exp().contiguous()
opacities = torch.sigmoid(scene["opacity"]).contiguous()
shs = scene["shs"].contiguous()

# ===== 2. Check radii distribution =====
print("\n2. Gaussian Radii Distribution")
print("-"*60)
for label, (cam_obj, res_str) in [
    ("Native", (cam, f"{W_native}x{H_native}")),
    ("1080p", (cam_1080, f"{W}x{H}")),
]:
    vmat = cam_obj.world_view_transform.unsqueeze(0).contiguous()
    Kmat = cam_obj.K.unsqueeze(0).contiguous()
    _, _, radii_ffp, _, _ = gsplat.fully_fused_projection(
        means=means3d, covars=None, quats=quats, scales=scales_exp,
        viewmats=vmat, Ks=Kmat, width=cam_obj.image_width, height=cam_obj.image_height,
        eps2d=0.3, near_plane=0.01, far_plane=1e10,
        radius_clip=0.0, packed=False, sparse_grad=False,
        calc_compensations=False, camera_model="pinhole", opacities=opacities,
    )
    # radii_ffp shape: [1, N]
    r = radii_ffp.squeeze().float().cpu()
    r_valid = r[r > 0]
    print(f"   {res_str}:")
    print(f"     Non-zero radii: {len(r_valid)}/{N_G} ({(len(r_valid)/N_G*100):.1f}%)")
    if len(r_valid) > 0:
        print(f"     Mean radius: {r_valid.mean().item():.2f}px")
        print(f"     P50/P90/P95/P99: {r_valid.quantile(0.5).item():.1f}/{r_valid.quantile(0.9).item():.1f}/{r_valid.quantile(0.95).item():.1f}/{r_valid.quantile(0.99).item():.1f}")
        print(f"     Max radius: {r_valid.max().item():.1f}px")
        # Expected tile intersections: each Gaussian hits roughly (2r/tile_size + 1)^2 tiles
        tile_t = 16
        expected_tiles_per_g = ((2 * r_valid / tile_t + 1) ** 2).sum().item()
        print(f"     Expected tile intersections (tile={tile_t}): {expected_tiles_per_g:.0f}")
        # With large-radius cut
        large = r_valid[r_valid > 4]
        print(f"     Gaussians with radius > 4px: {len(large)} ({len(large)/N_G*100:.1f}%)")
        large_exp = ((2 * large / tile_t + 1) ** 2).sum().item()
        print(f"     Expected tile hits from large Gaussians: {large_exp:.0f}")

# ===== 3. Compare FULL pass at both resolutions =====
print("\n3. Full Rasterization Comparison")
print("-"*60)

for label, cam_obj, res_str in [
    ("Native", cam, f"{W_native}x{H_native}"),
    ("1080p", cam_1080, f"{W}x{H}"),
]:
    vmat = cam_obj.world_view_transform.unsqueeze(0).contiguous()
    Kmat = cam_obj.K.unsqueeze(0).contiguous()
    bg = torch.zeros(1, 3, device=DEVICE)
    tw = math.ceil(cam_obj.image_width / 16)
    th = math.ceil(cam_obj.image_height / 16)

    out, alpha, meta = gsplat.rasterization(
        means=means3d, quats=quats, scales=scales_exp, opacities=opacities, colors=shs,
        viewmats=vmat, Ks=Kmat, width=cam_obj.image_width, height=cam_obj.image_height,
        near_plane=0.01, far_plane=1e10, radius_clip=0.0, eps2d=0.3,
        sh_degree=3, packed=False, tile_size=16, backgrounds=bg,
        render_mode="RGB", sparse_grad=False, absgrad=False, rasterize_mode="classic")
    torch.cuda.synchronize()

    n_int = meta["isect_ids"].shape[0]
    tile_ct = torch.zeros(tw*th, dtype=torch.int64, device="cpu")
    for tid in (meta["isect_ids"] % (tw*th)).cpu():
        tile_ct[tid.long()] += 1

    print(f"   {res_str}:")
    print(f"     Total intersections: {n_int:,}")
    print(f"     Grid: {tw}x{th} = {tw*th} tiles")
    print(f"     Active tiles: {(tile_ct>0).sum().item()}/{tw*th}")
    tcn = tile_ct.numpy()
    print(f"     Mean/tile: {tcn.mean():.1f}")
    print(f"     P50/P90/P99: {int(np.percentile(tcn,50))}/{int(np.percentile(tcn,90))}/{int(np.percentile(tcn,99))}")
    print(f"     Max/tile: {int(tcn.max())}")

    # Radii from meta
    r2 = meta["radii"].squeeze().float().cpu()
    r2_valid = r2[r2 > 0]
    print(f"     Mean 2D radius (rx, ry avg): {r2_valid.mean().item():.2f}")

# ===== 4. Synthetic scene test (400K, untrained) =====
print("\n4. Checking synthetic/untrained scene characteristics")
print("-"*60)
# Check if there's a synthetic PLY
for p in ["data/scene.ply", "data/synthetic/stress_400k.ply"]:
    fp = os.path.join(repo, p)
    if os.path.exists(fp):
        fsize = os.path.getsize(fp) / 1e6
        print(f"   Found: {p} ({fsize:.1f}MB)")
        # Quick load just for header
        with open(fp, 'rb') as f:
            header = f.read(500).decode('ascii', errors='replace')
        # Count vertices
        for line in header.split('\n'):
            if 'element vertex' in line:
                n = int(line.split()[-1])
                print(f"   Vertices: {n}")
    else:
        print(f"   Not found: {p}")

# ===== 5. Check old profile artifacts =====
print("\n5. Looking for previous profile data")
print("-"*60)
old_paths = [
    "reports/epic05/phase8e_per_kernel_forward_timing.md",
    "results/epic05/phase8e",
]
for p in old_paths:
    fp = os.path.join(repo, p)
    if os.path.exists(fp):
        print(f"   EXISTS: {p}")
    else:
        print(f"   NOT FOUND: {p}")

print("\n" + "="*70)
print("Done.")
