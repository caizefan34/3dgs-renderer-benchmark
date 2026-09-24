"""Probe gsplat 1.5.3: fully_fused_projection and isect_tiles signatures."""
import torch
import gsplat
import inspect

print("=== fully_fused_projection ===")
sig = inspect.signature(gsplat.fully_fused_projection)
print(f"  {sig}")
print(f"  doc: {gsplat.fully_fused_projection.__doc__[:300]}")

print()
print("=== fully_fused_projection_with_ut ===")
sig = inspect.signature(gsplat.fully_fused_projection_with_ut)
print(f"  {sig}")

print()
print("=== isect_tiles ===")
sig = inspect.signature(gsplat.isect_tiles)
print(f"  {sig}")
print(f"  doc: {gsplat.isect_tiles.__doc__[:300]}")

print()
print("=== isect_offset_encode ===")
sig = inspect.signature(gsplat.isect_offset_encode)
print(f"  {sig}")

print()
print("=== rasterize_to_pixels ===")
sig = inspect.signature(gsplat.rasterize_to_pixels)
print(f"  {sig}")

print()
print("=== accumulate ===")
sig = inspect.signature(gsplat.accumulate)
print(f"  {sig}")

torch.manual_seed(42)
N = 100
x = 0.1 * torch.randn(N, 3, device="cuda")
s = torch.rand(N, 3, device="cuda") * 0.05
q = torch.randn(N, 4, device="cuda")
q = q / q.norm(dim=1, keepdim=True)
viewmat = torch.eye(4, device="cuda").unsqueeze(0)
viewmat[0, 2, 3] = 5.0
K = torch.tensor([[500., 0., 960.], [0., 500., 540.], [0., 0., 1.]], device="cuda").unsqueeze(0)
width, height = 1920, 1080
tile_size = 16

print()
print("=== fully_fused_projection output ===")
proj = gsplat.fully_fused_projection(
    means=x, quats=q, scales=s,
    viewmats=viewmat, Ks=K,
    width=width, height=height,
    eps2d=0.1,
)
print(f"len: {len(proj)}")
for i, r in enumerate(proj):
    if hasattr(r, 'shape'):
        print(f"  [{i}]: shape={r.shape}, dtype={r.dtype}")
    else:
        print(f"  [{i}]: {type(r)}")

means2d, depths, radii, conics, num_tiles_hit, cov2d = proj
print(f"  means2d: {means2d.shape}, {means2d.dtype}")
print(f"  depths: {depths.shape}, {depths.dtype}")
print(f"  radii: {radii.shape}, {radii.dtype}")
print(f"  conics: {conics.shape}, {conics.dtype}")
print(f"  num_tiles_hit: {num_tiles_hit}")
print(f"  cov2d: {cov2d.shape}, {cov2d.dtype}")

print()
tile_width = 120
tile_height = 68
print(f"=== isect_tiles call ===")
isects = gsplat.isect_tiles(
    means2d, radii, depths,
    tile_size=tile_size,
    tile_width=tile_width,
    tile_height=tile_height,
    sort=True, packed=True,
    n_images=1,
)
print(f"type: {type(isects)}, len: {len(isects) if isinstance(isects, tuple) else 'N/A'}")
if isinstance(isects, tuple):
    for i, r in enumerate(isects):
        if hasattr(r, 'shape'):
            print(f"  [{i}]: shape={r.shape}, dtype={r.dtype}")
        else:
            print(f"  [{i}]: {type(r)}")
elif isinstance(isects, dict):
    for k, v in isects.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")
