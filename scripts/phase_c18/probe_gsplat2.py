"""Probe gsplat 1.5.3 correct API: fully_fused_projection, isect_tiles, etc."""

import torch
import gsplat

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

print("=== fully_fused_projection ===")
proj_result = gsplat.fully_fused_projection(
    means=x, quats=q, scales=s,
    viewmats=viewmat, Ks=K,
    width=width, height=height,
    eps2d=0.1, tile_size=tile_size,
)
print(f"type: {type(proj_result)}, len: {len(proj_result) if isinstance(proj_result, tuple) else 'N/A'}")
if isinstance(proj_result, tuple):
    for i, r in enumerate(proj_result):
        if hasattr(r, 'shape'):
            print(f"  [{i}]: shape={r.shape}, dtype={r.dtype}")
        else:
            print(f"  [{i}]: {r}")

print()
print("=== isect_tiles ===")
# What does project_gaussians actually return?
# Let's check rasterization info keys
rendered, render_alphas, info = gsplat.rasterization(
    means=x, quats=q, scales=s,
    opacities=torch.ones(N, 1, device="cuda") * 0.5,
    colors=torch.randn(N, 3, device="cuda"),
    viewmats=viewmat, Ks=K,
    width=width, height=height,
    tile_size=tile_size, packed=True, sh_degree=0,
    radius_clip=0.0, eps2d=0.1, render_mode="RGB",
)
print(f"info keys: {sorted(info.keys())}")
for k, v in info.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"  {k}: {type(v)}")

print()
print("=== isect_tiles (standalone) ===")
# First project to get means2d etc
means2d = x[:, :2]  # simplified
radii = torch.ones(N, device="cuda") * 10
depths = x[:, 2]
# But we actually need fully_fused_projection first
print("Checking fully_fused_projection_ut...")
proj_ut = gsplat.fully_fused_projection_with_ut(
    means=x, quats=q, scales=s,
    opacities=torch.ones(N, 1, device="cuda") * 0.5,
    viewmats=viewmat, Ks=K,
    width=width, height=height,
    eps2d=0.1, tile_size=tile_size,
)
print(f"fully_fused_projection_with_ut len: {len(proj_ut)}")
for i, r in enumerate(proj_ut):
    if hasattr(r, 'shape'):
        print(f"  [{i}]: shape={r.shape}, dtype={r.dtype}")
    else:
        print(f"  [{i}]: {type(r)}")
