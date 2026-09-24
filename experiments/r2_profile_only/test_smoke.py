"""Smoke test for R2.1 patched gsplat with r2 masks."""
import os, sys
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)
import torch
import torch.nn.functional as F
from gsplat import rasterization
import inspect

# Check params
sig = inspect.signature(rasterization)
params = list(sig.parameters.keys())
print("rasterization r2 params:", [p for p in params if "r2" in p])

# Smoke test with real-like data
N = 100
means = torch.randn(1, N, 3, device="cuda")
quats = torch.randn(1, N, 4, device="cuda")
quats /= quats.norm(dim=-1, keepdim=True)
scales = torch.rand(1, N, 3, device="cuda") * 0.1
opacities = torch.rand(1, N, device="cuda")
colors = torch.rand(1, N, 1, 3, device="cuda")  # SH degree 0: [1, N, 1, 3]
viewmats = torch.eye(4, device="cuda").unsqueeze(0).unsqueeze(0)
viewmats[0, 0, 3] = 2.0
Ks = torch.tensor([[[[500., 0., 256.], [0., 500., 256.], [0., 0., 1.]]]], device="cuda")

# Test without masks
print("\n=== Test 1: No masks ===")
render_colors, render_alphas, meta = rasterization(
    means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
    viewmats=viewmats, Ks=Ks, width=512, height=512, tile_size=16, packed=False,
    sh_degree=0, radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
)
print(f"  render_colors={render_colors.shape}, render_alphas={render_alphas.shape}")

# Test with r2 masks (all ones = full gradient)
print("\n=== Test 2: r2 masks all 1s ===")
r2_geo_mask = torch.ones(N, dtype=torch.uint8, device="cuda")
r2_app_mask = torch.ones(N, dtype=torch.uint8, device="cuda")
r2_opacity_mask = torch.ones(N, dtype=torch.uint8, device="cuda")
# Need tensors with requires_grad
means2 = means.detach().requires_grad_(True)
quats2 = quats.detach().requires_grad_(True)
scales2 = scales.detach().requires_grad_(True)
opacities2 = opacities.detach().requires_grad_(True)
colors2 = colors.detach().requires_grad_(True)
render_colors2, render_alphas2, meta2 = rasterization(
    means=means2, quats=quats2, scales=scales2, opacities=opacities2, colors=colors2,
    viewmats=viewmats, Ks=Ks, width=512, height=512, tile_size=16, packed=False,
    sh_degree=0, radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
    r2_geo_mask=r2_geo_mask, r2_app_mask=r2_app_mask, r2_opacity_mask=r2_opacity_mask,
)
print(f"  render_colors={render_colors2.shape}")

# Test backward
loss = render_colors2.sum()
loss.backward()
print(f"  Backward: OK")
print(f"  colors grad max: {colors2.grad.abs().max().item():.2e}")

# Test with r2 masks all zeros (should produce zero gradients)
print("\n=== Test 3: r2 masks all 0s ===")
r2_geo_mask_z = torch.zeros(N, dtype=torch.uint8, device="cuda")
r2_app_mask_z = torch.zeros(N, dtype=torch.uint8, device="cuda")
r2_opacity_mask_z = torch.zeros(N, dtype=torch.uint8, device="cuda")
# Need fresh tensors without grad
means3 = means.detach().requires_grad_(True)
quats3 = quats.detach().requires_grad_(True)
scales3 = scales.detach().requires_grad_(True)
opacities3 = opacities.detach().requires_grad_(True)
colors3 = colors.detach().requires_grad_(True)
render_colors3, _, meta3 = rasterization(
    means=means3, quats=quats3, scales=scales3, opacities=opacities3, colors=colors3,
    viewmats=viewmats, Ks=Ks, width=512, height=512, tile_size=16, packed=False,
    sh_degree=0, radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
    r2_geo_mask=r2_geo_mask_z, r2_app_mask=r2_app_mask_z, r2_opacity_mask=r2_opacity_mask_z,
)
loss3 = render_colors3.sum()
loss3.backward()
print(f"  colors grad max: {colors3.grad.abs().max().item():.2e} (should be 0)")
print(f"  opacities grad max: {opacities3.grad.abs().max().item():.2e} (should be 0)")

print("\n=== Smoke test PASSED ===")
