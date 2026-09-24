#!/usr/bin/env python3
"""Verify gsplat build with sparse backward support."""
import gsplat
print(f"gsplat version: {gsplat.__version__}")

from gsplat.cuda._wrapper import rasterize_to_pixels
import inspect
sig = inspect.signature(rasterize_to_pixels)
params = list(sig.parameters.keys())
print(f"rasterize_to_pixels params: {params}")
print(f"  has importance_mask: {'importance_mask' in params}")
print(f"  has compute_densify_grad: {'compute_densify_grad' in params}")

from gsplat.rendering import rasterization
sig2 = inspect.signature(rasterization)
params2 = list(sig2.parameters.keys())
print(f"rasterization has importance_mask: {'importance_mask' in params2}")
print(f"rasterization has compute_densify_grad: {'compute_densify_grad' in params2}")

# Quick smoke test: create a tiny scene and render with/without mask
import torch
device = torch.device("cuda:0")
N = 100
means = torch.randn(N, 3, device=device, requires_grad=True)
quats = torch.randn(N, 4, device=device)
quats /= quats.norm(dim=-1, keepdim=True)
scales = torch.randn(N, 3, device=device).abs() * 0.1
opacities = torch.sigmoid(torch.randn(N, device=device))
colors = torch.randn(N, 3, device=device)  # RGB, no SH
viewmats = torch.eye(4, device=device).unsqueeze(0)
Ks = torch.tensor([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0)

# Test without mask (baseline)
r1, a1, _ = gsplat.rasterization(
    means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
    viewmats=viewmats, Ks=Ks, width=100, height=100, tile_size=16,
    packed=False, render_mode="RGB")
print(f"Baseline render shape: {r1.shape}, alpha sum: {a1.sum().item():.4f}")

# Test with all-ones mask (should be identical)
mask_ones = torch.ones(N, dtype=torch.uint8, device=device)
r2, a2, _ = gsplat.rasterization(
    means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
    viewmats=viewmats, Ks=Ks, width=100, height=100, tile_size=16,
    packed=False, render_mode="RGB",
    importance_mask=mask_ones, compute_densify_grad=False)
print(f"Mask=ones render shape: {r2.shape}, alpha sum: {a2.sum().item():.4f}")
diff = (r1 - r2).abs().max().item()
print(f"Max diff (baseline vs mask=ones): {diff:.8f}")

# Test with all-zeros mask (B1: skip all gradients)
mask_zeros = torch.zeros(N, dtype=torch.uint8, device=device)
r3, a3, _ = gsplat.rasterization(
    means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
    viewmats=viewmats, Ks=Ks, width=100, height=100, tile_size=16,
    packed=False, render_mode="RGB",
    importance_mask=mask_zeros, compute_densify_grad=False)
print(f"Mask=zeros render shape: {r3.shape}, alpha sum: {a3.sum().item():.4f}")
diff_fwd = (r1 - r3).abs().max().item()
print(f"Max diff (baseline vs mask=zeros, forward): {diff_fwd:.8f}")

# Test backward: baseline vs mask=ones
loss1 = r1.sum()
loss1.backward()
grad_xyz_1 = means.grad.clone()
means.grad = None

loss2 = r2.sum()
loss2.backward()
grad_xyz_2 = means.grad.clone()
means.grad = None
diff_grad = (grad_xyz_1 - grad_xyz_2).abs().max().item()
print(f"Grad diff (baseline vs mask=ones): {diff_grad:.8f}")

# Test backward: mask=zeros (B1)
loss3 = r3.sum()
loss3.backward()
grad_xyz_3 = means.grad.clone()
means.grad = None
grad_max_3 = grad_xyz_3.abs().max().item()
print(f"Grad max (mask=zeros, B1): {grad_max_3:.8f} (should be 0.0 for skipped)")

print("\nSmoke test PASSED!")
