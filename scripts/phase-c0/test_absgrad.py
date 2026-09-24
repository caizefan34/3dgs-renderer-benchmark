import sys
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/src")
import torch
from gsplat import rasterization

N = 100
means = torch.nn.Parameter(torch.randn(N, 3, device="cuda"))
quats = torch.zeros(N, 4, device="cuda"); quats[:, 0] = 1
scales = torch.full((N, 3), -3.0, device="cuda")
opacities = torch.ones(N, device="cuda")
colors = torch.zeros(N, 1, 3, device="cuda")
viewmat = torch.eye(4, device="cuda").unsqueeze(0)
K = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]], dtype=torch.float32, device="cuda").unsqueeze(0)

# Test with absgrad=True
r, _, meta = rasterization(
    means=means, quats=quats, scales=torch.exp(scales),
    opacities=opacities, colors=colors,
    viewmats=viewmat, Ks=K,
    width=800, height=600,
    packed=False, sh_degree=0, absgrad=True,
)
print(f"means2d requires_grad: {meta['means2d'].requires_grad}")
print(f"means2d has absgrad attr: {hasattr(meta['means2d'], 'absgrad')}")

loss = r[0].sum()
loss.backward()

print(f"means.grad is not None: {means.grad is not None}")
# Check if absgrad is now populated
if hasattr(meta['means2d'], 'absgrad'):
    ag = meta['means2d'].absgrad
    print(f"means2d.absgrad is not None: {ag is not None}")
    if ag is not None:
        print(f"  shape: {ag.shape}")
        print(f"  dtype: {ag.dtype}")
        print(f"  non-zero count: {(ag > 0).sum().item()}")
        print(f"  mean: {ag.float().mean().item():.6f}")
        print(f"  max: {ag.float().max().item():.6f}")

# Also try retain_grad approach
means2 = torch.nn.Parameter(torch.randn(N, 3, device="cuda"))
r2, _, meta2 = rasterization(
    means=means2, quats=quats, scales=torch.exp(scales),
    opacities=opacities, colors=colors,
    viewmats=viewmat, Ks=K,
    width=800, height=600,
    packed=False, sh_degree=0,
)
meta2['means2d'].retain_grad()
loss2 = r2[0].sum()
loss2.backward()
print(f"\nretain_grad approach:")
print(f"  means2d.grad: {meta2['means2d'].grad}")
print(f"  means2d.grad is not None: {meta2['means2d'].grad is not None}")
