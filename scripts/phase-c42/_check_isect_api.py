import torch
from gsplat import fully_fused_projection

N = 100
means = torch.randn(N, 3, device="cuda")
quats = torch.nn.functional.normalize(torch.randn(N, 4, device="cuda"), dim=-1)
scales = torch.exp(torch.randn(N, 3, device="cuda") * 0.5)
viewmats = torch.eye(4, device="cuda").unsqueeze(0)
Ks = torch.tensor([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], dtype=torch.float32, device="cuda").unsqueeze(0)

# Non-packed mode
result_npk = fully_fused_projection(
    means=means, covars=None, quats=quats, scales=scales,
    viewmats=viewmats, Ks=Ks, width=1920, height=1080,
    eps2d=0.1, radius_clip=0.0, packed=False)
print(f"Non-packed: {len(result_npk)} values")
for i, r in enumerate(result_npk):
    if isinstance(r, torch.Tensor):
        print(f"  [{i}]: shape={r.shape}, dtype={r.dtype}")
    else:
        print(f"  [{i}]: {type(r)}")

# Packed mode
result_pk = fully_fused_projection(
    means=means, covars=None, quats=quats, scales=scales,
    viewmats=viewmats, Ks=Ks, width=1920, height=1080,
    eps2d=0.1, radius_clip=0.0, packed=True)
print(f"\nPacked: {len(result_pk)} values")
for i, r in enumerate(result_pk):
    if isinstance(r, torch.Tensor):
        print(f"  [{i}]: shape={r.shape}, dtype={r.dtype}")
    else:
        print(f"  [{i}]: {type(r)}")
