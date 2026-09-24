import sys
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/src")
from gsplat import rasterization
import inspect
sig = inspect.signature(rasterization)
params = list(sig.parameters.keys())
print("rasterization params:", params)

# Check if means2d is available
src = inspect.getsource(rasterization)
for keyword in ["means2d", "radii", "tiles_per_gauss", "visibility_filter", "return"]:
    lines = [l.strip() for l in src.split("\n") if keyword in l.lower()]
    if lines:
        print(f"\n{keyword}:")
        for l in lines[:5]:
            print(f"  {l}")

# Check return type
print("\n=== Testing basic render ===")
import torch
N = 100
means = torch.randn(N, 3, device="cuda")
quats = torch.zeros(N, 4, device="cuda"); quats[:, 0] = 1
scales = torch.full((N, 3), -3.0, device="cuda")
opacities = torch.ones(N, device="cuda")
colors = torch.zeros(N, 1, 3, device="cuda")
viewmat = torch.eye(4, device="cuda").unsqueeze(0)
K = torch.tensor([[800, 0, 400], [0, 800, 300], [0, 0, 1]], dtype=torch.float32, device="cuda").unsqueeze(0)

r, _, meta = rasterization(
    means=means, quats=quats, scales=torch.exp(scales),
    opacities=opacities, colors=colors,
    viewmats=viewmat, Ks=K,
    width=800, height=600,
    packed=False, sh_degree=0,
)
print(f"render shape: {r.shape}")
print(f"meta keys: {list(meta.keys()) if isinstance(meta, dict) else type(meta)}")
if isinstance(meta, dict):
    for k, v in meta.items():
        if hasattr(v, "shape"):
            print(f"  meta[{k}]: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  meta[{k}]: {type(v)} = {v}")
