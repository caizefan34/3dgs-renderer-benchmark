"""Probe gsplat 1.5.3 API surface."""
import gsplat
import inspect

print("=== gsplat ops ===")
ops = [x for x in dir(gsplat) if not x.startswith("_")]
for o in sorted(ops):
    try:
        sig = str(inspect.signature(getattr(gsplat, o)))
        print(f"  {o}{sig}"[:200])
    except Exception:
        print(f"  {o}  (no signature)")

print()
print("=== project_gaussians doc ===")
try:
    print(gsplat.project_gaussians.__doc__[:500])
except:
    pass

print()
print("=== rasterization doc ===")
try:
    print(gsplat.rasterization.__doc__[:500])
except:
    pass

print()
print("=== Testing project_gaussians output ===")
import torch
torch.manual_seed(42)
x = 0.01 * torch.randn(20, 3, device="cuda")
s = torch.rand(20, 3, device="cuda") * 0.1
q = torch.randn(20, 4, device="cuda")
q = q / q.norm(dim=1, keepdim=True)
viewmat = torch.eye(4, device="cuda").unsqueeze(0)
viewmat[0, 0, 3] = 2.0
viewmat[0, 1, 3] = 0.0
viewmat[0, 2, 3] = 5.0
K = torch.tensor([[500., 0., 960.], [0., 500., 540.], [0., 0., 1.]], device="cuda").unsqueeze(0)
width, height = 1920, 1080
tile_size = 16

result = gsplat.project_gaussians(
    means=x, scales=s, quats=q,
    viewmats=viewmat, Ks=K,
    width=width, height=height,
    tile_size=tile_size,
)

print(f"project_gaussians returns type: {type(result)}")
if isinstance(result, tuple):
    print(f"  tuple length: {len(result)}")
    for i, r in enumerate(result):
        print(f"  [{i}] shape={r.shape}, dtype={r.dtype}")
elif isinstance(result, dict):
    print(f"  keys: {list(result.keys())}")
    for k, v in result.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")
else:
    print(f"  value: {result}")

print()
print("=== Testing rasterization info keys ===")
# Check what info dict contains
x2 = torch.zeros(10, 3, device="cuda")
s2 = torch.ones(10, 3, device="cuda") * 0.01
q2 = torch.zeros(10, 4, device="cuda")
q2[:, 0] = 1.0
opacities = torch.ones(10, 1, device="cuda") * 0.5
colors = torch.randn(10, 3, device="cuda")
sh_degree = 0

rendered, render_alphas, info = gsplat.rasterization(
    means=x2, quats=q2, scales=s2,
    opacities=opacities, colors=colors,
    viewmats=viewmat, Ks=K,
    width=width, height=height,
    tile_size=tile_size, packed=True, sh_degree=sh_degree,
    radius_clip=0.0, eps2d=0.1, render_mode="RGB",
)
print(f"info keys: {sorted(info.keys())}")
for k, v in info.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"  {k}: {type(v)}")
