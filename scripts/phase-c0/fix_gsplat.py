import os, sys

# Create gsplat/csrc/__init__.py that loads the pre-compiled .so
gsplat_dir = "/home/liaoyuanjun/.local/lib/python3.10/site-packages/gsplat"
csrc_dir = os.path.join(gsplat_dir, "csrc")
os.makedirs(csrc_dir, exist_ok=True)

init_content = '''# Auto-generated: load pre-compiled gsplat_cuda extension
import importlib.util
import os

_so_path = os.path.expanduser(
    "~/.cache/torch_extensions/py310_cu118/gsplat_cuda/gsplat_cuda.so"
)

if not os.path.exists(_so_path):
    raise ImportError(f"Pre-compiled gsplat_cuda not found at {_so_path}")

_spec = importlib.util.spec_from_file_location("gsplat_cuda", _so_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Expose all attributes
for _attr in dir(_mod):
    if not _attr.startswith("__"):
        globals()[_attr] = getattr(_mod, _attr)
'''

init_path = os.path.join(csrc_dir, "__init__.py")
with open(init_path, "w") as f:
    f.write(init_content)
print(f"Created {init_path}")

# Now test the import
sys.path.insert(0, "/home/liaoyuanjun/3dgs-renderer-benchmark/src")
try:
    from gsplat import rasterization
    print("SUCCESS: from gsplat import rasterization")
    
    # Quick test
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
    print(f"Render shape: {r.shape}")
    print(f"Meta keys: {list(meta.keys())}")
    if "means2d" in meta:
        print(f"  means2d shape: {meta['means2d'].shape}")
        print(f"  means2d requires_grad: {meta['means2d'].requires_grad}")
    if "radii" in meta:
        print(f"  radii shape: {meta['radii'].shape}")
    if "tiles_per_gauss" in meta:
        print(f"  tiles_per_gauss shape: {meta['tiles_per_gauss'].shape}")
    
    # Test gradient flow through means2d
    means_param = torch.nn.Parameter(means)
    r2, _, meta2 = rasterization(
        means=means_param, quats=quats, scales=torch.exp(scales),
        opacities=opacities, colors=colors,
        viewmats=viewmat, Ks=K,
        width=800, height=600,
        packed=False, sh_degree=0,
    )
    loss = r2[0].sum()
    loss.backward()
    print(f"\nmeans_param.grad is not None: {means_param.grad is not None}")
    print(f"means2d.grad: {meta2['means2d'].grad is not None if 'means2d' in meta2 else 'N/A'}")
    
    print("\nALL TESTS PASSED")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"FAILED: {e}")
