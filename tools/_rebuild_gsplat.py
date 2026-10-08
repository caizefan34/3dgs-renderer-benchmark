#!/usr/bin/env python3
"""
Phase V3: Trigger JIT rebuild, fix build.ninja, run ninja, verify C1 build.
Run from VS 2022 x64 developer command prompt.
"""
import sys, os, subprocess

JIT_DIR = os.path.expandvars(
    r'%LOCALAPPDATA%\torch_extensions\torch_extensions\Cache\py313_cu130\gsplat_cuda')

os.environ['VSLANG'] = '1033'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Step 1: Trigger JIT to generate build.ninja (will fail, that's OK)
print("Step 1: Triggering JIT to generate build.ninja...")
import gsplat.cuda._backend as b
# It will fail, but artifacts should be there

# Step 2: Patch build.ninja
ninja_path = os.path.join(JIT_DIR, 'build.ninja')
if not os.path.exists(ninja_path):
    print(f"ERROR: {ninja_path} not found!")
    sys.exit(1)

print(f"Step 2: Patching {ninja_path}")
with open(ninja_path, 'r') as f:
    content = f.read()

token = '-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING'
if token not in content:
    content = content.replace(
        '--expt-relaxed-constexpr',
        '--expt-relaxed-constexpr ' + token
    )
    with open(ninja_path, 'w') as f:
        f.write(content)
    print("  Patched build.ninja")
else:
    print("  Already patched")

# Step 3: Run ninja
print("Step 3: Running ninja...")
result = subprocess.run(
    ['ninja', '-v', '-j', '8'],
    cwd=JIT_DIR,
    capture_output=True, text=True, timeout=600
)

# Show last lines
lines = (result.stdout + result.stderr).split('\n')
last_lines = [l for l in lines if l.strip()]
for l in last_lines[-20:]:
    print(f"  {l}")

print(f"ninja exit code: {result.returncode}")

if result.returncode == 0:
    pyd_files = [f for f in os.listdir(JIT_DIR) if f.endswith('.pyd')]
    print(f"SUCCESS! Built: {pyd_files}")
    print(f"Size: {os.path.getsize(os.path.join(JIT_DIR, pyd_files[0])) / 1024 / 1024:.1f} MB")
    
    # Step 4: Test
    print("Step 4: Testing import...")
    for mod in list(sys.modules.keys()):
        if 'gsplat' in mod:
            del sys.modules[mod]
    
    from gsplat.cuda._backend import _C
    print(f"  Import OK: {type(_C)}")
    
    # Quick test
    import torch
    DEVICE = 'cuda'
    h, w = 540, 960
    n = 5000
    torch.manual_seed(42)
    means = torch.randn(1, n, 3, device=DEVICE) * 2
    quats = torch.randn(1, n, 4, device=DEVICE)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.rand(1, n, 3, device=DEVICE) * 0.02 + 0.005
    opacities = torch.sigmoid(torch.randn(1, n, device=DEVICE)).unsqueeze(-1)
    colors = torch.rand(1, n, 3, device=DEVICE)
    viewmat = torch.eye(4, device=DEVICE).unsqueeze(0)
    fx = w / 2.0
    K = torch.tensor([[fx, 0, w/2], [0, fx, h/2], [0, 0, 1]], device=DEVICE, dtype=torch.float32).unsqueeze(0)
    img_size = torch.tensor([h, w], device=DEVICE)
    
    from gsplat import rasterization
    with torch.no_grad():
        renders, alphas = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=colors, viewmat=viewmat, K=K, img_size=img_size,
            tile_size=16, packed=True,
        )
    print(f"  Render OK: {renders.shape}, {alphas.shape}")
    print("ALL GOOD: C1 build works!")
else:
    sys.exit(1)
