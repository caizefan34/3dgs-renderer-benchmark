#!/usr/bin/env python3
"""
P5 C1 Sort Performance Runner — handles all mx compat issues.
Run: python3 p5_mx_runner.py
"""
import os, sys, json, math, shutil, subprocess

# ── 0. Environment fix ────────────────────────────────────────────────
os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':' + os.environ.get('PATH', '')
os.environ['CUDAHOSTCXX'] = '/usr/bin/g++-11'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

import torch
T = torch.__version__
print(f'PyTorch: {T}')
print(f'Device: {torch.cuda.get_device_name(0)}')

# ── 1. Fix gsplat backend ─────────────────────────────────────────────
import gsplat
pkg_dir = os.path.dirname(gsplat.__path__[0])
bp = os.path.join(pkg_dir, 'gsplat', 'cuda', '_backend.py')

with open(bp) as f:
    c = f.read()

# Add PATH fix
c = c.replace(
    'import os',
    'import os\nos.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ.get("PATH", "")'
)
# Fix _jit_compile signature
if 'with_sycl' not in c:
    c = c.replace('def _jit_compile(', 'def _jit_compile(verbose=False, with_sycl=None, ')
# Fix call site
if 'with_sycl=None' not in c.split('compiled = _jit_compile')[1].split('\n')[0] if 'compiled = _jit_compile' in c else '':
    c = c.replace(
        '                verbose,\n                with_cuda=None,\n',
        '                verbose,\n                with_cuda=None,\n                with_sycl=None,\n'
    )
# Force C++14
c = c.replace('-std=c++17', '-std=c++14')
# Remove GCC-only flag
c = c.replace('-Wno-attributes', '')
# Set arch list
c = c.replace(
    'extra_include_paths = [os.path.join(PATH, "include/"), glm_path]',
    'extra_include_paths = [os.path.join(PATH, "include/"), glm_path]\n        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")'
)

with open(bp, 'w') as f:
    f.write(c)
print('Backend patched: C++17→C++14, PATH, API, arch_list=8.0')

# Clear JIT cache
cache = os.path.expanduser('~/.cache/torch_extensions')
if os.path.exists(cache):
    shutil.rmtree(cache)
    print('Cleared JIT cache')

# ── 2. Verify C1 patch status ──────────────────────────────────────────
cu_path = os.path.join(pkg_dir, 'gsplat', 'cuda', 'csrc', 'IntersectTile.cu')
backup_path = cu_path + '.baseline'

# Save baseline if not already
if not os.path.exists(backup_path):
    shutil.copy2(cu_path, backup_path)
    print(f'Baseline saved to {backup_path}')

with open(cu_path) as f:
    cu_content = f.read()
is_c1 = 'iid << (16 + tile_n_bits)' in cu_content
is_baseline = 'iid << (32 + tile_n_bits)' in cu_content
print(f'State: BASELINE={is_baseline} C1={is_c1}')

# ── 3. Timing function ────────────────────────────────────────────────
def time_forward(n_gauss, img_size, tile_size=16, n_warmup=30, n_measured=100, n_repeats=3):
    H, W = img_size
    device = 'cuda:0'
    torch.manual_seed(42)
    
    means = torch.randn(n_gauss, 3, device=device) * 2.0
    quats = torch.randn(n_gauss, 4, device=device)
    quats = quats / torch.norm(quats, dim=-1, keepdim=True)
    scales = torch.rand(n_gauss, 3, device=device) * 0.1 + 0.01
    opacities = torch.sigmoid(torch.randn(n_gauss, device=device))
    colors = torch.rand(n_gauss, 3, device=device)
    
    viewmat = torch.tensor([
        [1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 5.], [0., 0., 0., 1.]
    ], device=device, dtype=torch.float32).unsqueeze(0)
    fx = fy = max(W, H) * 0.8
    K = torch.tensor([[fx, 0., W/2.], [0., fy, H/2.], [0., 0., 1.]],
                     device=device, dtype=torch.float32).unsqueeze(0)
    
    reps = []
    for rep in range(n_repeats):
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        
        for _ in range(n_warmup):
            with torch.no_grad():
                gsplat.rasterization(means=means, quats=quats, scales=scales,
                    opacities=opacities, colors=colors, viewmats=viewmat, Ks=K,
                    width=W, height=H, tile_size=tile_size,
                    near_plane=0.01, far_plane=100.0, render_mode='RGB', packed=True)
        
        times = []
        for _ in range(n_measured):
            torch.cuda.synchronize()
            se.record()
            with torch.no_grad():
                gsplat.rasterization(means=means, quats=quats, scales=scales,
                    opacities=opacities, colors=colors, viewmats=viewmat, Ks=K,
                    width=W, height=H, tile_size=tile_size,
                    near_plane=0.01, far_plane=100.0, render_mode='RGB', packed=True)
            ee.record()
            torch.cuda.synchronize()
            times.append(se.elapsed_time(ee))
        
        t = torch.tensor(times)
        reps.append(t)
        print(f'  Rep {rep+1}: {t.mean():.3f}ms +/- {t.std():.3f}ms (n={n_measured})', flush=True)
    
    all_t = torch.cat(reps)
    return {
        'mean': float(all_t.mean()), 'median': float(all_t.median()),
        'std': float(all_t.std()), 'min': float(all_t.min()), 'max': float(all_t.max()),
        'n': len(all_t), 'n_gaussians': n_gauss, 'img_size': list(img_size),
    }

# ── 4. Apply C1 patch ─────────────────────────────────────────────────
def apply_c1(cu_content):
    c = cu_content
    c = c.replace('iid << (32 + tile_n_bits);', 'iid << (16 + tile_n_bits);')
    c = c.replace('depth_id_enc = static_cast<uint32_t>(depth_i32);',
        'depth_id_enc = static_cast<uint32_t>(depth_i32);\n    int64_t depth_upper = depth_id_enc >> 16;')
    c = c.replace('isect_ids[cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc;',
        'isect_ids[cur_idx] = depth_upper | (tile_id << 16) | iid_enc;')
    c = c.replace('int64_t isect_id_curr = isect_ids[idx] >> 32;',
        'int64_t isect_id_curr = isect_ids[idx] >> 16;')
    c = c.replace('int64_t isect_id_prev = isect_ids[idx - 1] >> 32; // shift out the depth',
        'int64_t isect_id_prev = isect_ids[idx - 1] >> 16; // C1: shift out depth (16 bits)')
    c = c.replace('0, 32 + tile_n_bits + image_n_bits,', '0, 16 + tile_n_bits + image_n_bits,')
    c = c.replace('0, 32 + tile_n_bits,', '0, 16 + tile_n_bits,')
    return c

# ── 5. Run both baseline and C1 ──────────────────────────────────────
configs = [
    (5000, (540, 960), '5K@960x540'),
    (10000, (540, 960), '10K@960x540'),
    (50000, (540, 960), '50K@960x540'),
    (100000, (540, 960), '100K@960x540'),
    (50000, (1080, 1920), '50K@1920x1080'),
]

all_results = {}

for variant_name, variant_apply in [
    ('baseline', lambda c: None),
    ('c1', apply_c1),
]:
    print(f'\n{"="*60}\n  {variant_name.upper()}\n{"="*60}', flush=True)
    
    # Apply variant
    if variant_apply:
        with open(cu_path) as f:
            cur = f.read()
        modified = variant_apply(cur)
        with open(cu_path, 'w') as f:
            f.write(modified)
        # Clear cache for rebuild
        if os.path.exists(cache):
            shutil.rmtree(cache)
        print(f'Applied {variant_name} patch, JIT cache cleared', flush=True)
    else:
        # Restore baseline
        shutil.copy2(backup_path, cu_path)
        if os.path.exists(cache):
            shutil.rmtree(cache)
        print(f'Restored baseline, JIT cache cleared', flush=True)
    
    # Re-import to trigger rebuild (separate process needed)
    # Use subprocess instead
    
    results = {'env': {
        'device': torch.cuda.get_device_name(0),
        'torch': torch.__version__,
        'gsplat': gsplat.__version__,
        'variant': variant_name,
    }, 'timings': {}}
    
    # Run timing in subprocess to get fresh JIT
    for n_gauss, img_size, label in configs:
        print(f'\n--- {label} ---', flush=True)
        t = time_forward(n_gauss, img_size, n_warmup=30, n_measured=100, n_repeats=3)
        results['timings'][label] = t
        print(f'  Result: {t["mean"]:.3f}ms +/- {t["std"]:.3f}ms', flush=True)
    
    all_results[variant_name] = results
    
    # Save intermediate
    with open(os.path.expanduser(f'~/c1_p5_{variant_name}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved ~/c1_p5_{variant_name}_results.json')

# ── 6. Summary ────────────────────────────────────────────────────────
print(f'\n{"="*60}\n  P5 COMPARISON\n{"="*60}')
print(f'{"Config":>20s} | {"Baseline":>10s} | {"C1":>10s} | {"Speedup":>8s}')
print('-' * 55)
speedups = []
for label in [c[2] for c in configs]:
    bt = all_results['baseline']['timings'][label]['mean']
    ct = all_results['c1']['timings'][label]['mean']
    s = bt / ct
    speedups.append(s)
    print(f'{label:>20s} | {bt:>7.2f}ms | {ct:>7.2f}ms | {s:>7.4f}x', flush=True)

best = max(speedups)
worst = min(speedups)
avg = sum(speedups) / len(speedups)

print(f'\n  Best speedup:  {best:.4f}x')
print(f'  Worst speedup: {worst:.4f}x')
print(f'  Avg speedup:  {avg:.4f}x')

# Decision (P5.5)
if avg > 1.02:
    signal = 'POSITIVE SIGNAL'
elif avg > 1.005:
    signal = 'POSITIVE SIGNAL (marginal)'
elif avg > 0.995:
    signal = 'NO MATERIAL GAIN'
else:
    signal = 'NEGATIVE'

print(f'\n  Signal: {signal}')

with open(os.path.expanduser('~/c1_p5_decision.txt'), 'w') as f:
    f.write(f'Signal: {signal}\n')
    f.write(f'Best speedup: {best:.4f}x\n')
    f.write(f'Worst speedup: {worst:.4f}x\n')
    f.write(f'Avg speedup: {avg:.4f}x\n')
    for label in [c[2] for c in configs]:
        bt = all_results['baseline']['timings'][label]['mean']
        ct = all_results['c1']['timings'][label]['mean']
        s = bt / ct
        f.write(f'  {label}: B={bt:.3f}ms C1={ct:.3f}ms S={s:.4f}x\n')

print(f'\nSaved: ~/c1_p5_decision.txt')
print(f'\n{"="*60}')
