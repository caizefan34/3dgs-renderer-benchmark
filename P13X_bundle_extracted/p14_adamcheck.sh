#!/bin/bash
cat > /tmp/adamcheck.py <<'EOF'
import sys, torch
sys.path.insert(0, '/root/epic05_frozen/higs_c0_final30k_worktree')
import importlib.util
spec = importlib.util.spec_from_file_location('gsplat_cuda', '/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
sys.modules['gsplat_cuda'] = mod
print('adam op available:', hasattr(mod, 'adam'))
# quick functional check of the fused masked adam vs reference
N, M = 4096, 3
torch.manual_seed(0)
p = torch.randn(N, M, device='cuda')
g = torch.randn(N, M, device='cuda')
m0 = torch.randn(N, M, device='cuda')
v0 = torch.rand(N, M, device='cuda') + 0.1
vis = (torch.rand(N, device='cuda') > 0.7)
lr, b1, b2, eps, t = 1e-3, 0.9, 0.999, 1e-15, 16001
# reference (torch Adam semantics, bias-corrected, masked)
p_ref, m_ref, v_ref = p.clone(), m0.clone(), v0.clone()
gg = g * vis.unsqueeze(-1)
m_ref = b1 * m_ref + (1 - b1) * gg
v_ref = b2 * v_ref + (1 - b2) * gg * gg
mh = m_ref / (1 - b1 ** t); vh = v_ref / (1 - b2 ** t)
p_ref = p_ref - lr * mh / (vh.sqrt() + eps)
p_ref = torch.where(vis.unsqueeze(-1), p_ref, p)  # frozen rows untouched
# fused
p_f = p.clone(); m_f = m0.clone(); v_f = v0.clone()
mod.adam(p_f, g, m_f, v_f, vis, lr, b1, b2, eps)
dp_active = (p_f[vis] - p_ref[vis]).abs().max()
dfrozen = (p_f[~vis] - p[~vis]).abs().max()
dm_frozen = (m_f[~vis] - m0[~vis]).abs().max()
dv_frozen = (v_f[~vis] - v0[~vis]).abs().max()
print(f'active-row max diff vs ref: {dp_active:.3e}')
print(f'frozen param delta: {dfrozen:.3e}  frozen m delta: {dm_frozen:.3e}  frozen v delta: {dv_frozen:.3e}')
EOF
/root/miniforge3/envs/gsplat/bin/python /tmp/adamcheck.py
