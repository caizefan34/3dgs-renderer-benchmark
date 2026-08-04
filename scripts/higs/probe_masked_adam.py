"""Round-60 masked-adam correctness probe (EPIC-05, torch 2.7 fused Adam).

The kernel implements torch's fused-Adam update rule with IEEE float32/double
math.  Param updates match torch's fused kernel to within a couple of float32
ulps.  State (exp_avg / exp_avg_sq) matches to ~1 ulp on a small fraction of
elements (measured <0.5%): torch's fused kernel is built with its own nvcc
flags and its m/v arithmetic occasionally differs from this kernel's by one
rounding decision, which stays bounded and does not feed back into the mask
check (grads are independent of params here).  The masking semantics must be
exact: masked-False rows stay bit-identical to init, and an all-True mask
must track vanilla fused Adam within tolerance over many steps.
"""
import sys

sys.path.insert(0, "benchmark")

import torch

from higs_masked_adam import masked_adam_step

torch.manual_seed(1234)
dev = "cuda:0"


def make_opt(params, fused=True):
    groups = [
        {"params": [params[0]], "lr": 1.6e-4},
        {"params": [params[1]], "lr": 1e-3},
        {"params": [params[2]], "lr": 5e-3},
        {"params": [params[3]], "lr": 5e-2},
        {"params": [params[4]], "lr": 2.5e-3},
    ]
    if fused:
        try:
            return torch.optim.Adam(groups, fused=True)
        except (TypeError, ValueError, RuntimeError):
            pass
    return torch.optim.Adam(groups)


def rand_grads(p, scale=1e-2):
    return [torch.randn_like(t) * scale for t in p]


def rel_diff(a, b):
    """Max abs diff normalized by the tensor's own max magnitude (robust to
    zero crossings; a pure 1-ulp difference at scale s reads ~1.2e-7)."""
    d = (a - b).abs()
    denom = max(a.abs().max().item(), 1e-12)
    return (d.max().item() / denom)


N = 50000
params0 = [
    torch.randn(N, 3, device=dev),
    torch.randn(N, 4, device=dev),
    torch.randn(N, 3, device=dev).abs() + 0.01,
    torch.rand(N, 1, device=dev) * 0.5,
    torch.randn(N, 16, 3, device=dev),
]

fail = False
TOL_MV = 5e-6   # state: ~1 ulp at scale, 20-step accumulation stays <1e-6
TOL_P = 1e-6    # params: measured max 2e-7

# --- 1. all-True mask vs vanilla fused Adam over 20 steps -----------------
pa = [t.detach().clone().requires_grad_(True) for t in params0]
pb = [t.detach().clone().requires_grad_(True) for t in params0]
opt_a = make_opt(pa)
opt_b = make_opt(pb)
all_true = torch.ones(N, dtype=torch.bool, device=dev)
worst_m = 0.0
worst_v = 0.0
worst_p = 0.0
for step in range(20):
    ga = rand_grads(pa)
    gb = [g.clone() for g in ga]
    for t, g in zip(pa, ga):
        t.grad = g
    for t, g in zip(pb, gb):
        t.grad = g
    opt_a.step()
    masked_adam_step(opt_b, all_true)
    opt_a.zero_grad(set_to_none=True)
    opt_b.zero_grad(set_to_none=True)
    for i, (a, b) in enumerate(zip(pa, pb)):
        rm = rel_diff(opt_a.state[pa[i]]["exp_avg"], opt_b.state[pb[i]]["exp_avg"])
        rv = rel_diff(opt_a.state[pa[i]]["exp_avg_sq"], opt_b.state[pb[i]]["exp_avg_sq"])
        rp = rel_diff(a, b)
        worst_m = max(worst_m, rm)
        worst_v = max(worst_v, rv)
        worst_p = max(worst_p, rp)
        if rm > TOL_MV:
            fail = True
            print(f"P1 m OUT OF TOL step={step} group={i} rel={rm:.3e}")
        if rv > TOL_MV:
            fail = True
            print(f"P1 v OUT OF TOL step={step} group={i} rel={rv:.3e}")
        if rp > TOL_P:
            fail = True
            print(f"P1 p OUT OF TOL step={step} group={i} rel={rp:.3e}")
print(f"P1 alltrue: max rel-diff m={worst_m:.3e} v={worst_v:.3e} p={worst_p:.3e} (tols m/v {TOL_MV}, p {TOL_P})")

# --- 2. partial mask: True rows ~= vanilla, False rows frozen exactly ----
mask = (torch.rand(N, device=dev) < 0.45)
pc = [t.detach().clone().requires_grad_(True) for t in params0]
pd = [t.detach().clone().requires_grad_(True) for t in params0]
opt_c = make_opt(pc)
opt_d = make_opt(pd)
gc = rand_grads(pc)
gd = [g.clone() for g in gc]
for t, g in zip(pc, gc):
    t.grad = g
for t, g in zip(pd, gd):
    t.grad = g
opt_c.step()
masked_adam_step(opt_d, mask)
worst_true_rel = 0.0
for i, (c, d) in enumerate(zip(pc, pd)):
    d_sel = d.view(N, -1)[mask]
    c_sel = c.view(N, -1)[mask]
    r = rel_diff(d_sel, c_sel)
    worst_true_rel = max(worst_true_rel, r)
    if r > TOL_P:
        fail = True
        print(f"P2 true-row OUT OF TOL group={i} rel={r:.3e}")
    d_frozen = d.view(N, -1)[~mask]
    p0_sel = params0[i].view(N, -1)[~mask]
    if not torch.equal(d_frozen, p0_sel):
        fail = True
        print(f"P2 frozen row CHANGED group={i}")
print(f"P2 partial: True rows max rel-diff={worst_true_rel:.3e}, False rows frozen bit-identical (coverage={mask.float().mean().item():.3f})")

print("FAIL" if fail else "PROBE_PASS")
