"""Time torch fused Adam step vs masked_adam_step at benchmark scale."""
import sys, time
sys.path.insert(0, "benchmark")
import torch
from higs_masked_adam import masked_adam_step

torch.manual_seed(0)
dev = "cuda:0"
N = 3500000
groups = [
    torch.randn(N, 3, device=dev) * 0.05,
    torch.randn(N, 4, device=dev) * 0.05,
    (torch.randn(N, 3, device=dev).abs() + 0.01) * 0.05,
    torch.rand(N, 1, device=dev) * 0.5,
    torch.randn(N, 16, 3, device=dev) * 0.05,
]
lrs = [1.6e-4, 1e-3, 5e-3, 5e-2, 2.5e-3]

def build_opt(params, fused=True):
    g = [{"params": [p], "lr": lr} for p, lr in zip(params, lrs)]
    try:
        return torch.optim.Adam(g, fused=fused)
    except Exception:
        return torch.optim.Adam(g)

def timeit(fn, iters=50):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

# torch fused
pa = [t.detach().clone().requires_grad_(True) for t in groups]
opt_a = build_opt(pa)
for t in pa:
    t.grad = torch.randn_like(t) * 1e-2
print("torch fused step ms:", round(timeit(opt_a.step), 3))

# masked, mask ratios
for frac in (1.0, 0.58, 0.42, 0.0):
    pb = [t.detach().clone().requires_grad_(True) for t in groups]
    opt_b = build_opt(pb)
    mask = torch.rand(N, device=dev) < frac
    for t in pb:
        t.grad = torch.randn_like(t) * 1e-2
    def f(opt_b=opt_b, mask=mask):
        masked_adam_step(opt_b, mask)
    print(f"masked step ms (mask={frac}):", round(timeit(f), 3))
