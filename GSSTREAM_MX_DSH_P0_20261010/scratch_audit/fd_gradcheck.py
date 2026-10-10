#!/usr/bin/env python3
"""INDEPENDENT finite-difference gradient check for cpu_oracle.

Rationale: the dense-vs-sparse parity test in cpu_oracle.py shares the SAME
compositing + reverse-mode gradient code on both paths, so it can only detect
row-interval culling bugs, NOT a wrong alpha formula or a wrong gradient. This
script checks the analytic gradients against central finite differences of the
toy forward render -- the independence the parity test lacks.

CPU-only, no GPU. Perturbs opacity, mean2d, conic, and color and compares.
"""
import numpy as np
from cpu_oracle import evaluate, CUTOFF, TERMINATION

rng = np.random.default_rng(7)

def build(n=6, H=16, W=16, seed=0):
    g = np.random.default_rng(seed)
    mu = g.uniform(2, W-2, (n, 2))
    P = np.stack([g.uniform(0.05, 0.5, n), g.uniform(-0.05, 0.05, n),
                  g.uniform(0.05, 0.5, n)], axis=1)
    # keep SPD
    a, b, c = P[:, 0], P[:, 1], P[:, 2]
    bad = (a*c - b*b) <= 0
    while bad.any():
        P[bad, 1] *= 0.3
        a, b, c = P[:, 0], P[:, 1], P[:, 2]
        bad = (a*c - b*b) <= 0
    o = g.uniform(0.05, 0.9, n)
    col = g.uniform(0, 1, (n, 3))
    d = g.uniform(0.1, 50, n)
    return mu, P, o, col, d, H, W

def fd_grad(f, x, eps=1e-6):
    """central FD gradient of scalar f wrt array x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps; fp = f()
        x[idx] = orig - eps; fm = f()
        x[idx] = orig
        grad[idx] = (fp - fm) / (2*eps)
        it.iternext()
    return grad

mu, P, o, col, d, H, W = build()
base = dict(mu=mu.copy(), P=P.copy(), o=o.copy(), col=col.copy(), d=d.copy(),
            H=H, W=W)

def forward(**kw):
    st = dict(base); st.update(kw)
    r = evaluate(st['mu'], st['P'], st['o'], st['col'], st['d'], st['H'], st['W'], True)
    # scalar objective = 0.5 * sum((rgb-target)^2), matching toy MSE in evaluate
    target = np.full((st['H'], st['W'], 3), .4)
    return 0.5*np.sum((r['rgb']-target)**2)

# analytic
r = evaluate(mu, P, o, col, d, H, W, True)

print("=== Finite-difference vs analytic (toy RGB-MSE objective) ===")
worst = {}
for name, key in [('opacity', 'o'), ('mean2d', 'mu'), ('conic', 'P'), ('color', 'col')]:
    x0 = base[key]
    fd = fd_grad(lambda k=key: forward(**{k: x0}), x0)
    an = r[{'opacity':'gopa','mean2d':'gmean','conic':'gcol' if False else 'gopa','color':'gcol'}[name]] if False else None
    analytic = {'opacity': r['gopa'], 'mean2d': r['gmean'],
                'conic': None, 'color': r['gcol']}[name]
    if analytic is None:
        print(f"{name:8s}: no analytic conic grad exposed by oracle (expected)")
        continue
    denom = np.maximum(np.abs(fd).max(), 1e-12)
    rel = np.max(np.abs(analytic - fd)) / denom
    worst[name] = rel
    print(f"{name:8s}: max|analytic-fd|={np.max(np.abs(analytic-fd)):.3e}  rel={rel:.3e}")

# The oracle exposes gcol, gopa, gmean (NOT conic grads). FD check those three.
print()
print("=== Verdict ===")
ok = all(v < 1e-4 for v in worst.values())
print("PASS: analytic opacity/mean2d/color grads match finite differences"
      if ok else "FAIL: gradient mismatch")
print("(conic gradients are not exposed by the oracle; only mean2d/opacity/color are)")
