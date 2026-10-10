#!/usr/bin/env python3
"""
WV-HIER-R1-MX  Stage A — Hierarchical Work-Constrained Sampling (HWCS)
Part 1: Hierarchical sampling variance derivation + Monte Carlo validation.

Scientific scope: ANALYTIC + SIMULATION ONLY. No GPU, no CUDA, no training.
This module proves the math the task requires:
  * child marginal inclusion probability p = q * r
  * HT weight = 1/(q*r) = 1/p
  * joint inclusion probability of two children in the SAME parent
  * explicit covariance term in gradient variance (NOT independent-Bernoulli)
  * the exact convex-blend identity:
        Var_HWCS = (1-alpha)*Var_nativeG4 + alpha*Var_G16
        alpha = r*(1-q)/(1-p)
"""
import numpy as np

MASK = np.random.default_rng(20261008)


# ----------------------------------------------------------------------------
# Analytic variance of the hierarchical (two-stage) tile estimator.
#
# Notation (per scalar gradient component; sum over components for NMSE):
#   child t has contribution a_t ; parent T has b_T = sum_{t in T} a_t
#   g = sum_t a_t = sum_T b_T                       (dense gradient)
#   HWCS: parent active w.p. q (M_T~Bern(q)); child active w.p. r given parent
#         active (m_{t|T}~Bern(r)); child indicator xi_t = M_T * m_{t|T}
#         marginal p = q*r ; HT weight 1/p ; estimator ghat = sum_t (xi_t/p) a_t
# ----------------------------------------------------------------------------
def analytic_variances(a_child, parents_per_child, q, r):
    """Return dict of analytic per-component variances.

    a_child:            (C,) contributions of C child tiles (one param comp)
    parents_per_child:  (C,) int parent index in [0,P) for each child
    q, r:               parent / conditional child inclusion probs
    """
    a = np.asarray(a_child, dtype=np.float64)
    pid = np.asarray(parents_per_child, dtype=np.int64)
    p = q * r
    # per-component sums of squares
    S1 = float(np.sum(a * a))                      # sum_child a_t^2
    b = np.zeros(pid.max() + 1, dtype=np.float64)
    np.add.at(b, pid, a)
    S2 = float(np.sum(b * b))                      # sum_parent b_T^2
    # native G4: iid Bernoulli(p) over children
    var_g4 = (1.0 - p) / p * S1
    # G16: iid Bernoulli(p) over parents  (unit = whole parent block)
    var_g16 = (1.0 - p) / p * S2
    # HWCS: two-stage with explicit within-parent covariance
    alpha = r * (1.0 - q) / (1.0 - p)
    var_hwcs = (1.0 - alpha) * var_g4 + alpha * var_g16
    # expanded form (shows the covariance term explicitly):
    #   Var = (1-p)/p S1 + r(1-q)/p * (S2 - S1)
    cov_within = S2 - S1                           # sum_T sum_{t!=t'} a_t a_t'
    var_hwcs_expanded = (1.0 - p) / p * S1 + r * (1.0 - q) / p * cov_within
    return dict(p=p, S1=S1, S2=S2, cov_within=cov_within, alpha=alpha,
                var_g4=var_g4, var_g16=var_g16,
                var_hwcs=var_hwcs, var_hwcs_expanded=var_hwcs_expanded)


# ----------------------------------------------------------------------------
# Monte Carlo estimator of the same quantities (empirical Var of ghat).
# ----------------------------------------------------------------------------
def mc_variances(a_child, parents_per_child, q, r, n=400000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a_child, dtype=np.float64)
    pid = np.asarray(parents_per_child, dtype=np.int64)
    C = a.size
    p = q * r
    P = pid.max() + 1
    # parent active draws
    M = (rng.random((n, P)) < q).astype(np.float64)
    # child conditional draws
    m = (rng.random((n, C)) < r).astype(np.float64)
    xi = M[np.arange(n)[:, None], pid[None, :]] * m     # (n,C) child indicators
    # estimators
    ghat_g4 = ((rng.random((n, C)) < p).astype(np.float64) / p) @ a
    b_by_parent = np.array([np.sum(a[pid == t]) for t in range(P)])
    ghat_g16 = ((rng.random((n, P)) < p).astype(np.float64) / p) @ b_by_parent
    ghat_hwcs = (xi / p) @ a
    g = float(np.sum(a))
    def var(gh):
        return float(np.mean((gh - g) ** 2))
    return dict(var_g4=var(ghat_g4), var_g16=var(ghat_g16),
                var_hwcs=var(ghat_hwcs))


if __name__ == "__main__":
    print("=" * 74)
    print("STAGE A PART 1 — Hierarchical variance derivation validation")
    print("=" * 74)
    # Synthetic scene: 40 parents, 16 children each, controlled S1 vs S2.
    P, C = 40, 640
    pid = np.repeat(np.arange(P), C // P)
    # Give children same-sign contributions within a parent (positively
    # correlated), which is the realistic 3DGS case locally.
    rng = np.random.default_rng(7)
    parent_bias = rng.normal(0.0, 1.0, size=P)
    child = parent_bias[pid] + rng.normal(0.0, 0.3, size=C)
    for (q, r, name) in [(0.5, 0.5, "HWCS-1"), (0.75, 1.0 / 3.0, "HWCS-2")]:
        an = analytic_variances(child, pid, q, r)
        mc = mc_variances(child, pid, q, r, n=200000, seed=11)
        print(f"\n--- {name}: q={q}, r={r:.4f}, p=q*r={an['p']:.4f} ---")
        print(f"  alpha = r(1-q)/(1-p) = {an['alpha']:.6f}")
        print(f"  S1(child sq)={an['S1']:.4f}  S2(parent sq)={an['S2']:.4f}"
              f"  within-parent cov term={an['cov_within']:.4f}")
        print(f"  ANALYTIC  Var: G4={an['var_g4']:.5f}  HWCS={an['var_hwcs']:.5f}"
              f"  G16={an['var_g16']:.5f}")
        print(f"  MONTECARLO Var: G4={mc['var_g4']:.5f}  HWCS={mc['var_hwcs']:.5f}"
              f"  G16={mc['var_g16']:.5f}")
        print(f"  blend check (expanded==compact): "
              f"{abs(an['var_hwcs']-an['var_hwcs_expanded']):.2e}")
        print(f"  MC vs analytic (HWCS rel err): "
              f"{abs(mc['var_hwcs']-an['var_hwcs'])/an['var_hwcs']*100:.2f}%")
    print("\nInterpretation:")
    print("  * alpha in (0,1): HWCS variance is a convex blend of native-G4 and")
    print("    G16 variance. Since S2>=S1 for same-sign local contributions,")
    print("    Var_G16 >= Var_HWCS >= Var_nativeG4.")
    print("  * HWCS-1 (lower q) -> larger alpha -> closer to G16 (higher var).")
    print("  * HWCS-2 (higher q) -> smaller alpha -> closer to native G4.")
    print("  * Endpoint limits: q=p,r=1 -> HWCS==G16 ; q=1,r=p -> HWCS==nativeG4.")
