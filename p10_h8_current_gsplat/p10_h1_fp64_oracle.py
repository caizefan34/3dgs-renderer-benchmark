#!/usr/bin/env python3
"""P10 Gate H1 — FP64 sufficient-statistic oracle for the H8-MR geometry adjoint.

Pure-algebra test, no GPU and no CUDA source needed. It re-derives, in exact
IEEE float64, what the two backward implementations must produce for the same
set of per-(gaussian, intersection) contributions:

  baseline (direct)     per intersection:
      v_sigma   = -opac * vis * v_alpha
      v_mean    = (v_sigma*(A*dx + B*dy), v_sigma*(B*dx + C*dy))
      v_conic   = (0.5*v_sigma*dx*dx, v_sigma*dx*dy, 0.5*v_sigma*dy*dy)
      v_opacity = vis * v_alpha

  H8-MR (moment)        per intersection, accumulate:
      Sx = v_sigma*dx   Sy = v_sigma*dy
      Sxx = v_sigma*dx*dx  Sxy = v_sigma*dx*dy  Syy = v_sigma*dy*dy
      R0 = vis * v_alpha
    then reconstruct ONCE per gaussian:
      v_mean    = (A*Sx + B*Sy, B*Sx + C*Sy)      <- cross term B has NO 1/2
      v_conic   = (0.5*Sxx, Sxy, 0.5*Syy)
      v_opacity = R0

Both must agree to floating-point reduction precision. The interaction
predicate is reproduced exactly: only intersections with sigma >= 0,
alpha >= ALPHA_THRESHOLD and opac*vis <= MAX_ALPHA enter either accumulator.

Counts explicitly covered: 1,2,3,7,16,31,32,33,64,127,256,257,1024 (warp and
reduction-size boundaries +/-1) plus randomized counts, plus a negative control
that proves the oracle detects the classic 0.5-on-the-cross-term bug.
"""

import argparse
import json
import math
import platform
import sys

import numpy as np

ALPHA_THRESHOLD = np.float32(1.0 / 255.0)
MAX_ALPHA = np.float32(0.99)
EDGE_COUNTS = [1, 2, 3, 7, 16, 31, 32, 33, 64, 127, 256, 257, 1024]


def rng_conics(rng, G):
    """Random positive-definite inverse covariances (A, B, C) with A*C - B^2 > 0."""
    a = rng.uniform(0.05, 4.0, size=G)
    c = rng.uniform(0.05, 4.0, size=G)
    rho = rng.uniform(-0.85, 0.85, size=G)
    b = rho * np.sqrt(a * c)
    return np.stack([a, b, c], axis=1)


def make_case(rng, counts, dt, extra_noncontrib=0):
    """One gaussian per count entry; returns per-interaction arrays plus ids."""
    G = len(counts)
    conic = rng_conics(rng, G)
    dx_list, dy_list, opac_list, valpha_list, gid_list = [], [], [], [], []
    for g, n in enumerate(counts):
        total = int(n) + int(extra_noncontrib)
        if total == 0:
            continue
        scale = rng.uniform(1.0, 8.0)
        dx = rng.uniform(-scale, scale, size=total)
        dy = rng.uniform(-scale, scale, size=total)
        # span the opacity range so the predicate both admits and rejects
        opac = rng.uniform(0.01, 0.999, size=total)
        valpha = rng.uniform(-3.0, 3.0, size=total)
        dx_list.append(dx)
        dy_list.append(dy)
        opac_list.append(opac)
        valpha_list.append(valpha)
        gid_list.append(np.full(total, g, dtype=np.int64))
    if not dx_list:
        z = np.zeros(0, dtype=dt)
        return (np.zeros((G, 3), dtype=dt), z.copy(), z.copy(), z.copy(), z.copy(), z.copy())
    return (
        conic.astype(dt),
        np.concatenate(dx_list).astype(dt),
        np.concatenate(dy_list).astype(dt),
        np.concatenate(opac_list).astype(dt),
        np.concatenate(valpha_list).astype(dt),
        np.concatenate(gid_list),
    )


def evaluate(conic, dx, dy, opac, valpha, gid, G, dt):
    """Return (baseline, moments, n_contrib, n_skipped) for the given interactions."""
    A = conic[gid, 0]
    B = conic[gid, 1]
    C = conic[gid, 2]
    # forward weight evaluation, same formulas/order as the CUDA kernels
    sigma = dt(0.5) * (A * dx * dx + C * dy * dy) + B * dx * dy
    vis = np.exp(-sigma.astype(np.float64)).astype(dt) if dt is np.float32 else np.exp(-sigma)
    alpha = np.minimum(dt(MAX_ALPHA), opac * vis)
    contributing = (sigma >= dt(0)) & (alpha >= dt(ALPHA_THRESHOLD))

    v_sigma = -opac * vis * valpha
    base_m = np.zeros((G, 2), dtype=dt)
    base_c = np.zeros((G, 3), dtype=dt)
    base_o = np.zeros(G, dtype=dt)
    mom_m = np.zeros((G, 2), dtype=dt)
    mom_c = np.zeros((G, 3), dtype=dt)
    mom_o = np.zeros(G, dtype=dt)

    # geometry-branch predicate, identical to `if(opac * vis <= MAX_ALPHA)`
    geom = contributing & (opac * vis <= dt(MAX_ALPHA))
    for g in np.nonzero(np.bincount(gid[geom], minlength=G))[0]:
        m = geom & (gid == g)
        av, bv, cv = A[m], B[m], C[m]
        dxv, dyv, vsv = dx[m], dy[m], v_sigma[m]
        r = vis[m] * valpha[m]
        # ---- baseline: full geometry adjoint per intersection
        np.add.at(base_m, g, np.stack([(vsv * (av * dxv + bv * dyv)).sum(),
                                       (vsv * (bv * dxv + cv * dyv)).sum()]))
        base_c[g, 0] += (dt(0.5) * vsv * dxv * dxv).sum()
        base_c[g, 1] += (vsv * dxv * dyv).sum()
        base_c[g, 2] += (dt(0.5) * vsv * dyv * dyv).sum()
        base_o[g] += r.sum()
        # ---- H8-MR: opacity-absorbed sufficient statistics
        Sx = (vsv * dxv).sum()
        Sy = (vsv * dyv).sum()
        Sxx = (vsv * dxv * dxv).sum()
        Sxy = (vsv * dxv * dyv).sum()
        Syy = (vsv * dyv * dyv).sum()
        av0, bv0, cv0 = conic[g, 0], conic[g, 1], conic[g, 2]
        mom_m[g, 0] += av0 * Sx + bv0 * Sy
        mom_m[g, 1] += bv0 * Sx + cv0 * Sy
        mom_c[g, 0] += dt(0.5) * Sxx
        mom_c[g, 1] += Sxy
        mom_c[g, 2] += dt(0.5) * Syy
        mom_o[g] += r.sum()
    return (
        {"means2d": base_m, "conics": base_c, "opacity": base_o},
        {"means2d": mom_m, "conics": mom_c, "opacity": mom_o},
        int(geom.sum()),
        int((~contributing).sum()),
    )


def add_wrong_cross(moments, conic, dx, dy, opac, valpha, gid, G, dt):
    """Negative control: reconstruction with an illegal 0.5 on the cross term."""
    A, B, C = conic[gid, 0], conic[gid, 1], conic[gid, 2]
    sigma = dt(0.5) * (A * dx * dx + C * dy * dy) + B * dx * dy
    vis = np.exp(-sigma.astype(np.float64)).astype(dt) if dt is np.float32 else np.exp(-sigma)
    alpha = np.minimum(dt(MAX_ALPHA), opac * vis)
    v_sigma = -opac * vis * valpha
    geom = (sigma >= dt(0)) & (alpha >= dt(ALPHA_THRESHOLD)) & (opac * vis <= dt(MAX_ALPHA))
    wrong = np.zeros((G, 2), dtype=dt)
    for g in np.nonzero(np.bincount(gid[geom], minlength=G))[0]:
        m = geom & (gid == g)
        v = v_sigma[m]
        Sx = (v * dx[m]).sum()
        Sy = (v * dy[m]).sum()
        dSx = (v * dx[m]).sum()  # deliberately wrong
        wrong[g, 0] += conic[g, 0] * dSx + dt(0.5) * conic[g, 1] * Sy
        wrong[g, 1] += dt(0.5) * conic[g, 1] * dSx + conic[g, 2] * Sy
    return wrong


def metric(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return {"max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0, "cosine": 1.0, "support_mismatch": 0}
    d = b - a
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    return {
        "max_abs": float(np.max(np.abs(d))),
        "mean_abs": float(np.mean(np.abs(d))),
        "rel_l2": float(np.linalg.norm(d) / max(an, 1e-300)),
        "cosine": float(np.dot(a, b) / max(an * bn, 1e-300)),
        "support_mismatch": int(np.count_nonzero((np.abs(a) > 1e-30) != (np.abs(b) > 1e-30))),
    }


def run_case(name, rng, counts, dt, extra_noncontrib, log):
    conic, dx, dy, opac, valpha, gid = make_case(rng, counts, dt, extra_noncontrib)
    G = len(counts)
    base, mom, n_contrib, n_skipped = evaluate(conic, dx, dy, opac, valpha, gid, G, dt)
    per_tensor = {}
    worst = 0.0
    for k in ("means2d", "conics", "opacity"):
        mt = metric(base[k], mom[k])
        per_tensor[k] = mt
        worst = max(worst, mt["rel_l2"])
    # negative control: the oracle must reject the 0.5-on-cross-term variant
    wrong = add_wrong_cross(mom, conic, dx, dy, opac, valpha, gid, G, dt)
    ctrl = metric(base["means2d"], wrong)
    entry = {
        "case": name,
        "dtype": "float64" if dt is np.float64 else "float32",
        "n_gaussians": G,
        "counts": [int(c) for c in counts][:64],
        "n_interactions": int(dx.size),
        "n_contributing_geometry": n_contrib,
        "n_predicate_skipped": n_skipped,
        "per_tensor": per_tensor,
        "worst_rel_l2": worst,
        "negative_control_cross_term": ctrl,
    }
    log.append(entry)
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()

    log = []
    results = {"edge_cases": [], "random_counts": [], "separation": [], "summary": {}}

    # --- 1. interaction-count edge cases (warp/reduction boundaries +/-1)
    for n in EDGE_COUNTS:
        rng = np.random.default_rng(1000 + n)
        e64 = run_case(f"edge_n{n}", rng, [n], np.float64, extra_noncontrib=13, log=log)
        e32 = run_case(f"edge_n{n}", rng, [n], np.float32, extra_noncontrib=13, log=log)
        e64["fp32"] = e32["per_tensor"]
        e64["fp32_worst_rel_l2"] = e32["worst_rel_l2"]
        results["edge_cases"].append(e64)

    # --- 2. randomized interaction counts
    rng = np.random.default_rng(20260925)
    for t in range(args.trials):
        n = int(rng.integers(0, 3000))
        e64 = run_case(f"rand{t}_n{n}", rng, [n], np.float64, extra_noncontrib=int(rng.integers(0, 200)), log=log)
        e32 = run_case(f"rand{t}_n{n}", rng, [n], np.float32, extra_noncontrib=0, log=log)
        e64["fp32_worst_rel_l2"] = e32["worst_rel_l2"]
        results["random_counts"].append(e64)

    # --- 3. many gaussians: per-gaussian separation (no cross-contamination)
    rng = np.random.default_rng(777)
    counts = rng.integers(0, 300, size=4096).tolist()
    s64 = run_case("separation_4096", rng, counts, np.float64, extra_noncontrib=5, log=log)
    s32 = run_case("separation_4096", rng, counts, np.float32, extra_noncontrib=5, log=log)
    s64["fp32_worst_rel_l2"] = s32["worst_rel_l2"]
    results["separation"].append(s64)

    # --- verdict inputs
    fp64_tol = 1e-9          # pure reduction-order noise in float64
    fp32_tol = 1e-4          # reassociation-level noise in float32
    fails = []
    for grp in ("edge_cases", "random_counts", "separation"):
        for e in results[grp]:
            if e["worst_rel_l2"] > fp64_tol:
                fails.append({"case": e["case"], "scope": grp, "why": "fp64 rel_l2 above reduction tolerance"})
            if e["negative_control_cross_term"]["rel_l2"] < 1e-6:
                fails.append({"case": e["case"], "scope": grp, "why": "oracle failed to reject wrong cross term"})
    fp64_vals = [e["worst_rel_l2"] for e in log if e["dtype"] == "float64"]
    fp32_vals = [e["worst_rel_l2"] for e in log if e["dtype"] == "float32"]
    ctrl_vals = [e["negative_control_cross_term"]["rel_l2"] for e in log]
    fp32_max = max(fp32_vals or [0.0])
    fp64_max = max(fp64_vals or [0.0])
    results["summary"] = {
        "verdict": "PASS" if not fails else "FAIL",
        "logical_failures": fails,
        "n_cases": len(log),
        "fp64_max_rel_l2": fp64_max,
        "fp64_tolerance": fp64_tol,
        "fp32_max_rel_l2": fp32_max,
        "fp32_tolerance": fp32_tol,
        "fp32_within_reassociation_tolerance": bool(fp32_max <= fp32_tol),
        "negative_control_min_rel_l2": min(ctrl_vals or [0.0]),
        "negative_control_n_rejected": int(sum(1 for v in ctrl_vals if v >= 1e-6)),
        "edge_counts": EDGE_COUNTS,
        "random_trials": args.trials,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "note": (
            "H8-MR is an exact regrouping of the same linear per-intersection terms; "
            "float64 differences are accumulation-order noise only, and float32 "
            "differences are the same reassociation level the production kernel "
            "already exhibits run-to-run under atomic scatter."
        ),
    }
    with open(args.out, "w") as f:
        json.dump({"summary": results["summary"], "cases": log, "groups": results}, f, indent=2)
    s = results["summary"]
    print(f"H1 FP64 oracle: {s['verdict']}  cases={s['n_cases']}  "
          f"fp64_max_rel_l2={s['fp64_max_rel_l2']:.3e}  fp32_max_rel_l2={s['fp32_max_rel_l2']:.3e}")
    if fails:
        for f_ in fails[:10]:
            print("  FAIL:", f_)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
