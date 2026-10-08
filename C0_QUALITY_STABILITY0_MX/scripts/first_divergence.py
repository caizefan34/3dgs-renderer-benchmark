#!/usr/bin/env python3
"""C0-QUALITY-STABILITY0-MX first-divergence analysis.
For each scene x seed, compare A0 vs C0 per-iteration trajectories (n_gs, loss,
n_visible) from results.json's per_iter list, and find the first iteration where
they diverge beyond a relative tolerance. Also compare densification_events
(iter + n_after) to locate the first densification/pruning divergence.

Classification is OBSERVED from the recorded per_iter arrays.
"""
import argparse, json, os

SCENES = ["room", "bicycle", "garden"]
SEEDS = [42, 43, 44]
SEED42_ROOT = "/mnt/storage_pool/liaoyuanjun/mx_h8/incoming/derived/C0_SAME_PARENT_CAUSAL_LADDER_MX/raw"
ROOT4344 = "/dev/shm/c0_quality_stability/runs"
CAMPAIGN = "/mnt/storage_pool/liaoyuanjun/C0_QUALITY_STABILITY0_MX"

# A trajectory "diverges" at the first iter where BOTH arms have logged and the
# relative difference in n_gs exceeds tol (population is the discrete carrier of
# the training trajectory). We report the first such iter per pair.
NGS_TOL = 0.0   # any absolute difference in n_gs (densification is discrete)
LOSS_TOL = 1e-4  # relative loss difference threshold


def results_path(scene, seed, arm):
    base = SEED42_ROOT if seed == 42 else ROOT4344
    return os.path.join(base, f"{scene}_s{seed}_{arm}", "results.json")


def load(path):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            return None
    return None


def per_iter_map(r):
    pi = r.get("per_iter") or []
    return {e["iter"]: e for e in pi}


def dens_map(r):
    de = r.get("densification_events") or []
    return {e["iter"]: e for e in de}


def first_divergence(scene, seed):
    ra = load(results_path(scene, seed, "a0"))
    rc = load(results_path(scene, seed, "c0"))
    out = {"scene": scene, "seed": seed,
           "a0_present": ra is not None, "c0_present": rc is not None}
    if not ra or not rc:
        out["status"] = "MISSING_ARM"
        return out
    ma, mc = per_iter_map(ra), per_iter_map(rc)
    common = sorted(set(ma) & set(mc))
    div = None
    first_loss_div = None
    for it in common:
        na, nc = ma[it].get("n_gs"), mc[it].get("n_gs")
        if na is not None and nc is not None and na != nc:
            div = {"iter": it, "n_gs_a0": na, "n_gs_c0": nc,
                   "abs_diff": nc - na, "rel_diff": (nc - na) / na if na else None,
                   "loss_a0": ma[it].get("loss"), "loss_c0": mc[it].get("loss")}
            break
    for it in common:
        la, lc = ma[it].get("loss"), mc[it].get("loss")
        if la is not None and lc is not None and la > 0:
            rel = abs(lc - la) / la
            if rel > LOSS_TOL:
                first_loss_div = {"iter": it, "loss_a0": la, "loss_c0": lc, "rel_diff": rel}
                break
    # densification divergence
    da, dc = dens_map(ra), dens_map(rc)
    dcommon = sorted(set(da) & set(dc))
    dens_div = None
    for it in dcommon:
        if da[it].get("n_after") != dc[it].get("n_after"):
            dens_div = {"iter": it, "n_after_a0": da[it].get("n_after"),
                        "n_after_c0": dc[it].get("n_after")}
            break
    out.update({
        "first_population_divergence": div,
        "first_loss_divergence_rel": first_loss_div,
        "first_densification_divergence": dens_div,
        "n_common_iters": len(common),
        "n_a0_dens_events": len(da), "n_c0_dens_events": len(dc),
    })
    # classify the first divergent variable
    if div is not None:
        out["first_divergent_variable"] = "n_gaussians (population)"
        out["first_divergence_iter"] = div["iter"]
        # was it before or at the first densification?
        first_dens = min(dcommon) if dcommon else None
        out["relative_to_first_densification"] = (
            "at_or_before_first_densification" if (first_dens and div["iter"] <= first_dens)
            else "after_first_densification")
    elif first_loss_div is not None:
        out["first_divergent_variable"] = "loss"
        out["first_divergence_iter"] = first_loss_div["iter"]
    else:
        out["first_divergent_variable"] = "NONE_OBSERVED"
        out["first_divergence_iter"] = None
    out["status"] = "OK"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=CAMPAIGN)
    args = ap.parse_args()
    results = []
    for sc in SCENES:
        for sd in SEEDS:
            results.append(first_divergence(sc, sd))
    obj = {"artifact": "QUALITY_FIRST_DIVERGENCE", "campaign": "C0_QUALITY_STABILITY0_MX",
           "n_gs_tol": NGS_TOL, "loss_rel_tol": LOSS_TOL,
           "definition": "first iteration where A0/C0 per-iter n_gs differ (population is the discrete carrier); also first loss divergence (rel>tol) and first densification n_after divergence",
           "results": results}
    json.dump(obj, open(os.path.join(args.outdir, "QUALITY_FIRST_DIVERGENCE.json"), "w"), indent=2)
    for r in results:
        print(f"{r['scene']}/s{r['seed']}: var={r.get('first_divergent_variable')} "
              f"iter={r.get('first_divergence_iter')} popdiv={(r.get('first_population_divergence') or {}).get('iter')}")
    return obj


if __name__ == "__main__":
    main()
