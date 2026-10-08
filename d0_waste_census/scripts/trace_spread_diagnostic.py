#!/usr/bin/env python3
"""D0 trace-spread diagnostic (run on EPIC-05; reported evidence only).

Pre-registered companion to artifacts/D0_TRACE_VALIDITY_SCREEN_PREREG.json.
Reports, per scene, the full pairwise spread structure of the 3000-iter smoke
ensemble: within/cross group maxima (loss, dN, dN_rel), per-run mean distances,
two-sided and max-statistic permutation tests (20000 relabelings, seed
20260928), and the pre-registered 2.0x screen outcome.  Does not gate anything;
writes /root/epic05_d0/artifacts/D0_TRACE_SPREAD_DIAGNOSTIC.json.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = "/root/epic05_d0"
RUNS = os.path.join(ROOT, "runs")
ART = os.path.join(ROOT, "artifacts")
REF_DIR = "/root/epic05_p12/runs/p12a/p12a_c0f_s42_{scene}"
SCREEN_FACTOR = 2.0
SEED = 20260928
N_PERM = 20000


def load_run(rid):
    d = os.path.join(RUNS, rid)
    with open(os.path.join(d, "results.json")) as f:
        tr = json.load(f)
    return {"id": rid,
            "per_iter": tr.get("per_iter", []),
            "dens_events": tr.get("densification_events", [])}


def load_ref_prefix(scene, max_iter=3000):
    d = REF_DIR.format(scene=scene)
    with open(os.path.join(d, "results.json")) as f:
        tr = json.load(f)
    rows = [r for r in tr.get("per_iter", []) if r["iter"] <= max_iter]
    return {"id": "ref30k_prefix", "per_iter": rows, "dens_events": None}


def sample_table(run):
    it = np.array([r["iter"] for r in run["per_iter"]], dtype=np.int64)
    lo = np.array([r["loss"] for r in run["per_iter"]], dtype=np.float64)
    ng = np.array([r["n_gs"] for r in run["per_iter"]], dtype=np.int64)
    return it, lo, ng


def pair_dist(a, b):
    ita, la, na = a
    itb, lb, nb = b
    k = min(ita.size, itb.size)
    if not np.array_equal(ita[:k], itb[:k]):
        raise ValueError("iteration grids differ")
    dn = np.abs(na[:k] - nb[:k])
    denom = max(int(na[k - 1]), int(nb[k - 1]), 1)
    return (float(np.abs(la[:k] - lb[:k]).max()), int(dn.max()),
            float(dn.max() / denom))


def discover(prefix):
    return sorted(os.path.basename(p)
                  for p in glob.glob(os.path.join(RUNS, prefix + "*"))
                  if os.path.isdir(p))


def dist_matrices(runs, keys, tabs):
    """Symmetric dicts {(a,b): (max_loss, max_dN, dN_rel)} for a<b, as flat
    arrays aligned to pair index lists, plus helpers for the permuter."""
    n = len(runs)
    M_loss = np.zeros((n, n))
    M_dn = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ml, _, mdn = pair_dist(tabs[keys[i]], tabs[keys[j]])
            M_loss[i, j] = M_loss[j, i] = ml
            M_dn[i, j] = M_dn[j, i] = mdn
    return M_loss, M_dn


def perm_stats(M, n_clean, n_traced, rng):
    """Observed stats + permutation distributions.  Statistic 1 (mean diff):
    mean cross-group distance - mean within-group distance.  Statistic 2:
    max cross-group distance.  Returns dict with observed values and p-values
    (two-sided for the mean diff, one-sided upper for the max)."""
    n = n_clean + n_traced
    labels = np.zeros(n, dtype=bool)   # True = traced
    labels[n_clean:] = True
    idx = np.arange(n)

    def stats(lab):
        ct = M[np.ix_(lab, ~lab)]
        cc_idx = idx[~lab]
        tt_idx = idx[lab]
        cc = M[np.ix_(cc_idx, cc_idx)]
        tt = M[np.ix_(tt_idx, tt_idx)]
        cc_vals = cc[np.triu_indices(cc.shape[0], 1)]
        tt_vals = tt[np.triu_indices(tt.shape[0], 1)]
        ct_vals = ct.reshape(-1)
        within = np.concatenate([cc_vals, tt_vals])
        mean_diff = float(ct_vals.mean() - within.mean()) if within.size else 0.0
        max_ct = float(ct_vals.max()) if ct_vals.size else 0.0
        return mean_diff, max_ct

    obs_mean, obs_max = stats(labels)
    perm_mean = np.empty(N_PERM)
    perm_max = np.empty(N_PERM)
    for p in range(N_PERM):
        perm = rng.permutation(n)
        lab = np.zeros(n, dtype=bool)
        lab[perm[:n_traced]] = True
        pm, px = stats(lab)
        perm_mean[p] = pm
        perm_max[p] = px
    p_mean = float((np.abs(perm_mean) >= abs(obs_mean)).mean())
    p_max = float((perm_max >= obs_max).mean())
    return {"mean_diff_observed": obs_mean, "p_two_sided_mean_diff": p_mean,
            "max_ct_observed": obs_max, "p_one_sided_max_ct": p_max,
            "n_perm": N_PERM}


def scene_report(scene):
    clean_ids = ["ref30k_prefix"] + [r for r in discover("C3" + scene)]
    traced_ids = discover("T3" + scene)
    keys = clean_ids + traced_ids
    runs = {rid: load_run(rid) for rid in keys if rid != "ref30k_prefix"}
    runs["ref30k_prefix"] = load_ref_prefix(scene)
    tabs = {rid: sample_table(runs[rid]) for rid in keys}
    n_c, n_t = len(clean_ids), len(traced_ids)
    M_loss, M_dn = dist_matrices(keys, keys, tabs)
    n = len(keys)
    groups = {"cc": [(i, j) for i in range(n) for j in range(i + 1, n)
                     if i < n_c and j < n_c],
              "tt": [(i, j) for i in range(n) for j in range(i + 1, n)
                     if i >= n_c and j >= n_c],
              "ct": [(i, j) for i in range(n) for j in range(i + 1, n)
                     if (i < n_c) != (j < n_c)]}

    def agg(pairs, M):
        vals = [M[i, j] for i, j in pairs]
        return {"n_pairs": len(vals),
                "max": float(max(vals)) if vals else 0.0,
                "median": float(np.median(vals)) if vals else 0.0,
                "mean": float(np.mean(vals)) if vals else 0.0}

    out = {"clean_samples": clean_ids, "traced_samples": traced_ids}
    out["maxima"] = {}
    for key in ("cc", "tt", "ct"):
        out["maxima"][key] = {"loss": agg(groups[key], M_loss),
                              "dN_rel": agg(groups[key], M_dn)}
    env_loss = max(out["maxima"]["cc"]["loss"]["max"],
                   out["maxima"]["tt"]["loss"]["max"])
    env_dn = max(out["maxima"]["cc"]["dN_rel"]["max"],
                 out["maxima"]["tt"]["dN_rel"]["max"])
    ct_loss = out["maxima"]["ct"]["loss"]["max"]
    ct_dn = out["maxima"]["ct"]["dN_rel"]["max"]
    out["screen_2x"] = {
        "factor": SCREEN_FACTOR,
        "rule": "INVALID iff ct_max > factor * env_max, env = max(cc_max, "
                "tt_max); literal 1x value reported (see prereg)",
        "loss": {"ct_max": ct_loss, "env_max": env_loss,
                 "ratio": (ct_loss / env_loss if env_loss else None),
                 "inside_envelope_1x": bool(ct_loss <= env_loss),
                 "pass": bool(ct_loss <= SCREEN_FACTOR * env_loss)},
        "dN_rel": {"ct_max": ct_dn, "env_max": env_dn,
                   "ratio": (ct_dn / env_dn if env_dn else None),
                   "inside_envelope_1x": bool(ct_dn <= env_dn),
                   "pass": bool(ct_dn <= SCREEN_FACTOR * env_dn)},
    }
    rng = np.random.default_rng(SEED)
    out["permutation"] = {
        "seed": SEED,
        "dN_rel": perm_stats(M_dn, n_c, n_t, rng),
        "loss": perm_stats(M_loss, n_c, n_t, rng),
    }
    per_run = {}
    for i, rid in enumerate(keys):
        same = [j for j in range(n_c) if j != i] if i < n_c else \
            [j for j in range(n_c, n) if j != i]
        other = list(range(n_c, n)) if i < n_c else list(range(n_c))
        per_run[rid] = {
            "group": "clean" if i < n_c else "traced",
            "mean_loss_within": float(np.mean([M_loss[i, j] for j in same]))
            if same else None,
            "mean_loss_cross": float(np.mean([M_loss[i, j] for j in other]))
            if other else None,
            "mean_dN_rel_within": float(np.mean([M_dn[i, j] for j in same]))
            if same else None,
            "mean_dN_rel_cross": float(np.mean([M_dn[i, j] for j in other]))
            if other else None,
        }
    out["per_run_mean_distances"] = per_run
    return out


def main():
    if len(sys.argv) > 1:
        scenes = sys.argv[1:]
    else:
        scenes = ["room", "bicycle"]
    out = {"artifact": "D0_TRACE_SPREAD_DIAGNOSTIC.json",
           "prereg": "D0_TRACE_VALIDITY_SCREEN_PREREG.json",
           "seed": SEED, "n_permutations": N_PERM,
           "note": "evidence only; the gate lives in the analyzers and is the "
                   "pre-registered 2.0x screen",
           "scenes": {}}
    for s in scenes:
        out["scenes"][s] = scene_report(s)
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "D0_TRACE_SPREAD_DIAGNOSTIC.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
