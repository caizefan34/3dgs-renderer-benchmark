#!/usr/bin/env python3
"""N1 Stages 4 + 5 + heterogeneity analysis (CPU, on remote artifacts).

Inputs (produced by the measurement passes):
  N1_PROBE_LIST.json, N1_ACTION_RESULTS_{room,bicycle}.json,
  N1_ACTION_TIMING_{room,bicycle}.json, N1_GRADIENT_CONFLICT_{room,bicycle}.json,
  N1_SKETCHES_{scene}.npz, N1_VIEW_LOSSES_{scene}.json

Outputs:
  N1_ACTION_HETEROGENEITY.json   (H1/H2/H3 + aggregates)
  N1_SKETCH_PREDICTIVITY.json    (stale vs fresh Spearman)
  N1_BUDGET_FRONTIER.json        (MCKP DP + baselines, rho in {.60,.75,.90})

Frozen rules come from N1_SPEC_OPERATIONALIZATION.json (frozen before any
measurement); every threshold below is that frozen value, inlined verbatim.
Delta_L(A1) per anchor := mean of the 3 replica deltas. T(a) := Stage-3
median total wall excluding held-out evaluation and restore. E(A0) := 0.
"""
import json
import math
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n1_core as C

SCENES = ("room", "bicycle")
RHO = (0.60, 0.75, 0.90)
GRID = 0.01  # ms cost grid for the MCKP DP (frozen)
NEG = -1e18


def load(path):
    with open(path) as f:
        return json.load(f)


def build_action_table():
    rows = {s: {} for s in SCENES}
    tiers = {s: {} for s in SCENES}  # (ckpt, aid, partner) -> tier
    for s in SCENES:
        d = load(os.path.join(C.ARTIFACTS, f"N1_ACTION_RESULTS_{s}.json"))
        for r in d["rows"]:
            k = f"{s}|{r['checkpoint']}|{r['anchor_id']}"
            rows[s].setdefault(k, {})[r["action"]] = r
        pl = load(os.path.join(C.ARTIFACTS, "N1_PROBE_LIST.json"))["scenes"][s]
        for a in pl["anchors"]:
            for p in a["pool"]:
                tiers[s][(a["checkpoint"], a["anchor_index"], p["partner"])] = p["tier"]
    timing = {s: {} for s in SCENES}
    for s in SCENES:
        d = load(os.path.join(C.ARTIFACTS, f"N1_ACTION_TIMING_{s}.json"))
        for cell, v in d["cells"].items():
            ckpt, act = cell.split("_", 1)
            timing[s][(int(ckpt), act)] = v["total_wall_ms"]["median"]
    return rows, timing, tiers


def keys_sorted(rows, s):
    return sorted(rows[s].keys(),
                  key=lambda k: (int(k.split("|")[1]), int(k.split("|")[2])))


def d_a1_of(cell):
    return float(np.mean([cell[f"A1_r{i}"]["delta_fused"] for i in (1, 2, 3)]))


def best_a2(cell, T_a2, tiers, ck, aid):
    pool = [r for r in cell.values() if r["action"] == "A2_pool"]
    bb = max(pool, key=lambda r: r["delta_fused"] / T_a2)
    bb = dict(bb)
    bb["tier"] = tiers[(ck, aid, bb["partner"])]
    return bb


# ------------------------------------------------------------------ H1/H2/H3
def heterogeneity(rows, timing, tiers):
    out = {"definition_notes": {
        "Delta_L(A1)": "mean of the 3 replica deltas per anchor",
        "sigma_replica": "1.4826 * median over anchors of MAD of the 3 replica "
                         "deltas (MAD about each anchor's own median)",
        "H1": "skip iff Delta_L(A1) <= 0 OR <= 0.10*sigma_replica; gate >=20%/scene",
        "H2": "Delta_L(A1)>0 AND E(A2-best)>=1.20*E(A1) AND "
              "Delta_L(A2-best)-Delta_L(A1)>sigma_replica; gate >=15%/scene",
        "H3": "argmax E over {A0:0, A1, A2-best}, first-occurrence tie-break; "
              "gates per scene: A0>=15%, A1>=30%, A2-best>=10%",
    }, "scenes": {}}
    for s in SCENES:
        keys = keys_sorted(rows, s)
        per_ckpt = {}
        h1_flags, h1_flags_alt = [], []
        h2_flags, prefer_counts = [], {"A0": 0, "A1": 0, "A2_best": 0}
        e_a1_all, e_a2b_all, e_a2r_all = [], [], []
        d1_all, d2r_all, d2b_all = [], [], []
        tier_hist = {"high": 0, "medium": 0, "low": 0}
        h2_detail = []
        for ck in C.ANCHOR_CKPTS:
            ks = [k for k in keys if int(k.split("|")[1]) == ck]
            T = {a: timing[s][(ck, a)] for a in ("A0", "A1", "A2")}
            d_a1 = [d_a1_of(rows[s][k]) for k in ks]
            mad = [float(np.median(np.abs(
                np.array([rows[s][k][f"A1_r{i}"]["delta_fused"] for i in (1, 2, 3)])
                - np.median([rows[s][k][f"A1_r{i}"]["delta_fused"]
                             for i in (1, 2, 3)])))) for k in ks]
            sigma_replica = 1.4826 * float(np.median(mad))
            sigma_onestep = float(np.std(d_a1, ddof=1))
            d_a2r = [rows[s][k]["A2_random"]["delta_fused"] for k in ks]
            b_best, e_a2b, d_a2b = [], {}, []
            for k in ks:
                ck_i, aid_i = int(k.split("|")[1]), int(k.split("|")[2])
                bb = best_a2(rows[s][k], T["A2"], tiers[s], ck_i, aid_i)
                b_best.append(bb)
                e_a2b[k] = bb["delta_fused"] / T["A2"]
                d_a2b.append(bb["delta_fused"])
                tier_hist[bb["tier"]] += 1
            for i, k in enumerate(ks):
                e1 = d_a1[i] / T["A1"]
                e_a1_all.append(e1)
                e_a2b_all.append(e_a2b[k])
                e_a2r_all.append(d_a2r[i] / T["A2"])
                d1_all.append(d_a1[i])
                d2r_all.append(d_a2r[i])
                d2b_all.append(d_a2b[i])
                h1_flags.append(bool(d_a1[i] <= 0.0 or d_a1[i] <= 0.10 * sigma_replica))
                h1_flags_alt.append(bool(d_a1[i] <= 0.10 * sigma_onestep))
                h2 = bool(d_a1[i] > 0.0 and e_a2b[k] >= 1.20 * e1
                          and d_a2b[i] - d_a1[i] > sigma_replica)
                h2_flags.append(h2)
                if h2:
                    h2_detail.append({"anchor": k, "E_A1": e1, "E_A2best": e_a2b[k],
                                      "dA1": d_a1[i], "dA2best": d_a2b[i]})
                cands = [("A0", 0.0), ("A1", e1), ("A2_best", e_a2b[k])]
                prefer_counts[max(cands, key=lambda c: c[1])[0]] += 1
            per_ckpt[str(ck)] = {
                "T_ms": T, "sigma_replica": sigma_replica,
                "sigma_onestep": sigma_onestep,
                "median_dA1": float(np.median(d_a1)),
                "trimmed_mean_dA1": float(stats.trim_mean(d_a1, 0.10)),
                "sign_rate_dA1": float(np.mean(np.array(d_a1) > 0)),
                "median_dA2_random": float(np.median(d_a2r)),
                "trimmed_mean_dA2_random": float(stats.trim_mean(d_a2r, 0.10)),
                "sign_rate_dA2_random": float(np.mean(np.array(d_a2r) > 0)),
                "median_dA2_best": float(np.median(d_a2b)),
                "sign_rate_dA2_best": float(np.mean(np.array(d_a2b) > 0)),
            }
        n = len(keys)
        out["scenes"][s] = {
            "n_anchors": n,
            "per_checkpoint": per_ckpt,
            "H1_skip_share": float(np.mean(h1_flags)),
            "H1_gate_20pct_pass": bool(np.mean(h1_flags) >= 0.20),
            "H1_secondary_share_sigma_onestep": float(np.mean(h1_flags_alt)),
            "H2_share": float(np.mean(h2_flags)),
            "H2_gate_15pct_pass": bool(np.mean(h2_flags) >= 0.15),
            "H2_examples": h2_detail[:10],
            "H3_prefer_share": {k: v / n for k, v in prefer_counts.items()},
            "H3_gates": {"A0_ge_15pct": bool(prefer_counts["A0"] / n >= 0.15),
                         "A1_ge_30pct": bool(prefer_counts["A1"] / n >= 0.30),
                         "A2best_ge_10pct": bool(prefer_counts["A2_best"] / n >= 0.10)},
            "A2_best_tier_histogram": tier_hist,
            "pooled": {
                "median_E_A1": float(np.median(e_a1_all)),
                "median_E_A2_random": float(np.median(e_a2r_all)),
                "median_E_A2_best": float(np.median(e_a2b_all)),
                "median_dA1": float(np.median(d1_all)),
                "median_dA2_random": float(np.median(d2r_all)),
                "median_dA2_best": float(np.median(d2b_all)),
                "trimmed_mean_dA1": float(stats.trim_mean(d1_all, 0.10)),
                "trimmed_mean_dA2_random": float(stats.trim_mean(d2r_all, 0.10)),
                "trimmed_mean_dA2_best": float(stats.trim_mean(d2b_all, 0.10)),
                "sign_rate_dA1": float(np.mean(np.array(d1_all) > 0)),
                "sign_rate_dA2_random": float(np.mean(np.array(d2r_all) > 0)),
                "sign_rate_dA2_best": float(np.mean(np.array(d2b_all) > 0)),
            },
        }
    return out


# ------------------------------------------------------------------ sketches
def spear(x, y):
    r = stats.spearmanr(x, y)
    return float(r.statistic if hasattr(r, "statistic") else r.correlation)


def sketch_predictivity():
    out = {"definition": load(os.path.join(C.N1_ROOT, "N1_SPEC_OPERATIONALIZATION.json"))["sketch"],
           "scenes": {}}
    for s in SCENES:
        skz = np.load(os.path.join(C.ARTIFACTS, f"N1_SKETCHES_{s}.npz"))
        conf = load(os.path.join(C.ARTIFACTS, f"N1_GRADIENT_CONFLICT_{s}.json"))["rows"]

        def pred(ckpt, v1, v2):
            return 1.0 - float(skz[f"{ckpt}_{v1}"] @ skz[f"{ckpt}_{v2}"])

        rows = [{"ckpt": r["checkpoint"], "v1": r["v1"], "v2": r["partner"],
                 "tier": r["tier"], "actual": 1.0 - r["global_cosine"],
                 "pred_stale": pred(C.SKETCH_SRC[r["checkpoint"]], r["v1"], r["partner"]),
                 "pred_fresh": pred(r["checkpoint"], r["v1"], r["partner"])}
                for r in conf]
        A = np.array([r["actual"] for r in rows])
        Pst = np.array([r["pred_stale"] for r in rows])
        Pf = np.array([r["pred_fresh"] for r in rows])
        sc = {"n_pairs": len(rows),
              "spearman_stale_vs_true": spear(Pst, A),
              "spearman_fresh_vs_true": spear(Pf, A),
              "pearson_stale_vs_true": float(np.corrcoef(Pst, A)[0, 1]),
              "pearson_fresh_vs_true": float(np.corrcoef(Pf, A)[0, 1]),
              "per_checkpoint": {}, "per_tier": {}}
        for ck in C.ANCHOR_CKPTS:
            idx = [i for i, r in enumerate(rows) if r["ckpt"] == ck]
            sc["per_checkpoint"][str(ck)] = {
                "spearman_stale": spear(Pst[idx], A[idx]),
                "spearman_fresh": spear(Pf[idx], A[idx])}
        for tier in ("high", "medium", "low"):
            idx = [i for i, r in enumerate(rows) if r["tier"] == tier]
            sc["per_tier"][tier] = {
                "spearman_stale": spear(Pst[idx], A[idx]),
                "spearman_fresh": spear(Pf[idx], A[idx])}
        out["scenes"][s] = sc
    return out


# ------------------------------------------------------------------ frontier
def mckp_grid(anchors, budget_ms):
    """Exact MCKP DP on the frozen 0.01ms grid. anchors: [action->(delta,cost)]."""
    W = int(round(budget_ms / GRID))
    dp = np.full(W + 1, NEG)
    dp[0] = 0.0
    choices = []
    for cell in anchors:
        nd = np.full(W + 1, NEG)
        ch = np.full(W + 1, -1, dtype=np.int8)
        for ai, (a, (d, c)) in enumerate(cell.items()):
            cw = int(round(c / GRID))
            if cw > W:
                continue
            shifted = np.full(W + 1, NEG)
            shifted[cw:] = dp[:W + 1 - cw] + d
            better = shifted > nd
            nd[better] = shifted[better]
            ch[better] = ai
        dp = nd
        choices.append((list(cell.keys()), ch,
                        {a: int(round(c / GRID)) for a, (d, c) in cell.items()}))
    best_w = int(np.argmax(dp))
    total = float(dp[best_w])
    chosen = []
    w = best_w
    for actions, ch, cwmap in reversed(choices):
        ai = int(ch[w])
        if ai < 0:
            return 0.0, []
        a = actions[ai]
        chosen.append(a)
        w -= cwmap[a]
    chosen.reverse()
    return (total, chosen) if w == 0 else (0.0, [])


def fixed01_best(V1, C1, budget):
    best = 0.0
    for f in range(0, 1001):
        fr = f / 1000.0
        if fr * C1 <= budget:
            best = max(best, fr * V1)
    return best


def fixed012_best(V1, C1, V2, C2, budget):
    """Best fixed random 0/1/A2 mixture: 2-simplex vertex/edge enumeration."""
    cands = [(0.0, 0.0)]
    f1 = min(1.0, budget / C1) if C1 > 0 else 0.0
    cands.append((f1, 0.0))
    f2 = min(1.0, budget / C2) if C2 > 0 else 0.0
    cands.append((0.0, f2))
    if abs(C1 - C2) > 1e-12:  # budget line x edge f1+f2=1
        x = (budget - C2) / (C1 - C2)
        if 0.0 <= x <= 1.0:
            cands.append((x, 1.0 - x))
    best = 0.0
    for f1, f2 in cands:
        if f1 + f2 <= 1.0 + 1e-12 and f1 * C1 + f2 * C2 <= budget + 1e-9:
            best = max(best, f1 * V1 + f2 * V2)
    return best


def frontier(rows, timing, tiers):
    out = {"rho": list(RHO), "grid_ms": GRID,
           "labels": {"primary": "OFFLINE_ASSIGNMENT_UPPER_BOUND",
                      "oracle_partner": "OFFLINE_ASSIGNMENT_ORACLE_PARTNER_UPPER_BOUND"},
           "strong_gate_rule": "both scenes at rho=0.75 (primary set): "
                               "progress >= 0.95*all-A1 AND >= 1.10*max(fixed01, "
                               "oracle01-control) at the same budget",
           "scenes": {}}
    for s in SCENES:
        keys = keys_sorted(rows, s)
        cells = {}
        for rho in RHO:
            for setname, a2name in (("primary", "A2_random"),
                                    ("oracle_partner", "A2_best")):
                anchors, V1l, C1l, V2l, C2l = [], [], [], [], []
                for k in keys:
                    ck = int(k.split("|")[1])
                    T = {a: timing[s][(ck, a)] for a in ("A0", "A1", "A2")}
                    cell = {"A0": (0.0, T["A0"])}
                    d1 = d_a1_of(rows[s][k])
                    cell["A1"] = (d1, T["A1"])
                    if a2name == "A2_random":
                        cell[a2name] = (rows[s][k]["A2_random"]["delta_fused"], T["A2"])
                    else:
                        ck_i, aid_i = int(k.split("|")[1]), int(k.split("|")[2])
                        bb = best_a2(rows[s][k], T["A2"], tiers[s], ck_i, aid_i)
                        cell[a2name] = (bb["delta_fused"], T["A2"])
                    anchors.append(cell)
                    V1l.append(d1); C1l.append(T["A1"])
                    V2l.append(cell[a2name][0]); C2l.append(T["A2"])
                V1, C1 = sum(V1l), sum(C1l)
                V2, C2 = sum(V2l), sum(C2l)
                budget = rho * C1
                total, chosen = mckp_grid(anchors, budget)
                base_fixed01 = fixed01_best(V1, C1, budget)
                base_fixed012 = fixed012_best(V1, C1, V2, C2, budget)
                ctrl_total, _ = mckp_grid(
                    [{"A0": c["A0"], "A1": c["A1"]} for c in anchors], budget)
                cellout = {
                    "budget_ms": budget,
                    "assignment_progress": total,
                    "action_shares": {a: (chosen.count(a) / len(chosen)
                                          if chosen else None)
                                      for a in ("A0", "A1", a2name)},
                    "all_A1_progress": V1,
                    "baseline_fixed01_progress": base_fixed01,
                    "baseline_fixed012_progress": base_fixed012,
                    "baseline_oracle01_control_progress": ctrl_total,
                    "assignment_over_allA1": (total / V1) if V1 else None,
                    "assignment_over_max_control": (
                        total / max(base_fixed01, ctrl_total)
                        if max(base_fixed01, ctrl_total) > 0 else None),
                }
                if rho == 0.75 and setname == "primary":
                    cellout["strong_gate_pass"] = bool(
                        total >= 0.95 * V1
                        and total >= 1.10 * max(base_fixed01, ctrl_total))
                cells[f"rho{rho}_{setname}"] = cellout
        out["scenes"][s] = cells
    return out


def main():
    rows, timing, tiers = build_action_table()
    het = heterogeneity(rows, timing, tiers)
    C.json_dump(het, os.path.join(C.ARTIFACTS, "N1_ACTION_HETEROGENEITY.json"))
    skp = sketch_predictivity()
    C.json_dump(skp, os.path.join(C.ARTIFACTS, "N1_SKETCH_PREDICTIVITY.json"))
    fr = frontier(rows, timing, tiers)
    C.json_dump(fr, os.path.join(C.ARTIFACTS, "N1_BUDGET_FRONTIER.json"))
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
