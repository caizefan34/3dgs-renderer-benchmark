"""P15-P0 Step 4a - offline EdgeBudget Pareto-oracle analysis.

Reads ONLY the six export summaries and their NPZs (no GPU, no training).
Writes artifacts/P15_P0_PARETO.json and artifacts/P15_P0_VERDICT.json.

FROZEN definitions (preregistered before any P15-P0 export result):

  S_i          progress proxy: P14 selector energy (float64 [N]).
  c_i          cost: exact Gaussian-tile intersection count (16K initial
               state, 8 cluster cameras).
  E(A), C(A)   energy share  sum(S_i in A)/sum(S)  and cost share
               sum(c_i in A)/sum(C) of subset A.

  Reference family R1 (P14 selector sets): energy-top-q, q in Q5
  {0.10,0.20,0.25,0.30,0.50}, ordering identical to the P14 selector.
  Reference family R2 (informational): cheapest-q sets by c_i ascending.

  Oracle frontier (cost-aware, frozen):
    ratio order r_i = S_i/c_i for c_i>0; c_i==0 and S_i>0 -> +inf (first);
    S_i<=0 items excluded from accumulation.
    E_lp(b)  fractional-knapsack upper bound at cost budget b (split the
             marginal ratio item) - the oracle upper bound.
    E_int(b) integer greedy prefix at cost budget b (realizable).
    Budget grid B_GRID = 0.05..1.00 step 0.05; headline budgets 0.30
    (P14 O2 cost bar) and 0.50 (P14 O2 family hard-drop line).

  Bars (per cell, from P14 measured H=32 Stage A arms):
    q_viab(cell)     = smallest q in Q5 with measured R_train >= 0.85
    E_bar_viab(cell) = E_ref_p14(q_viab)   [None if no q reaches 0.85]
    E_bar_prog(cell) = E_ref_p14(0.20)     [P14's near-full-progress set;
                       measured median R_train = 0.98]
    (informational 0.80-bar counts also computed, not used for verdict)

  Counts:  n_strong = #cells with E_lp(0.30) >= E_bar_prog
           n_viab   = #cells with E_lp(0.30) >= E_bar_viab
           n_relax  = #cells with E_lp(0.50) >= E_bar_viab

  Verdict (frozen, evaluated in this order):
    n_strong >= 5   -> P0_STRONG_CANDIDATE   (STOP, return for review)
    n_relax  <= 1   -> P0_HARD_DROP          (STOP the EdgeBudget path)
    else            -> P0_HOLD_WEAK_PARETO   (STOP, no training)

  Any missing/invalid export (count != 6, gate_pass false, npz sha
  mismatch, summary/npz q-ref disagreement) -> P0_EXPORT_INVALID.

  Epistemic note (frozen): E is a NECESSARY-condition proxy for progress
  retention; P15-P0 runs no optimization arms and cannot certify real
  progress. P0_STRONG_CANDIDATE means "not falsified at the oracle
  level", never "proven". No novelty/speedup claim is made anywhere.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

Q5 = (0.10, 0.20, 0.25, 0.30, 0.50)
B_GRID = [round(0.05 * i, 2) for i in range(1, 21)]
B_COST = 0.30          # P14 O2 gate level
B_RELAX = 0.50         # P14 O2 family hard-drop line
R_TRAIN_BAR = 0.85     # P14 O1 candidate/viability bar
R_TRAIN_BAR_INFO = 0.80
N_EXPECTED_CELLS = 6
SCENES = ("room", "bicycle")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jwrite(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[p15] wrote {path}", flush=True)


def k_of_q(q, N):
    return int(np.floor(q * N + 0.5))


def ratio_order(S, c):
    """Frozen cost-aware ordering. Returns index array of S>0 items and
    the aligned S/c arrays."""
    pos = np.nonzero(S > 0)[0]
    Sp = S[pos]
    cp = c[pos]
    r = np.where(cp > 0, Sp / np.maximum(cp, 1).astype(np.float64),
                 np.inf)
    order = np.lexsort((pos, -Sp, -r))
    return pos[order], Sp[order], cp[order]


def staircase(Sp, cp, Stot, Ctot):
    cumE = np.cumsum(Sp) / Stot
    cumC = np.cumsum(cp.astype(np.float64)) / float(Ctot)
    return cumE, cumC


def e_lp(cumE, cumC, b):
    """Fractional-knapsack upper bound at budget b."""
    n = len(cumC)
    if n == 0:
        return 0.0
    j = int(np.searchsorted(cumC, b, side="right"))
    if j >= n:
        return float(cumE[-1])
    e0 = float(cumE[j - 1]) if j > 0 else 0.0
    c0 = float(cumC[j - 1]) if j > 0 else 0.0
    dc = float(cumC[j]) - c0
    if dc <= 0.0:
        return float(cumE[j])
    return e0 + (b - c0) / dc * (float(cumE[j]) - e0)


def e_int(cumE, cumC, b):
    j = int(np.searchsorted(cumC, b, side="right"))
    return float(cumE[j - 1]) if j > 0 else 0.0


def c_star(cumE, cumC, bar):
    """Minimal cost budget at which E_lp reaches 'bar'. Returns
    (c_star_lp, c_star_int) or (None, None) if unreachable."""
    if bar is None or len(cumC) == 0:
        return None, None
    m = int(np.searchsorted(cumE, bar, side="left"))
    if m >= len(cumE):
        return None, None
    c_int = float(cumC[m])
    e0 = float(cumE[m - 1]) if m > 0 else 0.0
    c0 = float(cumC[m - 1]) if m > 0 else 0.0
    de = float(cumE[m]) - e0
    dc = c_int - c0
    c_lp = c_int if de <= 0 else c0 + (bar - e0) / de * dc
    return c_lp, c_int


def load_exports(exports_dir, summaries_dir):
    """Returns {key: {"summary": obj, "summary_path":..., "npz": path}}
    and a list of findings for anything missing/invalid."""
    findings = []
    out = {}
    pat = re.compile(r"^p15_export_(room|bicycle)_cell([0-2])\.json$")
    for p in sorted(glob.glob(os.path.join(summaries_dir,
                                           "p15_export_*.json"))):
        base = os.path.basename(p)
        m = pat.match(base)
        if not m:
            findings.append(f"unexpected summary filename: {base}")
            continue
        scene, ci = m.group(1), int(m.group(2))
        s = json.load(open(p))
        if s.get("scene") != scene or int(s.get("cell", -1)) != ci:
            findings.append(f"{base}: scene/cell mismatch inside JSON")
        if s.get("gate", {}).get("gate_pass") is not True:
            findings.append(f"{base}: gate_pass != true")
        v = s.get("verdict")
        if v != "P0_EXPORT_VALID":
            findings.append(f"{base}: verdict={v}")
        npz_path = os.path.join(exports_dir,
                                f"p15_export_{scene}_cell{ci}.npz")
        if not os.path.exists(npz_path):
            findings.append(f"{base}: npz missing {npz_path}")
        else:
            sha = sha256_file(npz_path)
            rec = s.get("npz", {})
            if sha != rec.get("sha256"):
                findings.append(f"{base}: npz sha mismatch "
                                f"{sha[:16]} != {str(rec.get('sha256'))[:16]}")
        out[f"{scene}_c{ci}"] = {"summary": s, "summary_path": p,
                                 "npz": npz_path}
    for sc in SCENES:
        for ci in range(3):
            k = f"{sc}_c{ci}"
            if k not in out:
                findings.append(f"missing export: {k}")
    if len(out) != N_EXPECTED_CELLS:
        findings.append(f"expected {N_EXPECTED_CELLS} exports, "
                        f"found {len(out)}")
    return out, findings


def analyze_cell(key, rec):
    s = rec["summary"]
    d = np.load(rec["npz"])
    S = d["S"].astype(np.float64)
    c = d["c"].astype(np.int64)
    N = int(d["N"])
    assert N == len(S) == len(c), f"{key}: N/S/c length mismatch"
    Stot = float(S.sum())
    Ctot = int(c.sum())
    assert Ctot > 0, f"{key}: zero total cost"

    # ---- R1: P14 energy-top-q sets in P15 cost space ------------------
    order_S = np.argsort(-S, kind="stable")
    q_tab = {}
    for q in Q5:
        K = k_of_q(q, N)
        sel = order_S[:K]
        q_tab[str(q)] = {
            "K": int(K),
            "C_ref": float(c[sel].sum()) / float(Ctot),
            "E_ref": float(S[sel].sum()) / Stot,
            "R_train_p14": s["p14_anchor"][str(q)]["R_train"],
            "E_ref_p14": s["p14_anchor"][str(q)]["E_ref_p14"],
        }
        # cross-check against the export's own q_ref record
        assert abs(q_tab[str(q)]["C_ref"]
                   - s["q_ref_points"][str(q)]["C_ref"]) < 1e-9, key
        assert abs(q_tab[str(q)]["E_ref"]
                   - s["q_ref_points"][str(q)]["E_ref"]) < 1e-9, key

    # ---- R2: cheapest-q sets (informational) --------------------------
    order_c = np.argsort(c, kind="stable")
    cheap = {}
    for q in Q5:
        K = k_of_q(q, N)
        sel = order_c[:K]
        cheap[str(q)] = {"E_of_cheapest": float(S[sel].sum()) / Stot,
                         "C_of_cheapest": float(c[sel].sum()) / float(Ctot)}

    # ---- oracle frontier ----------------------------------------------
    idx, Sp, cp = ratio_order(S, c)
    cumE, cumC = staircase(Sp, cp, Stot, Ctot)
    frontier = {str(b): {"E_lp": e_lp(cumE, cumC, b),
                         "E_int": e_int(cumE, cumC, b)}
                for b in B_GRID}
    E_lp_cost = e_lp(cumE, cumC, B_COST)
    E_lp_relax = e_lp(cumE, cumC, B_RELAX)

    # ---- bars from P14 measured anchors --------------------------------
    q_viab = None
    for q in Q5:
        rt = q_tab[str(q)]["R_train_p14"]
        if rt is not None and rt >= R_TRAIN_BAR:
            q_viab = q
            break
    E_bar_viab = (q_tab[str(q_viab)]["E_ref_p14"]
                  if q_viab is not None else None)
    E_bar_prog = q_tab["0.2"]["E_ref_p14"]
    q_viab80 = None
    for q in Q5:
        rt = q_tab[str(q)]["R_train_p14"]
        if rt is not None and rt >= R_TRAIN_BAR_INFO:
            q_viab80 = q
            break
    E_bar_viab80 = (q_tab[str(q_viab80)]["E_ref_p14"]
                    if q_viab80 is not None else None)

    c_star_viab_lp, c_star_viab_int = c_star(cumE, cumC, E_bar_viab)
    c_star_prog_lp, c_star_prog_int = c_star(cumE, cumC, E_bar_prog)
    c_star_viab80_lp, c_star_viab80_int = c_star(cumE, cumC, E_bar_viab80)

    if q_viab is not None and c_star_viab_int:
        gain_viab = q_tab[str(q_viab)]["C_ref"] / c_star_viab_int
    else:
        gain_viab = None
    if c_star_prog_int:
        gain_prog = q_tab["0.2"]["C_ref"] / c_star_prog_int
    else:
        gain_prog = None

    strong = (E_bar_prog is not None and E_lp_cost >= E_bar_prog)
    viab = (E_bar_viab is not None and E_lp_cost >= E_bar_viab)
    relax = (E_bar_viab is not None and E_lp_relax >= E_bar_viab)
    viab80 = (E_bar_viab80 is not None and E_lp_cost >= E_bar_viab80)

    # ---- cost concentration (informational) ---------------------------
    c_sorted = np.sort(c)[::-1]
    top1 = max(1, int(0.01 * N))
    top1_share = float(c_sorted[:top1].sum()) / float(Ctot)

    return {
        "scene": s["scene"], "cell": s["cell"],
        "seed_camera": s["seed_camera"],
        "cluster_cameras": s["cluster_cameras"],
        "N": N, "n_isects_total": Ctot,
        "gate_min_jaccard": min(v["jaccard"]
                                for v in s["gate"]["per_q"].values()),
        "gate_max_abs_dE": max(abs(v["dE"])
                               for v in s["gate"]["per_q"].values()),
        "S_total": Stot,
        "q_table": q_tab,
        "cheapest_q": cheap,
        "bars": {"q_viab": q_viab, "E_bar_viab": E_bar_viab,
                 "E_bar_prog": E_bar_prog,
                 "q_viab80": q_viab80, "E_bar_viab80": E_bar_viab80},
        "frontier": frontier,
        "headline": {
            "E_lp_at_0.30": E_lp_cost, "E_lp_at_0.50": E_lp_relax,
            "c_star_viab_lp": c_star_viab_lp,
            "c_star_viab_int": c_star_viab_int,
            "c_star_prog_lp": c_star_prog_lp,
            "c_star_prog_int": c_star_prog_int,
            "c_star_viab80_lp": c_star_viab80_lp,
            "C_ref_q20": q_tab["0.2"]["C_ref"],
            "C_ref_q10": q_tab["0.1"]["C_ref"],
            "gain_prog_factor": gain_prog,
            "gain_viab_factor": gain_viab,
        },
        "flags": {"strong": bool(strong), "viab": bool(viab),
                  "relax": bool(relax), "viab80": bool(viab80)},
        "info": {"top1pct_cost_share": top1_share,
                 "n_zero_cost": int((c == 0).sum()),
                 "n_zero_energy": int((S <= 0).sum())},
    }


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--summaries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verdict-out", required=True)
    args = ap.parse_args()

    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    frozen = {
        "q_grid": list(Q5), "budget_grid": B_GRID,
        "budget_cost_bar": B_COST, "budget_relax_line": B_RELAX,
        "r_train_bar": R_TRAIN_BAR, "r_train_bar_info": R_TRAIN_BAR_INFO,
        "progress_proxy": "P14 selector energy S_i (necessary-condition "
                          "proxy; no arms are run by P15-P0)",
        "cost": "exact Gaussian-tile intersection count (16K state, 8 "
                "cluster cameras)",
        "oracle": ("ratio-order fractional knapsack E_lp (upper bound) and "
                   "integer greedy E_int; budgets are cost shares"),
        "bars": ("E_bar_viab = E_ref_p14(q_viab), q_viab = smallest q in "
                 "grid with measured P14 R_train >= 0.85; E_bar_prog = "
                 "E_ref_p14(0.20)"),
        "verdict_rule": ("n_strong>=5 -> P0_STRONG_CANDIDATE; elif "
                         "n_relax<=1 -> P0_HARD_DROP; else "
                         "P0_HOLD_WEAK_PARETO"),
        "no_new_q_alpha_budget": True,
        "no_speedup_or_novelty_claim": True,
    }

    exports, findings = load_exports(args.exports, args.summaries)
    pareto = {"phase": "P15-P0", "step": "analysis", "created_utc": created,
              "frozen_rules": frozen,
              "inputs": {
                  "summaries": {os.path.basename(v["summary_path"]):
                                sha256_file(v["summary_path"])
                                for v in exports.values()},
                  "exports": {os.path.basename(v["npz"]):
                              sha256_file(v["npz"])
                              for v in exports.values()
                              if os.path.exists(v["npz"])},
              },
              "per_cell": {}, "aggregate": {},
              "verdict": None, "findings": findings}

    verdict = {"phase": "P15-P0", "created_utc": created,
               "identity": "see artifacts/P15_SOURCE_IDENTITY.json",
               "gate_chain": {}, "verdict": None, "stop_reason": None,
               "reminders": [
                   "no git commit", "no training",
                   "no new q/alpha/budget after results",
                   "no speedup claim", "no novelty claim",
               ]}

    if findings:
        pareto["verdict"] = "P0_EXPORT_INVALID"
        verdict["verdict"] = "P0_EXPORT_INVALID"
        verdict["stop_reason"] = "export validation failed"
        verdict["findings"] = findings
        verdict["gate_chain"]["exports"] = {"all_valid": False,
                                            "findings": findings}
        verdict["next"] = ("STOP: fix/redo exports; the P14 reproduction "
                           "gate is the validity precondition for any "
                           "Pareto statement")
        jwrite(args.out, pareto)
        jwrite(args.verdict_out, verdict)
        print("[p15] P0_EXPORT_INVALID: " + "; ".join(findings[:6]), flush=True)
        sys.exit(2)

    for key in sorted(exports):
        cell = analyze_cell(key, exports[key])
        pareto["per_cell"][key] = cell
        print(f"[p15] {key}: E_lp(0.30)={cell['headline']['E_lp_at_0.30']:.4f} "
              f"E_lp(0.50)={cell['headline']['E_lp_at_0.50']:.4f} "
              f"C_ref(q20)={cell['headline']['C_ref_q20']:.4f} "
              f"C*(viab)={cell['headline']['c_star_viab_int']} "
              f"C*(prog)={cell['headline']['c_star_prog_int']} "
              f"flags={cell['flags']}", flush=True)

    pc = pareto["per_cell"]
    cells = sorted(pc)
    n_strong = sum(1 for k in cells if pc[k]["flags"]["strong"])
    n_viab = sum(1 for k in cells if pc[k]["flags"]["viab"])
    n_relax = sum(1 for k in cells if pc[k]["flags"]["relax"])
    n_viab80 = sum(1 for k in cells if pc[k]["flags"]["viab80"])

    agg = {
        "n_cells": len(cells),
        "n_strong": int(n_strong), "n_viab": int(n_viab),
        "n_relax": int(n_relax), "n_viab80_info": int(n_viab80),
        "median": {
            "E_lp_at_0.30": med([pc[k]["headline"]["E_lp_at_0.30"]
                                 for k in cells]),
            "E_lp_at_0.50": med([pc[k]["headline"]["E_lp_at_0.50"]
                                 for k in cells]),
            "C_ref_q10": med([pc[k]["headline"]["C_ref_q10"]
                              for k in cells]),
            "C_ref_q20": med([pc[k]["headline"]["C_ref_q20"]
                              for k in cells]),
            "C_ref_q25": med([pc[k]["q_table"]["0.25"]["C_ref"]
                              for k in cells]),
            "C_star_viab_int": med([pc[k]["headline"]["c_star_viab_int"]
                                    for k in cells]),
            "C_star_prog_int": med([pc[k]["headline"]["c_star_prog_int"]
                                    for k in cells]),
            "E_bar_viab": med([pc[k]["bars"]["E_bar_viab"]
                               for k in cells]),
            "E_bar_prog": med([pc[k]["bars"]["E_bar_prog"]
                               for k in cells]),
        },
        "per_scene": {},
    }
    for sc in SCENES:
        ks = [k for k in cells if k.startswith(sc)]
        agg["per_scene"][sc] = {
            "cells": ks,
            "E_lp_at_0.30": [pc[k]["headline"]["E_lp_at_0.30"] for k in ks],
            "E_lp_at_0.50": [pc[k]["headline"]["E_lp_at_0.50"] for k in ks],
            "C_ref_q20": [pc[k]["headline"]["C_ref_q20"] for k in ks],
            "C_star_prog_int": [pc[k]["headline"]["c_star_prog_int"]
                                for k in ks],
            "flags": {k: pc[k]["flags"] for k in ks},
        }
    pareto["aggregate"] = agg

    if n_strong >= 5:
        final = "P0_STRONG_CANDIDATE"
        reason = (f"E_lp(0.30) >= E_bar_prog in {n_strong}/6 cells: a "
                  "cost-aware subset reaching the q=0.20 energy level at "
                  "<=30% of intersections is not falsified at the oracle "
                  "level")
        nxt = ("STOP and return artifacts. Do not start P15-O1 until "
               "reviewed.")
    elif n_relax <= 1:
        final = "P0_HARD_DROP"
        reason = (f"E_lp(0.50) >= E_bar_viab in only {n_relax}/6 cells: "
                  "even at 50% of intersections the oracle cannot reach "
                  "the minimum P14-viability energy level")
        nxt = ("STOP P15 primitive EdgeBudget path. Do not tune "
               "alpha/budgets.")
    else:
        final = "P0_HOLD_WEAK_PARETO"
        reason = (f"frontier exists but is weak/partial (n_strong={n_strong}"
                  f"/6, n_viab={n_viab}/6, n_relax={n_relax}/6)")
        nxt = "STOP. No training."

    pareto["verdict"] = final
    pareto["verdict_statement"] = {"verdict": final, "stop_reason": reason,
                                   "next": nxt}
    verdict["gate_chain"]["exports"] = {
        "all_valid": True, "n_cells": len(cells),
        "min_jaccard": min(pc[k]["gate_min_jaccard"] for k in cells),
        "max_abs_dE": max(pc[k]["gate_max_abs_dE"] for k in cells),
    }
    verdict["gate_chain"]["pareto"] = {
        "n_strong": int(n_strong), "n_viab": int(n_viab),
        "n_relax": int(n_relax), "n_viab80_info": int(n_viab80),
        "median_E_lp_at_0.30": agg["median"]["E_lp_at_0.30"],
        "median_E_lp_at_0.50": agg["median"]["E_lp_at_0.50"],
        "median_C_ref_q20": agg["median"]["C_ref_q20"],
        "median_C_star_prog_int": agg["median"]["C_star_prog_int"],
    }
    verdict["verdict"] = final
    verdict["stop_reason"] = reason
    verdict["next"] = nxt
    jwrite(args.out, pareto)
    jwrite(args.verdict_out, verdict)
    print(f"[p15] VERDICT = {final} :: {reason}", flush=True)


if __name__ == "__main__":
    main()
