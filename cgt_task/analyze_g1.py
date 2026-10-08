#!/usr/bin/env python
"""G1 preregistered gate analysis + q_weighted + Amdahl minimum q (PER-TILE corrected).

Work unit = unique tile (view,tile_ti,tile_tj). Each tile has up to 3 records
(rank 4/8/16). All work-weighting and speedup math is per UNIQUE TILE (tile_work
counted once), not per record.

Preregistered gate (fixed before full run):
  VALUE    : max_abs <= 0.02  AND  psnr >= 30.0
  JACOBIAN : jac_abs <= max(REL_JAC*resp_range/TILE, ABS_JAC_FLOOR)
q_weighted(R) = work-weighted cumulative passing fraction (passes at any rank <= R).
Amdahl (inference): fwd_ratio = [sum_pass R_best*gamma + sum_fail M] / sum M;
  S = 1/fwd_ratio; R_best = smallest passing rank; M = tile_work (per unique tile).
Gate (fail-closed): PASS iff S(gamma_gate) >= TARGET for the best-R policy.
"""
from __future__ import annotations
import argparse, glob, json, math
from pathlib import Path
import numpy as np

TILE = 16
GATE_PSNR_DB = 30.0
GATE_MAX_ABS = 0.02
REL_JAC = 0.5
ABS_JAC_FLOOR = GATE_MAX_ABS / TILE
RANKS = (4, 8, 16)
TARGET_SPEEDUP = 1.25
GAMMA_GATE = 2.0
GAMMA_SENS = (1.0, 2.0, 4.0)


def jac_threshold(resp_range):
    return np.maximum(REL_JAC * np.asarray(resp_range) / TILE, ABS_JAC_FLOOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = sorted(glob.glob(args.inputs))
    combos = {}
    for fp in files:
        d = json.load(open(fp))
        key = (d["scene"], d["ckpt"])
        combos.setdefault(key, []).extend(d["records"])

    report = {"preregistered_gate": {
        "value": {"psnr_db": GATE_PSNR_DB, "max_abs": GATE_MAX_ABS},
        "jacobian": {"metric": "jac_abs (RMS/elem)", "rel_jac": REL_JAC,
                     "ABS_JAC_FLOOR": ABS_JAC_FLOOR, "tile": TILE},
        "ranks": list(RANKS), "target_speedup": TARGET_SPEEDUP,
        "gamma_gate": GAMMA_GATE, "gamma_sens": list(GAMMA_SENS),
        "note": "per-unique-tile work weighting (tile_work counted once)"},
        "combos": {}}

    agg_q = {R: [] for R in RANKS}
    agg_S = {g: [] for g in GAMMA_SENS}

    for (scene, ckpt), recs in sorted(combos.items()):
        # group by unique tile
        tiles = {}
        for r in recs:
            if "psnr" not in r:
                continue
            k = (r["view"], int(r["tile_ti"]), int(r["tile_tj"]))
            tiles.setdefault(k, {})[int(r["rank"])] = r
        keys = list(tiles.keys())
        M = np.array([tiles[k][max(RANKS)]["tile_work"] for k in keys], dtype=float)  # tile_work (same across ranks)
        # per-rank pass per tile
        passR = {}
        for R in RANKS:
            ok = np.zeros(len(keys), dtype=bool)
            for i, k in enumerate(keys):
                rec = tiles[k].get(R)
                if rec is None:
                    continue
                v = rec["max_abs"] <= GATE_MAX_ABS and rec["psnr"] >= GATE_PSNR_DB
                j = rec["jac_abs"] <= float(jac_threshold(rec["resp_range"]))
                ok[i] = bool(v and j)
            passR[R] = ok
        # cumulative passing at rank R = passes at any rank <= R
        cum = {}
        acc = np.zeros(len(keys), dtype=bool)
        for R in RANKS:
            acc = acc | passR[R]
            cum[R] = acc.copy()
        Itot = float(M.sum())
        q_w = {R: float(M[cum[R]].sum() / Itot) for R in RANKS}
        q_exact = {R: float(M[passR[R]].sum() / Itot) for R in RANKS}
        # best (smallest) passing rank per tile; 0 = never passes
        R_best = np.zeros(len(keys), dtype=int)
        for R in RANKS:
            R_best[(R_best == 0) & passR[R]] = R
        pass_w = R_best > 0
        # actual compressed-forward ratio per gamma (per unique tile)
        Sact = {}
        for g in GAMMA_SENS:
            comp_work = float(np.sum(np.where(pass_w, R_best * g, M)))
            fwd_ratio = comp_work / Itot
            Sact[g] = 1.0 / fwd_ratio if fwd_ratio > 0 else float("inf")
        # value-only diagnostic (per tile, cumulative)
        vacc = np.zeros(len(keys), dtype=bool)
        for R in RANKS:
            vr = np.zeros(len(keys), dtype=bool)
            for i, k in enumerate(keys):
                rec = tiles[k].get(R)
                if rec and rec["max_abs"] <= GATE_MAX_ABS and rec["psnr"] >= GATE_PSNR_DB:
                    vr[i] = True
            vacc = vacc | vr
        q_value_only = float(M[vacc].sum() / Itot)
        # incompressible work fraction
        incomp = float(M[~pass_w].sum() / Itot)
        # verdict: gate on ACTUAL S(gamma_gate) >= TARGET
        verdict = "DROP"; best_S = 0.0
        if Sact[GAMMA_GATE] >= TARGET_SPEEDUP:
            verdict = "PASS"; best_S = Sact[GAMMA_GATE]
        report["combos"][f"{scene}|{ckpt}"] = {
            "n_tiles": len(keys), "I_total": Itot, "incompressible_work_frac": incomp,
            "M_median": float(np.median(M)),
            "q_weighted_cumulative": {str(R): q_w[R] for R in RANKS},
            "q_weighted_exact_rank": {str(R): q_exact[R] for R in RANKS},
            "q_value_only_cumulative": q_value_only,
            "S_inference_actual": {str(g): Sact[g] for g in GAMMA_SENS},
            "verdict": verdict, "best_S_gamma_gate": best_S,
        }
        for R in RANKS:
            agg_q[R].append(q_w[R])
        for g in GAMMA_SENS:
            agg_S[g].append(Sact[g])

    agg = {}
    for R in RANKS:
        agg[f"q_weighted_cumulative_r{R}"] = {"mean": float(np.mean(agg_q[R])),
            "min": float(np.min(agg_q[R])), "max": float(np.max(agg_q[R])),
            "per_combo": {k: report["combos"][k]["q_weighted_cumulative"][str(R)] for k in report["combos"]}}
    for g in GAMMA_SENS:
        agg[f"S_inference_gamma{g}"] = {"mean": float(np.mean(agg_S[g])),
            "min": float(np.min(agg_S[g])), "max": float(np.max(agg_S[g])),
            "per_combo": {k: report["combos"][k]["S_inference_actual"][str(g)] for k in report["combos"]}}
    report["aggregate"] = agg
    combo_verdicts = [c["verdict"] for c in report["combos"].values()]
    report["G1_overall_verdict"] = "PASS" if combo_verdicts and all(v == "PASS" for v in combo_verdicts) else "DROP_REAL_SCENE_COMPRESSIBILITY"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print("=== G1 PREREGISTERED GATE ANALYSIS (per-tile corrected) ===")
    for k, c in report["combos"].items():
        print(f"\n[{k}] n_tiles={c['n_tiles']} I={c['I_total']:.0f} M_med={c['M_median']:.0f} incomp={c['incompressible_work_frac']:.4f} -> {c['verdict']}")
        print(f"  q_weighted(cumulative): " + " ".join(f"r{R}={c['q_weighted_cumulative'][str(R)]:.4f}" for R in RANKS))
        print(f"  q_value_only(cum)={c['q_value_only_cumulative']:.4f}")
        print(f"  S_inference: " + " ".join(f"g{g}={c['S_inference_actual'][str(g)]:.3f}" for g in GAMMA_SENS))
    print(f"\nG1_OVERALL: {report['G1_overall_verdict']}")


if __name__ == "__main__":
    main()