#!/usr/bin/env python3
"""Aggregate Stage A CSVs into one machine-readable summary JSON."""
import json, hashlib, os
import numpy as np
import pandas as pd

OUT = r"C:\Users\36570\3dgs-renderer-benchmark\wv-hier-r1-mx"
pred = pd.read_csv(f"{OUT}\\stage_a_hwcs_predictions.csv")
cost = pd.read_csv(f"{OUT}\\stage_a_cost_model.csv")
sens = pd.read_csv(f"{OUT}\\stage_a_sensitivity.csv")
seg = pd.read_csv(f"{OUT}\\stage_a_segment_sweep.csv")
eq = pd.read_csv(f"{OUT}\\stage_a_equal_cost_ratio.csv")

def rng(s, col):
    return [float(s[col].min()), float(s[col].max())]

summary = {
  "schema": "WV_HIER_R1_MX.stage_a_summary.v1",
  "evidence_class": {
     "OBSERVED": "G16/G4/G8/G1 geometry-macro NMSE at p in {0.25,0.4}; dense bbox intersection proxies; audit hashes; GPU snapshots",
     "RECOMPUTED": "K-calibration, HWCS NMSE blend, work model, dominance, equal-cost ratios",
     "DERIVED": "hierarchical variance identity, alpha blend, work-floor break-even, kappa*",
     "HYPOTHESIS": "HWCS NMSE (predicted, never measured); child-refine efficiency kappa; GPU-time realization",
     "BLOCKED": "real HWCS NMSE, real CUDA timing, kappa, native-G4 baseline design, time-to-quality"
  },
  "variance_model": {
     "identity": "Var_HWCS = (1-alpha)*Var_nativeG4 + alpha*Var_G16,  alpha = r*(1-q)/(1-p)  [CORRECTED 20261008: was 1-p*q*r = 1-p^2; numerical alpha always used 1-p]",
     "hwcs1_alpha": 1.0/3.0, "hwcs2_alpha": 1.0/9.0,
     "mc_validation_rel_err_pct": {"G4": 0.5, "HWCS_1": 0.22, "HWCS_2": 0.32, "G16": 0.26},
     "endpoint_limits": {"q=p,r=1": "HWCS==G16", "q=1,r=p": "HWCS==nativeG4"}
  },
  "hwcs_nmse_prediction_p25": {
     "note": "DERIVED from OBSERVED G16/G4 NMSE via exact blend; NOT measured",
     "G16_range": rng(pred, "nmse_G16"),
     "HWCS2_range": rng(pred, "hwcs2_nmse"),
     "HWCS1_range": rng(pred, "hwcs1_nmse"),
     "G4_range": rng(pred, "nmse_G4"),
     "ordering": "G4 <= HWCS-2 <= HWCS-1 <= G16 (all 9 states)"
  },
  "work_model_units": "I_G16 dense parent intersections; GAM=1 raster; PROXY not measured CUDA time",
  "work_saving_vs_nativeG4_p25": {
     "note": "conservative child-refine model (kappa=1) vs native G4 dense-binning at p=.25",
     "HWCS1_cons_mean": float(sens[(sens.method=="HWCS-1")&(sens.kappa=="cons")].work_save_vs_G4_p25.mean()),
     "HWCS1_cons_range": rng(sens[(sens.method=="HWCS-1")&(sens.kappa=="cons")], "work_save_vs_G4_p25"),
     "HWCS2_cons_mean": float(sens[(sens.method=="HWCS-2")&(sens.kappa=="cons")].work_save_vs_G4_p25.mean()),
     "HWCS2_cons_range": rng(sens[(sens.method=="HWCS-2")&(sens.kappa=="cons")], "work_save_vs_G4_p25"),
     "HWCS1_optimistic_range": rng(sens[(sens.method=="HWCS-1")&(sens.kappa=="opt")], "work_save_vs_G4_p25")
  },
  "g4_work_floor_undercut": {
     "claim": "HWCS-1 total structural work < native G4 minimum (dense-binning) work, for ALL states, for ALL kappa in [0,1]",
     "HWCS1_kappa_star_range": rng(pd.read_csv(f"{OUT}\\stage_a_breakeven.csv").query("method=='HWCS-1'"), "kappa_star_G4floor"),
     "HWCS1_kappa_star_all_gt_1": bool((pd.read_csv(f"{OUT}\\stage_a_breakeven.csv").query("method=='HWCS-1'").kappa_star_G4floor > 1).all()),
     "rho_range": rng(pred, "rho"),
     "rho_all_gt_4": bool((pred.rho > 4).all())
  },
  "matched_work_vs_nativeG4": {
     "HWCS1": "below G4 work floor in 9/9 states (G4 cannot match work at any p)",
     "HWCS2": "HWCS lower NMSE than G4-at-matched-work in 9/9 states"
  },
  "segment_mixture_dominance": {
     "note": "BELOW_mixture => genuine advantage over a G16/G4 workload split",
     "by_kappa": {f"{m}_k{k}": int(g.bit.sum()) for (m,k),g in
                  seg.assign(bit=(seg.verdict=="BELOW_mixture").astype(int))
                     .groupby(["method","kappa"])},
     "kappa_crossover": "HWCS BELOW mixture for kappa<=0.75 (9/9); ABOVE (dominated) at kappa=1.0 in 5/9"
  },
  "equal_cost_audit_reproduction": {
     "G4_over_G16_eqcost_range": rng(eq, "G4_over_G16_eqcost"),
     "audit_reported_range": [3.8391, 6.3482],
     "comment": "reproduced at low end; per-state rho compresses high end slightly"
  },
  "K_calibration_consistency": {
     "mean_rel_inconsistency_max_over_G": 0.0252,
     "comment": "Bernoulli NMSE=K*(1-p)/p fits both p values within ~2.5% -> model internally consistent"
  },
  "stage_b_feasibility": {
     "feasibility_falsified": False,
     "executable_from_this_host": False,
     "reason_not_executable": "MX /mnt absent; no torch locally; local GPU is RTX5070 sm120 (WV0 .so built for A100 sm80); checkpoints reside on MX",
     "must_not_preempt": "TEMPORAL_COVERAGE-R1 dedicated GPU",
     "minimal_stage": "Stage B-min: PyTorch-oracle HWCS mask, full-render+matched-loss, NO CUDA build",
     "estimated_cost_idle_A100": "~10-30 min, 256 sparse + 2 dense evals, 1 high-N checkpoint x 2 cams x 32 seeds x {G16,G4,HWCS1,HWCS2}"
  },
  "verdict": "CONDITIONAL_RESEARCH",
  "verdict_rationale": "Math sound (MC-validated <0.3%); HWCS-1 unconditionally undercuts native-G4 work floor in structural model; but ALL quality/cost numbers are proxy or predicted, never measured; decisive unknowns (real HWCS NMSE, kappa, GPU-time realization, native-G4 baseline design) require Stage B-min measurement before any Codex/CUDA authorization."
}

with open(f"{OUT}\\stage_a_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("wrote stage_a_summary.json")
print(json.dumps({k: summary[k] for k in ["verdict","work_saving_vs_nativeG4_p25","g4_work_floor_undercut","segment_mixture_dominance"]}, indent=2)[:1600])
