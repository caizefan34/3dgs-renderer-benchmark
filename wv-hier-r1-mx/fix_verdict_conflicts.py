#!/usr/bin/env python3
"""Fix the 5 residual conflicts named in the goal, in STAGE_A_VERDICT.json
and stage_a_summary.json. Uses dict manipulation (robust) and adds an explicit
resolution block. Does NOT change the CONDITIONAL_RESEARCH verdict.
"""
import json, os
OUT = r"C:\Users\36570\3dgs-renderer-benchmark\wv-hier-r1-mx"

# ---------------- STAGE_A_VERDICT.json ----------------
vp = os.path.join(OUT, "STAGE_A_VERDICT.json")
v = json.load(open(vp, encoding="utf-8"))

# (1) falsification_conditions: qualify the work-floor claim as N1 + cost-model
v["falsification_conditions"][1] = ("HWCS structural work >= Native-G4 work under the ASSUMED cost model "
    "AND the N1 dense-bin-then-mask baseline. NOTE: the undercut is a property of the structural "
    "cost model + N1 only; it is NOT a GPU-time result and does NOT apply if Native-G4 uses N2 "
    "(mask-then-bin). kappa is a cost-model parameter, not count-identifiable.")

# key_findings fixes
kf = v["key_findings"]
kf["work_saving_vs_nativeG4_p25_conservative"]["_conditioning"] = (
    "VALID ONLY under (a) the assumed structural cost model (bin+sort ~ linear in intersection "
    "entries, GAM=1) and (b) Native-G4 = N1 dense-bin-then-mask. If Native-G4 = N2 (mask-then-bin), "
    "Native-G4 avoids dense work and HWCS has NO structural edge. These are intersection-count "
    "proxies, NOT measured GPU time.")
kf["g4_work_floor_undercut"] = (
    "UNDER N1 + the assumed cost model only: HWCS-1 structural work < Native-G4 dense-binning floor "
    "in 9/9 states for every cost-model kappa in [0,1] (break-even kappa*=2-4/rho in [1.14,1.57], all "
    ">1; all states rho>4). This is NOT unconditional: it fails for N2 and is a proxy, not GPU time. "
    "kappa* is a cost-model break-even, NOT an empirically measurable threshold.")
kf["matched_work_vs_nativeG4"]["_conditioning"] = (
    "N1 + assumed cost model only. HWCS-2 result also requires the cost model; not a GPU-time claim.")
kf["segment_mixture_dominance"] = {
   "finding": kf["segment_mixture_dominance"],
   "_conditioning": ("Cost-model dependent (kappa sweep). 'Below mixture' means below the G16/G4 chord "
      "UNDER THE ASSUMED WORK MODEL; not a measured result and assumes N1.")
}

# (2)+(3) decisive_unmeasured: clarify kappa is a cost-model param, Stage B-min reports counts not kappa
v["decisive_unmeasured"][1] = ("child-refine operation-cost efficiency kappa: a COST-MODEL parameter "
    "that CANNOT be uniquely identified from intersection counts. Stage B-min reports child-refine "
    "ENTRY COUNTS (a structural count) and per-camera paired gradient statistics; it does NOT measure "
    "kappa. kappa requires Stage C CUDA timing with identical baselines.")

# (4) next_experiment: remove 'measure kappa (falsifier F2)'
v["next_experiment"] = ("Stage B-min (controlling protocol = WV_HIER_R1_MX_STAGE_B_REVISED): extend "
    "_wv0_grad.py with the HWCS two-stage mask using COMMON RANDOM NUMBERS; run G16/G4(N1)/HWCS-1/HWCS-2 "
    "at matched p=0.25 on bicycle__iter15000 x 2 cams x 32 paired seeds; measure per-camera real HWCS "
    "gradient NMSE + paired bootstrap CI (blend-identity check), and report child-refine/parent ENTRY "
    "COUNTS as STRUCTURAL COUNTS (NOT kappa, NOT GPU time). Report BOTH N1 and N2 Native-G4 structural "
    "work. All full-render+masked-loss timing labeled SUPERVISION_ORACLE. 0 training steps, no CUDA edit. "
    "If evidence insufficient -> ORACLE_INCONCLUSIVE. See STAGE_B_REPORT.md.")

# (3)+(5) answers: remove kappa<=0.5 CUDA GO gate; N2 independent baseline
v["answers_to_core_questions"]["4_worth_cuda_investment"] = (
    "NOT_YET. Stage B-min (gradient oracle + N1/N2 structural counts) first. A Codex CUDA prototype "
    "requires INDEPENDENT design + performance thresholds: (a) measured GPU kernel/F+B time vs BOTH N1 "
    "and N2 Native-G4 on matched p, (b) gradient-quality preservation vs G16, (c) a minimum E2E "
    "time-to-quality target. NO count-based kappa threshold (e.g. kappa<=0.5) is a valid CUDA GO gate; "
    "kappa is not count-identifiable. Any STRUCTURAL_POTENTIAL verdict is NOT CODEX_GO.")
v["answers_to_core_questions"]["1_can_HWCS_reduce_native_G4_cost"] = (
    "PLAUSIBLE_YES_IN_MODEL, N1+cost-model-conditional only - HWCS-1 undercuts the Native-G4 N1 "
    "dense-binning work floor by 22.8-31.3% in 9/9 states (structural proxy). Fails for N2. Not GPU time.")
v["answers_to_core_questions"]["2_credible_quality_advantage_at_matched_cost"] = (
    "IN-MODEL ONLY - under the assumed work model HWCS sits below the G16/G4 chord for kappa<=0.75 "
    "(9/9); but NMSE is blend-predicted not measured, ordering is scene-dependent, and at kappa=1.0 it "
    "is dominated in 5/9. Stage B-min must confirm with per-camera paired CIs.")

# rationale: qualify 'unconditionally'
v["rationale"] = v["rationale"].replace(
    "showed HWCS-1 unconditionally undercuts Native-G4's minimum structural work in 9/9 states",
    "showed HWCS-1 undercuts the Native-G4 N1 dense-binning work floor in 9/9 states UNDER THE ASSUMED "
    "STRUCTURAL COST MODEL (not N2, not GPU time)")

# add explicit resolution block
v["residual_conflicts_resolved_20261008"] = {
   "goal_requirements": {
      "1_kappa_not_count_identifiable": "RESOLVED - decisive_unmeasured + g4_work_floor_undercut + work_saving conditioning all state kappa is a cost-model parameter not identifiable from counts",
      "2_stage_b_min_no_kappa_measurement_claim": "RESOLVED - next_experiment now says Stage B-min reports child-refine ENTRY COUNTS (structural), explicitly NOT kappa",
      "3_remove_kappa_le_0.5_cuda_GO_gate": "RESOLVED - 4_worth_cuda_investment rewritten; no count-based kappa threshold; CUDA GO requires independent design+performance (measured GPU time vs N1 AND N2) thresholds",
      "4_native_g4_floor_N1_cost_model_limited": "RESOLVED - all floor/saving/segment findings carry N1 + cost-model conditioning; 'unconditionally' removed",
      "5_N2_independent_strong_baseline": "RESOLVED - N2 (mask-then-bin) named throughout as an independent baseline that must be audited; HWCS edge declared N2-negative"
   },
   "verdict_unchanged": "CONDITIONAL_RESEARCH (corrections tighten framing; nothing measured on GPU)",
   "controlling_stage_b_protocol": "WV_HIER_R1_MX_STAGE_B_REVISED_20261008"
}
json.dump(v, open(vp, "w", encoding="utf-8"), indent=2)
print("fixed", vp)

# ---------------- stage_a_summary.json ----------------
sp = os.path.join(OUT, "stage_a_summary.json")
s = json.load(open(sp, encoding="utf-8"))
s["work_saving_vs_nativeG4_p25"]["_conditioning"] = ("N1 + assumed structural cost model only; "
    "intersection-count proxy, NOT GPU time; fails for N2 mask-then-bin")
if isinstance(s.get("g4_work_floor_undercut"), dict):
    s["g4_work_floor_undercut"]["_conditioning"] = ("N1 + cost model only; kappa* is a cost-model "
        "break-even, NOT a measurable threshold; fails for N2")
s["variance_model"]["kappa_status"] = ("cost-model operation-efficiency parameter; NOT identifiable "
    "from intersection counts; Stage B-min reports structural ENTRY COUNTS only, not kappa")
s["controlling_stage_b_protocol"] = "WV_HIER_R1_MX_STAGE_B_REVISED_20261008"
s["residual_conflicts_resolved_20261008"] = v["residual_conflicts_resolved_20261008"]
json.dump(s, open(sp, "w", encoding="utf-8"), indent=2)
print("fixed", sp)

# validate
for f in (vp, sp):
    json.load(open(f, encoding="utf-8"))
print("JSON OK")
