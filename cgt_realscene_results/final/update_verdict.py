#!/usr/bin/env python3
"""Update CGT_MASTER_VERDICT.json + G2_G5_STATUS.json in place with fixed-teacher rerun results."""
import json, os, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
g1=json.load(open(os.path.join(HERE,"G1_REAL_SCENE_COMPRESSIBILITY.json")))
dense=json.load(open(os.path.join(HERE,"G2_CONTINUOUS_DOMAIN_CERTIFICATE.json")))
rig=json.load(open(os.path.join(HERE,"G2_RIGOROUS_CERTIFICATE.json")))
g3=json.load(open(os.path.join(HERE,"G3_GPU_MICROBENCH.json")))
bb=json.load(open(os.path.join(HERE,"G2_BENEFIT_BUDGET.json")))
combos=g1["combos"]
V=json.load(open(os.path.join(HERE,"CGT_MASTER_VERDICT.json")))
S=json.load(open(os.path.join(HERE,"G2_G5_STATUS.json")))

GPUS=[0,1,2,3,4,5,6]
V["gpus_used"]=GPUS
V["teacher_fix_note"]=("ROOT-CAUSE BUG FIXED + ALL RESULTS REGENERATED: tile_teacher in g1_compressibility.py "
 "used dx=px-mu (ignored py), evaluating the teacher along diagonal y=x instead of true 2D pixels. "
 "Fixed to concatenate([px,py]). g3 tile_teacher_torch had the same bug (fixed). All G1/G2/G3 raw records "
 "were regenerated with the corrected teacher; prior buggy outputs archived in G1_RAW_RECORDS/buggy_teacher/ "
 "and G1_RAW_RECORDS/stale_superseded_buggy_teacher/. G1 compressibility was essentially unchanged "
 "(q_weighted_r16 identical to 4dp); G2 fidelity metrics shifted (see G2 sections).")
V["overall"]="G1_PASS_G2_DENSE_AND_RIGOROUS_INTERVAL_CERTIFICATE_DONE_G3_MEASURED_LAUNCH_BOUND_G4_BATCHING_RECOVERABLE_NOT_A_SPEEDUP_CLAIM"
V["gates"]["G2_continuous_domain_certificate"]="DENSE_COVERAGE_EMPIRICAL_AND_RIGOROUS_INTERVAL_ARITHMETIC_DONE"
V["gates"]["G2_rigorous_interval_certificate"]="DONE_VALIDATED_PASS_RATE_BELOW_1_PROVES_NOT_EVERYWHERE"
# G1 summary refresh
V["G1_summary"]["q_weighted_r16"]={k:combos[k]["q_weighted_cumulative"]["16"] for k in combos}
V["G1_summary"]["S_inference_gamma2"]={k:combos[k]["S_inference_actual"]["2.0"] for k in combos}
V["G1_summary"]["S_inference_gamma4"]={k:combos[k]["S_inference_actual"]["4.0"] for k in combos}
V["G1_summary"]["incompressible_work_frac_range"]=[round(min(combos[k]["incompressible_work_frac"] for k in combos),4),round(max(combos[k]["incompressible_work_frac"] for k in combos),4)]
# G2 continuous domain certificate section
dp=dense["pooled_all_scenes"]; rp=rig["pooled_all_scenes"]
V["G2_continuous_domain_certificate"]={
 "status":"DENSE_COVERAGE_EMPIRICAL_AND_RIGOROUS_INTERVAL_ARITHMETIC_DONE",
 "evidence":"G2_CONTINUOUS_DOMAIN_CERTIFICATE.json (dense), G2_RIGOROUS_CERTIFICATE.json (rigorous)",
 "method_dense":"Deterministic dense sub-pixel grid (sub=4, 4096 pts/tile) per tile; 3 scenes x 10 tiles x 3 ranks = 90 tiles/rank.",
 "method_rigorous":"Branch-and-bound interval arithmetic over tile 2D ray domain; student interval box = convex hull of rank-R expert predictions; validated on synthetic (bound/pointmax ~1.0-4.6).",
 "pooled_dense_pass_rate":{r:dp[r]["dense_pass_rate"] for r in ("4","8","16")},
 "pooled_dense_max_abs_max":{r:dp[r]["dense_max_abs_max"] for r in ("4","8","16")},
 "pooled_rigorous_pass_rate":{r:rp[r]["pass_rate"] for r in ("4","8","16")},
 "pooled_rigorous_max_abs_max":{r:rp[r]["rigorous_max_abs_max"] for r in ("4","8","16")},
 "rigorous_validated":True,
 "finding":("DENSE (empirical) pooled pass_rate 0.98/0.90/0.90 (rank 4/8/16); worst-tile max|E,T| up to 0.78. "
   "RIGOROUS (interval-arithmetic, validated) pooled pass_rate 0.75/0.63/0.62; the bound PROVES the student "
   "does NOT reproduce the teacher everywhere on the continuous domain (fails 0.02 gate on a non-empty tile set). "
   "Higher rank does NOT monotonically improve either metric. This TEMPERS any 'student reproduces teacher everywhere' claim with a proof, not a sample."),
 "not_a_speedup_claim":"Fidelity certificate only; says nothing about GPU throughput."
}
# G3 section refresh (4 scenes)
ai=g3["aggregate_intrinsic_gamma"]; ap=g3["aggregate_per_tile_gamma"]; asf=g3["aggregate_S_measured_with_h"]
V["G3_gpu_microbenchmark"]={
 "status":"PARTIAL_MEASURED","evidence":"G3_GPU_MICROBENCH.json",
 "scenes_run":g3["scenes_run"],
 "intrinsic_gamma_R4_range":[ai["R4"]["min"],ai["R4"]["max"]],
 "per_tile_gamma_R4_range":[ap["R4"]["min"],ap["R4"]["max"]],
 "S_measured_with_h_R4_range":[asf["R4"]["min"],asf["R4"]["max"]],
 "h_overhead_per_tile_range":[g3["aggregate_h_overhead_per_tile"]["min"],g3["aggregate_h_overhead_per_tile"]["max"]],
 "finding":("4 scenes (bicycle, room r5k/r30k, garden). Intrinsic launch-free math gamma median R4=26.4x "
   "(range ~1.3-78x, scene-dependent via M_rep/R), but per-tile launch-bound gamma collapses to ~1.65-1.82x; "
   "measured per-tile S with dispatch overhead ~1.38-1.49x, far below 3x. Non-monotonic in rank."),
 "scenes_pending":[],
 "not_a_speedup_claim":"LOCAL per-tile kernel ratio, NOT end-to-end renderer speedup."
}
# critical caveats: replace the 'no rigorous certificate' line
V["critical_caveats"]=[
 "S_inference is an Amdahl UPPER BOUND assuming per-component cost gamma=2-4x Gaussian and ZERO overhead (tree traversal/gating/normalization/compositing/adaptive maintenance unmeasured). NOT a real GPU speedup.",
 "G2 benefit-budget adds a break-even overhead h_max: a future renderer must keep real overhead <= ~26.6% (gamma=2) / ~20.7% (gamma=4) for the 3x target to survive. Necessary condition, not evidence 3x is achievable.",
 "Fitting is per-view (teacher = that view's rasterization); measures low-rank compressibility of the local transfer response, NOT cross-view 3D-consistent representation (that is G5 native training).",
 "Held-out is held-out PIXELS within a view, not held-out VIEWS; plan sec.9 G1 'test new views / occlusion change' only partially addressed (no cross-view generalization certificate).",
 "G2 rigorous interval-arithmetic certificate (validated) PROVES the student does NOT reproduce the teacher everywhere on the continuous tile domain (pass_rate 0.62-0.75 < 1.0). This is a per-tile fixed-view bound, NOT a camera/view-variation bound and NOT a GPU throughput bound.",
 "Component-count reduction is not a GPU speedup; real speedup requires G3/G4 (and a fused spatial-tree kernel, not implemented).",
]
json.dump(V,open(os.path.join(HERE,"CGT_MASTER_VERDICT.json"),"w"),indent=1)
print("verdict updated; gpus",V["gpus_used"])

# ---- status refresh ----
S["G1_compressibility"]["status"]="PASS" if g1["G1_overall_verdict"]=="PASS" else "DROP"
S["G1_compressibility"]["evidence"]="G1_REAL_SCENE_COMPRESSIBILITY.json"
S["G1_compressibility"]["q_weighted_r16_range"]=[round(min(combos[k]["q_weighted_cumulative"]["16"] for k in combos),4),round(max(combos[k]["q_weighted_cumulative"]["16"] for k in combos),4)]
S["G1_compressibility"]["incompressible_work_frac_range"]=[round(min(combos[k]["incompressible_work_frac"] for k in combos),4),round(max(combos[k]["incompressible_work_frac"] for k in combos),4)]
S["G1_compressibility"]["S_inference_gamma2_range"]=[round(min(combos[k]["S_inference_actual"]["2.0"] for k in combos),2),round(max(combos[k]["S_inference_actual"]["2.0"] for k in combos),2)]
S["G1_compressibility"]["S_inference_gamma4_range"]=[round(min(combos[k]["S_inference_actual"]["4.0"] for k in combos),2),round(max(combos[k]["S_inference_actual"]["4.0"] for k in combos),2)]
cdc=S.get("G2_continuous_domain_certificate",{"status":"NOT_REACHED"})
cdc["status"]="DONE_DENSE_COVERAGE_AND_RIGOROUS_INTERVAL_ARITHMETIC"
cdc["dense_evidence"]="G2_CONTINUOUS_DOMAIN_CERTIFICATE.json"
cdc["rigorous_evidence"]="G2_RIGOROUS_CERTIFICATE.json"
cdc["pooled_dense_pass_rate"]={r:dp[r]["dense_pass_rate"] for r in ("4","8","16")}
cdc["pooled_rigorous_pass_rate"]={r:rp[r]["pass_rate"] for r in ("4","8","16")}
cdc["rigorous_validated"]=True
cdc["note"]=("Dense (empirical, 4096 pts/tile) pooled pass 0.98/0.90/0.90; rigorous (interval-arithmetic, validated) "
 "pooled pass 0.75/0.63/0.62 PROVES student does not reproduce teacher everywhere on the continuous domain. "
 "Both per-tile fixed-view; NOT view-variation or throughput bounds.")
S["G2_continuous_domain_certificate"]=cdc
json.dump(S,open(os.path.join(HERE,"G2_G5_STATUS.json"),"w"),indent=1)
print("status updated")
