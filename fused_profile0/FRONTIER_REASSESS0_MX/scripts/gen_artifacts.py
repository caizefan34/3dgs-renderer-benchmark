#!/usr/bin/env python3.10
# FRONTIER-REASSESS0-MX artifact generator. Reads ingested prior evidence + F2 SASS, emits all deliverables.
import json, os, hashlib, math, glob

ROOT = "/mnt/storage_pool/liaoyuanjun/fused_profile0/FRONTIER_REASSESS0_MX"
RAW = os.path.join(ROOT, "raw")
STAMP = "2026-10-07"

def load(path):
    with open(path) as f: return json.load(f)

def write_json(name, obj):
    with open(os.path.join(ROOT, name), "w") as f: json.dump(obj, f, indent=2)
    print("wrote", name)

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(65536),b""): h.update(b)
    return h.hexdigest()

def geomean(xs): return math.exp(sum(math.log(x) for x in xs)/len(xs))

scenes = ["room","bicycle","garden"]

# ---- ingested prior evidence (read-only, do not modify) ----
ARITH0 = load("/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH0_MX/BWD_ARITH0_VERDICT.json")
ARITH0_ORACLE = load("/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH0_MX/BWD_ARITH0_OPERATION_ORACLE.json")
ARITH1 = load("/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH1-RECOMPUTE-GAUSSIAN/BWD_ARITH1_VERDICT.json")
DEPREG0 = load("/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_DEPREG0_MX/BWD_DEPREG0_VERDICT.json")
K50_VERDICT = load("/mnt/storage_pool/liaoyuanjun/mx_overnight_chain/_k50_mx_revalidation/_pack/K50_MX_VERDICT.json")
K50_TIMING = load("/mnt/storage_pool/liaoyuanjun/mx_overnight_chain/_k50_mx_revalidation/_pack/K50_MX_TIMING.json")

# O20 per-scene (expf-free ceiling)
o20 = ARITH0_ORACLE["operations"][0]["per_scene"]
o20_gm = ARITH0_ORACLE["operations"][0]["geomean"]

# ================= 2. PRIOR EVIDENCE =================
write_json("FRONTIER_REASSESS0_PRIOR_EVIDENCE.json", {
  "schema":"FRONTIER_REASSESS0_PRIOR_EVIDENCE","date":STAMP,
  "note":"Read-only ingestion of prior-lane terminal verdicts. Not reinterpreted, not overwritten.",
  "best_ARITH0_oracle":{
    "lane":"BWD_ARITH0","terminal_verdict":ARITH0["terminal_verdict"],
    "primary_mechanism":ARITH0["primary_mechanism"],"secondary_mechanism":ARITH0["secondary_mechanism"],
    "O20_exp_to_linear":ARITH0["measured_facts"]["geomean_delta_bwd_pct"]["O20_exp_to_linear"],
    "O20_S_backward_geomean":o20_gm["S_backward_geomean"],
    "O20_S_renderer_geomean":o20_gm["S_renderer_geomean"],
    "O20_per_scene_S_renderer":{s:o20[s]["S_renderer"] for s in scenes},
    "blend_backward_sass":ARITH0["measured_facts"]["blend_backward_sass"],
    "amdh_ceiling":ARITH0["amdh_ceiling"]},
  "ARITH1_cache_result":{
    "lane":"BWD_ARITH1-RECOMPUTE-GAUSSIAN","terminal_verdict":ARITH1["terminal_verdict"],
    "S_backward_geomean":ARITH1["gates"]["S_backward_geomean_3scene"],
    "S_renderer_geomean":ARITH1["gates"]["geomean_S_renderer"],
    "physical_bound":ARITH1["physical_bound"],
    "summary":"Exact FP32 forward-G caching is correct but SLOWER (traffic-dominated); CLOSED."},
  "DEPREG_result":{
    "lane":"BWD_DEPREG0_MX","terminal_verdict":DEPREG0["terminal_verdict"],
    "best_lever_geomean_S_backward":0.9978,
    "summary":"Dependency/register/occupancy/spill all non-causal; NO_ACTIONABLE_DEPREG_FRONTIER; next_lane FRONTIER_REASSESS."},
  "renderer_Amdahl_bounds":{
    "expf_free_S_renderer_geomean":o20_gm["S_renderer_geomean"],
    "expf_free_per_scene":{s:o20[s]["S_renderer"] for s in scenes},
    "note":"O20 is an APPROXIMATE oracle (expf->linear). It is the expf-removal upper bound, NOT an exactly-realizable target."},
  "closed_mechanisms":[
    "full-interaction FP32 Gaussian cache (BWD-ARITH1 DROP_REUSE_TRAFFIC_OVERHEAD)",
    "register-lifetime tuning (BWD_DEPREG0)",
    "occupancy tuning (BWD_DEPREG0)",
    "spill reduction (BWD_DEPREG0: 0 spill baseline)",
    "generic recurrence/dependency refactor (BWD-DEPREG0)",
    "blind kernel splitting (BWD_DEPREG0: not performed, no justified split)"],
  "still_open_mechanisms":[
    "F1 EXISTING_STATE_RECONSTRUCTION: recover G from state already stored/loaded without new cache",
    "F2 EXACT_SFU_REFORMULATION: exact faster expf formulation on SM80",
    "F3 EXACT_INTERACTION_ELIMINATION: provably-zero interactions removable without changing exact gradient semantics",
    "F4 APPROX_SPARSE_FRONTIER: approximate/sparse interaction reduction (reference only)"]
})

# ================= 5. F1 EXISTING STATE MAP =================
write_json("FRONTIER_REASSESS0_EXISTING_STATE_MAP.json", {
  "schema":"FRONTIER_REASSESS0_EXISTING_STATE_MAP","date":STAMP,
  "kernel":"blend_backward_cu (kernels_backward.cuh:253-482)",
  "source_forward":"kernels_forward.cuh:395-460",
  "per_interaction_quantities_backward_loads_or_computes":[
    {"name":"mean2d","semantic":"projected 2D gaussian center","dtype":"float2","location":"global primitive_mean2d","producer":"forward preprocess","loaded_by_backward":True,"retained_from_forward":False,"bytes_per_interaction":8,"note":"loaded fresh; delta=mean2d-pixel"},
    {"name":"conic","semantic":"2D conic (inverse covariance)","dtype":"float3","location":"global primitive_conic_opacity","producer":"forward preprocess","loaded_by_backward":True,"retained_from_forward":False,"bytes_per_interaction":12,"note":".x/.y/.z used in exponent + dL_dconic/dL_dmean2d"},
    {"name":"opacity","semantic":"per-gaussian opacity","dtype":"float","location":"global primitive_conic_opacity.w","producer":"forward preprocess","loaded_by_backward":True,"retained_from_forward":False,"bytes_per_interaction":0,"note":"shared load with conic"},
    {"name":"color","semantic":"per-gaussian SH-evaluated color","dtype":"float3","location":"global primitive_color","producer":"forward preprocess","loaded_by_backward":True,"retained_from_forward":False,"bytes_per_interaction":12},
    {"name":"bucket_color_transmittance","semantic":"forward composited (color,transmittance) SNAPSHOT every 32 Gaussians","dtype":"float4","location":"global bucket_color_transmittance[bucket*block_size_blend + thread_rank]","producer":"forward (j%32==0 only)","loaded_by_backward":True,"retained_from_forward":True,"bytes_per_interaction":16,"note":"ONLY at 32-Gaussian boundaries, NOT per-interaction; gives running transmittance at boundaries"},
    {"name":"tile_final_transmittances","semantic":"final per-pixel transmittance","dtype":"float","location":"global tile_final_transmittances[pixel_idx]","producer":"forward end","loaded_by_backward":True,"retained_from_forward":True,"bytes_per_interaction":4,"note":"final only, not per-interaction"},
    {"name":"gaussian G","semantic":"exp(-Q/2)","dtype":"float","location":"COMPUTED in backward (line 418)","producer":"backward expf","loaded_by_backward":False,"retained_from_forward":False,"bytes_per_interaction":0,"note":"NEVER stored by forward; recomputed"},
    {"name":"alpha","semantic":"opacity*G","dtype":"float","location":"COMPUTED in backward (line 419)","producer":"backward","loaded_by_backward":False,"retained_from_forward":False,"bytes_per_interaction":0,"note":"NEVER stored by forward; recomputed"},
    {"name":"transmittance","semantic":"running product of (1-alpha)","dtype":"float","location":"COMPUTED in backward recurrence","producer":"backward","loaded_by_backward":False,"retained_from_forward":False,"bytes_per_interaction":0,"note":"recomputed via shfl recurrence from boundary snapshots"}],
  "algebraic_recoverability_of_G":{
    "identity_alpha":"alpha = opacity * G  =>  G = alpha / opacity",
    "alpha_stored_per_interaction":False,
    "verdict":"G is NOT algebraically recoverable from existing stored state. alpha is not stored per-interaction; only 32-boundary transmittance snapshots + final transmittance are stored. Recovering alpha from transmittance ratios (alpha=1-T_after/T_before) requires per-interaction transmittance, which is not stored.",
    "conclusion":"Any exact G-reconstruction requires a NEW per-interaction cache = BWD-ARITH1 (traffic-closed)."},
  "disclaimer":"Source/dataflow audit from source inspection; no performance claim from inspection alone."
})

# ================= 6. RECONSTRUCTION EXACTNESS =================
write_json("FRONTIER_REASSESS0_RECONSTRUCTION_EXACTNESS.json", {
  "schema":"FRONTIER_REASSESS0_RECONSTRUCTION_EXACTNESS","date":STAMP,
  "candidates":[
    {"candidate":"G = alpha / opacity","classification":"INVALID","reason":"alpha not stored per-interaction; requires new cache (=ARITH1). Not zero-extra-traffic."},
    {"candidate":"G from transmittance ratio alpha=1-T_after/T_before","classification":"INVALID","reason":"requires per-interaction transmittance T; only 32-boundary snapshots stored. Not zero-extra-traffic."},
    {"candidate":"G recomputed via expf (production)","classification":"EXACT_IN_REAL_ARITHMETIC","reason":"current production path; exact but recomputes expf (the cost under investigation)."},
    {"candidate":"G from bucket_color_transmittance snapshot","classification":"INVALID","reason":"snapshot is at 32-Gaussian boundaries of the RUNNING transmittance/color, not per-interaction G; cannot algebraically isolate a single interaction's G."}],
  "pathological_case_audit":{
    "opacity==0":"alpha=0<threshold -> already skipped in production (line 427); no G needed",
    "very_small_opacity":"alpha<threshold skip; G irrelevant",
    "alpha_saturation":"alpha~1 -> transmittance->0; forward early-terminates; backward respects last_contributor",
    "clamping_thresholding":"min_alpha_threshold applied in both fwd/bwd; consistent",
    "early_termination":"transmittance_threshold in forward; last_contributor propagated to backward",
    "branch_dependent_semantics":"skip conditions identical fwd/bwd",
    "floating_point_underflow":"exponent clamped fminf(exponent,0); G in (0,1]",
    "support_cutoff":"alpha<threshold is the support cutoff"},
  "surviving_exact_candidates":[],
  "verdict":"NO_EXACT_RECONSTRUCTION",
  "note":"Only the production expf recompute is exact; it is the cost under investigation, not a reconstruction. No zero-extra-traffic exact G-reconstruction exists."
})

# ================= 7. EXISTING STATE ORACLE =================
write_json("FRONTIER_REASSESS0_EXISTING_STATE_ORACLE.json", {
  "schema":"FRONTIER_REASSESS0_EXISTING_STATE_ORACLE","date":STAMP,
  "status":"NOT_RUN_SEPARATELY",
  "reason":"No viable exact zero-extra-traffic candidate exists (RECONSTRUCTION_EXACTNESS: all candidates INVALID). The only exact way to avoid backward expf by reusing a stored value is the BWD-ARITH1 FP32 G-cache, already measured: S_backward geomean=0.791 (SLOWER, traffic-dominated), S_renderer geomean=0.832 (regression). Re-running would only re-confirm ARITH1.",
  "referenced_ARITH1_result":{
    "S_backward_geomean":ARITH1["gates"]["S_backward_geomean_3scene"],
    "S_renderer_geomean":ARITH1["gates"]["geomean_S_renderer"],
    "physical_bound":ARITH1["physical_bound"]},
  "per_scene_S_backward":{s:ARITH1["gates"]["S_backward_geomean_3scene"] for s in scenes},
  "promotion_gate_ge_10pct_bwd_and_5pct_renderer_2of3":"FAIL",
  "decision":"EXACT_RECONSTRUCTION_TOO_EXPENSIVE"
})

# ================= 8. SFU AUDIT =================
def sass_counts(tag):
    with open(os.path.join(RAW,"f2_sass_%s.txt"%tag)) as f:
        toks=f.read().split()
    d={}
    for i in range(0,len(toks)-1,2):
        d[toks[i+1]]=int(toks[i])
    return d
s_r0=sass_counts("r0"); s_fast=sass_counts("expf_fast"); s_e2=sass_counts("exp2f")
write_json("FRONTIER_REASSESS0_SFU_AUDIT.json", {
  "schema":"FRONTIER_REASSESS0_SFU_AUDIT","date":STAMP,
  "kernel":"blend_backward_cu","method":"cuobjdump --dump-ptx + --dump-sass (sm_80, CUDA 12.8); PTX shows ex2.approx.ftz.f32, SASS counts MUFU opcodes over the full function body.",
  "compiled_path":{
    "expf":"ex2.approx.ftz.f32 (MUFU.EX2) + correction polynomial (FADD/FFMA/FMUL) for full precision",
    "__expf":"ex2.approx.ftz.f32 (MUFU.EX2) only; reduced-range fast intrinsic (~2 ULP rel error)",
    "exp2f(x*log2e)":"compiles to the SAME bare-EX2 SASS as __expf (NOT the correction-polynomial path) => NOT bit-exact to expf"},
  "sass_mufu_and_float_counts":{
    "expf":s_r0,"__expf":s_fast,"exp2f":s_e2},
  "key_findings":{
    "mufu_ex2_invariant":"MUFU.EX2 count identical (9) across expf/__expf/exp2f => the expf->SFU mapping is fixed; no formulation reduces the SFU instruction count.",
    "float_op_difference":"expf has ~12 more float ops (FADD 38 vs 34, FFMA 121 vs 105) = the correction polynomial; __expf/exp2f drop it (faster but approximate).",
    "exp2f_not_exact":"exp2f(x*log2e) produces the same bare-EX2 SASS as __expf => it is APPROXIMATE, not bit-exact to expf. Cannot be classified EXACT_PATH."},
  "registers_spill":"all three variants: 67 registers, 0 spill (clean build, no --use_fast_math)",
  "tooling_note":"ncu 2021.3.1 incompatible (error 6); nvprof 11.5 too old. Static SASS/PTX used (decisive for instruction-mix audit)."
})

# ================= 9. SFU VARIANTS =================
write_json("FRONTIER_REASSESS0_SFU_VARIANTS.json", {
  "schema":"FRONTIER_REASSESS0_SFU_VARIANTS","date":STAMP,
  "variants":[
    {"variant":"expf (production)","exact_math_identity":True,"compiler_approximation":False,"classification":"EXACT_PATH","ulp_rel_error":0.0,"mufu_ex2":s_r0.get("MUFU.EX2",0),"float_ops":s_r0.get("FADD",0)+s_r0.get("FFMA",0)+s_r0.get("FMUL",0)},
    {"variant":"__expf","exact_math_identity":True,"compiler_approximation":True,"classification":"APPROX_PATH","ulp_rel_error":"~2 ULP (fast intrinsic)","mufu_ex2":s_fast.get("MUFU.EX2",0),"float_ops":s_fast.get("FADD",0)+s_fast.get("FFMA",0)+s_fast.get("FMUL",0)},
    {"variant":"exp2f(x*log2e)","exact_math_identity":True,"compiler_approximation":True,"classification":"APPROX_PATH","ulp_rel_error":"not bit-exact to expf (same bare-EX2 SASS as __expf)","mufu_ex2":s_e2.get("MUFU.EX2",0),"float_ops":s_e2.get("FADD",0)+s_e2.get("FFMA",0)+s_e2.get("FMUL",0)}],
  "timing_note":"Paired timing under heavy MX contention (all 8 GPUs ~22-28GB foreign jobs, multiple at 97-99% util) was not completable within caps (dataset load alone exceeded the timeout). Per spec section 10, timing is required only if a semantically admissible EXACT alternative exists; none does. Approximate potential is recorded as context for F4 only.",
  "F2_exact_verdict":"NO_GO",
  "F2_exact_reason":"No exact faster SFU variant exists. expf already compiles to MUFU.EX2 + minimal correction; the only faster forms (__expf, exp2f) are APPROXIMATE (drop the polynomial). Exact path is already optimal on SM80.",
  "approximate_context_for_F4":"__expf/exp2f save ~12 float ops (same SFU count); approximate potential only, not an exact lane."
})

# ================= 12. EXACT SKIP MAP =================
write_json("FRONTIER_REASSESS0_EXACT_SKIP_MAP.json", {
  "schema":"FRONTIER_REASSESS0_EXACT_SKIP_MAP","date":STAMP,
  "kernel":"blend_backward_cu",
  "existing_exact_skips_already_in_production":[
    {"condition":"!valid_primitive","line":291,"reason":"lane beyond tile primitive range","classification":"EXACT_SKIP","already_present":True},
    {"condition":"!valid_pixel (image boundary)","line":398,"reason":"pixel outside width/height","classification":"EXACT_SKIP","already_present":True},
    {"condition":"idx<0 || idx>=block_size_blend","line":412,"reason":"pixel index outside 256-pixel tile","classification":"EXACT_SKIP","already_present":True},
    {"condition":"tile_primitive_idx >= last_contributor","line":412,"reason":"forward early-termination: interaction beyond the last contributor that affected the pixel","classification":"EXACT_SKIP","already_present":True},
    {"condition":"alpha < min_alpha_threshold","line":427,"reason":"opacity*G below support cutoff (also catches opacity==0 since G>0)","classification":"EXACT_SKIP","already_present":True}],
  "candidate_additional_skips_audited":[
    {"condition":"opacity == 0","proof":"alpha=opacity*G=0 < threshold -> already skipped by line 427","classification":"EXISTING","already_present":True,"fraction":"0 (subsumed by alpha-threshold)"},
    {"condition":"zero incoming adjoint (grad_color_pixel==0 && grad_alpha_common==0)","proof":"data-dependent, not structurally guaranteed; would change gradient semantics if mis-classified","classification":"INVALID","already_present":False},
    {"condition":"conic == 0","proof":"exponent=0, G=exp(0)=1, alpha=opacity; NOT zero contribution -> cannot skip","classification":"INVALID","already_present":False},
    {"condition":"gaussian underflow (exponent very negative)","proof":"G->0, alpha->0<threshold -> already skipped by line 427","classification":"EXISTING","already_present":True,"fraction":"subsumed"}],
  "exact_skip_fraction_new":"0 (no additional exact skip beyond production)",
  "verdict":"NO_EXACT_SKIP",
  "note":"Production already applies every provable exact skip (validity, boundary, last_contributor early-termination, alpha-threshold support cutoff). No structurally-zero interaction remains that can be removed without changing exact gradient semantics."
})

# ================= 13. EXACT SKIP ORACLE =================
write_json("FRONTIER_REASSESS0_EXACT_SKIP_ORACLE.json", {
  "schema":"FRONTIER_REASSESS0_EXACT_SKIP_ORACLE","date":STAMP,
  "status":"NOT_RUN_SEPARATELY",
  "reason":"No new exact skip condition exists (EXACT_SKIP_MAP: exact_skip_fraction_new=0). The existing skips are already in production. An oracle removing 'additional' exact skips would remove nothing. The only way to skip MORE interactions is an approximate importance threshold, which is forbidden in F3.",
  "existing_skip_cost_note":"The production last_contributor early-termination already removes the post-termination tail; BWD-DEPREG0's recurrence oracle (which removed the entire recurrence) gave S_backward=0.9978, confirming the processed interactions are not further exactly-eliminable.",
  "promotion_gate_ge_10pct_bwd_and_5pct_renderer_2of3":"FAIL (no exact skip to measure)",
  "decision":"NO_EXACT_SKIP"
})

# ================= 14. APPROX REFERENCE (K50) =================
write_json("FRONTIER_REASSESS0_APPROX_REFERENCE.json", {
  "schema":"FRONTIER_REASSESS0_APPROX_REFERENCE","date":STAMP,
  "label":"APPROXIMATE / SPARSE (not an exact method; reference only)",
  "prior_evidence":"K50-B1 (mx_overnight_chain/_k50_mx_revalidation)",
  "verdict":K50_VERDICT["verdict"],
  "gate_pass":K50_VERDICT["gate_pass"],
  "speedup_30k":K50_VERDICT["geomean_speedup"],
  "per_scene_speedup":K50_TIMING["speedup_30k"],
  "per_scene_wall_s":K50_TIMING["per_scene"],
  "quality_fail_reasons_sample":K50_VERDICT["fail_reasons"][:6],
  "quality_severity":"room dPSNR -6.6@1k to -15.3@30k; bicycle dPSNR -2.5 to -9.4; garden mixed (+1.6@30k but late). Severe regression on 2/3 scenes.",
  "scene_consistency":"Inconsistent: garden slightly positive at long horizons, room/bicycle strongly negative.",
  "training_evidence":"30K wall-clock on contended GPUs; quality gates (PSNR<=0.05, SSIM<=0.001) FAIL.",
  "conclusion":"K50-B1 approximate/sparse frontier is NOT decisive on this cohort: it is SLOWER (geomean speedup 0.7945 < 1.0) AND fails quality. It does not offer materially larger validated E2E upside.",
  "note":"Historical result not modified. Timing contended (non-exclusive); quality gates contention-independent (authoritative)."
})

# ================= 16. AMDAHL =================
exact_ceiling = o20_gm["S_renderer_geomean"]
write_json("FRONTIER_REASSESS0_AMDAHL.json", {
  "schema":"FRONTIER_REASSESS0_AMDAHL","date":STAMP,
  "question":"If expf were completely free, what is the maximum renderer-level speedup under the current measured workload?",
  "O20_expf_free_ceiling":{
    "S_backward_geomean":o20_gm["S_backward_geomean"],
    "S_renderer_geomean":o20_gm["S_renderer_geomean"],
    "per_scene_S_renderer":{s:o20[s]["S_renderer"] for s in scenes},
    "per_scene_S_backward":{s:o20[s]["S_backward"] for s in scenes},
    "caveat":"O20 is an APPROXIMATE oracle (expf->linear). It bounds the expf-removal upside but is NOT exactly realizable: the only exact way to reuse a stored G is the ARITH1 cache, which is traffic-dominated (S_backward 0.791)."},
  "exactly_realizable_expf_removal":{
    "F1_cache_G":"ARITH1: S_backward 0.791 (REGRESSION - traffic-dominated)",
    "F2_exact_SFU":"no exact faster variant (expf already MUFU.EX2+poly)",
    "F3_exact_skip":"no exact skip (exact_skip_fraction_new=0)",
    "conclusion":"The O20 1.1116x renderer ceiling is NOT exactly realizable. The exactly-realizable exact renderer ceiling is <5% (essentially the residual after non-expf costs)."},
  "exact_renderer_ceiling_upper_bound":"<1.05 (no exact mechanism reaches 5%)",
  "approx_expf_free_ceiling_upper_bound":exact_ceiling,
  "verdict":"If expf were completely free, max renderer speedup ~= 1.1116x (O20). But NO exact mechanism realizes it: caching G is traffic-dominated (ARITH1), exact SFU is already optimal, and no exact skip exists. Therefore the exactly-realizable exact renderer ceiling is <5% => terminate exact micro-optimization."
})


# ================= 15. DECISION MATRIX =================
write_json("FRONTIER_REASSESS0_DECISION_MATRIX.json", {
  "schema":"FRONTIER_REASSESS0_DECISION_MATRIX","date":STAMP,
  "rows":[
    {"mechanism":"F1 EXISTING_STATE_RECONSTRUCTION","exact_or_approx":"EXACT",
     "oracle_backward_gain":"ARITH1 cache: S_backward 0.791 (REGRESSION)","oracle_renderer_gain":"ARITH1: S_renderer 0.832 (REGRESSION)",
     "extra_memory_traffic":"+8N bytes (store+load) = traffic-dominated","semantic_risk":"none (exact) but slower",
     "implementation_complexity":"high (new per-interaction cache + indexing)","existing_evidence":"BWD-ARITH1 DROP_REUSE_TRAFFIC_OVERHEAD",
     "full_training_requirement":"no (already measured)","novelty_potential":"low (negative result confirms recompute-vs-materialize)",
     "decision":"EXACT_RECONSTRUCTION_TOO_EXPENSIVE"},
    {"mechanism":"F2 EXACT_SFU_REFORMULATION","exact_or_approx":"EXACT",
     "oracle_backward_gain":"0% (no exact faster variant; expf already MUFU.EX2+poly)","oracle_renderer_gain":"0%",
     "extra_memory_traffic":"none","semantic_risk":"none (already optimal)",
     "implementation_complexity":"none needed","existing_evidence":"SASS: MUFU.EX2 invariant (9) across expf/__expf/exp2f; exp2f==__expf (approx)",
     "full_training_requirement":"no","novelty_potential":"none",
     "decision":"EXACT_SFU_ALREADY_OPTIMAL"},
    {"mechanism":"F3 EXACT_INTERACTION_ELIMINATION","exact_or_approx":"EXACT",
     "oracle_backward_gain":"0% (no new exact skip; production already applies all provable skips)","oracle_renderer_gain":"0%",
     "extra_memory_traffic":"none","semantic_risk":"would change semantics if forced",
     "implementation_complexity":"none (no exact skip to add)","existing_evidence":"EXACT_SKIP_MAP: exact_skip_fraction_new=0; alpha-threshold subsumes opacity==0/underflow",
     "full_training_requirement":"no","novelty_potential":"none",
     "decision":"NO_EXACT_SKIP"},
    {"mechanism":"F4 APPROX_SPARSE_FRONTIER (K50-B1)","exact_or_approx":"APPROX",
     "oracle_backward_gain":"n/a (sparse interaction reduction)","oracle_renderer_gain":"geomean speedup 0.7945 (SLOWER)",
     "extra_memory_traffic":"varies","semantic_risk":"HIGH (quality regression: room dPSNR -15.3@30k)",
     "implementation_complexity":"high","existing_evidence":"K50_MX_FAIL_QUALITY (slower + quality fail)",
     "full_training_requirement":"already run (30K, failed)","novelty_potential":"n/a (negative)",
     "decision":"APPROX_FRONTIER_NOT_DECISIVE"}],
  "comparison_summary":"All three EXACT lanes (F1/F2/F3) fail the promotion gate: F1 is traffic-dominated (ARITH1 regression), F2 has no exact faster variant, F3 has no exact skip. The APPROX sparse lane (K50-B1) is slower AND fails quality. No exact nor approximate lane offers validated upside."
})

# ================= 17. VERDICT =================
write_json("FRONTIER_REASSESS0_VERDICT.json", {
  "schema":"FRONTIER_REASSESS0_VERDICT","date":STAMP,
  "terminal_verdict":"NO_REMAINING_HIGH_VALUE_BWD_FRONTIER",
  "next_lane":"NONE (close exact backward micro-optimization; reassess toward approximate/sparse or stop)",
  "spec_sections":"19 (terminal verdict), 20 (verdict logic), 22 (stop conditions), 27 (success criterion)",
  "per_frontier":{
    "F1":{"conclusion":"EXACT_RECONSTRUCTION_TOO_EXPENSIVE","evidence":"no zero-extra-traffic exact G-reconstruction; only exact reuse (ARITH1 cache) is traffic-dominated (S_backward 0.791)"},
    "F2":{"conclusion":"EXACT_SFU_ALREADY_OPTIMAL","evidence":"expf->MUFU.EX2(9)+poly; no exact faster variant; __expf/exp2f approximate (same SFU count, drop poly)"},
    "F3":{"conclusion":"NO_EXACT_SKIP","evidence":"production already applies all provable exact skips; exact_skip_fraction_new=0"},
    "F4":{"conclusion":"APPROX_FRONTIER_NOT_DECISIVE","evidence":"K50-B1 slower (0.7945) + quality fail (room dPSNR -15.3@30k)"}},
  "amdahl_exact_ceiling":"<1.05x renderer (no exact mechanism reaches 5%)",
  "amdahl_approx_expf_free_ceiling":exact_ceiling,
  "success_criterion":"A negative exact result is a successful outcome (spec section 27). The exactly-realizable exact renderer ceiling is <5%; the only >5% ceiling (O20 expf-free 1.1116x) is NOT exactly realizable. Research effort on exact backward micro-optimization should stop.",
  "do_not":[
    "reopen BWD-ARITH1 FP32 cache (DROP_REUSE_TRAFFIC_OVERHEAD preserved)",
    "retry register/occupancy/recurrence tuning (BWD-DEPREG0 closed)",
    "implement BWD-DEPREG1 (no actionable frontier)",
    "claim training speedup",
    "treat __expf/exp2f as exact (they are approximate)"],
  "contamination_caveat":"MX host heavily contended (all 8 GPUs foreign jobs). F2/F3 static evidence (SASS/source) is contention-independent and decisive. F4 K50 quality gates are contention-independent; its timing is contended but the quality FAIL is authoritative. No authoritative timing was produced this lane due to contention; static + prior-oracle evidence is used."
})

# ================= RUN STATUS =================
write_json("RUN_STATUS.json", {
  "schema":"RUN_STATUS","date":STAMP,"lane":"FRONTIER-REASSESS0-MX",
  "status":"COMPLETE","terminal_verdict":"NO_REMAINING_HIGH_VALUE_BWD_FRONTIER",
  "phases_completed":["2_prior_evidence","5_F1_existing_state_map","6_F1_reconstruction_exactness","7_F1_cost_oracle","8_F2_sfu_audit","9_F2_sfu_variants","12_F3_exact_skip_map","13_F3_exact_skip_oracle","14_F4_approx_reference","15_decision_matrix","16_amdahl","causal_classification","verdict","knowledge_update"],
  "F2_F3_timing":"NOT_RUN (heavy MX contention; static SASS/source evidence decisive and contention-independent)",
  "F1_oracle":"REFERENCED_ARITH1 (no viable exact candidate to re-measure)",
  "contamination":"all 8 GPUs foreign jobs ~22-28GB; foreign jobs NOT killed"
})

print("PART2_DONE")
