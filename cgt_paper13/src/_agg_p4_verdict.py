import json, glob, os
import numpy as np
HOME="/mnt/storage_pool/liaoyuanjun/cgt_paper13"
def gm(a): return float(np.exp(np.mean(np.log(np.array(a)))))
p2={}
for f in glob.glob(f"{HOME}/results/p2/cgt_*.json"):
    d=json.load(open(f))
    if d.get("status")=="OK": p2[d["scene"]]=d
scenes=sorted(p2.keys())
p3={}
for f in glob.glob(f"{HOME}/results/p3/e2e_*.json"):
    d=json.load(open(f))
    if d.get("status")=="OK": p3[d["scene"]]=d

# ---- P4 ablations ----
abl={
 "stage":"CGT-PAPER13 P4 paper ablations (RUN AFTER P2 frozen, per objective section 22)",
 "date_utc":"2026-10-04","host":"MX A100-PCIE-40GB, runtime liaoyuanjun_mx0qual",
 "ablation_scenes_fixed":["bicycle","truck","drjohnson"],
 "ablation_A_rank":{
   "status":"MEASURED_R4_ONLY_KERNEL_LIMITATION",
   "frozen_rank":4,
   "finding":"The frozen F0 fused CUDA kernel (f0_kernels.so) correctly supports ONLY R=4. "
             "Running the SAME kernel with R=8/R=16 produces catastrophic kernel-correctness failure "
             "(fused vs where(safe,student,teacher) reference max_abs ~0.86 >> 2e-5 gate). The kernel's "
             "coeffs stride / student-mixture loop is compiled for R=4; passing a larger R mis-reads the "
             "coeffs buffer. Therefore the rank ablation CANNOT be validly run with the frozen kernel.",
   "room_R4":{"S_mixed":2.9457,"kernel_pass":True,"kernel_max_abs":3.58e-7},
   "room_R8":{"S_mixed":1.0095,"kernel_pass":False,"kernel_max_abs":8.64e-1,"safe_fraction":0.46},
   "room_R16":{"S_mixed":1.0123,"kernel_pass":False,"kernel_max_abs":8.67e-1,"safe_fraction":0.34},
   "interpretation":"R=4 is the only valid rank for the frozen kernel. Upstream G1 CPU rank evidence (R=4/8/16) "
                    "does not transfer to the fused CUDA kernel without a kernel rebuild (forbidden in PAPER13).",
   "decision":"Do NOT replace frozen rank. R=4 remains the only production-valid rank.",
 },
 "ablation_B_cert_fallback":{
   "status":"CHARACTERIZED_FROM_P2_NOT_FULL_RUNTIME",
   "finding":"Certified mixed path is necessary: safe tiles (mean 61.7%) use the student fast path; "
             "unsafe tiles (mean weighted fallback 45.1%) REQUIRE teacher fallback for correctness. "
             "A no-fallback student-only variant would render unsafe tiles incorrectly (student approx "
             "error up to cert GATE=0.02, unbounded on hard tiles). Full runtime no-fallback ablation NOT_REACHED "
             "(E2E failure in P3 makes production ablation moot).",
   "safe_fraction_mean":float(np.mean([p2[s]["certificate"]["safe_fraction"] for s in scenes])),
   "weighted_fallback_mean":float(np.mean([p2[s]["certificate"]["weighted_fallback_fraction"] for s in scenes])),
 },
 "ablation_C_fused_translation":{
   "status":"NOT_MEASURED",
   "reason":"Requires a separate unfused CGT implementation (per-stage kernel launches). No unfused reference "
           "kernel exists in the frozen F0 tree; building one is new engineering that would also require re-validation. "
           "Deferred; not blocking the verdict.",
 },
}
open(f"{HOME}/results/PAPER13_ABLATIONS.json","w").write(json.dumps(abl,indent=2))

# ---- Verdict ----
S_mixed=np.array([p2[s]["S_mixed"] for s in scenes])
build_min=np.array([p3[s]["T_build_full_ms"]/60000.0 for s in scenes])
render_ratio=np.array([p3[s]["T_cgt_frame_ms"]/p3[s]["T_canonical_frame_ms"] for s in scenes])
verdict={
 "stage":"CGT-PAPER13 FINAL VERDICT (13-scene renderer paper-evidence lane)",
 "date_utc":"2026-10-04","host":"MX A100-PCIE-40GB, runtime liaoyuanjun_mx0qual",
 "classification":"DROP_RENDERER_GENERALIZATION",
 "one_line":"The frozen CGT-F0 renderer's mixed-forward kernel speedup (geomean 1.26x on 50-tile samples) does NOT "
            "generalize to full-frame end-to-end rendering: per-frame per-tile fit+certify build costs ~40 min/frame "
            "geomean and the tile-based frame render is ~6.7x slower than canonical gsplat, so novel-view E2E speedup is ~0.",
 "evidence_tiers":{
   "mixed_forward_S_mixed_geomean":float(gm(S_mixed.tolist())),
   "mixed_forward_tier":"meaningful (1.20-1.50x) on 50-tile samples, BUT kernel-only (build excluded)",
   "e2e_novel_view_S":"~0 for all 13 scenes (build + render >> canonical)",
   "e2e_build_geomean_min_per_frame":float(gm(build_min.tolist())),
   "e2e_render_ratio_geomean":float(gm(render_ratio.tolist())),
 },
 "observed":[
   "13/13 scenes have valid canonical checkpoints + data; F0 ingests all schemas (P0).",
   "MIXED_FORWARD (P2): S_mixed geomean=1.2606, median=1.0216, min=0.97 (counter/truck), max=3.03 (room).",
   "MIXED_FORWARD kernel correctness PASSES on all 13 scenes (max 1.85e-6 << 2e-5 gate) at R=4.",
   "MIXED_FORWARD safe fraction mean=0.617, weighted fallback mean=0.451.",
   "E2E (P3): full-frame build geomean=40.3 min/frame (0.9-301.7 min); CGT frame render 6.70x SLOWER than gsplat geomean.",
   "E2E novel-view S=0.000 for all 13; N_break=None for 12 scenes (CGT render slower than canonical); counter N_break=30236.",
   "F0 kernel is R=4-specific: R=8/R=16 fail kernel correctness (max_abs ~0.86).",
 ],
 "evidence":[
   "PAPER13_SOURCE_AUDIT.json (13/13 ingestible, cohorts).",
   "PAPER13_CGT_RESULTS.json (P2 mixed-forward + certificate).",
   "PAPER13_BASELINE_RESULTS.json (P1 canonical gsplat).",
   "PAPER13_BUILD_AMORTIZATION.json (P3 E2E build/render).",
   "PAPER13_CORRECTNESS.json, PAPER13_CERTIFICATE_STATS.json, PAPER13_ABLATIONS.json.",
 ],
 "inference":[
   "The CGT student+certificate compresses the per-tile render function and the fused kernel IS faster than the "
   "teacher kernel on small work-decile samples (mixed-forward).",
   "But the representation is screen-space tile-based: it must be REBUILT (fit+certify) per camera per frame. "
   "For novel-view synthesis (a new camera every frame) this build is unavoidable and dominates (~40 min/frame).",
   "The tile-based fused render (one block/tile, no cross-tile sort) is also ~6.7x slower than gsplat's optimized "
   "full-frame rasterizer, so even the render step loses to the strong canonical baseline.",
   "Thus the kernel-level speedup does not survive the full production frame path.",
 ],
 "hypothesis_not_measured":[
   "Whether a camera-independent CGT representation (amortized across views) could rescue E2E (would be a NEW method, "
   "forbidden in this frozen lane).",
   "Full-image GT dPSNR/dSSIM/LPIPS quality gates (NOT_REACHED; mixed-forward tile-level only).",
   "Nsight hardware counters (NOT_MEASURED; no profiler permission assumed).",
 ],
 "not_measured":[
   "PAPER13_MEMORY.json (peak VRAM per scene) - NOT_MEASURED.",
   "Ablation C (fused vs unfused) - NOT_MEASURED.",
   "Same-hardware external renderer reproduction (section 29 optional) - NOT_ATTEMPTED.",
 ],
 "consistency_with_upstream":[
   "Consistent with TI0 DROP_UPDATE_COST: representation refit/rebuild dominates.",
   "Consistent with G5 DROP_NATIVE_OPTIMIZATION: CGT is not a training primitive.",
   "Confirms CGT as a rendering representation with a real but non-generalizing kernel speedup.",
 ],
 "decision":"CLOSE the CGT-PAPER13 renderer generalization lane as DROP_RENDERER_GENERALIZATION. "
           "Do not promote to a larger multi-scene TI1. The frozen CGT renderer does not deliver a meaningful "
           "full-frame E2E acceleration against the strong canonical gsplat baseline.",
}
open(f"{HOME}/results/PAPER13_VERDICT.json","w").write(json.dumps(verdict,indent=2))
print("Wrote PAPER13_ABLATIONS.json + PAPER13_VERDICT.json")
print("VERDICT:",verdict["classification"])
print("S_mixed geomean=%.4f  E2E build geomean=%.1f min/frame  render ratio=%.2fx"%(gm(S_mixed.tolist()),gm(build_min.tolist()),gm(render_ratio.tolist())))