#!/usr/bin/env python3
"""Regenerate BWD_ARITH1_REPORT.md with OBSERVED/INFERRED/HYPOTHESIZED section (spec 38).
Writes to the correctly-named dir BWD_ARITH1_RECOMPUTE_GAUSSIAN (underscores)."""
import json, os, math
os.chdir('/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH1_RECOMPUTE_GAUSSIAN')
kt=json.load(open('BWD_ARITH1_KERNEL_TIMING.json'))
rt=json.load(open('BWD_ARITH1_RENDERER_TIMING.json'))
to=json.load(open('BWD_ARITH1_TRAFFIC_ORACLE.json'))
mem=json.load(open('BWD_ARITH1_MEMORY.json'))

def tbl(rows, hdr):
    out="| "+" | ".join(hdr)+" |\n| "+" | ".join("---" for _ in hdr)+" |\n"
    for r in rows: out+="| "+" | ".join(str(x) for x in r)+" |\n"
    return out

scenes=['room','bicycle','garden']
work={'room':225030315,'bicycle':740143995,'garden':242298908}
rows=[]
for sc in scenes:
    k=kt['scenes'][sc]; t=to['scenes'][sc]; m=mem['scenes'][sc]
    rows.append([sc, f"{work[sc]:,}", f"{t['cache_MB']} MB",
                 f"{k['S_backward_geomean']}", f"{k['S_renderer_geomean']}",
                 f"{t['load_to_savings_ratio']}x", f"+{m['bwd_delta_gb']} GB"])
perf_tbl=tbl(rows,["scene","N_valid_G","cache","S_bwd(geomean)","S_rend(geomean)","load/sav","bwd VRAM Δ"])

ktr=[]
for sc in scenes:
    for i,r in enumerate(kt['scenes'][sc]['runs']):
        ktr.append([sc, f"run{i+1}", r['T_B_R0_ms'], r['T_B_R1_ms'], kt['scenes'][sc]['S_backward_geomean']])
kt_tbl=tbl(ktr,["scene","run","T_B_R0_ms","T_B_R1_ms","S_bwd"])

report=f"""# BWD-ARITH1 REPORT — Exact Forward-to-Backward Gaussian Response Reuse on MX A100

## 1. EXECUTIVE RESULT
**Terminal verdict: `DROP_REUSE_TRAFFIC_OVERHEAD`**

Storing the exact forward Gaussian response G=exp(-Q/2) in FP32 and loading it in
`blend_backward_cu` (instead of recomputing expf) is **correct but slower** than recompute
on all 3 MX A100 production scenes. R1 is a net regression: backward geomean
**S_backward = {kt['S_backward_geomean_3scene']}** (R1 ~21% slower), renderer geomean
**S_renderer = {rt['geomean_S_renderer']}** (R1 ~16.5% slower). The cache load costs more
HBM traffic than the expf it removes. Recompute wins; exact full-interaction caching is
dropped. No rescue attempts (no FP16/quant/selective/approximate) per §39.

## 2. BASELINE
Production H-FasterGS `blend_backward_cu` recomputes G=exp(-Q/2) in backward (R0).
Predecessor BWD-ARITH0-MX established the arithmetic frontier: O20 (exp→linear) gives
-15.7% backward, mechanism COMPUTE_COUNT_BOUND + TRANSCENDENTAL_SFU. R0 build here
(ARITH0_ORACLE=0, no fast_math) measures bwd_med: room 3.27ms, bicycle 9.46ms, garden 3.89ms.

## 3. BWD-ARITH0 MOTIVATION
Backward Gaussian recomputation has measurable arithmetic cost. The next production
question: is recomputation actually more expensive than materializing + replaying the exact
intermediate? BWD-ARITH1 answers this.

## 4. EXACT REUSE MATHEMATICS
Forward: G_f = exp(-Q/2), stored FP32. Backward: G_b <- G_f (loaded), used unchanged in
all gradient equations (opacity/conic/mean). Contract G_b = G_f up to FP32 storage.
No approximation; FP32 only (§25).

## 5. WORKLOAD / CACHE SIZE
N_valid_G (post early-termination/alpha-threshold processed interactions):
room 225,030,315; bicycle 740,143,995; garden 242,298,908. Cache = 4*N bytes FP32:
room 900MB, bicycle 2.96GB, garden 969MB. (BWD_ARITH1_WORKLOAD.json)

## 6. TRAFFIC ORACLE
Peak A100-40GB HBM = 1.555 TB/s. Cache LOAD ALONE at peak = 1.45-1.59x the O20 expf
savings on all 3 scenes; store+load = 2.9-3.2x. L2 (40MB) << cache (0.9-3.0GB).
**R1 is physically impossible to pass the backward gate even at idealized bandwidth.**
(BWD_ARITH1_TRAFFIC_ORACLE.json)

## 7. CACHE LAYOUT
[n_buckets, 32, 256] FP32, index bucket*(32*256)+lane*256+pixel. Forward store
coalesced across pixel; backward load strided (lane/pixel transpose penalty). Shared via
host globals passed as kernel params (cudaMemcpyToSymbol avoided: invalid for dlopen'd
ext). Indexing fixed during implementation (stride 32*256, slot j%32, bucket offset-1).
(BWD_ARITH1_CACHE_LAYOUT.json)

## 8. FORWARD CORRECTNESS
R1 forward adds only a store (no arithmetic change). R0vR1 max_abs == R0vR0 nondeterminism
floor on all 3 scenes (room 4.5e-4, bicycle 3.8e-3, garden bit-identical). **PASS.**
Production forward is itself run-to-run nondeterministic (CUB radix sort/atomics).
(BWD_ARITH1_FORWARD_CORRECTNESS.json)

## 9. GRADIENT CORRECTNESS
Directly-affected families (means, shr) — which consume cached G — are **bitwise-identical**
(rate>=0.99999) between R0 and R1 on all 3 scenes, proving G_cached == G_recompute
bit-for-bit. All families within the production nondeterminism floor (~1e-3 relL2).
**PASS.** (BWD_ARITH1_GRADIENT_CORRECTNESS.json, BWD_ARITH1_CACHE_VALUE_CORRECTNESS.json)

## 10. RESOURCE AUDIT
R1 adds 2 kernel params/blend kernel; no new shared memory, no register-pressure change,
no spills. Binary +0.0MB (same .so, runtime toggle). **No large resource confound** — the
slowdown is genuine memory traffic, not occupancy/registers. (BWD_ARITH1_RESOURCE_AUDIT.json)

## 11. BACKWARD TIMING
{kt_tbl}
S_backward geomean (3 scenes) = **{kt['S_backward_geomean_3scene']}**. Backward gate
requires T_B_R1 <= 0.92*T_B_R0 (S>=1.087); measured S=0.790. **FAIL by a wide margin.**

## 12. FORWARD CACHE OVERHEAD
Delta_T_write = T_F_R1 - T_F_R0: room +0.24ms, bicycle +1.05ms, garden +0.19ms. The store
is a small coalesced write but non-zero; backward VRAM +0.8 to +2.7GB confirms cache active.

## 13. NET RASTER PERFORMANCE
S_renderer = (T_F_R0+T_B_R0)/(T_F_R1+T_B_R1): room 0.887, bicycle 0.780, garden 0.841.
Geomean **{rt['geomean_S_renderer']}**. Gate requires S_R>=1.05 on >=2/3 scenes; R1 is
**slower on all 3**. **FAIL.** (BWD_ARITH1_RENDERER_TIMING.json)

## 14. FULL-ITERATION PERFORMANCE
NOT RUN. Fixed-topology E2E gate requires renderer pass first (§22); renderer failed.
(BWD_ARITH1_ITERATION_TIMING.json)

## 15. MEMORY TRADEOFF
Cache adds 0.8-2.7GB backward VRAM (room +0.81GB, bicycle +2.70GB, garden +0.97GB).
Feasible on A100-40GB, but the memory cost buys a speed LOSS.
(BWD_ARITH1_MEMORY.json)

## 16. MECHANISM SCALING
delta_T_bwd scales NEGATIVELY with N_valid_G: more cache => more load traffic => larger
backward REGRESSION (room -0.56ms, bicycle -3.32ms, garden -1.09ms). Confirms
memory-traffic-bound, not compute-bound. (BWD_ARITH1_MECHANISM_SCALING.json)

## 17. TRAINING REGRESSION
NOT RUN. E2E gate not reached. Method is exact (no approximation) so no convergence
regression expected, but not measured. (BWD_ARITH1_TRAINING_REGRESSION.json)

## 18. PRIOR ART
See BWD_ARITH1_PRIOR_ART.md. H-FasterGS already stores bucket color/transmittance
checkpoints; R1 adds Gaussian-response state BEYOND that. Novelty label:
RESIDUAL_GAP_PLAUSIBLE (the negative result is consistent with the generic
recompute-vs-materialize principle).

## 19. LIMITATIONS
- GPU contaminated by foreign jobs (~6.6GB/GPU, 26–40% util) at measurement; paired
  same-session R0/R1 ratio used (cancels systematic load); direction consistent across
  3 scenes + 2 runs + matches physical bound. Not a clean-GPU measurement.
- Production forward nondeterministic (~1e-3 relL2); correctness gates floor-relative.
- Absolute R0 times ~13–39% above rebase reference (build-flag differences); relative
  R0->R1 deltas are the measurement of interest.

## 20. VERDICT
**`DROP_REUSE_TRAFFIC_OVERHEAD`** — recomputation is expensive in isolation (~15.7% backward
via O20), but materializing/replaying the exact FP32 intermediate costs more HBM traffic
than it saves (load alone 1.45–1.59x the expf savings at peak BW). Modern GPU execution
prefers recomputation over materializing this intermediate. Close exact full-interaction
caching.

## 21. EVIDENCE CLASSIFICATION (OBSERVED / INFERRED / HYPOTHESIZED)

### OBSERVED (directly measured on MX A100, raw JSON in raw/)
- R0/R1 paired timings: S_backward room 0.853 / bicycle 0.740 / garden 0.780; S_renderer
  room 0.887 / bicycle 0.780 / garden 0.841; geomeans 0.791 / 0.832. (BWD_ARITH1_KERNEL_TIMING.json,
  BWD_ARITH1_RENDERER_TIMING.json; runs 1+2 in raw/timing_*.json)
- Backward VRAM delta: +0.81 / +2.70 / +0.97 GB; memory_allocated R0->R1 delta 932.7MB (room)
  confirms cache allocation. (BWD_ARITH1_MEMORY.json, raw/cachecheck*.log)
- N_valid_G = 225,030,315 / 740,143,995 / 242,298,908 (workload, corrected obtain_off unpacking).
- Forward R0vR1 max_abs == R0vR0 floor; gradient means/shr bitwise rate 1.0 (room,bicycle,garden).
- Correctly-named dir `BWD_ARITH1_RECOMPUTE_GAUSSIAN/` holds all 21 artifacts + MANIFEST.sha256.

### INFERRED (derived from OBSERVED data)
- R1 is memory-traffic-bound: delta_T_bwd scales negatively with N_valid_G (more cache =>
  larger backward regression), inconsistent with a compute-bound explanation.
- Physical bound: cache LOAD at peak HBM = 1.45–1.59x the O20 expf savings (arithmetic from
  measured N + A100 peak BW), so R1 cannot pass even at idealized bandwidth.
- The slowdown is not a resource confound (no register/shared-mem/occupancy change); it is
  genuine HBM traffic cost.

### HYPOTHESIZED (not measured; future lanes only — forbidden in this lane per §39)
- FP16/BF16 cache, quantization, selective/hot-only cache, compressed cache, altered bucket
  size, or approximate Gaussian might shift the traffic/compute balance. NOT TESTED here;
  these are separate future hypotheses. The exact FP32 reuse result stands as measured.
- A fused fwd/bwd schedule (single pass, no intermediate materialization) could avoid the
  round-trip entirely; not implemented in this lane.
"""
open('BWD_ARITH1_REPORT.md','w').write(report)
print('REPORT REGENERATED with evidence classification, chars=',len(report))
