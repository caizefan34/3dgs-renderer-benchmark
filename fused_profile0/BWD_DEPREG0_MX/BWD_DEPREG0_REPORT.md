# BWD-DEPREG0-MX — Backward Dependency / Register-Pressure Frontier Decomposition

Lane: DIAGNOSTIC / ORACLE / MECHANISM IDENTIFICATION ONLY. No production optimization implemented before mechanism established.
Date: 2026-10-07. Hardware: MX A100-PCIE-40GB. Baseline: H-FasterGS (RASTER-REBASE0-MX) + FasterGSFused.

---

## 1. Executive Summary

Question: is the dominant remaining backward frontier in blend_backward_cu caused by serial dependency chains (H1), register pressure (H2), spills/local memory (H3), occupancy (H4), scheduler/issue (H5), or a mix (H6)?

Answer: NO. No dependency / register / spill / occupancy lever produces a material backward reduction. The recurrence-removal oracle removes the entire cross-lane and intra-lane recurrence yet changes backward latency by <=0.5%. Register count falls monotonically as gradient families are removed (color -6, opacity -3, conic -8) with no register cliff, no spill, and no latency change. The occupancy sweep (occ64, 67->64 regs) is flat; forcing 32 regs only adds spill.

Terminal verdict: NO_ACTIONABLE_DEPREG_FRONTIER. next_lane = FRONTIER_REASSESS.

OBSERVED (strongest DEPREG lever): dependency oracle geomean S_backward = 0.9978 across room/bicycle/garden.
OBSERVED: R0 = 67 registers/thread, 0 spill bytes, 1152 B smem, 0 barriers.
DERIVED: backward residual cost is not addressable by dependency/register/occupancy levers in this lane.

---

## 2. Frozen Baseline

Environment (frozen, from BWD_DEPREG0_BASELINE.json):
- GPU: NVIDIA A100-PCIE-40GB, UUID 84d2099e-9223-fb5f-f807-20e9932d07e0, CC 8.0.
- Python: /mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python3.10
- torch 2.9.1+cu128, CUDA runtime 12.8, driver 595.71.05.
- Build toolchain: higs-13scene-env nvcc (CUDA 12.8) with -Xptxas -v. System nvcc V11.5.119 is NOT used for timed builds.
- Module: _C_depreg0. Scenes room/bicycle/garden, view 0. warmup 50, iters 100. CUDA events, explicit sync outside measured interval.

R0 backward (ms), contention-robust min / contention-inflated median:
- room:      2.791 / 4.543   (N = 1,027,092)
- bicycle:  11.036 / 14.763  (N = 4,430,283)
- garden:    3.806 / 5.065   (N = 3,104,845)

Host is a shared 8xA100 with foreign/root training jobs present; bwd_min is used for cross-scene ratios, bwd_median is documented as contaminated.

---

## 3. Prior-Lane Context

- RASTER-REBASE0-MX: H-FasterGS selected as canonical raster baseline.
- BWD-ARITH0: BWD_ARITHMETIC_FRONTIER_FOUND; mechanism COMPUTE_COUNT_BOUND + TRANSCENDENTAL_SFU. O20 (expf -> linear diagnostic) = -15.7% backward, the only variant passing the promotion gate.
- BWD-ARITH1-RECOMPUTE-GAUSSIAN: DROP_REUSE_TRAFFIC_OVERHEAD. Caching exact FP32 G=exp(-Q/2) is correct but SLOWER (S_backward geomean 0.791). Recompute one expf < 4-byte/intersection HBM load on this workload. Gaussian FP32 caching is CLOSED and NOT reopened here.

BWD-DEPREG0 asks the harder production question: is the residual backward frontier dependency/register/spill/occupancy bound, and can any exact lever reach >=10% backward / >=5% renderer?

---

## 4. Research Hypotheses

H1 serial dependency frontier; H2 register-pressure frontier; H3 spill/local-memory; H4 occupancy; H5 scheduler/issue; H6 mixed. All tested against measured intervention evidence below.

---

## 5. Source Live-State Analysis

blend_backward_cu: rasterization/include/kernels_backward.cuh, lines 253-482. Launch <<<n_buckets, 32>>>: one warp (32 lanes = up to 32 Gaussians) per bucket; #pragma unroll over the 256-pixel tile (block_size_blend = 256). Full classification in BWD_DEPREG0_SOURCE_MAP.json. Key live state:
- Recurrence (L1): last_contributor, color_pixel_after, transmittance, and the shfl-carried grad_color_pixel/grad_alpha_common, updated every 32-lane step via warp.shfl_up and intra-lane via transmittance *= one_minus_alpha.
- Output accumulators (L3): dL_dmean2d_accum, dL_dconic_accum, dL_dopacity_accum, dL_dcolor_accum.
- Conic/mean2d Jacobian (L7/L4): gaussian_grad_helper = -alpha * dL_dalpha drives both dL_dconic and dL_dmean2d; these temporaries are live simultaneously with the recurrence across the unrolled body.
- Transcendental: gaussian = expf(fminf(exponent, 0.0f)).

The covariance-gradient machinery and the shfl recurrence are simultaneously live; whether that overlap is latency-critical is exactly what the probes test. OBSERVED from inspection only; no performance claimed.

---

## 6. Compiler Resource Analysis

ptxas -v (R0, logs/build_fast_driver.log): blend_backward_cu = 67 registers/thread, 0 spill stores, 0 spill loads, 1152 B shared memory, 0 barriers. preprocess_backward_cu = 80 regs, 0 spill. adam_step_invisible = 18 regs.

Per-variant registers (OBSERVED):
- r0: 67 reg, 0 spill
- dep (NORECUR): 64 reg, 0 spill
- nocolor: 61 reg, 0 spill
- noopacity: 64 reg, 0 spill
- noconic: 59 reg, 0 spill
- occ32 (maxrreg=32): 32 reg, 284 B store + 408 B load spill
- occ64 (maxrreg=64): 64 reg, 0 spill

Occupancy estimate: 67 regs -> ~30 blocks/SM (47%); a 1-warp/32-thread block caps at 32 blocks/SM (50%) hard limit. Occupancy is block-structured (one warp per bucket), not purely register-starved. See BWD_DEPREG0_RESOURCE_BASELINE.json.

---

## 7. Runtime Profiler Results

Runtime hardware profilers were UNAVAILABLE (documented, not hidden):
- Nsight Compute 2021.3.1 fails on this torch 2.9.1+cu128 / cudnn 9.10 build (application error code 6, zero kernels profiled; logs/ncu_room.log).
- nvprof v11.5 is too old for CUDA 12.8.

Fallback causal evidence is intervention-based paired interleaved CUDA-event timing (R0 vs variant, alternating iterations), which is contention-robust and satisfies the two-independent-evidence-class requirement. Per-scene records: BWD_DEPREG0_PROFILE_{ROOM,BICYCLE,GARDEN}.json.

OBSERVED R0 backward min (ms): room 2.791, bicycle 11.036, garden 3.806.

---

## 8. Cross-Scene Consistency

Dependency oracle S_backward (paired median): room 0.9943, bicycle 1.0007, garden 0.9984. Geomean 0.9978.
Occupancy occ64 S_backward (paired median): room 1.0011, bicycle 1.0006, garden 1.0000. Geomean 1.0006.
All three scenes agree: every DEPREG lever S_backward in [0.994, 1.001]. Direction is consistent; no scene shows a hidden cliff.

---

## 9. Gradient-Family Probes

Probes disable one gradient family each (ORACLE_ONLY / INVALID_FOR_TRAINING). Paired room timing + register counts:
- nocolor (BWD_DEPREG_NOCOLOR): 61 reg (-6), 0 spill, S_backward_paired 1.0009 (delta +0.0005 ms).
- noopacity (BWD_DEPREG_NOOPACITY): 64 reg (-3), 0 spill; register-only (paired timing not run; see README note in artifact).
- noconic (BWD_DEPREG_NOCONIC): 59 reg (-8), 0 spill, S_backward_paired 1.0011 (delta -0.0225 ms).

Findings: no register cliff, no spill emergence, no material latency change. Removing an entire gradient family does not speed backward up -> backward is not family-compute-bound in an exploitable way. See BWD_DEPREG0_GRADIENT_RESOURCE_PROBES.json.

---

## 10. Live-Range Probes

Dedicated hand-authored live-range shortening was not pursued; the gradient-family probes double as register-reduction interventions (family removal lowers registers with no cliff and no latency change). No register cliff, no spill, no latency change. See BWD_DEPREG0_LIVERANGE_PROBES.json. Diagnostic only; variants are not numerically complete.

---

## 11. Dependency Oracle

BWD_DEPREG_NORECUR removes the entire cross-lane shfl recurrence (transmittance/color/grad/last_contributor) and the intra-lane transmittance/color carry-over, setting transmittance = 1.0 and last_contributor = block_size_blend. ORACLE_ONLY / SEMANTICALLY_INVALID; timing only.

OBSERVED S_backward (paired median): room 0.9943, bicycle 1.0007, garden 0.9984; geomean 0.9978. Registers 64, 0 spill.

Key quantity (latency reduction / arithmetic removed): removing the whole recurrence yields <=0.5% latency change. Large recurrence removal, ~zero latency change -> dependency serialization is NOT the dominant frontier (H1 REJECTED). The recurrence is well-hidden by surrounding arithmetic and memory latency. See BWD_DEPREG0_DEPENDENCY_ORACLE.json.

---

## 12. Occupancy Sensitivity

- occ64 (maxrregcount=64, 67->64 regs, no spill): geomean S_backward 1.0006 (no help).
- occ32 (maxrregcount=32): forces 284 B store + 408 B load spill; not a production point, not timed as a candidate.

Per spec rule (higher occupancy + same/slower -> occupancy not causal): occ64 is flat -> occupancy is not causal (H4 REJECTED). The binding occupancy ceiling is the 1-warp-per-bucket block mapping, not register pressure. See BWD_DEPREG0_OCCUPANCY_SWEEP.json.

---

## 13. Split-Kernel Diagnostic

NOT_PERFORMED. Every bounded probe returned ~1.0x; no single gradient subsystem split is justified when removing the subsystem itself changes nothing. A split would only add intermediate HBM traffic and launch overhead with no mechanism to exploit. Optional Phase-H skipped deliberately. See BWD_DEPREG0_SPLIT_ORACLE.json.

---

## 14. Correctness

R0 (production) is bit-identical to canonical H-FasterGS forward and gradient; no macro defined. All gated variants (dep, nocolor, noopacity, noconic, occ32, occ64) are ORACLE_ONLY / SEMANTICALLY_INVALID and quarantined; never used for production or correctness claims. Gate: PASS. See BWD_DEPREG0_CORRECTNESS.json and BWD_DEPREG0_SOURCE_DIFF.json.

---

## 15. Causal Mechanism Analysis

Primary: NO_ACTIONABLE_DEPREG_FRONTIER (confidence HIGH).
Secondary: MEMORY_LATENCY_BOUND residual, with TRANSCENDENTAL_SFU (expf) residual from BWD-ARITH0.

Evidence classes (>=2 agree, per spec section 13):
- Intervention timing (dependency oracle ~1.0; gradient removal ~1.0; occupancy ~1.0) — OBSERVED.
- Compiler resource inventory (67 reg, 0 spill; family removal no cliff) — OBSERVED.

Hypothesis outcomes:
- H1 dependency-bound: REJECTED (oracle ~1.0).
- H2 register-bound: REJECTED (67 reg not severe; no cliff, no latency).
- H3 spill-bound: REJECTED (0 spill everywhere except forced occ32).
- H4 occupancy-bound: REJECTED (occ64 flat).
- H5 scheduler-bound: cannot be confirmed or denied without ncu (UNAVAILABLE); no intervention moves latency.
- H6 mixed: no single DEPREG resource dominates; residual sits in expf/SFU + memory latency + issue overhead.

See BWD_DEPREG0_MECHANISM.json.

---

## 16. Amdahl / Renderer Impact

Best oracle geomean S_backward = 0.9978 (dependency). Renderer speedup geomean ~1.00; maximum possible renderer speedup ~1.00x. Promotion gate FAILS (needs S_backward >= 1.10 AND S_renderer >= 1.05 on >= 2/3 scenes). Because backward does not move, the renderer cannot move via DEPREG levers. See BWD_DEPREG0_AMDAHL.json.

---

## 17. Candidate Ranking

Zero production candidates generated. No DEPREG mechanism reaches the promotion gate. The only measured lever with real arithmetic impact in this kernel family is the O20 expf->linear replacement from BWD-ARITH0 (a transcendental-arithmetic change, -15.7% backward), which is not a dependency/register/occupancy/spill lever and is already characterized in that lane. Generating a DEPREG1 here would be unsupported. See BWD_DEPREG0_CANDIDATES.json.

---

## 18. Limitations

- Shared MX host carries foreign/root training jobs (~6.6 GB/GPU on all 8 GPUs, high load avg). Medians are contaminated; min-based and paired interleaved ratios are used as contention-robust evidence. A clean-GPU re-run would tighten absolute numbers but is unlikely to flip a ~1.0 result to >=1.10.
- No ncu/stall/eligible-warp counters (tooling incompatible). H5 cannot be directly confirmed or denied; the null result is inferred from the absence of any latency-moving intervention.
- noopacity paired timing was not run (register count only); this does not affect the verdict because color and conic removal already bracket the family-removal signal at ~1.0.
- The dependency oracle is semantically invalid; it upper-bounds the recurrence contribution, which is ~0.

---

## 19. Terminal Verdict

NO_ACTIONABLE_DEPREG_FRONTIER (spec sections 18 and 19).

All stop conditions are true: register pressure not severe, no meaningful spills, occupancy adequate, dependency oracle < 5% (< 0.5%), live-range probes < 5%. The strongest DEPREG lever is ~0.2% (a null result), far below the >= 10% backward / >= 5% renderer success criterion. The lane terminates. Do NOT invent a BWD-DEPREG1.

---

## 20. Next Lane

next_lane = FRONTIER_REASSESS.

The measured backward residual is the expf/SFU transcendental cost already isolated by BWD-ARITH0 (O20 -15.7%) plus memory-latency/issue overhead. Reassessment should consider whether a transcendental-focused lane (e.g. SFU-throughput or expf approximation under a separate exactness contract) is warranted, and whether the BWD-ARITH1 traffic finding implies that recompute-vs-materialize tradeoffs, not dependency structure, are the productive axis. Gaussian FP32 caching remains CLOSED (DO NOT reopen).

---

## 21. Evidence Classification

OBSERVED:
- R0 = 67 registers, 0 spill, 1152 B smem, 0 barriers (ptxas -v).
- Dependency oracle S_backward paired-median room/bicycle/garden = 0.9943 / 1.0007 / 0.9984 (geomean 0.9978); 64 reg, 0 spill.
- occ64 S_backward = 1.0011 / 1.0006 / 1.0000 (geomean 1.0006); 64 reg, 0 spill.
- nocolor 61 reg / S 1.0009; noconic 59 reg / S 1.0011; noopacity 64 reg (register-only).
- occ32 forces 284 B store + 408 B load spill at 32 reg.
- ncu 2021.3.1 fails (error code 6); nvprof 11.5 too old.

DERIVED:
- geomean ratios and the cross-scene [0.994, 1.001] consistency band.
- Occupancy estimate (~30 blocks/SM at 67 reg; 32 blocks/SM hard cap).
- Amdahl renderer upper bound ~1.00x from S_backward ~1.0.

INFERRED:
- The recurrence is latency-hidden by arithmetic + memory latency (inferred from null oracle, not from stall counters).
- The residual backward cost is expf/SFU + memory latency + issue overhead (inferred; consistent with BWD-ARITH0 O20).

HYPOTHESIZED:
- H5 scheduler/issue behavior (UNVERIFIABLE without ncu).

ORACLE_ONLY:
- dep (NORECUR), nocolor, noopacity, noconic, occ32, occ64. Semantically invalid; used only for upper-bound timing. Never promoted.
