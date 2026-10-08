# DPC0 Report - Deterministic Primitive-Centric 3DGS Backward
Date: 2026-09-30 22:06  |  EPIC-05 A100 (remote, gsplat env)
## Verdict
**DPC0_TOO_SLOW**
all_promote_cells=False all_strong_cells=False backward_speedup_median=0.6333919732798792
## Global gates
- source_identity_pass: True
- build_integrity_pass: True
- same_gpu_determinism_pass: True
- gradient_correctness_pass: True
- one_step_equivalence_pass: True
- cross_a100_bitwise_equal: True
- directional_conclusive: True
- backward_speedup_median: 0.6333919732798792
- directional_rel_err_median: 0.21697164833902446
- directional_rel_err_p90: 1.9439555266818187
- nan_inf_count: 0
- peak_extra_vram_fraction: 0.24481900468450163
## Per-cell summary
| cell | same_gpu | cos_med | cos_p10 | relL2_med | relL2_p90 | 1step | speedup | cross | dir_med | dir_p90 | vram_frac | nan |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| room@10000 | True | 1.0000000 | 1.0000000 | 6.74e-06 | 1.25e-05 | True | 0.735 | True | 2.12e-01 | 1.19e+00 | 0.245 | 0 |
| room@15000 | True | 1.0000000 | 1.0000000 | 8.43e-06 | 2.87e-05 | True | 0.700 | True | 2.97e-01 | 3.34e+00 | 0.207 | 0 |
| bicycle@10000 | True | 1.0000000 | 1.0000000 | 9.27e-06 | 1.68e-05 | True | 0.608 | True | 1.99e-01 | 1.48e+01 | 0.088 | 0 |
| bicycle@15000 | True | 1.0000000 | 1.0000000 | 3.72e-06 | 1.05e-05 | True | 0.587 | True | 2.09e-01 | 7.01e-01 | 0.070 | 0 |

## Notes
- Determinism: 32x same-GPU byte-identical for every source; cross-A100 bitwise-equal across GPU1/2/3 (3 A100s measured per cell, identical gradient hashes; GPU0 is the same-GPU primary and was not in the cross set).
- Correctness: DPC gradient matches the fp32 mean of 8 authoritative C0 backward samples (cosine ~1.0, relL2 ~6e-6); one-step update lies within the C0 min/max envelope.
- Directional: FD in float32 on tiny Rademacher directional derivatives is noisy (rel err ~5e-2..1e2); diagnostic only, gates STRONG only. Not a DPC defect (C0 backward is itself an approximation of the forward-loss gradient).
- Performance: DPC total backward is ~1.4-1.7x SLOWER than atomic C0 backward (per-cell median speedup 0.59-0.74, global median 0.63) due to the deterministic partial-buffer + segmented-reduction reorganization. VRAM overhead ~25%.
- Performance metrics (gate): per prereg 6, the total-backward speedup and peak extra VRAM are measured (128 reps, all DPC prep/sort/CSR included in the timed path); speedup median 0.63, VRAM overhead 7-24.5% (<=40% pass).
- Performance breakdown (S9 diagnostics, DPC0_TIMING_BREAKDOWN.json): measured via CUDA events in a SEPARATE timed build (module dpc0timing_..._cuda, .so sha256 7561c675...); verified DPC .so (8742df90...) and frozen analyzer untouched. Per source cam[0], Stage-1 splits into prep ~0.04-0.07 ms, producer ~8.7-13.0 ms, sort ~0.24-0.34 ms, segmented-reduce ~1.4-1.6 ms; the partial producer dominates Stage-1. proj+SH residual (projection + SH VJP + fixed tree reductions outside Stage-1) ~3.1-8.8 ms. Intersections 3.0-5.2M/source, partial-record bytes 107-189 MB/source (partial_record_bytes = n_isects*pstride*4, pstride=9 floats), partials/Gaussian histogram: n_gaussians_with_partials 265K-845K, min 1, median 4-6, mean 6.2-11.3, p90 10-20, max ~1380. All gradients finite.
- Timing correction: the orchestrator per-source timing loop omitted switching the backward op to the DPC build for its dpc_ms loop, so its raw speedup (~1.0x) is a measurement artifact. Corrected timing (dpc0_timing_fix.py) is used here.
- Verdict is DPC0_TOO_SLOW: determinism + correctness achieved, but the deterministic backward does not beat the atomic C0 backward on these cells.
