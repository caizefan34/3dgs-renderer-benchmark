# MomentGrad-GS — Final Lane Verdict

Date: 2026-10-03
Lane: MOMENTGRAD-GS (independent; not combined with ClockGS/RetireGS)

## VERDICT: DROP_GRADIENT_FIDELITY

Stage M1 (quadrature oracle) did not meet the pre-registered fidelity gate, so per the
plan the lane STOPs at M1 with DROP_GRADIENT_FIDELITY. Stages M2-M7 were not executed
(M2-M5 are gated on M1 passing; M6/M7 are A100/CUDA-gated and BLOCKED on this host).

## Stage status
- M0 exact-gradient dataset: DONE (pure-torch oracle rasterizer; per-Gaussian opacity/scale/
  rotation/SH capture via moments; renderer state = tile membership, footprint, depth order,
  alpha, transmittance, pixel adjoint). Oracle kernel validated.
- M1 quadrature oracle: DONE -> GATE FAILED.
- M2 moment-field oracle: NOT RUN (gated on M1).
- M3 depth-aware adjoint compression: NOT RUN (gated on M1).
- M4 difficulty certificate + E2E headroom: NOT RUN (gated on M1).
- M5 convergence experiment: NOT RUN (gated on M1).
- M6 CUDA prototype: BLOCKED_IMPLEMENTATION (no MSVC `cl`; gsplat 1.4 CUDA backend cannot
  JIT-compile; host is RTX 5070 Laptop 8GB, not A100).
- M7 authoritative A100 performance: BLOCKED_IMPLEMENTATION / PARTIAL_BLOCKED_BY_GPU
  (requires clean/exclusive A100; not available).

## OBSERVED
- Median per-Gaussian cosine: opacity ~1.000, color 0.997-1.000, geometry(mean2d) 0.84-0.98.
- Global/aggregate cosine: 0.74-0.99 (gate requires >=0.995 global, >=0.99 geometry).
- Work reduction: 550x-2200x (gate requires >=4x) -> PASSES.
- K-sweep non-monotonic (K=9 often worse than K=4).

## EVIDENCE
- momentgrad_gs/out/m1_quadrature.json
- momentgrad_gs/out/M1_SMOOTH_CONTROL.json
- momentgrad_gs/out/M1_REPORT.md
- Single-Gaussian validation: quadrature/canonical ratio 1.004.

## INFERENCE
- Smooth-field control is near-perfect and monotonic (K4=0.995 -> K16=0.998), so the
  real-field failure is caused by NON-SMOOTHNESS of the alpha-compositing adjoint field
  (occlusion boundaries, tile edges, neighbor discontinuity), not by the quadrature.

## HYPOTHESIS (untested, future work)
- A certificate-gated variant (compress the typical smooth majority, fall back to canonical
  backward on hard occlusion-boundary Gaussians) or a depth-aware moment basis (M2/M3) might
  recover aggregate fidelity. Not pursued here because the pre-registered M1 gate failed and
  the plan forbids rescuing by escalating complexity past the gate.

## NOT_MEASURED
- M2-M5 metrics; CUDA kernel time/memory/cache/warp efficiency; A100 E2E speedup.

## Reproduce
    python momentgrad_gs/m1_run.py            # M1 quadrature vs canonical (3 scenes)
    python momentgrad_gs/m1_smooth_control.py # smooth-field control
