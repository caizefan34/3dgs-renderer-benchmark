# FIGURE CAPTION FACTS (authoritative only)

**Generated:** 2026-10-02  |  Every figure uses authoritative artifacts only; missing points are explicit PENDING/null. NO fabrication.

## Figure A — TFC geometry
- Scene `room`, 8 cameras, tile_size 16. Stock production per-axis AABB vs exact-ellipse tile membership.
- Pooled intersection reduction = 0.4241; mean-per-camera = 0.4244 (report separately).
- Candidate A_ellipse_exact: PSNR loss vs GT = 0.0 dB (bit-exact render); provably lossless (k-sigma ellipse == alpha=ALPHA_THRESHOLD iso-contour).
- Native intersection reduction 0.3927 (PBR0). Down2 not separately measured (null); PBR0 down2 realized Amdahl 1.0985 borderline-FAIL reported.

## Figure B — TFC performance scopes
- intersection-only: current-cohort geomean 1.0317x (PUB0); TFC2 kernel-level intersect speedup room 1.464 / garden 1.475 / bicycle 1.603.
- forward pipeline: TFC2 fwd speedup room 1.341 / garden 1.277 / bicycle 1.379.
- forward+backward (FROZEN PLY): legacy pooled e2e 1.3564x (footprint-stage); per-scene room 1.384 / garden 1.376 / bicycle 1.508. NOT a training step.
- production timing: current-cohort full-step 1.0317x; PBR0 native reconciled 1.1988x; total 30K wall-clock PENDING (TFC_FC0).

## Figure C — DFC mechanism
- DFC2 universal fixed thresholds {1e-7, 1e-6} on 6 held-out scenes. Quality PASS all scenes (PSNR loss <= 5e-6 dB).
- Processed-splat reduction 5.8%-23.4% (below the >=10%-on-all gate at 1e-7). Verdict: DOES_NOT_GENERALIZE_UNIVERSALLY.
- Full-step speedup per scene 1.006x-1.080x. Gradient cosine 0.99999986+, relL2 1.4e-5..2.9e-4.
- Held-out full GPU results PENDING (isolated node).

## Figure D — Training trajectory
- baseline vs TFC_FC0 and vs DFC_FC0: ALL curves PENDING (PSNR/SSIM/LPIPS/gaussian-population/cumulative-wall-time).
- No TFC_FC0/DFC_FC0 30K curves exist. No fabrication.

## Figure E — Negative-space analysis
- VOS: Amdahl-bound (optimizer ~15-19% of step, SparseGaussianAdam O(N)); VOS1 full-30K QUALITY_FAIL.
- IGF: DROP (no kernel, 0.46-0.92x). CGC: DROP (fidelity gates fail all thresholds). KFC: DROP (cannot launch real workload). AGS: DROP (0% tile reuse, 2.18x slower).

## HARDENING_V2 caption corrections
- Figure B: 1.3564x = forward-rendering-pipeline timing (NOT production). production_timing = PENDING TFC_FC0. PBR0 1.1988x / PUB0 1.0317x = Tier-C decomposed proxies.
- Figure C: DFC2 COMPLETED (measured, EXECUTED EPIC-05). DOES_NOT_GENERALIZE_UNIVERSALLY — quality PASS all scenes; splat reduction 5.8-23.4%; no uniform 10% floor.
