# Track B — ReplayGS / Radiance-Operator Replay (P0) — VERDICT: **DROP**

**Date:** 2026-09-30  **Host:** EPIC-05 A100 (4x A100-SXM4-80GB)  **Cells:** room/bicycle x 20K/25K

## Hypothesis
With geometry + opacity FROZEN and a fixed camera, the rendered RGB is linear in the SH
coefficients, so an appearance-only forward/backward can be replayed exactly from a cached
radiance operator cheaper than a native full step. The prior negative A0 result was blamed on
q_alpha; freezing geometry/opacity removes it and leaves an exact, cheaply-replayable color subproblem.

## Method
- Frozen renderer: `rasterize_gaussian_higs_dynamic` (higs_native backward), eps2d=0.3, SH degree 3.
- Operator built ONCE per camera by concatenating accepted hits into (rows, vals, masters);
  replay forward/VJP use one batched `spherical_harmonics` call over all hits (memory-bounded chunking).
- Per-gaussian color activation `c = clamp_min(SH_eval(dir)+0.5, 0)`; the per-hit weight w=T*alpha is
  frozen, color is re-evaluated each step; the SH adjoint is gated by (activated_color > 0).
- Loss unchanged: 0.8*L1 + 0.2*(1-SSIM). No densification. 8 frozen cameras/cell (last 2 held out).

## Stage B1 — Exact replay construction: **PASS (all cells)**
The replay operator is an EXACT reconstruction of the native SH gradient.

| cell | pooled sh_grad cosine | pooled rel_l2 | gate (cos>=0.9999, rel<=1e-3) |
|---|---|---|---|
| room_20000    | 0.99999999994 | 1.08e-5 | PASS |
| room_25000    | 0.99999999984 | 1.81e-5 | PASS |
| bicycle_20000 | 0.99999999987 | 1.62e-5 | PASS |
| bicycle_25000 | 0.99999999982 | 1.92e-5 | PASS |

The radiance-operator algebra is validated: frozen geometry/opacity => RGB linear in SH => exact replay.

## Stage B2 — Cost oracle: **FAIL (all cells), decisively**
`rho = T_replay / T_full` (median of >=100 timed iters after >=20 warmup, CUDA-synchronized).
Field `timing_ms` holds seconds; values below are converted to ms.

| cell | native full | native SH-only | replay SH-only | **rho** | train-cam0 hits | cache |
|---|---|---|---|---|---|---|
| room_20000    | 37.0 ms | 35.2 ms | 1290 ms | **34.88** | 290.0M | 11.6 GB |
| room_25000    | 15.6 ms | 14.9 ms | 1192 ms | **76.22** | 267.9M | 10.7 GB |
| bicycle_20000 | 33.5 ms | 30.2 ms | 1140 ms | **34.00** | 252.4M | 10.1 GB |
| bicycle_25000 | 33.1 ms | 29.8 ms | 1031 ms | **31.19** | 229.0M |  9.2 GB |

Gate: median rho <= 0.30. **All cells fail by ~100-250x.**

Root cause (profiling): the work is compute-bound on per-hit SH shading (229M-290M hits x 16 coeffs
x 3 ch). The native fused kernel does this in 15-46 ms without materializing intermediates; the
PyTorch replay materializes [n_hit,16,3] tensors and runs scattered index_add_ for the VJP, taking
~1.0-1.3 s. The cache (40 B/hit, 9-12 GB) fits the A100 (B5 OK), but the replay is far MORE expensive,
not cheaper. **This failure is fundamental/fair, not an artifact.**

## Stage B3 — Schedule utility oracle: **MIXED / mostly FAIL**
`U_K` = held-out loss improvement of (K replay SH steps + 1 full step) / (K+1 full steps).

| cell | held-out base | F1 (9 full steps) | U_2 | U_4 | U_8 | gate3 (>=1 K in {4,8} with U>=0.80) | gate4 (PSNR reg <=0.10 dB) | B3 pass |
|---|---|---|---|---|---|---|---|---|
| room_20000    | 0.03374 | +0.000569 | 0.612 | 0.717 | 1.560 | True | False (PSNR reg) | False |
| room_25000    | 0.03494 | +0.000127 | 2.737 | 2.846 | 2.665 | True | True | **True** |
| bicycle_20000 | 0.04920 | -0.000082 | nan | nan | nan | False (F1<=0) | True | False |
| bicycle_25000 | 0.04804 | -0.000520 | nan | nan | nan | False (F1<=0) | True | False |

- room_25000: the replay schedule gives ~2.7-2.8x the held-out improvement per step of full steps
  (U_4=2.85, U_8=2.67), PSNR-safe. The schedule IS more effective per step — but it is irrelevant
  because B2 fails.
- room_20000: U_8=1.56 but PSNR regresses >0.10 dB vs the matched full-step trajectory (gate4 fail).
- bicycle cells: the full-step baseline F1 does not improve held-out loss (F1<=0 => U undefined);
  extra full steps overfit the 6 train cameras, so the schedule-utility oracle cannot be measured.

## Verdict: **DROP**
PROMOTE requires B1 AND B2 AND B3 AND B4 AND B5. B1 is exact (PASS) and B5 cache fits (PASS), but
**B2 fails decisively in every cell (rho=31-76 vs gate <=0.30)**. ReplayGS is mathematically exact
but NOT cheaper: the PyTorch replay cannot beat the native fused SH-shading kernel. The B3 schedule
utility is moot. No threshold was tuned; the gate was applied as pre-registered.

## Novelty / crowding note
The exact frozen-geometry radiance operator (deliberate geometry/opacity freezing => exact reusable
appearance operator) is the candidate narrow boundary in NOVELTY_CROWDING_GUARD.md. However,
"Instant Colorization of Gaussian Splats (2026)" already does visibility-weighted least-squares on
fixed splats, which is close to this appearance-only replay idea. Since the track is DROP, no novelty
claim is made.

## Deliverables
REPLAYGS_SOURCE_IDENTITY / REPLAYGS_REPLAY_FIDELITY / REPLAYGS_COST / REPLAYGS_SCHEDULE_UTILITY /
REPLAYGS_VERDICT .json per cell, plus REPLAYGS_REPORT.md (this file). Logs in trackB/*.log.
