# Track C — Profiled-SH / VarPro-GS (P0) — VERDICT: **DROP**

**Date:** 2026-09-30  **Host:** EPIC-05 A100 (GPUs 1-4)  **Cells:** room/bicycle x 20K/25K

## Hypothesis
Given the fixed-geometry radiance operator, SH appearance can be optimized MUCH more effectively than
ordinary Adam, so the outer geometry/optimizer needs fewer expensive full backward passes. The inner
appearance block is conditionally linear; optimize it as an inner problem via the exact replay operator.

## Dependency rule (honored)
This track reuses Track B's EXACT frozen radiance operator (build_operator / replay_forward /
replay_sh_grad) but does NOT assume Track B's performance result. It builds a slow PyTorch operator
independently. The operator reproduces native SH gradients (cosine=1.0 vs native), satisfying the
"if the operator cannot reproduce native SH gradients, DROP" precondition.

## Method (Stage C0 — appearance-only condition test)
- Freeze means/scales/rotations/opacity; optimize only SH.
- Inner fit on 2 train cameras; held-out eval on 2 frozen cameras. Fixed inner budget = 4 cycles.
- A. native SH-Adam (full native render per train camera per step).
- C. replay SH-L-BFGS (replay operator per train camera per step + backtracking line search, Armijo).
- Original 0.8*L1+0.2*(1-SSIM) throughout; no surrogate loss; no closed-form claim for L1+SSIM.

## Replay fidelity: **PASS (all cells)**
Per-camera SH-gradient cosine vs native = 1.000000, rel_l2 ~1e-5 (operators exact, inherited from Track B).

## Inner optimization (held-out loss decrease, 4 cycles, 2 inner cams)

| cell | held-out base | native Adam | replay L-BFGS | **ratio (LBFGS/Adam)** | gate2 (>=1.5x in >=3/4 cells) |
|---|---|---|---|---|---|
| room_20000    | 0.03374 | 0.03284 (psnr 32.50) | 0.03276 (psnr 32.51) | **1.091** | FAIL |
| room_25000    | 0.03494 | 0.03474 (psnr 31.89) | 0.03491 (psnr 31.81) | **0.164** | FAIL |
| bicycle_20000 | 0.04920 | 0.04959 (psnr 28.55) | 0.04923 (psnr 28.62) | nan (Adam worse) | FAIL |
| bicycle_25000 | 0.04804 | 0.04812 (psnr 28.80) | 0.04798 (psnr 28.83) | nan (Adam worse) | FAIL |

Gate 2: one fixed optimizer+budget improves held-out loss >=1.5x vs SH-Adam in >=3/4 cells.
**0/4 cells reach 1.5x** (best is room_20000 at 1.09x). Gate 2 FAILS.

## Findings
- L-BFGS does not beat SH-Adam on held-out loss in any cell. room_25000: L-BFGS is 6x WORSE (0.16x).
- bicycle: SH-Adam itself DEGRADES held-out loss (overfits the 2 inner cameras); L-BFGS is roughly
  neutral. The appearance inner problem is dominated by overfitting to the 2 train cameras, and
  neither optimizer generalizes to the 2 held-out cameras.
- Gate 4 (PSNR regression <=0.10 dB) passes in all cells; gate 3 (outer-step oracle) was NOT run
  (cannot rescue a PROMOTE once gate 2 fails).

## Verdict: **DROP**
PROMOTE requires gates 1-4. Gate 1 (fidelity) passes, gate 4 (PSNR) passes, but **gate 2 fails
(0/4 cells >=1.5x)** and gate 3 (outer oracle) is not run. The conditionally-linear appearance block
is NOT optimized more effectively than Adam by L-BFGS at the fixed budget. No threshold tuned; the
pre-registered gate was applied.

## Caveat (P0 minimal scope)
C0 tests one optimizer (L-BFGS) at one fixed budget (4 cycles, 2 inner cams). C1 (budgets {2,4,8,16})
and C2 (outer-step oracle) were not exhaustively swept. The DROP is driven by gate 2, which is robust:
L-BFGS is not better than Adam in ANY cell, let alone by 1.5x in 3/4. The bicycle overfit signal
(Adam itself degrading held-out loss) suggests the inner-appearance framing needs more train cameras
to be a fair test, which is beyond P0 scope.

## Novelty / crowding note
Profiled-SH (conditionally linear appearance block as an inner problem via the exact replay operator)
is a candidate narrow boundary in NOVELTY_CROWDING_GUARD.md, but it overlaps "Instant Colorization of
Gaussian Splats (2026)" and generic second-order/VarPro claims explicitly listed as NOT defensible.
Since the track is DROP, no novelty claim is made.

## Deliverables
PROFILEDSH_SOURCE_IDENTITY / PROFILEDSH_REPLAY_FIDELITY / PROFILEDSH_INNER_OPT /
PROFILEDSH_OUTER_ORACLE / PROFILEDSH_VERDICT .json per cell, PROFILEDSH_PORTFOLIO_VERDICT.json, plus
PROFILEDSH_REPORT.md (this file). Logs in trackC/*.log.
