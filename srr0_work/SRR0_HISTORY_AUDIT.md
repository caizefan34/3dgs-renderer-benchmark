# SRR0 History Audit — Gate 1 (historical-negative applicability)

Date: 2026-10-01 (UTC)
Remote identity: dsw-933622-64cc845c7d-lqswc / A100-SXM4-80GB / source SHA e458ac6e41810ad6d9405687c1e698fadd28dac5

## Question
Does the sealed historical negative (F1 forward→backward state reuse) directly cover the
current FasterGSFused `preprocess_backward` recomputation target and cohort?

## Evidence inspected
- `results/a100/c17-c33/phase-c26/f1_state_reuse.json`
- `reports/phase-c26/c26_candidate_tournament.md`

## F1 (historical) — what it actually evaluated
| attribute | F1 (historical) | SRR0 (current target) |
|-----------|-----------------|-----------------------|
| backend | **gsplat** rasterization | **FasterGSFused** fused CUDA backend |
| cohort | room, 1,593,376 Gaussian, SH-degree 3, camera 5 | room 1,262,605 / garden 4,809,378 / bicycle 4,720,561, SH-degree 3 |
| recompute found | SH **color** recompute only (minor) | SH direction/basis + scale→variance + quat→rot + 3D covariance/RSS + camera projection + EWA Jacobian/2D-cov |
| saved-state traced | radii, means2d, depths, conics, opacities, tiles_per_gauss, isect_ids, flatten_ids, isect_offsets | n_touched_tiles, offset, screen_bounds, mean2d, conic_opacity, color (+ sort keys) |
| R (repeated/reconstructed ratio) | 0.269 | not applicable (different recompute set) |
| verdict | DROP — "backward does not substantially recompute forward work; SH color recomputation is minor" | — |

## Classification
**DIFFERENT_TARGET**

Reasoning:
1. **Different backend.** F1 measured `gsplat.rasterization`; SRR0 targets `FasterGSFused`'s fused
   `preprocess_backward_cu`. These are distinct CUDA implementations with different saved-state and
   different recompute sets.
2. **Different recomputed quantities.** F1's negative rests on SH-**color** recompute being minor and
   intersection state being fully saved. SRR0's recompute set is the global-3D transform/covariance/
   projection math (Q1–Q6 in SRR0_SOURCE_LOCK.json) — a different cost structure that F1 did not measure.
3. **Different cohort.** F1 used a 1,593,376-Gaussian room; SRR0 uses frozen iso checkpoints at
   1,262,605 / 4,809,378 / 4,720,561.

## Gate 1 result
`DIFFERENT_TARGET` → **NOT** `SAME_TARGET_SAME_BACKEND`. The historical negative does **not**
auto-drop SRR0. Per prereg, we do not reinterpret the old failure as success, nor do we claim it covers
the new target. Proceed to Gate 2 (source lock).

## Caveat
The F1 mechanism ("SH recompute is minor, intersection state fully saved") is *directionally* consistent
with SRR0's Gate 4 finding that the recompute is a tiny fraction of a launch-bound iteration. But
consistency is not coverage: F1 did not evaluate the FasterGSFused covariance/projection recompute, so it
cannot be used to drop SRR0. It is recorded as corroborating context only.
