# SRR0-P0 Execution Report — Selective Rematerialization Rebalancing

Date: 2026-10-01 (UTC)
Repo: C:\Users\36570\3dgs-renderer-benchmark
Remote identity (EPIC-05, proven before every experimental command):
- hostname: dsw-933622-64cc845c7d-lqswc
- GPU: NVIDIA A100-SXM4-80GB (×8)
- cwd: /root/epic05_p16rb/src/nerficg
- source git SHA: e458ac6e41810ad6d9405687c1e698fadd28dac5
- torch 2.5.1 / cuda 12.4 / python 3.10 (conda env p16_fastergs)

Local Windows = control-plane only. All GPU/CUDA/profiling ran on EPIC-05. No CUDA locally.
Frozen checkpoints only; no altered training.

## Verdict
`DROP_NO_COMPUTE_HEADROOM` (decided at Gate 4).

The FasterGSFused `preprocess_backward` recomputation is **not** a meaningful fraction of the
end-to-end iteration, because the iteration is **launch-bound** (CPU/kernel-launch overhead dominates),
not GPU-compute-bound. Even under the most optimistic assumption — 100% of the recompute removed and
fully on the critical path — only 1 of 3 scenes (garden) reaches the 1.04x E2E ceiling; the prereg
requires ≥2/3.

## Prereg gate sequence (sequential, early-exit)
| Gate | Test | Result |
|------|------|--------|
| 0 | remote only (EPIC-05) | PASS |
| 1 | historical-negative applicability | PASS — `DIFFERENT_TARGET` (F1 = gsplat, different cohort & recompute set); no auto-drop |
| 2 | source lock + freeze R1/R2/R3 | PASS — see SRR0_SOURCE_LOCK.json |
| 3 | prior-art audit | PASS — no direct public collision |
| 4 | target compute headroom | **FAIL → DROP_NO_COMPUTE_HEADROOM** |
| 5 | retained-state cost | NOT REACHED (advisory bytes in SRR0_STATE_BYTES.json) |
| 6 | exactness oracle | NOT REACHED (no altered training; frozen ckpt only) |
| 7 | economics/Amdahl | NOT REACHED (advisory ceiling in SRR0_AMDAHL.json) |

## Gate 4 — measurement and reasoning
Recompute target: `preprocess_backward_cu` (kernels_backward.cuh:15-250), which recomputes per touched
primitive: SH direction/basis backward (Q1), camera projection depth/x/y (Q2), variance=exp(2·raw_scale)
(Q3), R=quat→rot (Q4), 3D covariance/RSS (Q5), EWA Jacobian + 2D covariance (Q6).

Measured on EPIC-05 A100, frozen iso checkpoints (room 1,262,605 / garden 4,809,378 / bicycle 4,720,561
Gaussians, SH-degree 3), 6 views × 8 timing iters, behavior-inert harness (zero-init Adam moments,
native forward+backward, no kernel/checkpoint mutation):

| scene | recompute (ms) | busy GPU (ms) | GPU-stream (ms) | async/iter (ms) | sync wall (ms) |
|-------|---------------:|--------------:|----------------:|----------------:|---------------:|
| room | 1.422 | 36.6 | 139.7 | 130.5 | 124.0 |
| garden | 6.179 | 55.5 | 146.2 | 147.6 | 133.9 |
| bicycle | 3.593 | 50.8 | 145.2 | 130.9 | 124.2 |

Optimistic E2E ceiling = den / (den − recompute):

| scene | async | stream | sync-wall | busy-GPU (INVALID) |
|-------|------:|-------:|----------:|-------------------:|
| room | 1.0110 | 1.0103 | 1.0116 | 1.0405 |
| garden | 1.0437 | 1.0441 | 1.0484 | 1.1253 |
| bicycle | 1.0282 | 1.0254 | 1.0298 | 1.0761 |

Scenes ≥ 1.04x: **1/3 (garden only)**. Required: ≥2/3. → **DROP**.

### Why the iteration is launch-bound (resolves the handoff's open question)
GPU-stream (event) time ≈ 139-146 ms, but summed kernel busy-time ≈ 37-56 ms. The ~74% difference is
idle gaps between kernels (launch latency). Async steady-state (130-148 ms) ≈ sync wall (124-134 ms) ≫
busy compute. Therefore the wall-vs-GPU gap is **real (CPU/launch-bound)**, not a per-iter-sync artifact:
the GPU waits on the host, so removing GPU recompute cannot shorten E2E unless it was on the critical path.

The "busy-GPU" denominator that yields 1.04-1.13× (all pass) is **invalid**: it assumes zero inter-kernel
gaps, which the event/async measurement disproves. An E2E ceiling must be measured against actual
per-iteration time, not summed kernel busy-time.

## Corroboration with internal negative laws
- **LAW-PSC / PSC-GS (CLOSED):** global-3D work ≈ 4% of training; SRR0's recompute is global-3D and
  measures ~1-4% of E2E — consistent.
- **LAW-REPLAY / ReplayGS (CLOSED):** mathematical reusability ≠ GPU-cheap materialization. Even if the
  recompute were free, the retained-state traffic (Gate 5) would offset gains; SRR0_STATE_BYTES.json shows
  only R1 fits the 25% byte budget yet saves negligible compute.

## Decision
No compute headroom ⇒ SRR0 is not worth SRR1 minimal-retention engineering. Frozen verdict:
`DROP_NO_COMPUTE_HEADROOM`. No rescue, no retest (prereg early-exit). No CUDA was run locally; no
training was altered; checkpoints are frozen.

## Deliverables
SRR0_VERDICT.json, SRR0_REPORT.md, SRR0_HISTORY_AUDIT.md, SRR0_SOURCE_LOCK.json, SRR0_PRIOR_ART.md,
SRR0_PROFILE.json, SRR0_STATE_BYTES.json, SRR0_NUMERICAL_ORACLE.json, SRR0_AMDAHL.json, scripts/,
MANIFEST.sha256. Raw remote profiles in _remote/.
