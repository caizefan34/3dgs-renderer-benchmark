# Confirmatory Protocol (pre-registered) — 2026-08-06

This file freezes the confirmatory experimental design referenced by
`paper/claims.yaml` `required_gates`. It was written before the
confirmatory runs started; no post-hoc configuration selection is allowed
on the confirmatory seeds.

## 1. Objective

Run the frozen HiGS speed/quality configurations to the standard 30,000-step
convergence horizon on all five canonical scenes, add a held-out dataset
family that was not used in exploratory rounds, collect per-seed values with
paired block-bootstrap 95% confidence intervals, and repeat the frozen
protocol on a consumer GPU. This closes the audit P0/P1 items in
`docs/research-readiness-audit-2026-08-05.md`.

## 2. Frozen configurations (locked 2026-08-06)

Both arms use the round-65 final operating points (reproducible via
`--higs-quality-max` plus explicit overrides) executed by
`benchmark/run_higs_train_benchmark.py` with backend `higs_dynamic_ts`.

Common flags (identical for both arms):

| flag | value |
| --- | --- |
| `--steps` | 30000 |
| `--width` / `--height` | 1920 / 1080 |
| `--n-train` / `--n-eval` | 4 / 3 |
| `--densify-every` / `--densify-threshold` / `--prune-threshold` | 5 / 0.005 / 0.01 |
| `--anchor-densify` / `--anchor-densify-every` | on / 2 |
| `--tile-sampling-ratio` / `--sampling-mode` | 0.35 / error_guided |
| `--error-alpha` / `--error-refresh-every` / `--error-lambda` | 1.0 / 25 / 0.7 |
| `--eval-every` | 300 (100 eval points) |
| `--lr-decay` | 0.1 (decays to 10% over the 30k horizon) |
| `--densify-window` | 1500 |
| `--lpips-loss-weight` / `--lpips-loss-every` / `--lpips-full-res` | 0.1 / 25 / on |
| `--masked-adam` | on (both arms; this is the round-60 speed-max unit) |

Arm-specific flags:

| arm | flags |
| --- | --- |
| ctrl (full-resolution baseline) | none additional |
| pd (progressive-resolution × decay quality-max cell) | `--masked-adam-union-decay 0.99`, `--masked-adam-union-decay-eval-proj`, `--res-schedule <scene factor>:0,1.0:1500` |

Per-scene coarse factors (frozen from round-65): garden
`0.75:0,1.0:1500`; bicycle, bonsai, truck, train `0.5:0,1.0:1500`. The
train pd cell is run for completeness; round-65 found the decay opt-in does
not pay off for low-N train, so train is reported separately and excluded
from the strict-dominance claim.

## 3. Scenes, seeds, randomization

- Scenes: garden, bicycle, bonsai (mipnerf360) and truck, train
  (tanks_and_temples), plus the held-out family (see §6).
- Seeds: 3, 4, 5. Seeds 0/1/2 were used by exploratory rounds; the
  confirmatory seeds were never used to select the configuration.
- Pairing: one ctrl and one pd run per (scene, seed) pair.
- Randomization: arm order inside each pair is randomized with a fixed RNG
  seed (20260806); pairs are assigned to GPU slots round-robin and both arms
  of a pair run back-to-back on the same GPU inside the same launch wave.

## 4. Evaluation rules

- Quality is evaluated on the 3 held-out cameras every 300 steps; the final
  report uses the step-30000 eval.
- Reported metrics: `train_ms` (mean per-step optimizer latency), per-eval
  `wall_s` and `total_wall_s` wall-clock, PSNR, SSIM, LPIPS, final Gaussian
  count, peak VRAM, failure rate.
- Wall clock is measured inside the harness around training + eval; the
  launcher additionally records process wall time and GPU assignment.
- No early stopping; every run executes the full 30k horizon.

## 5. Analysis and success criteria (pre-registered)

- Analysis: `src/scripts/collect_confirmatory_results.py` builds a
  `runs` summary; `src/scripts/bootstrap_analysis.py` computes paired
  per-scene/per-seed deltas, a scene-level block bootstrap (2000 replicates,
  seed 0) 95% percentile CI, and a paired effect size.
- Primary outcome: paired `train_ms` delta (pd − ctrl), lower is better.
- Quality guardrail: the geometric-mean final PSNR delta must not fall below
  −0.05 dB and the LPIPS delta must not exceed +0.005; otherwise the cell is
  reported as a quality regression, not a dominance result.
- "Strict dominance" for a scene is declared only when the per-scene paired
  train_ms delta is negative for all three seeds AND the quality guardrail
  passes for that scene.
- Failure rule: a crashed run is retried up to 2 times; persistent failures
  are recorded in the failure rate and excluded from paired analysis with a
  documented reason.

## 6. Held-out dataset family

- Family: Deep Blending (playroom, drjohnson) — a handheld indoor capture
  family not present in any exploratory round.
- The family is prepared with the pinned official iteration-30000
  checkpoints and cameras from the Graphdeco model archive plus the official
  T&T+Deep-Blending input bundle, using `src/scripts/prepare_datasets.py`.
- Held-out scenes run the identical frozen protocol (30k steps, seeds 3/4/5,
  both arms) and are analyzed separately from the canonical five; they are
  evidence for generalizability, not for per-scene dominance counts.

## 7. Hardware legs

- Primary: EPIC-05, 8 × A100-SXM4-80GB (this matrix).
- Consumer: RTX 5070 Laptop 8GB — same frozen configs; the full 5-scene
  matrix runs at 720p (960×540) because several 1080p cells exceed 8GB VRAM
  (bicycle ctrl peaks ≈14.4GB at 1080p), plus a best-effort 1080p subset on
  VRAM-feasible cells. 720p/1080p cells are analyzed within resolution only.
- Second datacenter GPU: not available in this environment; documented as an
  open gate in the audit with a reproducible recipe.

## 8. Reproducibility

- Harness commit, launcher flags, per-run JSON and logs, the generated
  summary JSON, and the analysis outputs are all committed under
  `results/confirmatory-*` and `paper/tables/`.
- The claims manifest `freeze_status` becomes `frozen` only after the
  required gates pass with this protocol's artifacts as evidence.
