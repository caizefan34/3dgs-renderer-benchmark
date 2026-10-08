# DSH-4 — SkipGS × FasterGSFused Composition Baseline

## Goal

Measure the strongest orthogonal public skip baseline on top of the already reproduced
FasterGSFused system.

Compare:

A. FasterGSFused
B. FasterGSFused + official SkipGS

This is an EXTERNAL BASELINE experiment, not our method.

Official SkipGS:
https://github.com/ASU-ESIC-FAN-Lab/SkipGS

Freeze the exact SkipGS commit before running.

## Source identities

FasterGSFused:
- commit ae665af2f91a4973bb92c40fdc4cba8d12fe3c96

NeRFICG:
- e458ac6e41810ad6d9405687c1e698fadd28dac5

SkipGS:
- record exact branch + full commit SHA.

## Protocol

Scenes:
- room
- garden
- bicycle

Seed:
- 42

Training:
- 30K
- corrected holdout: original camera index `i % 8 == 0`
- holdout cameras excluded from training
- identical FasterGSFused configuration between A/B

Use SkipGS official default logic/settings unless an adapter is strictly required.

No threshold tuning after seeing results.

## Instrumentation

For every post-15K iteration save compact records:

- iteration
- camera/view ID
- current loss
- EMA/reference used by SkipGS
- skip decision
- backward executed yes/no
- optimizer step yes/no
- iteration wall
- backward+optimizer wall if available

Also save:
- total skip ratio
- backward ratio
- per-view skip distribution

## Metrics

Per scene:

- full 30K wall
- 15K→30K wall
- speedup A vs B
- peak VRAM
- final Gaussian count
- corrected true-holdout PSNR
- corrected true-holdout SSIM
- corrected true-holdout LPIPS
- skip ratio
- backward ratio

## Fail-closed rules

- no threshold shopping
- no hidden scene exclusions
- no changing FasterGSFused quality settings
- no local GPU
- timing only on uncontended A100
- keep failed quality scenes

## Required outputs

SGF_ENVIRONMENT.json
SGF_SOURCE_LOCK.json
SGF_CAMERA_SPLIT.json
SGF_CONFIG.json
SGF_DECISION_TRACE.json
SGF_TIMING.json
SGF_QUALITY.json
SGF_SUMMARY.json
SGF_REPORT.md
MANIFEST.sha256

## Return

1. FasterGSFused + SkipGS incremental speedup per scene.
2. Quality delta per scene.
3. Skip/backward ratio.
4. Whether the composition is truly additive.
5. Preserve iteration decision trace for later AABS orthogonality analysis.
6. Artifact paths + manifest verification.
