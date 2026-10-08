# DSH-5 — AdamW-GS / Sparse-Adam Strong Optimizer Baseline Reproduction

## Goal

Reproduce the closest PUBLIC optimizer prior art on EPIC-05 so future GLM/AABS claims are compared
against a real strong optimizer baseline rather than paper numbers.

Official repository:
https://github.com/EliottDJay/3DGSMCMC_AdamWGS

Freeze exact commit before scientific execution.

## Required arms

At minimum, if official configs permit clean isolation:

A. official dense/vanilla optimizer reference in the repository
B. Sparse Adam component
C. full AdamW-GS

If the repository does not expose a clean isolated Sparse-Adam arm:
- do not invent one;
- report unavailable and run the officially exposed closest ablation.

## Scenes

- room
- garden
- bicycle

Seed:
- 42

Two evidence lanes if practical:

1. OFFICIAL protocol reproduction
2. CORRECTED holdout lane with `i % 8 == 0` excluded from training

Never mix metrics across lanes.

## Measure

- full training wall
- optimizer wall if instrumentable without algorithm changes
- peak VRAM
- Gaussian count trajectory
- final Gaussian count
- true-holdout PSNR / SSIM / LPIPS
- visible/updated primitive ratio for Sparse Adam if exposed
- optimizer-state traffic proxy if available

## Scientific purpose

Answer:

- How much speed does official Sparse Adam actually deliver?
- What quality effect does it have?
- How much does full AdamW-GS gain over Sparse Adam alone?
- Is the optimization primitive-level only, or is any per-attribute execution suppression present?

The final question is observational; do not modify the method to create attribute masks.

## Required outputs

AWG_ENVIRONMENT.json
AWG_SOURCE_LOCK.json
AWG_CONFIGS.json
AWG_OFFICIAL_RESULTS.json
AWG_CORRECTED_RESULTS.json
AWG_TIMING.json
AWG_QUALITY.json
AWG_SUMMARY.json
AWG_REPORT.md
MANIFEST.sha256

## Rules

- official code/config first
- no threshold rescue
- no hidden failed scene
- no local GPU
- external baseline only; do not label any AdamW-GS component as ours

## Return

Exact commit, faithful reproduction status, per-arm speed/quality, optimizer-specific observations,
artifact paths, manifest verification.
