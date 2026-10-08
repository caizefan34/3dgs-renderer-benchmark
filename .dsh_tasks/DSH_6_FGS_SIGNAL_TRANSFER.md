# DSH-6 — Strong-Backend Signal Transfer Audit for GLM/AABS

## Goal

Determine whether the signals currently being studied on frozen C0 survive in a MODERN strong
backend (FasterGSFused) before any future integration work is attempted.

This task does NOT implement GLM or AABS.
It is metadata-only instrumentation of FasterGSFused.

## Source

Use exact FasterGSFused:
ae665af2f91a4973bb92c40fdc4cba8d12fe3c96

Use a separate instrumentation worktree/build.

## Scenes

room / garden / bicycle
seed42
corrected holdout protocol

## Part A — GLM signal transfer

Reconstruct a lightweight lifecycle/age signal using stable row/event bookkeeping available in the
FasterGSFused trainer.

At representative phases:
- 5K
- 10K
- 14.5K
- 20K
- 25K

Measure among VISIBLE Gaussians:
- age distribution
- fraction age >=3000
- effective update magnitude / gradient-norm statistics where available
- mature-candidate fraction
- candidate fraction weighted by fused-backward work/intersections if available

Do NOT execute cadence skipping.

## Part B — AABS signal transfer

For the four frozen groups:
- XYZ
- SHAPE
- OPACITY
- APPEARANCE

Record per-visible-Gaussian update/gradient magnitude summaries sufficient to estimate whether mixed
stable/active attribute states still exist under FasterGSFused.

Use the same frozen stability definitions from AABS0 if AABS0 has already frozen them.
If AABS0 is still running, collect RAW statistics only; do not invent new thresholds.

Do NOT suppress any gradients.

## Part C — transferability report

Compare signal distributions qualitatively/numerically to C0:

- GLM mature-candidate fraction
- AABS attribute disagreement/mixed-state fraction
- work-weighted candidate fraction

The goal is not a verdict on quality.
The goal is to answer whether future method integration into FasterGSFused is structurally plausible.

## Required outputs

FGST_ENVIRONMENT.json
FGST_SOURCE_LOCK.json
FGST_RAW_TRACE.json
FGST_GLM_SIGNAL.json
FGST_AABS_SIGNAL.json
FGST_TRANSFER_SUMMARY.json
FGST_REPORT.md
MANIFEST.sha256

## Stop/interpretation rules

- metadata-only instrumentation
- no training-policy changes
- no cadence changes
- no gradient masks
- no quality claim
- no local GPU
- no manual data cherry-picking

Return:
- whether GLM signal survives;
- whether AABS mixed-attribute signal survives;
- whether strong-backend integration is structurally worth attempting;
- artifact paths and manifest verification.
