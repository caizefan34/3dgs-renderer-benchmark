# DSH-3 — FGS-H0: FasterGSFused Strong-Backend Headroom Audit

## Goal

Profile the already reproduced FasterGSFused backend on EPIC-05 A100 and determine where meaningful
training headroom still exists.

This is a PROFILING/ORACLE task only.
Do not propose or implement a new method.

## Source lock

Use the exact already-reproduced FasterGSFused source identity:
- branch: FasterGSFused
- commit: ae665af2f91a4973bb92c40fdc4cba8d12fe3c96
- NeRFICG: e458ac6e41810ad6d9405687c1e698fadd28dac5

Do not change algorithmic settings.

## Scenes

- room
- garden
- bicycle

Seed:
- 42

Use the corrected `i % 8 == 0` holdout split used in the strong baseline reproduction.

Authoritative profiling must use an uncontended EPIC-05 A100.

## Stage-A profile

Measure phase-weighted training wall for:

1. projection / preprocess
2. key generation / sorting / binning
3. forward blending
4. loss
5. fused backward + optimizer
6. densification / topology
7. checkpoint/logging overhead
8. other

Report separately for:
- 0–5K
- 5–15K
- 15–30K
- full 30K weighted total

Use CUDA events or the framework's authoritative GPU timing where possible.
Do not use Python wall time for sub-kernel attribution.

## Stage-B fused backward+optimizer decomposition

Using a separate timing/instrumentation build, decompose the fused stage as far as source structure
permits:

- shared compositing / common backward math
- XYZ gradient/update
- scale+rotation gradient/update
- opacity gradient/update
- SH gradient/update
- optimizer-only residual/state update
- memory movement / writeback
- unavoidable shared work

If exact per-attribute wall isolation is impossible because of fusion:
- use compile-time ablation variants in a separate module;
- include all common work;
- explicitly label marginal timing as DIFFERENTIAL, not additive.

Do not mutate the verified production FasterGSFused build.

## Stage-C zero-cost upper bounds

Compute full-training Amdahl upper bounds for:

- ZERO_LOSS
- ZERO_SORT
- ZERO_DENSIFICATION
- ZERO_FUSED_OPTIMIZER_ONLY
- ZERO_SH_BRANCH
- ZERO_GEOMETRY_BRANCH
- ZERO_OPACITY_BRANCH
- ZERO_ALL_ATTRIBUTE_SPECIFIC
- ZERO_FULL_FUSED_BACKWARD_OPTIMIZER

For each:
- removable fraction
- maximum theoretical full-training speedup

## Output

FGSH0_ENVIRONMENT.json
FGSH0_SOURCE_LOCK.json
FGSH0_STAGE_PROFILE.json
FGSH0_FUSED_BREAKDOWN.json
FGSH0_AMDAHL_ORACLES.json
FGSH0_SUMMARY.json
FGSH0_REPORT.md
MANIFEST.sha256

## Return

Answer:

1. What is the dominant remaining cost in FasterGSFused?
2. What absolute full-training headroom remains for:
   - GLM-like cadence reduction?
   - AABS-like attribute branch suppression?
3. Which branch has the highest measured marginal cost?
4. Is there any path with >=5% realistic E2E headroom?
5. Artifact paths + manifest verification.

Do not optimize anything.
