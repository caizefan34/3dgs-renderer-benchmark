# C0-QUALITY-STABILITY0-R — Correction Log

Non-destructive report correction of C0_QUALITY_STABILITY0_MX. Every changed statement is listed as ORIGINAL / PROBLEM / CORRECTED / EVIDENCE.

## Correction 1

ORIGINAL:

>
Pairs passing all frozen gates

PROBLEM:

>
Original QUALITY_REPORT.md stated 6/9 pass, 3/9 fail.

CORRECTED:

>
RECOUNT from authoritative QUALITY_30K_METRICS.json shows only 3/9 pass all three frozen thresholds; 6/9 fail (3 of which fail all three). The original count was a REPORT_ACCOUNTING_ERROR.

EVIDENCE:

>
3/9 pass-all; 6/9 fail-any; 3/9 fail-all-three; 1/9 catastrophic.

## Correction 2

ORIGINAL:

>
Catastrophic pair count

PROBLEM:

>
Original report narrative was ambiguous about how many pairs breach the -0.5 dB catastrophic threshold.

CORRECTED:

>
Exactly 1 pair (garden seed43, dPSNR = -0.5843 dB) is below the -0.5 dB catastrophic threshold.

EVIDENCE:

>
N_catastrophic = 1

## Correction 3

ORIGINAL:

>
'No systematic negative effect'

PROBLEM:

>
Original statistical wording risked reading 'mean CI crosses zero' as 'no systematic negative effect'.

CORRECTED:

>
CORRECTED: failure to establish a mean degradation is NOT evidence of quality equivalence / non-inferiority. The two are explicitly distinguished.

EVIDENCE:

>
QUALITY0R_STATISTICAL_INTERPRETATION.json allowed/forbidden wording + distinction block

## Correction 4

ORIGINAL:

>
garden interpretation

PROBLEM:

>
Original prose: 'garden drives nearly all the negative signal'.

CORRECTED:

>
CORRECTED: garden contains BOTH the largest negative delta (garden s42 -0.4963) AND the largest positive response (garden s44 +0.4686). Preferred: 'garden dominates the variance and both tails of the quality response'.

EVIDENCE:

>
garden dPSNR range [-0.5843, +0.4686]; non-garden range [-0.0854, -0.0097]; garden spread 1.053 dB vs non-garden 0.076 dB.

## Correction 5

ORIGINAL:

>
fixed-state section

PROBLEM:

>
Original 'Fixed-State Numerical Equivalence' block mixed iter15000/iter30000 loss rel-diff values into fixed-state evidence.

CORRECTED:

>
CORRECTED: iter15000 (0.0859) and iter30000 (0.0401) loss rel-diff are TRAJECTORY-STATE comparisons (independently trained states) and are REMOVED from the fixed-state block; reclassified under TRAJECTORY_DIVERGENCE. Only same-state tests remain as fixed-state evidence.

EVIDENCE:

>
QUALITY0R_STATE_CLASSIFICATION.json

## Correction 6

ORIGINAL:

>
algebraic exactness wording

PROBLEM:

>
Original exactness justification derived ALGEBRAICALLY_EXACT partly from identical binary hash.

CORRECTED:

>
CORRECTED: algebraic exactness is supported by the DERIVATION of the F9/Scalar/H8 transforms; binary identity (9baf8655) only establishes same substrate/implementation family. The two claims are separated.

EVIDENCE:

>
QUALITY0R_EXACTNESS_AUDIT.json layers.ALGEBRAICALLY_EXACT

## Correction 7

ORIGINAL:

>
quality-equivalence wording

PROBLEM:

>
Original FULL_TRAINING_QUALITY_EQUIVALENT = DERIVED_FROM_MULTI_SEED_VERDICT (ambiguous support).

CORRECTED:

>
CORRECTED: NOT_ESTABLISHED. Quality equivalence is not inferred merely because the mean CI crosses zero; frozen non-inferiority gates are not consistently passed (3/9).

EVIDENCE:

>
FULL_TRAINING_QUALITY_EQUIVALENT = NOT_ESTABLISHED; QUALITY_NEUTRALITY_NOT_ESTABLISHED

## Correction 8

ORIGINAL:

>
CVPR viability / paper-role wording

PROBLEM:

>
Original campaign framed C0 speed (1.0669x B1A->C0) and left C0's performance role implicit.

CORRECTED:

>
CORRECTED: fair same-parent A0->C0 speed is ~1.09% (GM 1.01089); quality neutrality NOT established; C0 is a SUPPORTING exact-adjoint/dataflow contribution, not the main performance headline. Evidence quality = ACCEPTABLE, contribution strength = WEAK (kept separate).

EVIDENCE:

>
QUALITY0R_PUBLICATION_CLAIMS.json

## Terminal status

COMPLETE_REPORT_CORRECTION

## Preserved (not changed)

OFFICIAL_CAMPAIGN_VERDICT = QUALITY_SEED_SENSITIVE (frozen, preserved)
QUALITY_SEED_SENSITIVE_WITH_NEGATIVE_DIRECTIONAL_BIAS (added publication-facing characterization, NOT a replacement)