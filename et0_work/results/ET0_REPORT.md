# EventTrust-GS ET0 - Renderer-State Trust Oracle (REPORT)

- Generated: 2026-09-30 12:04:28 (EPIC-05 A100, remote only)
- Verdict: **ET0_DROP**
- Prereg: EVENTTRUST_ET0_PACKET_20260930 (FROZEN)

## Question
Along a REAL native Adam direction d from theta_0, which better predicts
breakdown of the first-order local model: (A) squared Hellinger distance H,
or (B) renderer-state churn C? If churn is not materially more predictive,
EventTrust is closed.

## Frozen system / identities
- C0 extension SHA256: d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11
- core gsplat_cuda.so SHA256: 935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e
- Cells: room/bicycle @ 10K/15K/20K (8 source + 8 true-holdout cameras each)
- Alpha sweep: 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00
- tau_C = 0.10; tau_H = median H(alpha=1) = 0.003512
- All identity gates passed (fail-closed); checkpoint hashes match frozen values.

## Per-cell summary (8 source-camera probes/cell)
| cell | rho_EC med | delta_rho med | valid AUC | G_C med | truehold dPSNR med |
|------|-----------|---------------|-----------|---------|--------------------|
| room@10000 | 0.774 | 0.000 | 0 | 2.73 | -0.126 |
| room@15000 | 0.452 | 0.000 | 0 | 2.96 | -0.109 |
| room@20000 | 0.357 | 0.000 | 0 | 3.18 | -0.164 |
| bicycle@10000 | 0.940 | 0.000 | 0 | 2.71 | -0.043 |
| bicycle@15000 | 0.429 | 0.000 | 1 | 3.31 | -0.061 |
| bicycle@20000 | 0.548 | 0.000 | 0 | 3.31 | -0.016 |

## Why ET0_DROP
ET0_PROMOTE requires EVERY cell: valid_rho>=6, rho_EC>=0.55, delta_rho>=0.10,
valid_auc>=4, AUC_C>=0.70, delta_AUC>=0.05, G_C>=1.15, truehold dPSNR>=-0.05 dB,
invalid_fraction<=0.10.

- **delta_rho = 0.0 in all 6 cells.** Within each probe, C(alpha) and H(alpha)
  are both monotonically increasing in alpha, so they are rank-equivalent and
  Spearman rho_EC == rho_EH exactly -> delta_rho = 0 < 0.10. Fails the
  delta_rho gate in every cell.
- **valid_auc_count = 0 (5 cells) / 1 (bicycle@15000), all < 4.** The model
  error E(alpha) never reaches the fixed BREAK threshold (E>=0.50) and there
  are no sign failures, so there are no positive BREAK labels; with identical
  labels the ROC-AUC is NOT_DEFINED and excluded. Fails valid_auc>=4.
- invalid_fraction = 0.0 and valid_rho_count = 8 in every cell: the data is
  clean. The DROP is a scientific property of the frozen within-probe design
  (monotone C and H -> rank-equivalent; E below breakdown -> no labels), not a
  measurement failure.

## Interpretation (ET0 s11)
PROMOTE would mean renderer churn is a promising trust signal. DROP closes
EventTrust: churn is NOT materially more predictive than Hellinger along the
native Adam direction in this frozen C0 system.

## Precedence for CR0 (CR0_PREREG s7)
ET0 DROP mechanically caps CellReuse-GS CR0 at **CR0_CONDITIONAL_DROP**,
even if local CR0 numbers look positive. See CR0_REPORT.md.

## Outputs
ET0_SOURCE_IDENTITY.json, ET0_ENVIRONMENT.json, ET0_CAMERA_SEMANTICS.json,
ET0_ADAM_DIRECTIONS.json, ET0_ALPHA_LOSSES.json, ET0_RENDERER_CHURN.json,
ET0_HELLINGER.json, ET0_MODEL_ERROR.json, ET0_TRUE_HOLDOUT.json,
ET0_SUMMARY.json, ET0_VERDICT.json, ET0_REPORT.md, raw logs, ET0_MANIFEST.json.
