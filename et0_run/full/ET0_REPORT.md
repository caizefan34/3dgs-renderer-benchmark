# EventTrust-GS ET0 — Renderer-State Trust Oracle (FROZEN) — REPORT

**Verdict: ET0_DROP**  (analyzer-owned, ET0_PREREG.md S10/S11)

## 1. Execution environment
- Compute: remote **EPIC-05** A100-80GB x8 (`8.130.30.251:1024`). **No local GPU used** (per run goal).
- Python env: `/root/miniforge3/envs/gsplat/bin/python`; Python 3.10.21; torch 2.9.1+cu128; CUDA available.
- Frozen renderer: C0 ext SHA256 `d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11`, core `935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e` — all matched.
- No production CUDA modified; renderer driven read-only via the A0/IR0 capture patch. No densification. No optimizer change.

## 2. Identity gate (fail-closed)
| cell | ckpt_sha256 | num_points |
|---|---|---|
| room@10000 | `9aa00f0ddf28b6c097e3dc8f41dbb107cd9bf175750503dc330aa1d6ab579fc4` | 710,652 |
| room@15000 | `b276b338f5a444d8ea1be390d21fd3e67679b7d862652f0babd131db9434e5dd` | 841,781 |
| room@20000 | `c4e41cfef79e50980406fc1fa148af8d04bc8b37b801c36da5cddb2ce8058718` | 841,781 |
| bicycle@10000 | `26bf0b959f10f0ce526c496d0dc7ca8e62b0ba93efc83d58b7f9440d58c2f4e9` | 2,960,907 |
| bicycle@15000 | `273971d3867c1d2a08b7246803265539d74756b8eca4e3f01c6a8d149b7fd6f7` | 3,654,566 |
| bicycle@20000 | `96f136cb331a19900a41a516d53314d522b4de5a3250ddbba8a232b610b398f0` | 3,654,566 |

All 6 checkpoint hashes independently re-verified with `sha256sum` and via `a0_lib.load_model`; all match the authoritative IR0 record. C0 ext + core `.so` pins matched `bootstrap_c0()` for every cell.

## 3. Protocol
- Cells: room@10K/15K/20K, bicycle@10K/15K/20K.
- Source cameras: 8 deterministic midpoint-stratified train cams/cell (exclude `i%8==0`). True-holdout: 8 midpoint-stratified `i%8==0` cams/cell.
- Direction: one exact native Adam step on a clone; `d = theta_adam - theta_0` (NOT raw gradient).
- Alpha sweep: `{0.25,0.50,0.75,1.00,1.50,2.00,3.00,4.00}`; `theta(a)=theta_0+a*d`.
- Churn (frozen): `C = 0.5*C_tile + 0.5*C_order`; `tau_C = 0.1`. Churn key multiplier fixed at `base = N+1`.
- Hellinger (frozen): squared Hellinger between N(mu0,S0),N(mu1,S1) in fp64; `H = mean_i H2_i`. SH/opacity excluded.
- tau_H (cohort comparator): median `H(a=1)` over all 48 probes = `3.511539e-03`.
- Breakdown: `E(a) >= 0.50` OR sign-failure. Predictors C(a), H(a); ROC-AUC when both labels present.

## 4. Per-cell aggregate (8 source probes each)
| cell | rho_EC_med | delta_rho_med | valid_AUC | AUC_C_med | delta_AUC_med | G_C_med | G_C_p10 | holdout_dPSNR_dB | invalid_frac |
|---|---|---|---|---|---|---|---|---|---|
| room@10000 | 0.774 | 0.000 | 0/8 | n/a | n/a | 2.734 | 2.320 | -0.1257 | 0.000 |
| room@15000 | 0.452 | 0.000 | 0/8 | n/a | n/a | 2.961 | 2.690 | -0.1088 | 0.000 |
| room@20000 | 0.357 | 0.000 | 0/8 | n/a | n/a | 3.177 | 2.687 | -0.1642 | 0.000 |
| bicycle@10000 | 0.940 | 0.000 | 0/8 | n/a | n/a | 2.706 | 2.056 | -0.0430 | 0.000 |
| bicycle@15000 | 0.429 | 0.000 | 1/8 | 1.000 | 0.000 | 3.313 | 2.630 | -0.0610 | 0.000 |
| bicycle@20000 | 0.548 | 0.000 | 0/8 | n/a | n/a | 3.310 | 3.038 | -0.0162 | 0.000 |

## 5. Interpretation — why ET0_DROP
1. **delta_rho = 0.000 in every cell.** Along a single native Adam ray, both renderer-state churn `C(a)` and Gaussian Hellinger `H(a)` are strictly monotone in the scalar step `alpha`. They are therefore rank-equivalent, so Spearman `rho_EC == rho_EH` identically and `delta_rho = 0` by construction. Churn carries **no incremental rank information** about first-order model error over Hellinger.
2. **No first-order breakdown labels.** `E(a) < 0.50` for every probe and no sign-failures occurred (`n_positive_labels = 0` in 5/6 cells; 1/8 in bicycle@15000). With a single class, ROC-AUC is NOT_DEFINED, so `valid_auc_count = 0` in 5/6 cells. The comparison the protocol needs (AUC_C vs AUC_H) is undefined.
3. **Churn threshold never binds.** `C(a)` never exceeds `tau_C = 0.10` even at `alpha = 4.0` (max C ~0.010), so `alpha_C = 4.0` (the full sweep) in every probe. The churn-derived safe step is unconstrained and overshoots: `G_C ~ 2.7-3.3` (source loss keeps dropping) but true-holdout PSNR delta is **negative** in all 6 cells (-0.016 to -0.164 dB), failing the S10 PSNR gate.
4. **Hellinger comparator behaves identically.** `alpha_H` (tau_H-calibrated) also lands at 0.25-0.5; both signals select comparably over-conservative steps, again giving no advantage to churn.

Conclusion: renderer-state churn is **not** a materially better predictor of first-order model breakdown than squared Hellinger distance along the native Adam direction. Per ET0_PREREG S11, **EventTrust is closed** and CR0 is not promoted. No forbidden rescue was used (no new alphas, no churn-weight/tau_C tuning, no Hellinger redefinition, no production CUDA).

## 6. Required outputs (ET0_PREREG S13)
| artifact | sha256 |
|---|---|
| ET0_SOURCE_IDENTITY.json | `1d0335a3b06e9a37c2c14e7b63a2dd1c4a5c41df5d5d2795a656516dc691fe0c` |
| ET0_ENVIRONMENT.json | per-cell `ET0_ENVIRONMENT_<cell>.json` (6 files) |
| ET0_CAMERA_SEMANTICS.json | `52c92b6f52d897628f27cac6b179ab346b94e30504d40bd4dc6b01c3867b0eff` |
| ET0_ADAM_DIRECTIONS.json | `c2705de9ed376c9bf7617cf44d5d8596781a189954a2dd8a4b3b0a3b83a5cc82` |
| ET0_ALPHA_LOSSES.json | `11a7543a503dcd00cdff34f7ef038d5a5d8cf43b75d085228054b441e2ab1e06` |
| ET0_RENDERER_CHURN.json | `302b6e05e5b36d2881e82f7cde4251697e16d6c42d7a62e582728b3ec76e8e5e` |
| ET0_HELLINGER.json | `e7cc6862f6b6c38c4af7f52a641f87c898e38b202d583c65914493a352265caa` |
| ET0_MODEL_ERROR.json | `570caf5a5d981d3b680a37e8c1eca52f68c273016ffa7bbbcceadccb8c6b2c41` |
| ET0_TRUE_HOLDOUT.json | `08a2c5b815ea53f1fab8dfc59e87d7cd26cc2d78c52a3a5cc68be452b43003bf` |
| ET0_SUMMARY.json | `326fc54101ef006b968337c7f65663c235e374c819b0136c8e3addaa2403d9d2` |
| ET0_VERDICT.json | `3c25662841b7628aecab9b07e26bd92fcf93ffe10752397aa91715d5d3a4a11e` |
| ET0_REPORT.md | (this file) |

Per-cell files: `ET0_ENVIRONMENT_<cell>.json`, `ET0_SWEEP_<cell>.json`, `ET0_H1_<cell>.json`, `ET0_SAFESTEP_<cell>.json` (6 cells each). Raw logs: `sweep_<cell>.log`, `safestep_<cell>.log`, `aggregate.log`, `verdict.log`. Full SHA256 inventory in `ET0_ARTIFACT_MANIFEST.json` (35 JSON artifacts).

---
Generated by `et0_finalize.py` + `et0_analyze.py` (frozen packet analyzer). Analyzer owns the verdict; no manual override.
