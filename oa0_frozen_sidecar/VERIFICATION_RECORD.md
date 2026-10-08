# OA0 Frozen Analysis Sidecar v1 — Intake Verification Record

Recorded: 2026-09-29 · Status: **VERIFIED_AND_ARCHIVED** (pre-result; no OA0 data exist)

Source: `C:\Users\36570\Downloads\OA0_Frozen_Analysis_Sidecar_v1.zip`
- sha256 `b1d5bbb25f746edda7c147e3aad5151d66daabce59a2bd2c1053f7f470a30ea8`, 111474 bytes
- archived unmodified into this directory; 8/8 manifest SHA256 re-verified after copy.

## 1. Integrity

| file | sha256 (verified) |
|---|---|
| N1D_LABELS_FROZEN.csv | `4188c8743264e7095e50b0851954456152be331d4a144e5b8a8fb43d556e91f2` |
| N1D_LABEL_SUMMARY.json | `c12a38fdad64c475d23e7bc1d85f1e0b6a1197186b0c3142645af74121236a27` |
| README.md | `b06ee9416b9ba781a358f1504100fe0a8ab23374e8318157fdf22611f8dd03f4` |
| UNIT_TEST.json | `f26f14ae58ffbaa79e00fe14d578abf6d88166a72b0d03d2abb9d4273498580c` |
| _smoke_analysis.json | `1fc1c2bf0f91990d720274866dba7c89ac44721a4aa3d44dd9d5d2b448aec42a` |
| _synthetic_features.csv | `10b0884a8b71c74f8cf188f832187784745a9c259baf4ab7dbad872d50ce207c` |
| oa0_analyze_frozen.py | `d81c45c10417e5f1e807d30fcd7ca6cb44a915698cedced463cf98083dda80fb` |
| q0_oa0_bridge_analyze_frozen.py | `7dfa7dadb84720be8fa4d3dfa6cf289a4102cf3840e048c700bb8bb88d0180a3` |

Provenance inputs re-hashed at intake:

| input | sha256 |
|---|---|
| `N1D_RETURN/artifacts/N1D_ACTION_RESULTS.parquet` | `389fc1d81ae61d8919fd122cd4fd9f8f895c32540af3eb6f5400d88c70a892bc` |
| `N1D_RETURN/artifacts/MANIFEST_N1D.json` | `9f0cdd20675382c8be994ec299b3e63cbc156c8260bd8dfc1dcfb28120092bec` |

## 2. Smoke unit test (analyzer mechanics)

Re-ran `oa0_analyze_frozen.py --features _synthetic_features.csv --labels N1D_LABELS_FROZEN.csv`
on 2026-09-29 (python 3.13.13, numpy 2.5.3, pandas 3.0.3, scipy 1.18.0, sklearn 1.9.0).

Reproduces `UNIT_TEST.json` exactly: verdict `OA0_STRONG_OPTIMIZER_PATHOLOGY`,
prevalence {room 0.90625, bicycle 0.6302083333333334}, both transfer ROC-AUC 1.0,
Brier 1.721573520619527e-05 / 3.36188258732024e-06 (identical to float repr).
Synthetic AUC=1.0 is by construction — mechanics only, no scientific content.

## 3. Frozen-label provenance (all 384 rows reproduce bit-exact)

`N1D_LABELS_FROZEN.csv` is an exact deterministic reduction of
`N1D_ACTION_RESULTS.parquet` over the N1D action family
{A0, A1-1.00, A1-0.75, A1-0.50, A1-0.25}. Recovered definitions, validated
0 mismatches per column with `atol=1e-15`:

| column | definition |
|---|---|
| `v1` | N1D anchor `v1` |
| `sigma_scene` | N1 replica noise: room 8.6e-10, bicycle 1.4e-9 (matches MANIFEST_N1D.json) |
| `native_progress` | `delta_fused` of A1-1.00 |
| `damped_best_progress` | max progress over α ∈ {0, 0.25, 0.50, 0.75} (includes A0) |
| `overshoot_gain` | `damped_best_progress − native_progress` |
| `best_alpha` / `best_progress_all_alpha` | argmax / max over all five α incl. 1.00 |
| `native_harmful` | `native_progress < 0` |
| `damping_rescue` | `best_progress_all_alpha > native_progress` — **not** gated on `native_harmful`; 9 of 295 rescues (room 174, bicycle 121) are anchors where native improved but a damped α improved more |

`N1D_LABEL_SUMMARY.json` reproduces exactly from the frozen CSV alone
(prevalence/counts/rates/medians/`best_alpha_counts`/by-checkpoint; the frozen
by-checkpoint blocks omit count and mean fields — not mismatches).

Coverage: 384 rows = room 192 + bicycle 192; 64 per checkpoint per scene
(15K/20K/25K). Frozen best_alpha counts — room: {0:157, .25:8, .5:4, .75:5, 1:18};
bicycle: {0:112, .25:5, .5:1, .75:3, 1:71}.

## 4. Pre-result hazards (flagged before OA0 data return, thresholds untouched)

1. **Q0↔OA0 bridge is dead on arrival with the frozen Q0 verdict.**
   `q0_oa0_bridge_analyze_frozen.py` sets `qok = verdict ∈ {STRONG, MODERATE}`.
   Q0 closed as `Q0_NO_MATERIAL_QUOTIENT_SIGNAL` in all 4 cells
   (`_moonshot/Q0_RESULTS.md`), which is not in `qok` → the bridge emits
   `OA0_Q0_NO_GEOMETRY_LINK` by construction regardless of OA0 feature values.
   Only a Q0 rerun with a STRONG/MODERATE verdict could change this branch.
2. **Group-name join hazard.** Sidecar expects groups
   `xyz, scale, quat, opacity, sh_dc, sh_rest`; Q0 artifacts use
   `xyz, log_scale, quaternion, opacity, sh_dc, sh_rest`. A naive join on
   `group` silently loses scale/quat rows. Bridge table must also supply
   `null_frac` (Q0 `Q0_ADAM_PROJECTION_*.parquet` has per-cluster
   `null_frac` with `kind ∈ {high, control}`, `step ∈ cam_*` → needs
   deterministic aggregation to scene/checkpoint/group).
3. Bridge scope detail: bridge consistency checks use checkpoints 20000 and
   25000 only, while the frozen labels span 15K/20K/25K.
4. Analyzer hard-failure clauses to respect at execution time: all six groups
   must be present (else `SystemExit`); any non-finite value in the four
   primary features is a hard data-quality failure; feature/label join must not
   lose rows (`validate="many_to_one"`, row-count check).

## 5. What this record does not establish

- No OA0 execution has run; the four feature names
  (`lowv_q1_update_energy_share`, `zero_grad_update_energy_share`,
  `moment_conflict_update_energy_share`, `update_top1_energy_share`) appear
  nowhere in this repo. No capture code, no `OA0_ANCHOR_FEATURES.csv`, no OA0
  prereg exists locally as of this date — feature *definitions* must come from
  the OA0 execution spec, not from this sidecar.
- No analyzer threshold was modified; the frozen verdict branches were read,
  not changed.
- Nothing was committed to git.

## 6. Replay

    python oa0_frozen_sidecar/verify_oa0_sidecar.py

Checks: (a) 8/8 manifest hashes, (b) smoke re-run vs UNIT_TEST.json,
(c) 384-label reconstruction vs N1D_ACTION_RESULTS.parquet,
(d) summary reproduction vs N1D_LABEL_SUMMARY.json. Prints PASS/FAIL per check.
