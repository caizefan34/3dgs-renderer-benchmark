# MG0_OneView_GS_HotCV — DSH Return Package

**Verdict (mechanical, frozen prereg §8 gate): `MG0_NO_USEFUL_VARIANCE_SIGNAL`.**
No K ∈ {2048, 8192, 32768} passed the actionable gate in all four cells
(room@20K, room@25K, bicycle@20K, bicycle@25K). Per prereg §8/§9 the one-step
integration test was **not run** (`run_one_step=false`), and
`MG0_ONE_STEP.parquet` / `MG0_B1_NOISE.json` are therefore not produced
(they are conditional deliverables).

## Package

| item | value |
|---|---|
| tarball (in this folder) | `mg0_oneview_hotcv_return_20260929.tar.gz` |
| sha256 | `c294c427c08d123743902ecad771bb91f6a4b827ca874f040e40fe7f1e10c440` |
| bytes | 100,354,242 |
| files in tar | 154 |
| box source | `/root/epic05_mg0` (tar of `artifacts/ controls/ logs/ scripts/`) |

Contents (extracted here):

- `artifacts/` — 44 files: all required `MG0_*.json` deliverables
  (`MG0_SOURCE_IDENTITY`, `MG0_CHECKPOINT_IDENTITY`, `MG0_CAMERA_SET`,
  `MG0_HOTSET_15K/20K/25K`, `MG0_VARIANCE_ROOM/BICYCLE`, `MG0_GROUP_METRICS`,
  `MG0_COST`, `MG0_MEMORY`, `MG0_CONTAMINATION`, `MANIFEST_MG0.json`), plus
  `MG0_GATE.json`, `MG0_VERDICT.json`, `MG0_SPEC_OPERATIONALIZATION.json`,
  per-stage raw JSONs, and the six frozen score arrays `MG0_SCORE_*.npy`.
- `controls/` — 4 stored FP16 control tables (246 MiB each):
  `{room,bicycle}_{15000,20000}_controls.pt` (`hot_ids` [32768],
  `c_fp16` [64, 32768, 59], `cbar` exact FP32 means).
- `logs/` — 77 files: per-stage logs, `pipeline.log`, contamination CSVs
  (`cont_*`, GPU/UUID level), and `attempt1/`, `attempt2/` (pre-recorded
  failed launches, see below).
- `scripts/` — the 9 executed scripts + 2 post-hoc helpers
  (`mg0_diag_shapes.py`, `mg0_pull_report.py`).

## Integrity

- Manifest re-verification over the extracted tree: **126/129 entries match**
  (sha256+bytes). The 3 non-matching entries are self-referential and benign:
  `artifacts/MANIFEST_MG0.json` (hashed before its own rewrite),
  `logs/finalize_2.log`, `logs/pipeline.log` (both grew after the manifest
  snapshot; `pipeline.log` gained its final `DONE` line).
- `MG0_BLOCKER.json` does not exist; Stage 0 gate passed for both scenes
  (`stage0_ok=true`, all bit-exact restore flags true).
- GPU discipline: whole campaign ran on GPU 2 only
  (`bcca92b9-0234-7ec9-a2f0-a63925d52146`, `CUDA_VISIBLE_DEVICES=2`); the
  contamination CSVs show GPUs 0/1 held by other tenants, untouched.

## Headline numbers (details in MG0_REPORT.md)

Four evaluation cells, controls 5,000 iterations stale:

| cell | K | adam_var_ratio | median cos gain | raw_var_ratio |
|---|---|---|---|---|
| room@20000 | 2048 / 8192 / 32768 | 3.195e+05 (all K) | +0.0002 / −0.0002 / −0.0015 | 1.07 / 1.14 / 1.17 |
| room@25000 | 2048 / 8192 / 32768 | 1.006 (all K) | −0.0003 / −0.0007 / −0.0009 | 1.02 / 1.03 / 1.04 |
| bicycle@20000 | 2048 / 8192 / 32768 | 9.144 (all K) | −0.0000 / +0.0001 / −0.0000 | 1.0004 / 1.0004 / 1.0079 |
| bicycle@25000 | 2048 / 8192 / 32768 | 1.000 (all K) | −0.0000 / −0.0000 / −0.0000 | 1.002 / 1.002 / 1.003 |

- Estimator never reduced variance: `raw_var_ratio > 1` in **12/12** cells.
- Cost side was never the binding constraint: isolated control maintenance
  ≤ 0.94 % of native A1 iteration wall; projected full-camera table memory
  0.044–1.127 GiB (cap 1.5 GiB).

## Deviations / notes (full list in MG0_REPORT.md)

1. Two launches precede the recorded run (fixed bugs: `_opacity` row packing;
   a `["global"]` key on the G-norm line). Their logs are archived in
   `logs/attempt1/`, `logs/attempt2/`; the returned artifacts are from the
   completed run (2026-09-29 01:32–01:43 UTC).
2. Mechanical operationalizations where the prereg leaves implementation
   freedom are recorded verbatim in `MG0_SPEC_OPERATIONALIZATION.json`
   (no threshold was changed).
3. The 15K score is extremely concentrated: room — 9 rows carry 50 % of the
   total score, 330 rows 99 %; bicycle — 1 row 50 %, 18 rows 99 %. Recorded
   as an observation; the frozen rule was not altered.
4. No garden, no full training, no K/threshold change, no second fresh view,
   no loss/resolution change, no test-view comparison, no novelty claim.

**Return to ChatGPT and STOP.**
