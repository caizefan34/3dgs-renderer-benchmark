# MG0 — OneView-GS / HotCV — Execution Report (DSH)

Frozen prereg: 2026-09-29 (`MG0_PREREG.md`). Executed as an execution engine;
no redesign, no verdict reinterpretation. All Booleans computed mechanically
from the frozen §8 formulas (`MG0_GATE.json`).

- Box: EPIC-05, GPU 2 only (`bcca92b9-0234-7ec9-a2f0-a63925d52146`,
  A100-SXM4-80GB, `CUDA_VISIBLE_DEVICES=2`).
- Recorded run: 2026-09-29 01:32:30–01:43:58 UTC (`logs/pipeline.log`);
  two earlier launches aborted on code bugs and were re-run from scratch
  (see Deviations).
- Return package: `return/mg0_oneview_hotcv_return_20260929.tar.gz`
  sha256 `c294c427c08d123743902ecad771bb91f6a4b827ca874f040e40fe7f1e10c440`.

## Stage 0 — identity / restore audit (PASS)

All frozen file hashes verified:

| artifact | sha256 (match) |
|---|---|
| P12 trainer | `675b0267…` |
| N1 save-only snapshot trainer | `49a79538…` |
| c0_ext .so | `d26cccf4…` |
| core gsplat .so | `935ea6c0…` |
| fused-ssim .so (P11 lock) | `dd6a4c79…` |
| HIGS renderer module | `d599aae3…` |
| refv1 gaussian_model / config / colmap_reader | `68731e37…` / `ea8e8704…` / `b30a2a34…` |
| phase7 dataset.py | `e6d8bfb1…` |

Per scene and checkpoint (15K/20K/25K): exact Gaussian tensors (room N=841,781;
bicycle N=3,654,566 — constant across checkpoints), Adam first/second moments,
group step values 14856 / 19856 / 24856 (5 groups, all equal), active SH
degree 3, `reload_bit_exact`, `restore_after_random_perturbation_bit_exact`,
`restore_after_real_action_bit_exact` — all true. Loss path re-verified:
0.8·L1 + 0.2·(1−SSIM) (fused, padding=same) at 1080p with the frozen c0
render branch (eps2d=0.3). Camera ordering: GTDataset index space ==
cameras.json order == training stream (311/311 and 194/194 matched).
`MG0_BLOCKER.json` not written — exact state was available.

## Stage 1 — frozen camera sets (64/scene, before any gradient)

Rule `j_k = floor((k+0.5)·N/n)`: room (311 cams) ids `[2, 7, …, 308]`;
bicycle (194 cams) ids `[1, 4, …, 192]`. Identical cameras at all three
checkpoints. `anchors_32` (odd positions) and disjoint `internal_eval_16`
frozen here, before gradients. → `MG0_CAMERA_SET.json`.

## Stages 2–4 — hot controls and estimator evaluation

Hot sets K=2048/8192/32768 by descending fp64 score
`score_j = mean_i ||P_t(g_i,j) − mean_v P_t(g_v,j)||²`,
`P_t(x)=x/(sqrt(v_t)+eps)` (per-group eps), two-pass fp64 streaming.
FP16 control tables stored at 15K and 20K (raw gradient rows of the 64 views,
H_32768 prefixes, exact FP32 means `c_bar`). Estimator arithmetic:
`v = (g − c_i) + c_bar` in fp32 on the hot rows only.

**Score concentration (observation, 15K):** room — 9 rows carry 50 % of the
total score, 330 rows 99 % (single largest row 22.1 %); bicycle — 1 row 50 %,
18 rows 99 % (largest 57.3 %). `n_nonfinite = 0` in both scenes.

Four cells (controls 5,000 iterations stale; no metric stored at 15K):

| cell | K | adam_var_ratio | median cos gain | p10 gain | pos-dot | pos-dot raw | raw_var_ratio | norm ratio to G |
|---|---|---|---|---|---|---|---|---|
| room@20000 | 2048 | 3.195e+05 | +0.0002 | −0.0066 | 1.000 | 1.000 | 1.0724 | 5.10 |
| room@20000 | 8192 | 3.195e+05 | −0.0002 | −0.0090 | 1.000 | 1.000 | 1.1400 | 5.10 |
| room@20000 | 32768 | 3.195e+05 | −0.0015 | −0.0131 | 1.000 | 1.000 | 1.1677 | 5.13 |
| room@25000 | 2048 | 1.006 | −0.0003 | −0.0013 | 1.000 | 1.000 | 1.0176 | 5.21 |
| room@25000 | 8192 | 1.006 | −0.0007 | −0.0060 | 1.000 | 1.000 | 1.0296 | 5.24 |
| room@25000 | 32768 | 1.006 | −0.0009 | −0.0139 | 1.000 | 1.000 | 1.0412 | 5.38 |
| bicycle@20000 | 2048 | 9.144 | −0.0000 | −0.0001 | 1.000 | 1.000 | 1.0004 | 5.67 |
| bicycle@20000 | 8192 | 9.144 | +0.0001 | −0.0000 | 1.000 | 1.000 | 1.0004 | 5.67 |
| bicycle@20000 | 32768 | 9.144 | −0.0000 | −0.0009 | 1.000 | 1.000 | 1.0079 | 5.67 |
| bicycle@25000 | 2048 | 1.000 | −0.0000 | −0.0001 | 0.969 | 0.969 | 1.0015 | 6.47 |
| bicycle@25000 | 8192 | 1.000 | −0.0000 | −0.0003 | 0.969 | 0.969 | 1.0020 | 6.47 |
| bicycle@25000 | 32768 | 1.000 | −0.0000 | −0.0008 | 0.969 | 0.969 | 1.0032 | 6.47 |

- `raw_var_ratio > 1` in 12/12 cells → the stale-control correction did not
  reduce raw gradient variance anywhere; Adam-coordinate variance was equal
  (ratio ≈ 1.0) or worse (up to 3.2e+05).
- room@20000 breakdown (K=32768 per group): the Adam blow-up localizes to the
  SH groups (`sh_dc` 3.193e+05, `sh_rest` 3.195e+05) driven by rows with
  near-zero second moments; `xyz`/`scale`/`quat`/`opacity` ratios 1.57–2.33.
- Full per-group, per-view detail: `MG0_VARIANCE_ROOM.json`,
  `MG0_VARIANCE_BICYCLE.json`, `MG0_GROUP_METRICS.json`;
  frozen rankings: `MG0_HOTSET_15K/20K/25K.json`, `MG0_SCORE_*.npy`.

## Stage 5 — isolated control-maintenance cost

18 cells (2 scenes × 3 checkpoints × 3 K), 10 warmups + 1000 timed repeats
(CUDA-event pairs + host wall), isolated buffer ops only — no renderer, no
optimizer step, no parameter update.

- Event-median per op: 0.090–0.094 ms (K=2048/8192), 0.115–0.117 ms (K=32768).
- Normalized to native A1 wall medians (room 12.43/15.17/15.44 ms,
  bicycle 30.20/32.32/28.14 ms): **max ratio 0.0094 (0.94 %)**, host-wall
  max 0.0097 — far under the 10 % / 15 % caps.
- Bytes per op at K=32768: 46,399,488 B (read FP16 c_i + FP32 c_bar + FP32
  current rows; write FP32 temp + FP16 store + c_bar update).
- Peak additional GPU memory ≤ 29.9 MB; analytic resident buffers 37.1 MB.
- Projected full-camera table memory (FP16 rows + FP32 c_bar + ids):
  room 0.070 / 0.282 / 1.127 GiB; bicycle 0.044 / 0.177 / 0.706 GiB
  (K=2048/8192/32768) — all ≤ 1.5 GiB.
  → `MG0_COST.json`, `MG0_MEMORY.json`.

## Gate (mechanical, §8) and verdict

Per-K over the four cells, the strong gate (≤0.50 / ≥+0.12 / pos-dot not
decreased / ≤1.5 GiB / ≤10 %) and actionable gate (≤0.70 / ≥+0.07 / ≤1.5 GiB /
≤15 %) both fail for every K — decisively on `adam_var_ratio` and
`median_cosine_gain`; memory and cost conditions pass everywhere.

`MG0_GATE.json`: `verdict = MG0_NO_USEFUL_VARIANCE_SIGNAL`, `K_star = null`,
`run_one_step = false`.
`MG0_VERDICT.json`: **`MG0_NO_USEFUL_VARIANCE_SIGNAL`** — "no K <= 32768 passed
the actionable gate; STOP OneView-GS; no trainer implemented."

Stage 6 (one-step B1/B2 integration) was not executed, per §8/§9;
`MG0_ONE_STEP.parquet` and `MG0_B1_NOISE.json` are not produced (conditional).

## Deviations / notes

1. **Launch history.** Attempt 1 (01:21 UTC) aborted in Stage 0: row packing
   assumed `_opacity` as [N,1] but the model keeps it [N] (`logs/attempt1/`).
   Attempt 2 (01:24 UTC) aborted at the last write of `mg0_variance room 20000`
   on a missing `["global"]` key in the G-norm summary line
   (`logs/attempt2/`). Both fixed; the recorded run (attempt 3) re-ran every
   stage from scratch, and all returned artifacts are from that run.
2. **Operationalizations** (recorded in `MG0_SPEC_OPERATIONALIZATION.json`,
   no threshold changed): row layout xyz(3)|scale(3)|quat(4)|opacity(1)|
   sh_dc(3)|sh_rest(45); two-pass fp64 score; `c_bar` = fp32 mean of the
   stored fp16 values; positive-dot = fraction of `dot(v_i,G)>0` (global);
   "median improvement beyond noise" = median paired diff > max subset
   B1-repeat span; "no catastrophic instability" = all params finite and
   per-group median ‖Δθ‖_B2 ≤ 3× median ‖Δθ‖_B1; cost normalization vs
   N1 `{ckpt}_A1` total_wall_ms.median (event-median ratio primary, host-wall
   ratio also reported); no new control store at 25K (no 30K consumer).
3. **Manifest self-reference.** `MANIFEST_MG0.json`, `logs/finalize_2.log`,
   `logs/pipeline.log` hash-mismatch against the manifest snapshot because
   they were written/appended after the snapshot; all other 126 entries verify.
4. **Post-finalize helpers.** `scripts/mg0_diag_shapes.py` and
   `scripts/mg0_pull_report.py` were added after `finalize_2`, so they are in
   the tar but not in `MANIFEST_MG0.json`.
5. **Scope compliance.** No garden; no full training; K values unchanged
   (2048/8192/32768); thresholds unchanged; no additional compression scheme;
   no second fresh current view in HotCV; loss/resolution unchanged
   (0.8·L1+0.2·(1−SSIM), 1080p, frozen c0 renderer); no controller training;
   training views only (no formal test views); no novelty or publication
   claim. GPU 2 only; GPUs 0/1 held by other tenants and untouched
   (contamination CSVs).

**Return to ChatGPT and STOP.**
