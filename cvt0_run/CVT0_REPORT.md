# CVT-GS CVT0 — Cross-View Transfer-Oracle Report

**Verdict: CVT0_DROP** (owned by `cvt0_analyze.py`, the frozen analyzer)

## 1. Question
For a fixed model state, does the choice of which single training camera
generates the next Adam step matter for OTHER views, and does a peer-view
transfer oracle beat (a) uniform-random source selection and (b) the
highest-current-loss source? CVT0 is an oracle/opportunity study only; it does
NOT implement a production camera scheduler.

## 2. Execution
- Host: dsw-933622-64cc845c7d-lqswc (remote A100-SXM4-80GB; **never local GPU** per directive).
- Torch 2.9.1+cu128, CUDA True, python 3.10.21.
- Frozen C0 source: c0_ext `.so` = `d26cccf46c7932bfb5fd44ee...`,
  core `gsplat_cuda.so` = `935ea6c0dd04f34e26407afd...` — both match the
  authoritative pins (`CVT0_SOURCE_IDENTITY.json`, all_match_prereg=true).
- Native loss = **fused-ssim** (padding=same, so `dd6a4c79...`), the exact
  training loss that produced the frozen n1snap checkpoints
  (`final_status.json` loss=fused); L = 0.8*L1 + 0.2*(1-SSIM); eps2d=0.3,
  tile16, black background, topology frozen.
- One exact native Adam step per source camera from an identical cloned
  optimizer state (`StateMaster` bit-exact restore); NO densify/prune/clone/split.
- All 6 required checkpoints (room/bicycle @ 10K/15K/20K) present; SHA256
  verified against authoritative pins (fail-closed gate passed).

## 3. Design (frozen protocol)
- TRUE HOLDOUT = canonical cameras with dataset index `i % 8 == 0`;
  TRAIN POOL = the rest. Asserted `train ∩ holdout = ∅`.
- 24 stratified midpoint positions `q_k=floor((k+0.5)*N/24)` into the ordered
  train pool (dedup by advancing); panel_p = selected k with `k mod 3 == p`
  → 3 panels × 8 unique train cameras. True-holdout eval panel = 8 stratified
  midpoint quantiles into the ordered holdout pool.
- Per cell & panel: base eval on 8 panel + 8 holdout cams; then for each of the
  8 sources, ONE native Adam step from θ0, forward-only eval on all 16 cams;
  `M_ij = L_j(θ0) − L_j(θ_i⁺)`.
- Selectors: RANDOM = mean over sources; HARD-LOSS = argmax current loss
  (mandatory baseline; TurboGS already uses error-aware anchor); ORACLE =
  argmax peer-transfer T_i using TRAIN peers only. True holdout NEVER used for
  selection.

## 4. Per-cell results (median over 3 panels)
| cell | panels | T_oracle | G_random | G_hard | G_true | H_oracle>0 | F_pos | rho_TH | complete |
|------|--------|----------|----------|--------|--------|-----------|-------|--------|----------|
| room@10000 | 3 | 0.000049 | 1.7490 | 2.3791 | 0.9432 | 2/3 | 0.571 | 0.429 | yes |
| room@15000 | 3 | -0.000020 | 0.7100 | 0.6031 | -1.6331 | 0/3 | 0.286 | -0.095 | yes |
| room@20000 | 3 | -0.000024 | 0.7391 | 0.9948 | 0.7613 | 1/3 | 0.429 | -0.024 | yes |
| bicycle@10000 | 3 | 0.000024 | 0.6971 | 0.9242 | 0.4111 | 2/3 | 0.429 | 0.452 | yes |
| bicycle@15000 | 3 | 0.000019 | 1.2727 | 1.4051 | 0.0527 | 0/3 | 0.429 | 0.310 | yes |
| bicycle@20000 | 3 | 0.000030 | 1.3824 | 2.0508 | -0.0933 | 3/3 | 0.571 | -0.048 | yes |

## 5. Gate assessment (frozen thresholds)
| gate (strong / promote) | r10 | r15 | r20 | b10 | b15 | b20 |
|------|---|---|---|---|---|---|
| T_oracle_median>0 (strong>=0.0) | Y | . | . | Y | Y | Y |
| G_random_median (strong>=0.6) | Y | Y | Y | Y | Y | Y |
| G_hard_median (MANDATORY) (strong>=0.35) | Y | Y | Y | Y | Y | Y |
| G_true_median (strong>=0.4) | Y | . | Y | Y | . | . |
| H_oracle_positive_count (strong>=3) | . | . | . | . | . | Y |
| F_pos_median (strong>=0.75) | . | . | . | . | . | . |
| rho_TH_median (strong>=0.4) | Y | . | . | Y | . | . |

The mandatory `G_hard` gate is the one CVT0 exists to test (TurboGS already uses
an error-aware anchor). Here `G_hard_median` clears the 0.35 strong floor in every
cell, so the transfer oracle does beat the highest-current-loss source on the
normalized peer metric. The DROP is therefore NOT from `G_hard`; it comes from the
gates that require the advantage to be *consistent and specific*:
- `F_pos_median` (fraction of peers the oracle source actually improves) never
  reaches the 0.65 promote floor in any cell (0.29–0.57) — the oracle's
  advantage is diffuse, not a targeted improvement.
- `H_oracle_positive_count` (true-holdout panels where the oracle helps) is 0–1
  in 5/6 cells; only bicycle@200K reaches 3.
- `T_oracle_median > 0` fails for room@15000/20000; `G_true_median` is
  ≤0 for room@15000 and bicycle@15000/20000; `rho_TH_median` is negative in
  4/6 cells — the train-panel transfer signal does not generalize to held-out
  views.

## 6. Diagnostics (report-only, NOT used for selection)
Median across all 18 panels:
- pose-distance vs |M_ij| Pearson = **-0.162** (transfer is not explained by camera proximity)
- source current loss vs T_i Pearson = **0.138**
- source self-improvement S_i vs T_i Pearson = **-0.058**

## 7. Timing
- Median one native Adam step ≈ **389.9 ms**; median one forward-only
  eval ≈ 16 renders. Full 6-cell grid (18 panels × 8 sources) ≈ a few minutes.
- CVT0 makes NO E2E speedup claim.

## 8. Interpretation
The transfer oracle does beat the highest-current-loss source on the normalized
peer metric (`G_hard` gate passes everywhere), so cross-view transfer is *not*
purely the hard-view signal. But the advantage is not actionable: it is diffuse
(`F_pos` 0.29–0.57, far below 0.65), it rarely helps the true held-out views
(`H_oracle_positive_count` 0–1 in 5/6 cells), and the train-panel transfer signal
does not generalize to held-out views (`G_true_median` ≤0 in 3/6 cells,
`rho_TH_median` negative in 4/6). Geometry proximity is a weak correlate
(pose-vs-|M| Pearson ≈ −0.16), so transfer is not merely viewpoint distance
either. Per prereg §9 and DSH_HANDOFF item 8, **CVT0_DROP closes this
mechanism**. No rescue: panels/gates/holdout/checkpoints are unchanged.

## 9. Provenance
- Hashes: `CVT0_SOURCE_IDENTITY.json`, `CVT0_SHA256_MANIFEST.json`.
- Raw matrices/selectors/evals: `CVT0_TRANSFER_MATRICES.json`,
  `CVT0_SELECTORS.json`, `CVT0_BASE_LOSSES.json`, `CVT0_TRUE_HOLDOUT.json`.
- Logs: `cvt0_raw.log`, `cvt0_run.log`.
