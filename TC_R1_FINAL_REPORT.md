# TEMPORAL_COVERAGE_R1_MX — Stage-6 Scientific Report

**Task:** TEMPORAL-COVERAGE-R1-MX (10K A/B/W temporal-coverage campaign)
**Host:** MX (36.140.146.31:26372, user liaoyuanjun), A100-PCIE-40GB, torch 2.9.1+cu128, CUDA 12.8
**Date:** 2026-10-08
**Correctness basis:** SPARSE_BACKWARD_CORRECTNESS_GATE = SPARSE_BACKWARD_PASS
(see `SPARSE_BACKWARD_CORRECTNESS_GATE_REPORT.md`).

---

## 1. Final verdict

**VERDICT: `TC_R1_DROP`**

- B_vs_A gate pass: **False** (mean ΔPSNR -0.054 dB [gate≥+0.10: False], scenes≥0.05: 0/3 [gate≥2/3: False], mean ΔLPIPS -0.0015 [gate≤+0.005], max scene ΔLPIPS +0.0011 [gate≤+0.010], mean speed B/A 0.950 [gate≥0.95], no-NaN True)
- B_vs_W within 0.05 dB band: **True** (mean ΔPSNR B−W +0.006 dB)
- time-to-quality (B reaching A's final PSNR, per scene): garden: A 44.9s → B 47.8s (Δ+2.9s); bicycle: A 64.7s → B 66.8s (Δ+2.1s); room: A 50.9s → B 53.3s (Δ+2.4s)
- per-scene ΔPSNR B−A: garden -0.002, bicycle -0.077, room -0.084
- per-scene ΔPSNR B−W: garden +0.108, bicycle -0.033, room -0.057
- B better than A (mean ΔPSNR B−A ≥ +0.10): False

**Interpretation.** With only two seeds this is a *directional* result, not a
statistically powered claim. The verdict applies the pre-registered GO/DROP
gates to the real per-run records; no threshold was modified and no metric was
hand-entered.

---

## 2. B vs A paired results (per scene, per seed)

LPIPS: lower is better; a positive ΔLPIPS is a regression.

| Scene | Seed | A PSNR | B PSNR | ΔPSNR | A SSIM | B SSIM | ΔSSIM | A LPIPS | B LPIPS | ΔLPIPS | A wall_s | B wall_s | B/A spd | A ttq | B ttq | A N | B N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| garden | 0 | 11.882 | 12.080 | 0.198 | 0.2864 | 0.2873 | 0.0009 | 0.9261 | 0.9295 | 0.0035 | 399.0 | 420.3 | 0.95 | 86.6 | 47.8 | 57701 | 57656 |
| garden | 1 | 12.160 | 11.958 | -0.202 | 0.2879 | 0.2844 | -0.0035 | 0.9282 | 0.9225 | -0.0056 | 395.3 | 424.9 | 0.93 | 43.6 | 45.4 | 56823 | 57971 |
| bicycle | 0 | 13.395 | 13.249 | -0.146 | 0.3109 | 0.3081 | -0.0028 | 0.5876 | 0.5831 | -0.0045 | 554.3 | 582.0 | 0.95 | 64.7 | 66.8 | 368478 | 384957 |
| bicycle | 1 | 13.264 | 13.257 | -0.008 | 0.3100 | 0.3082 | -0.0018 | 0.5877 | 0.5831 | -0.0046 | 557.1 | 577.0 | 0.97 | 64.9 |  | 371490 | 381337 |
| room | 0 | 16.534 | 16.830 | 0.296 | 0.6343 | 0.6387 | 0.0043 | 0.5332 | 0.5278 | -0.0054 | 469.8 | 494.9 | 0.95 | 50.9 |  | 75863 | 78561 |
| room | 1 | 16.785 | 16.322 | -0.463 | 0.6409 | 0.6324 | -0.0084 | 0.5288 | 0.5365 | 0.0077 | 470.7 | 494.5 | 0.95 | 50.9 | 53.3 | 75611 | 78887 |

**Seed-mean B−A:** ΔPSNR -0.054 dB · ΔSSIM -0.0019 ·
ΔLPIPS -0.0015 · speed B/A 0.950×

---

## 3. B vs W paired results (per scene, per seed)

| Scene | Seed | W PSNR | B PSNR | ΔPSNR(B-W) | W LPIPS | B LPIPS | ΔLPIPS(B-W) | W wall_s | B wall_s | B/W spd | W ttq | B ttq | W N | B N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| garden | 0 | 12.302 | 12.080 | -0.222 | 0.9276 | 0.9295 | 0.0020 | 419.7 | 420.3 | 1.00 | 45.8 |  | 57979 | 57656 |
| garden | 1 | 11.520 | 11.958 | 0.438 | 0.9321 | 0.9225 | -0.0095 | 413.6 | 424.9 | 0.97 | 45.6 |  | 57632 | 57971 |
| bicycle | 0 | 13.301 | 13.249 | -0.052 | 0.5827 | 0.5831 | 0.0004 | 581.1 | 582.0 | 1.00 | 67.6 | 66.8 | 389023 | 384957 |
| bicycle | 1 | 13.270 | 13.257 | -0.014 | 0.5838 | 0.5831 | -0.0006 | 578.0 | 577.0 | 1.00 | 67.9 | 66.3 | 381592 | 381337 |
| room | 0 | 16.594 | 16.830 | 0.236 | 0.5300 | 0.5278 | -0.0023 | 490.2 | 494.9 | 0.99 | 54.5 |  | 79365 | 78561 |
| room | 1 | 16.671 | 16.322 | -0.350 | 0.5297 | 0.5365 | 0.0068 | 488.9 | 494.5 | 0.99 | 52.8 | 53.3 | 79114 | 78887 |

**Seed-mean B−W:** ΔPSNR +0.006 dB · ΔLPIPS -0.0005

---

## 4. Pre-registered B vs A gate evaluation

| gate | threshold | value | pass |
|---|---|---|---|
| mean ΔPSNR | ≥ +0.10 dB | -0.054 dB | False |
| scenes ΔPSNR ≥ +0.05 | ≥ 2/3 | 0/3 | False |
| mean ΔLPIPS | ≤ +0.005 | -0.0015 | True |
| max scene ΔLPIPS | ≤ +0.010 | +0.0011 | True |
| mean speed B/A | ≥ 0.95 | 0.950× (training compute) | False |
| no NaN / collapse | — | no_nan=True | True |

---

## 5. B vs W mechanism-equivalence audit

`results/BW_MECHANISM_AUDIT.json`. Operating point: 1920×1080, tile 16×16,
ratio 0.25, 4 groups, seed 20261008+991, 40 visits.

- **Shared coverage structure:** same fixed partition, group_size
  2040, 8160 tiles, budget 0.25,
  theoretical marginal 0.25 for both; both exactly complementary within a
  cycle; revisit cycle = 4 visits for both.
- **Long-run inclusion probability:** B 0.2500 ± 0.0000, W 0.2500 ± 0.0000
  (identical, exactly balanced).
- **Intra-cycle group-emission order:** B = fixed cyclic rotation
  `(phase0+v) % n_groups` (source-verified `B_is_fixed_rotation=
  True`); W = random deck
  `randperm(n_groups)` per cycle (source-verified `W_is_random_deck=
  True`).
- **Partition reshuffle:** both reshuffle the tile partition at each cycle
  boundary (`B_reshuffles=True`,
  `W_reshuffles=True`),
  which is the sole source of the nonzero consecutive-visit overlap
  (B mean 0.033, W mean
  0.077); within a cycle both are exactly
  complementary (0 overlap).
- **Realized mask hashes differ** (`mask_sequences_differ=
  True`) because the visit order
  differs, not the coverage.

**Conclusion.** B and W are the *same unbiased sampled-tile-mean estimator* with
identical partition / budget / marginal / complementarity / revisit-cycle; they
differ only in the temporal ORDER of group visits. For the training loss the
expected gradient is identical, so any B−W outcome difference with two seeds is
noise, not a verifiable advantage. W is a fair random-without-replacement
control. **MECHANISM_EQUIVALENT = True →
`TC_R1_BASELINE_EQUIVALENT`.**

---

## 6. Sparse-Gate reconciliation notes (requirement 7)

### 6a. 74.9% intersection-skipped (gate) vs isect_frac ≈ 0.289 (campaign)

These are different quantities on different denominators:

- **Gate 74.9% skipped:** single camera (cam 0), fixed 40 000-Gaussian on-screen
  subset, ONE sparse render. `skip = (n_isects_full − n_isects_sparse) /
  n_isects_full = 290639 / 388116 = 74.88%` (processed ≈ 25.1%). Denominator =
  that single camera's dense (Gaussian, tile) intersection count.
- **Campaign isect_frac ≈ 0.289:** `mean over all sampled training steps` of
  `n_isects / n_isects_full`, where each step renders **all 32 train cameras**
  with the full, densifying Gaussian set (≈57 k–370 k). It is *identical*
  (0.2890) across all arms/scenes because the rasterizer's realized
  intersection-fraction for ratio-0.25 tile sampling is a deterministic function
  of the tile budget (25% of tiles → ≈0.289 of intersections; tiles are sampled
  by count, and the intersection-weighted fraction differs slightly from 0.25).

Both confirm ~71–75% of (Gaussian, tile) intersections are skipped under the
0.25 budget; the small numeric gap is the camera/subset/averaging basis, not a
discrepancy.

### 6b. Cross-renderer colors ~5% — NOT attributed to forward SH convention

The prior RENDERER_BACKWARD_CORRECTNESS_GATE (§3.1, §3.4) shows B (gsplat) and
C (HiGS) forward renders are **bit-identical** (renderMAE = 0.0, maxerr = 0.0),
yet the colors *gradient* differs by relL2 ≈ 5.0e-2 (cosine 0.9987) while
means/quats/scales/opac match to ~1e-7. Because the forward is identical, the
~5% is a **backward color/SH-gradient-path** difference, not a forward
difference. Candidate causes (SH basis evaluation order/precision in the
backward, intermediate dtype in SH-coefficient gradient accumulation,
color-gradient↔coefficient association) are **hypotheses requiring targeted
evidence** (per-SH-band gradient comparison, FD on SH coefficients) — not
asserted here. The same-renderer sparse-vs-dense colors relL2 = 2.2e-07
(`SPARSE_BACKWARD_CORRECTNESS_GATE_REPORT.md`) stands: the HiGS sparse backward
is color-gradient-correct.

---

## 7. Provenance & integrity

- **18/18 runs** completed at 10K iterations (eval curve last step = 10000,
  10 eval points each), all rc=0, no CUDA errors / OOM / retries.
- Renderer SO `gsplat_scene_cuda.so` sha256
  `e4b0bc5b…90627c8` matches `TC_R1_ENV_FREEZE.json` (unmodified).
- Raw data mtime 2026-09-21 (untouched). Core training functions byte-identical
  to repo HEAD (`SPARSE_GATE_CODE_AUDIT.json`).
- **SHA256 manifest:** `results/TC_R1_SHA256_MANIFEST.json` (all raw runs +
  result JSONs + this report).
- Aggregated with `aggregate_r1.py` (pre-registered gates, unmodified) →
  `results/TC_R1_VERDICT.json` + `results/TC_R1_*.csv`.

## 8. Constraints respected

No 30K launched, no CUDA modified, no sampling-strategy change, no exploratory
training added. Work-Gradient frozen; WV-HIER and C51-CD0 independent. Only two
seeds → no statistical-significance or CVPR-level-innovation claim is made.

## 9. Artifacts

- `results/TC_R1_VERDICT.json` — machine verdict + gates
- `results/TC_R1_EXPERIMENT_MATRIX.csv`, `TC_R1_PER_RUN_RESULTS.csv`,
  `TC_R1_QUALITY_CURVES.csv`, `TC_R1_TIMING.csv` — full per-run records
- `results/BW_MECHANISM_AUDIT.json` — B vs W mechanism audit
- `results/TC_R1_SHA256_MANIFEST.json` — SHA256 of all runs + outputs
- `SPARSE_BACKWARD_CORRECTNESS_GATE_REPORT.md` + `SPARSE_GATE_CODE_AUDIT.json`
- `WITHDRAWN_ARTIFACT_B_vs_C.md` — retracted near-orthogonal conclusion
