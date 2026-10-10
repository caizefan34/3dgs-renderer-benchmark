# HIGS R16 Opacity Semantics Investigation — Final Report

## Verdict: OPACITY_SEMANTICS_CONFIRMED

**Date:** 2026-10-08  
**Platform:** MX / A100-PCIE-40GB  
**CUDA binary (unchanged):** `b64d02c1cdad51bfc9ff0c2c435f9a4c1d1603466700194fb30f1c2348bcf2c3`  
**Checkpoint (unchanged):** `/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt`

---

## Root Cause Identified

HiGS stores opacity as **raw logits** (mean = −2.886, 916/1000 sampled are negative). The CUDA counting kernel receives raw logits directly and its conic threshold test `opacity <= alpha` (0.00392) evaluates the raw value — the kernel never applies sigmoid, so most Gaussians pass the alpha threshold and get counted at full geometric extent.

## Three GPU Replay Experiments

| Experiment | GPU count sum | Oracle count sum | Mismatch pairs | Mismatch ratio | Count vs raw |
|---|---|---|---|---|---|
| **RAW** (opacity as stored) | 1,804,447 | 132,796 | 915 | 91.5% | baseline |
| **Sigmoid** (logit→alpha) | 1,453,128 | 1,453,128 | 0 | 0.0% | −360,251 (−19.96%) |
| **Sigmoid + threshold skip** | 1,453,128 | 1,453,128 | 0 | 0.0% | −360,251 (−19.96%) |

- RAW exactly reproduces the R15 capture (`captured_gpu_exact_match: true`).
- Sigmoid: **GPU == Oracle exactly** (`ALL_COMPARABLE_COUNTS_EQUAL`). The CUDA AccuTile algorithm and the independent geometry oracle agree perfectly once opacity is properly activated.
- Sigmoid + threshold skip: identical to sigmoid — the CUDA kernel already skips below-threshold Gaussians internally.

## Gaussian 1196 (the R15 first mismatch)

| Property | Value |
|---|---|
| Raw opacity | −4.595 |
| Sigmoid-activated opacity | 0.00998 |
| Radius (px) | 1448 × 928 |
| Projected center | (1524.4, 62.0) |
| Depth | 7.615 |
| RAW GPU count | 1796 |
| RAW Oracle count | 0 |
| Sigmoid GPU count | 0 |
| Sigmoid Oracle count | 0 |

## CPU Opacity Audit (1000 sampled primitives)

- Raw opacity: min −5.284, median −3.786, mean −2.886, max 9.778
- Raw negative/nonpositive count: **916/1000**
- Activated (sigmoid): min 0.00505, median 0.0222, mean 0.1393, max 0.99994
- NaN/Inf in raw opacity: 0
- Nonpositive radii: 0

## Interpretation

The R15 "geometric discrepancy" was not a counting bug. It was caused by feeding unactivated logits into a geometry oracle that assumes activated opacity semantics. When opacity is correctly sigmoid-activated before the intersection computation, GPU counts match the independent oracle exactly for all 1000 primitives.

## Impact on the 4.41M OOM

At 580K/B1: raw GPU count = 1,804,447. The full 4.41M workload extrapolates to ~14.9B intersections (R10 OOM) under raw-opacity semantics. If the production forward path applies sigmoid before calling `isect_tiles` (as the standard gsplat pipeline does), the real workload count would be ~20% lower (~11.9B) — still OOM on 40GB, meaning the hierarchy/tiling strategy itself drives the memory blow-up.

## Deliverables

- `R16_OPACITY_SEMANTICS_REPORT.json` — machine-readable results
- `R16_REPLAY_COUNTS.csv` — three-mode comparison table
- Raw artifacts on MX at `/mnt/storage_pool/liaoyuanjun/HIGS_R16_OPACITY_SEMANTICS_MX/`:
  - `oracle_raw/oracle_summary.json`
  - `oracle_sigmoid/oracle_summary.json`
  - `oracle_sigmoid_threshold_skip/oracle_summary.json`
  - `raw_replay_capture.pt`, `sigmoid_replay_capture.pt`, `sigmoid_threshold_skip_replay_capture.pt`
  - `run.log`

## Next Steps

1. Verify whether `rasterize_gaussian_higs_frozen` applies sigmoid to opacity before calling `isect_tiles` in the production path.
2. If sigmoid IS applied in production: the R10 111.28 GiB OOM reflects true intersection load → hierarchy redesign is the only path (R17+).
3. If sigmoid is NOT applied in production: fix the missing activation in HiGS counting — this is a genuine correctness bug affecting rendering quality, not just memory.