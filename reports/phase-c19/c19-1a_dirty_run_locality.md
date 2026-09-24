# C19-1A — Dirty-Run Locality Gate

**Scene:** `room`  **Steps:** 100+50  **GPU:** NVIDIA A100-PCIE-40GB
**gsplat:** 1.5.3  **Date:** 2026-09-08T00:53:07.609332+00:00  **Duration:** 553s

---

## Decision: **RUN-LEVEL NO-GO**

**Trigger:** P50 dirty entry ratio 23.81% > 20% threshold (primary). Repair/full ratio P95=0.993 (entire tile). Run-level repair ≈ full-tile repair.

---

## 1. Method

For each consecutive pair (t,t+1) across 100 measurement steps:

1. Identify (Gaussian,Tile) pairs common to both steps via searchsorted
2. For each tile: reconstruct reused entries in their PREV sorted order with **CURR depth values**
3. Detect adjacent inversions: `d[i] > d[i+1]` in the prev-order sequence
4. Build dirty runs: merge connected inversion spans `[i,i+1]` into contiguous dirty regions
5. Measure: run sizes, counts, separation, minimum repair region, expansion ratio
6. Check run independence: sort each run independently and check cross-boundary violations

---

## 2. Correctness

| Check | Value |
|:---|---:|
| Tiles analyzed | 816,000 |
| Repair tiles detected | 687,870 |
| Non-repair tiles | 128,130 |

---

## 3. Three Quantities (Section 7)

| Quantity | Value | Interpretation |
|:---|---:|:---|
| Repair tile ratio | 0.8430 | Fraction of tiles with any inversion |
| Dirty entry ratio (repair tiles) | 0.2841 | Fraction of entries in repair tiles that are dirty |
| **Global dirty entry ratio** | **0.2547** | **Fraction of ALL intersection entries in dirty runs** |
| C18-2 repair tile ratio | 0.8240 | (previous conservative metric) |
| C18-2 repair entry ratio | 0.9112 | (previous conservative metric) |

---

## 4. Aggregate — Repair Tiles (Section 5)

| Metric | Mean | P10 | P25 | P50 | P75 | P90 | P95 | P99 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dirty entry ratio | 0.2407 | 0.0217 | 0.0805 | 0.2381 | 0.3731 | 0.4710 | 0.5218 | 0.6042 |
| Run count | 39.0982 | 2.0000 | 9.0000 | 29.0000 | 56.0000 | 90.0000 | 112.0000 | 176.0000 |
| Mean run size | 2.6358 | 2.0000 | 2.1000 | 2.5294 | 2.9868 | 3.4516 | 3.7742 | 4.4286 |
| Largest run ratio | 0.0189 | 0.0063 | 0.0104 | 0.0168 | 0.0249 | 0.0339 | 0.0404 | 0.0543 |
| Violation density | 0.1300 | 0.0110 | 0.0412 | 0.1250 | 0.2015 | 0.2603 | 0.2922 | 0.3463 |
| Repair expansion ratio | 7.6711 | 1.8122 | 2.3432 | 3.3750 | 6.4688 | 15.4667 | 26.1250 | 73.2758 |
| Repair / full tile ratio | 0.8218 | 0.3268 | 0.8359 | 0.9518 | 0.9786 | 0.9894 | 0.9933 | 0.9992 |

---

## 5. Aggregate — All Tiles (Section 5)

| Metric | Mean | P10 | P25 | P50 | P75 | P90 | P95 | P99 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dirty entry ratio | 0.2029 | 0.0000 | 0.0248 | 0.1802 | 0.3482 | 0.4559 | 0.5104 | 0.5970 |
| Largest run ratio | 0.0159 | 0.0000 | 0.0066 | 0.0144 | 0.0231 | 0.0323 | 0.0388 | 0.0526 |

---

## 6. Violation Density (Section 8)

| Mean | P10 | P25 | P50 | P75 | P90 | P95 | P99 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Violation density | 0.1300 | 0.0110 | 0.0412 | 0.1250 | 0.2015 | 0.2603 | 0.2922 | 0.3463 |

P50=0.1250 — **Moderate** — noticeable but not dominant

---

## 7. Tile Workload Buckets (Section 6)

| Bucket | Tiles | Mean Entries | Repair Ratio | DER Mean | DER P50 | Runs | Run Size | LRR Mean | LRR P50 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-32 | 34 | 32 | 0.1471 | 0.0093 | 0.0000 | 0.15 | 0.29 | 0.0093 | 0.0000 |
| 33-64 | 25,574 | 51 | 0.5112 | 0.0334 | 0.0317 | 0.83 | 1.06 | 0.0213 | 0.0312 |
| 65-128 | 45,216 | 99 | 0.7113 | 0.0589 | 0.0374 | 2.66 | 1.53 | 0.0193 | 0.0198 |
| 129-256 | 206,414 | 201 | 0.7584 | 0.1425 | 0.0971 | 11.46 | 1.84 | 0.0177 | 0.0159 |
| 257-512 | 331,965 | 370 | 0.8749 | 0.2081 | 0.2036 | 27.66 | 2.27 | 0.0156 | 0.0146 |
| 513-1024 | 185,127 | 675 | 0.9397 | 0.2920 | 0.3153 | 64.00 | 2.74 | 0.0134 | 0.0130 |
| 1025-2048 | 21,300 | 1256 | 1.0000 | 0.4343 | 0.4605 | 152.65 | 3.51 | 0.0122 | 0.0115 |
| 2049-4096 | 370 | 2211 | 1.0000 | 0.5998 | 0.5971 | 290.40 | 4.61 | 0.0109 | 0.0106 |

---

## 8. Run Interaction (Section 12)

| Metric | Value |
|:---|---:|
| Repair tiles checked | 687,870 |
| Tiles with interacting runs | 544,092 |
| Interaction ratio | 0.790981 |
| Boundary violations | 8198581 |
| Clean-region violations | 16326147 |

⚠️ Runs interact — independent run sorting alone may be insufficient.

---

## 9. Theoretical Sort Reduction (Section 11)

| Strategy | Sort volume | Reduction vs baseline |
|:---|---:|---:|
| Baseline (full repair tile sort) | 289,192,567 | — |
| Min repair region sort | 253,532,623 | 12.33% |
| Dirty-only sort | 82,172,912 | 71.59% |

---

## 10. Separation Between Dirty Runs (Section 9)

| Mean | P10 | P25 | P50 | P75 | P90 | P95 | P99 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean gap sizes | 5.3081 | 1.0000 | 1.0000 | 3.0000 | 5.0000 | 11.0000 | 18.0000 | 45.0000 |
Gaps: P50=3.0, P90=11.0 — well-separated

---

## 11. Training Phase Analysis (Section 13)

| Phase | Steps | Repair Ratio | Global DER | Mean Run Size | Largest Run Ratio |
|:---|---:|---:|---:|---:|---:|
| early | 50-83 | 0.9614 | 0.3042 | 2.67 | 0.0197 |
| middle | 83-116 | 0.8356 | 0.2082 | 2.51 | 0.0159 |
| late | 116-149 | 0.7328 | 0.2522 | 2.74 | 0.0212 |

---

## 12. Stress Analysis — Top 20 (Section 14)

### By Dirty Entry Ratio

| # | Step | Tile | Entries | Inv | Runs | Dirty | DER | MaxRun | LRR | ViolDen |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 60 | 7847 | 1672 | 734 | 217 | 1264 | 0.7560 | 53 | 0.0317 | 0.4500 |
| 2 | 60 | 7967 | 1393 | 607 | 169 | 1053 | 0.7559 | 25 | 0.0179 | 0.4463 |
| 3 | 59 | 7607 | 1777 | 785 | 227 | 1342 | 0.7552 | 28 | 0.0158 | 0.4530 |
| 4 | 59 | 7727 | 1823 | 810 | 249 | 1375 | 0.7543 | 35 | 0.0192 | 0.4530 |
| 5 | 62 | 7967 | 1352 | 588 | 182 | 1012 | 0.7485 | 41 | 0.0303 | 0.4444 |
| 6 | 59 | 7847 | 1692 | 739 | 224 | 1266 | 0.7482 | 26 | 0.0154 | 0.4476 |
| 7 | 57 | 7727 | 1846 | 803 | 238 | 1377 | 0.7459 | 29 | 0.0157 | 0.4469 |
| 8 | 57 | 7847 | 1719 | 742 | 232 | 1280 | 0.7446 | 24 | 0.0140 | 0.4438 |
| 9 | 60 | 7727 | 1802 | 774 | 250 | 1340 | 0.7436 | 26 | 0.0144 | 0.4403 |
| 10 | 56 | 7847 | 1714 | 746 | 223 | 1271 | 0.7415 | 34 | 0.0198 | 0.4440 |
| 11 | 62 | 7847 | 1622 | 695 | 216 | 1202 | 0.7411 | 28 | 0.0173 | 0.4388 |
| 12 | 51 | 7727 | 1904 | 813 | 265 | 1410 | 0.7405 | 30 | 0.0158 | 0.4318 |
| 13 | 61 | 7967 | 1363 | 591 | 191 | 1008 | 0.7395 | 30 | 0.0220 | 0.4410 |
| 14 | 64 | 7513 | 844 | 359 | 108 | 624 | 0.7393 | 23 | 0.0273 | 0.4294 |
| 15 | 58 | 7967 | 1440 | 628 | 188 | 1064 | 0.7389 | 19 | 0.0132 | 0.4502 |
| 16 | 62 | 7487 | 1672 | 720 | 226 | 1235 | 0.7386 | 31 | 0.0185 | 0.4401 |
| 17 | 57 | 7607 | 1779 | 759 | 246 | 1313 | 0.7381 | 25 | 0.0141 | 0.4370 |
| 18 | 61 | 7607 | 1733 | 744 | 241 | 1277 | 0.7369 | 30 | 0.0173 | 0.4387 |
| 19 | 75 | 7727 | 1723 | 737 | 206 | 1269 | 0.7365 | 32 | 0.0186 | 0.4374 |
| 20 | 61 | 7727 | 1771 | 756 | 245 | 1304 | 0.7363 | 26 | 0.0147 | 0.4370 |

### By Largest Run Ratio

| # | Step | Tile | Entries | Inv | Runs | Dirty | DER | MaxRun | LRR | ViolDen |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 55 | 7802 | 144 | 34 | 12 | 61 | 0.4236 | 25 | 0.1736 | 0.2378 |
| 2 | 115 | 4390 | 118 | 28 | 13 | 53 | 0.4492 | 18 | 0.1525 | 0.2435 |
| 3 | 119 | 1542 | 133 | 30 | 16 | 59 | 0.4436 | 20 | 0.1504 | 0.2326 |
| 4 | 119 | 1422 | 144 | 32 | 19 | 63 | 0.4375 | 20 | 0.1389 | 0.2254 |
| 5 | 68 | 5629 | 134 | 22 | 12 | 44 | 0.3284 | 18 | 0.1343 | 0.1654 |
| 6 | 59 | 7922 | 149 | 28 | 14 | 52 | 0.3490 | 20 | 0.1342 | 0.1905 |
| 7 | 115 | 4270 | 115 | 25 | 13 | 48 | 0.4174 | 15 | 0.1304 | 0.2212 |
| 8 | 61 | 8057 | 155 | 28 | 13 | 54 | 0.3484 | 20 | 0.1290 | 0.1842 |
| 9 | 119 | 2988 | 47 | 4 | 2 | 8 | 0.1702 | 6 | 0.1277 | 0.0870 |
| 10 | 142 | 2632 | 47 | 4 | 2 | 8 | 0.1702 | 6 | 0.1277 | 0.0870 |
| 11 | 119 | 2989 | 48 | 4 | 2 | 8 | 0.1667 | 6 | 0.1250 | 0.0851 |
| 12 | 142 | 2152 | 48 | 5 | 3 | 10 | 0.2083 | 6 | 0.1250 | 0.1064 |
| 13 | 142 | 2154 | 48 | 4 | 2 | 8 | 0.1667 | 6 | 0.1250 | 0.0851 |
| 14 | 142 | 2272 | 48 | 5 | 3 | 10 | 0.2083 | 6 | 0.1250 | 0.1064 |
| 15 | 142 | 2631 | 48 | 4 | 2 | 8 | 0.1667 | 6 | 0.1250 | 0.0851 |
| 16 | 115 | 5628 | 289 | 84 | 37 | 157 | 0.5433 | 36 | 0.1246 | 0.2947 |
| 17 | 60 | 7504 | 244 | 69 | 33 | 127 | 0.5205 | 30 | 0.1230 | 0.2875 |
| 18 | 119 | 3108 | 49 | 3 | 1 | 6 | 0.1224 | 6 | 0.1224 | 0.0625 |
| 19 | 142 | 2034 | 49 | 5 | 3 | 10 | 0.2041 | 6 | 0.1224 | 0.1042 |
| 20 | 142 | 2273 | 49 | 4 | 2 | 8 | 0.1633 | 6 | 0.1224 | 0.0870 |

---
## 13. Final Decision

**RUN-LEVEL NO-GO**

### Key evidence

- **Global dirty entry ratio**: 0.2547 (25.47%) of ALL intersection entries
  are part of dirty runs.
- **P50 dirty entry ratio in repair tiles**: 0.2381 (23.81%) — 
  exceeds the 20% NO-GO threshold. This is the primary trigger.
- **P95 dirty entry ratio**: 0.5218 (52.18%) — very high.
- **Repair / full tile ratio P50**: 0.9518 — 
  the minimum repair region covers 95% of the tile on average. P95=0.993 (essentially full tile).
- **Repair expansion ratio P50**: 3.38x — 
  dirty runs expand 3.4× beyond the raw inversion span to achieve correct ordering. Because
  the runs are numerous (P50=29 per tile) and close together, the "minimum repair region"
  quickly expands to cover most of each tile.

**Situation B** 🔴 — 82.4% of tiles need repair AND dirty regions are LARGE. The run-level repair strategy would sort almost as many entries as the full-tile repair. Candidate should be killed.
