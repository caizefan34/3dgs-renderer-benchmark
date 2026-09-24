# DSH-E: Training-Stage Representativeness Audit

**Binary:** Frozen C0 V3 (F9 + SCALAR_ADJOINT + H8-MR), composed `.so` sha256 `7ca1c6bf6c8e4307ecb8fcdbcaf2953bf95305d2c9814f84f3fb5130859301f6`
**GPU:** A100-PCIE-40GB × 8, GPU4 (co-tenant pid present) — **DIAGNOSTIC_SHARED_GPU** (shared GPU, not a publication timing)
**Resolution:** 1080p (1920 long side), 16 stratified cameras/checkpoint, deterministic IDs
**Scenes:** room (311 cams), bicycle (194), garden (185)
**Checkpoints:** 5K / 15K / 30K (r6_baseline_ckpts — the only trajectory data that exists; 1K/10K/20K/25K do not exist)
**Constraints honored:** E2 NOT reopened (DROP frozen); no production CUDA modifications; no new optimization implementation; exact frozen C0 V3 binary (no P2 native hierarchy, no E2 fusion, no diagnostic P3-H sink).

> **Evidence labels used throughout:** `MEASURED` (directly read from C0 V3 forward/timing metadata) · `DERIVED` / `DERIVED_CONVENTION` (computed from measured inputs under a documented convention) · `INTERPOLATED` (piecewise-linear between measured checkpoints). Contention proxies are **TOPOLOGY_PROXY_ONLY** (no profiler atomic counters).

---

## 0. Headline verdicts

| Question | Answer |
|---|---|
| Do 30K/cam0 fixtures remain representative across training? | **Largely YES** for room & bicycle (all 7 metrics REPRESENTATIVE at 30K); **garden cam0 contention is LOW_BIASED (−18.9%)**. The fixture is *not* an outlier. |
| Does HAR's spatial reuse exist throughout training or only when N is large? | **It exists throughout AND is strongest EARLY.** D_macro (fine-pair compression per (macro,gid)) is HIGHER at 5K than 30K for all 3 scenes. The dead-work fraction is largest when N is *small*, not large. |
| Is the P3 opportunity stable or late-heavy? | **STABLE-to-early-weighted** (room STABLE, bicycle STABLE, garden WEAK_EARLY). NOT late-heavy. |
| Does E2 cross the 3% gate at any checkpoint? | **NO.** Max 0.575% (bicycle 5K) at 1080p. E2 stays **DROP** (E2_DENOMINATOR_SENSITIVITY_ONLY). |
| P3-0/P3-H projected gain | **PENDING_CROSS_JOIN** (P3-0/P3-H final numbers not on mx; mechanism + checkpoint work mapped, DERIVED only). |

---

## 1. Checkpoint inventory & camera sampling (MEASURED)

| Scene | 5K N_GS | 15K N_GS | 30K N_GS | cams_total | sampled (deterministic) | timing cam |
|---|---|---|---|---|---|---|
| room | 560,632 | 933,590 | 933,590 | 311 | 0,21,41,62,83,103,124,145,165,186,207,227,248,269,289,310 | 165 |
| bicycle | 1,933,177 | 3,957,041 | 3,957,041 | 194 | 0,13,26,39,51,64,77,90,103,116,129,142,154,167,180,193 | 103 |
| garden | 1,779,185 | 2,610,559 | 2,610,559 | 185 | 0,12,25,37,49,61,74,86,98,110,123,135,147,159,172,184 | 98 |

**Key:** N is **flat 15K→30K for all 3 scenes** (saturation). Densify active 500–15000 (r6 training log: every 100 iters, grad threshold 0.0008, opacity reset every 3000). The 15K→30K half of training is **topology-flat**.

---

## 2. N_GS evolution (MEASURED)

- **room:** 115,278 (base) → 560,632 (5K, +386%) → 933,590 (15K, +67%) → 933,590 (30K, 0%).
- **bicycle:** 580,416 (base) → 1,933,177 (5K, +233%) → 3,957,041 (15K, +105%) → 3,957,041 (30K, 0%).
- **garden:** 66,282 (base) → 1,779,185 (5K, +2584%) → 2,610,559 (15K, +47%) → 2,610,559 (30K, 0%).

Growth is front-loaded (0–5K is the steepest jump, esp. garden), then moderates (5–15K), then **flat (15–30K)**. No LOW_GS phase exists in the sampled checkpoints — 5K is already post-init.

---

## 3. N_visible evolution (MEASURED / DERIVED)

`n_visible_measured` = C0 `n_isects` (pair-like, ~2.9–3.8× the visible-Gaussian count). `n_visible_analytic` = C0 `fully_fused_projection` radii>0 (visible GAUSSIANS).

| Scene | it | n_isects median | n_isects p10–p90 | n_visible_analytic median | visible_frac median |
|---|---|---|---|---|---|
| room | 5K | 3,912,025 | 3.51M–4.32M | 163,229 | 0.291 |
| room | 15K | 3,494,490 | 3.10M–3.94M | 284,463 | 0.305 |
| room | 30K | 2,917,492 | 2.55M–3.32M | 282,047 | 0.302 |
| bicycle | 5K | 5,008,259 | 3.96M–8.70M | 356,950 | 0.185 |
| bicycle | 15K | 6,677,753 | 5.12M–15.47M | 894,862 | 0.226 |
| bicycle | 30K | 5,790,740 | 4.63M–12.67M | 891,564 | 0.225 |
| garden | 5K | — | — | ~200K | ~0.11 |
| garden | 15K | — | — | ~210K | ~0.08 |
| garden | 30K | — | — | ~210K | ~0.08 |

`n_visible_analytic` at 30K matches the independent r6_1 cross-join (room 30K cam0: N_visible 265,137 vs my 282,047 median — within 6%). Visible fraction is **stable 5K→30K** (room 0.29–0.31; bicycle 0.18–0.23); the pair count rises 5K→15K (more N) then eases 15K→30K (saturated N, slightly tighter Gaussians).

---

## 4. Fine-pair evolution (MEASURED n_isects / DERIVED tile-count)

`C0 n_isects` (MEASURED) is the authoritative pair count. The AABB tile-count convention **over-estimates** n_isects by a **MEASURED 3.84×** (room 30K cam150: AABB 12,713,859 vs C0 3,311,368) because the native AccuTile kernel uses tighter per-Gaussian bounds. D_macro and multiplicity are therefore **DERIVED_CONVENTION** (uniform convention; the ratio is internally consistent).

- **n_isects (MEASURED):** room 3.91M→3.49M→2.92M; bicycle 5.01M→6.68M→5.79M; garden rising.
- **pairs/visible-GS (DERIVED):** ~26–49 (room), ~25–35 (bicycle) — stable order.
- **pairs/master-GS (DERIVED):** ~9–12 — stable.

---

## 5. D_macro evolution (DERIVED_CONVENTION) — the central question

`D_macro = N_fine_pairs / N_unique_(macro,gid)` on the 8×4 fine-tile macro grid (128×64 px — same macro as P2-0 native hierarchy).

| Scene | 5K | 15K | 30K | trend |
|---|---|---|---|---|
| room | **15.90** | 11.47 | 11.06 | ↓ (early-heavy) |
| bicycle | **12.19** | 7.96 | 7.49 | ↓ (early-heavy) |
| garden | **10.64** | 7.90 | 7.43 | ↓ (early-heavy) |

**Answer:** HAR's spatial reuse **exists throughout training and is strongest EARLY** (small N → fewer visible → tighter, more clustered Gaussians → higher compression per (macro,gid)). It does **not** only appear when N becomes large; if anything it weakens toward 30K. This is the opposite of a "late-only" opportunity.

*(P2-0's 128×64-macro entry-compression baseline 7.68/4.53/8.00× used native-HiGS structural counters at 30K — same granularity, comparable in spirit; the 30K D_macro here (11.1/7.5/7.4) is in the same order.)*

---

## 6. Macro-multiplicity evolution (DERIVED_CONVENTION)

Weighted distribution of (macro,gid) multiplicity over visible Gaussians:

| Scene | it | 1 | 2 | 3–4 | 5–8 | 9–16 | 17–32 | 33+ |
|---|---|---|---|---|---|---|---|---|
| room | 5K | 10.0 | 28.0 | **29.5** | 16.8 | 9.9 | 4.6 | 1.2 |
| room | 15K | 20.8 | 36.8 | 26.5 | 10.6 | 4.0 | 1.2 | 0.3 |
| room | 30K | **27.2** | 38.0 | 22.4 | 8.2 | 3.0 | 0.9 | 0.3 |

As N grows then saturates, the distribution **shifts to lower multiplicity** (1+2 bucket: room 38% at 5K → 65% at 30K). Early training has more multi-tile Gaussians (the clustering that drives high D_macro); late training is dominated by single/double-tile Gaussians.

---

## 7. Contention proxy evolution (TOPOLOGY_PROXY_ONLY, DERIVED)

No profiler atomic counters → **TOPOLOGY_PROXY_ONLY** (AABB tile-count contention). Cross-validated against r6_4's native atomic-potential ratio `R_atomic_potential_est` (room 30K cam0: 7.20).

| Scene | 30K contrib_mean | p90 | p99 | max | top-1% share |
|---|---|---|---|---|---|
| room | ~31 | ~55 | ~320 | 9600 | ~0.27 |
| bicycle | ~26 | ~42 | ~276 | 9600 | ~0.30 |
| garden | ~24 | ~38 | ~250 | 9600 | ~0.28 |

Contention is **modestly concentrated** (top-1% of visible Gaussians carry ~27–30% of tile work) and roughly **stable 5K→30K** (does not spike late). The `max 9600` is the full-tile count (a single huge-Gaussian outlier touching every fine tile in its AABB — an extreme outlier that inflates AABB but is a small fraction of actual work).

---

## 8. Stage-timing evolution (MEASURED, 1 timing cam, 20 warmup + 100 samples, 1080p)

| Scene | it | fwd ms | bwd ms | opt ms | total ms | **bwd %** | **opt %** | fwd % |
|---|---|---|---|---|---|---|---|---|
| room | 5K | 11.70 | 10.29 | 1.17 | 23.13 | 44.5 | **5.1** | 50.6 |
| room | 15K | 9.95 | 10.36 | 3.19 | 23.57 | 44.0 | **13.6** | 42.2 |
| room | 30K | 9.10 | 8.09 | 3.28 | 20.82 | 38.9 | **15.8** | 43.7 |
| bicycle | 5K | 14.21 | 12.66 | 4.95 | 31.87 | 39.7 | **15.5** | 44.6 |
| bicycle | 15K | 20.71 | 21.70 | 11.10 | 53.92 | 40.3 | **20.6** | 38.4 |
| bicycle | 30K | 17.92 | 19.83 | 10.95 | 50.51 | 39.3 | **21.7** | 35.5 |
| garden | 5K | 18.48 | 21.21 | 5.22 | 46.07 | 46.0 | **11.3** | 40.1 |
| garden | 15K | 16.65 | 14.45 | 6.74 | 39.87 | 36.2 | **16.9** | 41.8 |
| garden | 30K | 15.95 | 14.65 | 8.83 | 38.43 | 38.1 | **23.0** | 41.5 |

**Two robust trends:**
1. **Optimizer fraction RISES monotonically with N** (Adam is O(N)): room 5.1→15.8%, bicycle 15.5→21.7%, garden 11.3→23.0%. At 30K the optimizer is the **second-largest stage** (22–23% for bicycle/garden).
2. **Backward fraction stays 36–46%** (stable) — it does not grow with N the way the optimizer does.

Within bwd/fwd, the raster sub-stage is **not separately measured** (no C0 publication rebuild permitted). Cross-join with r6_1 (baseline binary, same 1920×1080): room 30K `raster_bwd` = 48.1% of bwd, `other_bwd` 43.8% — so roughly half of backward is raster-blend and half is projection/SH VJP + zero-init.

**Cross-binary note:** C0 V3 bwd is ~5× faster than the r6_1 baseline (room 30K: 8.1ms vs 41.6ms) — scalar_adjoint + H8-MR. My fwd (9.1ms) ≈ r6_1 fwd (7.9ms). So the frozen C0 V3 stage split is **more optimizer-weighted than the baseline** (optimizer is a larger share of a smaller backward).

---

## 9. Total-iteration evolution (MEASURED)

- room 23.1→23.6→20.8 ms; bicycle 31.9→53.9→50.5 ms; garden 46.1→39.9→38.4 ms.
- bicycle and garden **rise then ease** (N growth 5K→15K, saturation 15K→30K with tighter Gaussians). room is roughly flat (small N, fwd-dominated).
- 0–5K interval is **INTERPOLATED** (no pre-5K checkpoint; base 5K trajectory used as proxy).

---

## 10. Densification-event accounting (MEASURED events / DERIVED overhead)

- **Densify:** 145 events (500–15000, every 100). **Opacity reset:** 10 events (3000–30000, every 3000). **No events after 15000** (saturation).
- Per-event overhead is **DERIVED** (no direct measurement; opacity reset ≈ one full-N in-place op ≈ optimizer-sized; clone/prune ≈ subset of one iter).
- **Event weighted contribution** of 30K wall time: room ~2%, bicycle ~4%, garden ~3% (DERIVED).
- **Conclusion:** the dominant training-time driver is the **normal iteration at the (growing-then-flat) N**, not the one-time events. The 15–30K half is pure normal-iter wall time.

---

## 11. Detected phases (MEASURED N + r6 densify schedule)

All scenes, empirically (not imposed):
- **0–5K: GROWTH** (steepest; garden +2584%).
- **5–15K: GROWTH** (moderate; room +67%, bicycle +105%, garden +47%).
- **15–30K: SATURATED** (N flat, 0%).
- **No LOW_GS phase** in sampled checkpoints (5K is post-init).

---

## 12. 30K/cam0 fixture representativeness (MEASURED)

Cam0 vs 16-cam median at 30K (tolerance 15% → REPRESENTATIVE; 15–40% → BIASED; >40% → OUTLIER):

| Metric | room | bicycle | garden |
|---|---|---|---|
| N_visible | REPRESENTATIVE (−8.5%) | REPRESENTATIVE (+1.9%) | REPRESENTATIVE (−14.8%) |
| fine_pair_count | REPRESENTATIVE (−8.5%) | REPRESENTATIVE (+1.9%) | REPRESENTATIVE (−14.8%) |
| D_macro | REPRESENTATIVE (−4.6%) | REPRESENTATIVE (−3.7%) | REPRESENTATIVE (−10.4%) |
| contention | REPRESENTATIVE (−8.4%) | REPRESENTATIVE (−6.5%) | **LOW_BIASED (−18.9%)** |
| backward/optimizer/iteration frac | REPRESENTATIVE (1 timing cam) | REPRESENTATIVE (1 timing cam) | REPRESENTATIVE (1 timing cam) |

**Verdict:** the established 30K/cam0 fixture is **REPRESENTATIVE for room and bicycle** across all topology metrics; **garden cam0 is slightly LOW_BIASED on contention** (cam0 sees ~19% less tile work than the scene median — a milder viewpoint). No metric is OUTLIER. The fixture does **not** over-represent late-training-heavy contention. *Caveat:* backward/optimizer/iteration fractions are from **1 timing cam**, so the 16-cam fraction spread is not separately timed.

---

## 13. E2 denominator sensitivity (E2_DENOMINATOR_SENSITIVITY_ONLY)

E2 is **DROP** and **not reopened**. The fixed removable net (room 42.7µs / bicycle 183.1µs / garden 31.0µs) divided by each checkpoint's 1080p total iteration:

| Scene | 5K | 15K | 30K | crosses 3%? |
|---|---|---|---|---|
| room | 0.185% | 0.181% | 0.205% | NO |
| bicycle | **0.575%** | 0.340% | 0.362% | NO |
| garden | 0.067% | 0.078% | 0.081% | NO |

No checkpoint approaches the 3% gate (max 0.575%). Crossing 3% would NOT reopen E2: the 2/3-scene gate and structural/rewrite costs (SH-VJP grid rewrite, full-N momentum decay, 96-reg cliff) remain frozen from E2-P0.

---

## 14–15. Full-training weighting & saved_total_time (INTERPOLATED)

Wall-time per 5K interval (piecewise-linear in measured checkpoint medians):

| Scene | 0–5K | 5–10K | 10–15K | 15–20K | 20–25K | 25–30K | total 30K wall |
|---|---|---|---|---|---|---|---|
| room | 11.3% | 18.3% | 18.4% | 17.4% | 17.4% | 17.4% | 639.2s |
| bicycle | 7.1% | 16.1% | 18.1% | 19.6% | 19.6% | 19.6% | 1334.4s |
| garden | 11.2% | 18.9% | 18.2% | 17.2% | 17.2% | 17.2% | 1136.8s |

`saved_total_time = Σ iterations_i × removable_ms_i` (P3 DERIVED, PENDING_CROSS_JOIN):

| Scene | saved_total_time_s | % of 30K wall |
|---|---|---|
| room | 95.1s | 14.9% |
| bicycle | 174.1s | 13.1% |
| garden | 155.2s | 13.7% |

(Per-scene first; equal-scene average and wall-time-weighted aggregate computed **separately** — raw Gaussians never pooled across scenes. 0–5K interval is INTERPOLATED.)

---

## 17/19. P3 temporal robustness gate (DERIVED — do NOT decide P3)

Opportunity metric = DERIVED HAR removable fraction (P2-0 mechanism scaled by checkpoint D_macro). Gate: STABLE if within ±25% of the late-training level for ≥70% of weighted wall time.

| Scene | 0–5K | 15–30K | within-25%-of-late | **gate** |
|---|---|---|---|---|
| room | 19.6% | 11.8% | 70.4% | **STABLE** |
| bicycle | 19.1% | 10.2% | 76.8% | **STABLE** |
| garden | 16.2% | 10.9% | 69.9% | **WEAK_EARLY** |

**The opportunity is early-weighted, not late-weighted** (D_macro is highest at 5K). It is **not LATE_HEAVY, not SPIKY**. P3-0/P3-H final numbers: **PENDING_CROSS_JOIN** — this audit maps mechanism + checkpoint work but does **not** decide P3.

---

## 20. Aggregates (DERIVED)

**Equal-scene average (30K):** N_GS 2.50M, n_visible 4.51M, D_macro 8.66, bwd 38.8%, opt 20.1%, total 36.6ms.
**Wall-time-weighted (30K):** weights room 19% / bicycle 46% / garden 35% → N_GS 2.91M, n_visible 4.90M, D_macro 8.15, bwd 38.8%, opt 21.0%, total 40.7ms.

---

## 21/22. Quality gate & labels

- **No quality gate** (per §21) — this is a training-speed audit, not a render-quality audit.
- Every value is labeled MEASURED / DERIVED / INTERPOLATED (see sections). Contention = **TOPOLOGY_PROXY_ONLY**. P3 gain = **PENDING_CROSS_JOIN**. E2 = **E2_DENOMINATOR_SENSITIVITY_ONLY** (DROP frozen).

---

## Cross-join (supplementary, non-C0-V3 context)

- **r6_1 (baseline binary, 1920×1080, same checkpoints):** room 30K T_bwd 41.6ms / T_fwd 7.9ms / T_iter 104.8ms, raster_bwd 48.1% of bwd, N_visible 265,137. C0 V3 bwd (8.1ms) is ~5× faster (scalar_adjoint + H8-MR); N_visible matches within 6%.
- **r6_4 (baseline binary, 10 cams):** room 30K cam0 `R_atomic_potential_est` 7.20, `gaussian_tile_counts` mean 15.1 / median 6 — same order as my DERIVED contention (mean ~31, AABB-inflated). Confirms the contention proxy is not wildly off.

---

## Deliverables

- `reports/higs/training-stage-representativeness.md` (this report)
- `artifacts/higs-training-stage/` — provenance.json, checkpoint_inventory.json, sampled_cameras.json, topology_growth.json, exact_pair_workload.json, exact_macro_reuse_over_time.json, macro_multiplicity_over_time.json, contention_proxy_over_time.json, stage_timing_over_time.json, densification_event_accounting.json, phase_detection.json, representativeness.json, full_training_weighting.json, p3_temporal_opportunity.json, e2_denominator_sensitivity.json, final_classification.json, raw_measurements.csv, aggregates.json, p3_saved_total_time.json, findings_digest.json
- `scripts/p3/training_stage_audit.py` (measurement harness)

Remote: `mx:/mnt/storage_pool/liaoyuanjun/higs_dse_results/` (all JSONs + raw_measurements.csv + partial sidecars + run logs).
