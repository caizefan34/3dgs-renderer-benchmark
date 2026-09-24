# H7-B0 — HiGS Macro-Owned Gaussian Backward Roofline Oracle

**Gate: ROOFLINE_ORACLE (read-only). Production H7-B CUDA kernel NOT implemented.**

**Date:** 2026-09-21
**Host:** mx (A100-PCIE-40GB, cuda:4)
**Representation:** H3-FWD-1A-R2 exact macro — `macro_sorted_ids` + `uint32 fine_tile_masks` (from `higs_train_macro_f4`), gsplat_cuda.so sha256 `361b216b…448c98`
**last_ids:** H5 forward `rasterize_to_pixels_3dgs` (`LAST_ID_ABSOLUTE`)
**Method:** read-only structural oracle, vectorized numpy, no production CUDA modified.

**Verdict: `H7B0_WEAK` — H7-B production CUDA is NOT authorized.**

---

## Provenance & authoritative anchors

| Scene | macro entries | fine pairs | B2 backward (ms) | B2 fwd raster (ms) | C_macro (ms) |
|-------|---------------|------------|------------------|--------------------|--------------|
| room   | 124,017 | 953,144 | 2.870 | 0.6717 | 0.259144 |
| bicycle| 311,610 | 1,412,189 | 3.775 | 0.9103 | 0.151552 |
| garden | 66,682  | 533,928 | 1.51  | 0.4127 | 0.039936 |

All figures recomputed from the exact R2 representation (not pre-R2).

---

## 1. Exact geometric reuse (fine_tile_mask popcount)

Per-macro-entry `fine_tile_mask` popcount distribution.

| Metric | room | bicycle | garden |
|--------|------|---------|--------|
| mean | 7.69 | 4.53 | 8.01 |
| p10 | 1 | 1 | 1 |
| p25 | 2 | 2 | 2 |
| p50 | 5 | 3 | 5 |
| p75 | 10 | 5 | 11 |
| p90 | 18 | 9 | 20 |
| p95 | 26 | 14 | 26 |
| p99 | 32 | 31 | 32 |
| max | 32 | 32 | 32 |

Bucket fractions (entry %):

| Bucket | room | bicycle | garden |
|--------|------|---------|--------|
| 1 | 10.8% | 19.5% | 11.3% |
| 2 | 14.8% | 25.3% | 14.5% |
| 3–4 | 20.9% | 24.7% | 19.4% |
| 5–8 | 24.6% | 19.8% | 23.2% |
| 9–16 | 17.5% | 7.1% | 17.8% |
| 17–24 | 5.7% | 1.7% | 8.2% |
| 25–32 | 5.7% | 1.8% | 5.6% |

Raw geometric cross-fine-tile reuse is confirmed near the expected ~4.5–8× but is concentrated: most entries cover few tiles, with a long tail of heavy 25–32-tile owners.

## 2. Backward-effective reuse (not just geometric)

Combining `fine_tile_mask` with `last_ids` (`L_tile[ft] = max_pixel(last_ids)`, then per-macro-entry `rank ≤ L_local[ft]` gating):

| Scene | R_geom | R_backward | backward-active macro entries | backward-active fine pairs | dead macro fraction |
|-------|--------|-----------|-------------------------------|---------------------------|---------------------|
| room | 7.686 | **6.866** | 119,696 | 821,777 | 3.48% |
| bicycle | 4.532 | **4.365** | 306,512 | 1,337,828 | 1.64% |
| garden | 8.007 | **7.721** | 62,148 | 479,864 | 6.80% |

R_backward is the real reuse ceiling: only ~4.4–7.7× per macro owner is backward-active. Real reuse still comfortably exceeds 2.0× on all 3 scenes → **Gate A PASS**.

## 3. 32-G group structure

Partition depth-sorted macro entries into 32-G owner groups.

| Metric (mean) | room | bicycle | garden |
|---------------|------|---------|--------|
| n_groups | 4,048 | 9,918 | 2,246 |
| active fine tiles / group | 24.98 | 22.32 | 25.53 |
| active Gaussian lanes / tile | 8.13 | 6.04 | 8.37 |
| mask density / group | 0.0298 | 0.0307 | 0.0290 |
| active tiles / lane (utilization) | 6.63 | 4.29 | 7.20 |

The 32-G owner warp is reasonably dense per tile (~6–8 active lanes of 32) and each lane covers ~4–7 active tiles, so the owner warp is not pathologically empty. But mask density is dominated by the 1/32=0.03125 quantization (most groups at the 0.03125 median), i.e. coupled tiles dominate.

## 4. Logical attribute-load roofline

Accounting model (means2d=2, conic=3, opacity=1, color=3, id=1 slots; 4B/slot → 40B/unit):
baseline = 1 ownership/load unit per backward-active fine-tile×Gaussian; H7-B ideal = 1 unit per active macro owner.

| Scene | baseline units | H7 ideal units | logical reduction % | baseline load slots | H7 slots |
|-------|----------------|----------------|--------------------|--------------------|----------|
| room | 821,777 | 119,696 | **85.4%** | 8,217,770 | 1,196,960 |
| bicycle | 1,337,828 | 306,512 | **77.1%** | 13,378,280 | 3,065,120 |
| garden | 479,864 | 62,148 | **87.0%** | 4,798,640 | 621,480 |

Logical load reduction is 77–87% across all 3 scenes.

## 5. Cache-aware physical traffic ceiling — MANDATORY

Working set = `n_visible × 40B`. All 3 scenes fit << L2 (A100 40MB): room 1.8MB, bicycle 7.3MB, garden 0.98MB. Cross-tile reuse is therefore served by L1/L2 resident Gaussian attribute state; the logical reuse largely does **not** reach DRAM.

| Scene | logical/L2 request reduction | DRAM attr bytes (baseline) | DRAM reduction ceiling |
|-------|------------------------------|---------------------------|------------------------|
| room | 85.4% | 1,796,320 | **0.0%** |
| bicycle | 77.1% | 7,261,000 | **0.0%** |
| garden | 87.0% | 979,320 | **0.0%** |

**Physical DRAM reduction ceiling ≈ 0%.** L2 request reduction is high (77–87%), but on-device backward is compute-bound and the attribute bytes are already L1/L2-resident. Hardware L1/L2/DRAM replay measurement is deferred but the analytic cache model already caps DRAM saving at ~0 → **Gate B (physical DRAM ≥25%) FAIL** (passes only via the gradient 35% door, see §6).

## 6. Gradient-write / atomic roofline

Logical gradient ownership units: fine-tile owner flushes vs one aggregate macro-owner flush.

| Scene | v_means2d/conic/color/opacity current flushes | macro-owner flushes | reduction factor | reduction % |
|-------|------------------------------------------------|--------------------|------------------|-------------|
| room | 821,777 | 119,696 | 6.87 | **85.4%** |
| bicycle | 1,337,828 | 306,512 | 4.36 | **77.1%** |
| garden | 479,864 | 62,148 | 7.72 | **87.0%** |

Logical gradient-flush reduction is 77–87%, satisfying the **≥35%** door → **Gate B PASS** via gradient door. Note: on-device backward is compute-bound and atomics are only ~6% of backward time (H2-BWD-0), so this reduction translates to a small fraction of wall time (see §11). Actual hardware atomic counts are not separately measured.

## 7. Frontier-gating incremental value

Exact R2 `fine_tile_masks` reconstruct exactly the B2 fine pairs, so **HiGS mask ALONE removes 0 additional work**. All frontier gain comes from last-id gating.

| Scene | baseline fine pairs | mask+last_id pairs | last-id removed % | pixel visits delta | grad units removed % |
|-------|--------------------|--------------------|--------------------|--------------------|---------------------|
| room | 953,144 | 821,777 | **13.8%** | 0 | 13.8% |
| bicycle | 1,412,193 | 1,337,828 | **5.3%** | 0 | 5.3% |
| garden | 533,928 | 479,864 | **10.1%** | 0 | 10.1% |

Per-pixel arithmetic is unchanged (0%): the per-pixel reachable set is already fixed by last_ids, and tile-level frontier gating does not reduce per-pixel visits (they were already pruned by the exact per-pixel last-id frontier). Only attribute loads / gradient ownership units gain from last-id gating (5–14%).

## 8. Pixel-work warning

| Scene | Design A (scan all 256 px × active tiles) | Design B (scan baseline-active px only) | Design B same-as-baseline | Design A explosion |
|-------|-------------------------------------------|-----------------------------------------|---------------------------|--------------------|
| room | 210,374,912 | 178,086,883 | 1.0× | 1.18× |
| bicycle | 342,483,968 | 279,871,958 | 1.0× | 1.22× |
| garden | 122,845,184 | 105,000,851 | 1.0× | 1.17× |

Design A does **not catastrophically explode** (≤1.22×) because masked active-tile scanning is bounded; but it adds ≤22% redundant work for zero benefit. **Design B (reuse baseline-active pixel positions) is mandatory** — it adds no new pixel work and no new contribution metadata. Director will require Design B.

## 9. Owner-warp load balance — HARD GATE

lane work = per-Gaussian backward-active fine-tile count (Design-B pixel work scales with it).

| Scene | mean max/mean | **p90 max/mean** | p90 p90/mean | p99 p90/mean | max max/mean |
|-------|---------------|------------------|--------------|--------------|--------------|
| room | 3.36 | **4.97** | 2.61 | 3.63 | 32.0 |
| bicycle | 3.74 | **6.55** | 2.44 | 3.48 | 21.3 |
| garden | 3.21 | **4.38** | 2.51 | 4.00 | 32.0 |

`max_over_mean_per_group: hard_gate_p90_of_max_over_mean_le4` = **FALSE on all 3 scenes** (p90 max/mean 4.97 / 6.55 / 4.38, all > 4). p90/p90 mean stays ≤4 (2.44–2.61) and p99 mean ≤4, but the strict max-lane imbalance fails the 4× hard gate. There is no low-cost work-splitting remedy trialed in this phase — lane=max-over-mean is structurally driven by a minority of heavy 25–32-tile owners. **Gate C FAIL.**

## 10. No-new-metadata design

`required_persistent_metadata`: all derived from already-persisted H3-R2 (`macro_sorted_ids`, `fine_tile_masks`, `macro_offsets`) + H5 `last_ids` checkpoint. **new_persistent_bytes = 0**, no per-pixel×group masks, extra read/write cost = 0. Verdict: PASS (~0 new persistent metadata) → **Gate E PASS.**

## 11. Net overhead accounting (net roofline)

Model: backward is compute-bound; atomics ≈ 6% of backward; DRAM reduction ceiling ~0% (L1/L2 already capture cross-tile reuse). real backward saving = atomic-flush reduction only; optimistic credits half the logical L2 reduction into real time; pessimistic = 0. Net always subtracts forward-side exact-macro-F4 construction penalty C_macro.

| Scene | F+B (ms) | C_macro % of F+B | realistic backward saving (ms) | **realistic net F+B %** | optimistic net % | pessimistic net % |
|-------|----------|------------------|-------------------------------|------------------------|------------------|-------------------|
| room | 3.542 | 7.32% | 0.147 | **−3.16%** | +3.76% | −7.32% |
| bicycle | 4.685 | 3.23% | 0.175 | **+0.49%** | +6.70% | −3.23% |
| garden | 1.923 | 2.08% | 0.079 | **+2.02%** | +8.86% | −2.08% |

Optimistic nets are strong (+3.8 to +8.9%, hitting the ≥5% preferred gate), but they rely on crediting half the L2-load reduction into real backward time — contradicted by the §5 cache model. Realistic net F+B is **+0.5% / +2.0% / −3.2%**, with room negative. Only bicycle/+2% qualifies any value ≥3%? No — 0 of 3 scenes reach +3% realistic net → **Gate D FAIL.**

## 12. Innovation Gate — summary

| Gate | Threshold | room | bicycle | garden | Verdict |
|------|-----------|------|---------|--------|---------|
| A | R_backward ≥2 on ≥2/3 | 6.87 | 4.36 | 7.72 | PASS (3/3) |
| B | phys≥25% OR grad≥35% on ≥2/3 | 0/85.4 | 0/77.1 | 0/87.0 | PASS (grad door) |
| C | p90 max/mean ≤4× | 4.97 | 6.55 | 4.38 | **FAIL** |
| D | realistic net ≥3% on ≥2/3 | −3.16 | +0.49 | +2.02 | **FAIL** |
| E | no large metadata | — | — | — | PASS |
| **ALL** | A–E all true | | | | **FALSE** |

**Classification: `H7B0_WEAK`.**

Rationale: Logical/geometric compression is real (R_backward 4.4–7.7×, logical load/flush reduction 77–87%, zero new metadata), but it does **not** survive the physical/net accounting — cross-tile attribute reuse is already captured by L1/L2 (DRAM ceiling ~0%), the gradient reduction lands entirely in a compute-bound kernel's ~6% atomic share, warp own-lane imbalance fails the hard 4× gate, and the C_macro forward penalty makes realistic net F+B only −3.2% to +2.0%.

---

## Final answers (15 deliverables)

1. **Exact geometric reuse:** mean popcount 7.69 / 4.53 / 8.01; heavy-tailed; buckets tabulated in §1.
2. **Backward-effective reuse:** R_backward 6.87 / 4.36 / 7.72; backward-active fine pairs 821,777 / 1,337,828 / 479,864.
3. **32-G mask density:** ~0.029–0.031 per group; ~6–8 active lanes/tile; ~4.3–7.2 active tiles/lane.
4. **Logical attribute-load reduction:** 85.4 / 77.1 / 87.0% (means2d/conic/opacity/color/id all share per-attribute ratios).
5. **Physical DRAM/L2 ceiling:** L2 request reduction 77–87%, **DRAM reduction ceiling ~0%** (working set ≤7.3MB fits L2).
6. **Gradient flush/atomic reduction:** 85.4 / 77.1 / 87.0% logical (factor 6.87 / 4.36 / 7.72); hardware atomics not separately measured.
7. **Last-id frontier increment:** removes 13.8 / 5.3 / 10.1% of attribute loads & grad units beyond HiGS mask alone (mask alone adds 0); pixel visits unchanged.
8. **Pixel-work change:** Design A ≤1.22× (not catastrophic but wasteful); Design B = 1.0× (no new pixel work).
9. **Owner-warp load imbalance:** p90 max/mean = 4.97 / 6.55 / 4.38 — **all > 4×; GATE C FAIL**.
10. **Metadata requirement:** ~0 new persistent bytes; reuses H3-R2 macros + H5 last_ids; no per-pixel×group masks. Gate E PASS.
11. **Macro-F4 overhead included:** C_macro = 7.32% / 3.23% / 2.08% of F+B, subtracted in net model.
12. **Optimistic / realistic / pessimistic net F+B:** obs: +3.76 / +6.70 / +8.86%; realistic: **−3.16 / +0.49 / +2.02%**; pess: −7.32 / −3.23 / −2.08%.
13. **Classification:** **H7B0_WEAK**.
14. **H7-B production CUDA authorized:** **NO**.
15. **Artifact paths:**
   - Report: `reports/higs/h7-b0-macro-owner-roofline.md`
   - `artifacts/higs-h7-b0/exact_macro_reuse.json`
   - `artifacts/higs-h7-b0/backward_effective_reuse.json`
   - `artifacts/higs-h7-b0/group32_masks.json`
   - `artifacts/higs-h7-b0/logical_load_roofline.json`
   - `artifacts/higs-h7-b0/physical_traffic_roofline.json`
   - `artifacts/higs-h7-b0/gradient_flush_roofline.json`
   - `artifacts/higs-h7-b0/frontier_increment.json`
   - `artifacts/higs-h7-b0/pixel_work.json`
   - `artifacts/higs-h7-b0/warp_balance.json`
   - `artifacts/higs-h7-b0/metadata_cost.json`
   - `artifacts/higs-h7-b0/net_roofline.json`
   - `artifacts/higs-h7-b0/analysis.json`
   - `artifacts/higs-h7-b0/provenance.json`

## Recommendation

Do not build H7-B macro-ownership in its current form. If pursuit is later warranted, the only viable direction is a **work-split/load-balanced variant** (remedy §9, design B for §8), plus a real hardware L1/L2/DRAM replay measurement to displace the §5 analytic ceiling — but the net-accounting model (§11) already indicates the realistic opportunity is far below the 3% gate.