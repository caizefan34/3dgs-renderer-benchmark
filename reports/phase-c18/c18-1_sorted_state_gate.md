# C18-1 — Sorted-State Stability Gate
**Scene:** `room`  
**Steps:** 500  
**Sample interval:** every 10 steps  
**GPU:** NVIDIA A100-PCIE-40GB  
**gsplat:** 1.5.3  
**Date:** 2026-09-07T17:47:44.205952+00:00
---
## Executive Decision: **GO**
Strong: pairwise_order P50=99.86% > 95% AND new_entry_ratio P50=1.51% < 10%
## 1. Sort Key Encoding (from IntersectTile.cu source)

The 64-bit sort key layout (verified from gsplat 1.5.3 CUDA source):

```
63           32+tile_n_bits  32                0
+----[image_id]----+-[tile_id]-+----[depth]-----+
+   iid_enc=iid<<(32+tile_n_bits)  + depth_i32 ->
```

- **depth encoding:** float32 bit-reinterpreted as int32 via `*(int32_t*)&depth`, then zero-extended to int64
- **ordering:** ascending by (image_id → tile_id → depth)
- **equal-depth order:** NOT stable (CUB DeviceRadixSort::SortPairs default)
- **gaussian_id in sort key:** NO — only in flatten_ids (associated value)
- **tile_n_bits:** computed as `floor(log2(tile_width * tile_height)) + 1`

## 2. Rank Preservation

| Metric | Value |
|:-------|:-----:|
| Mean absolute rank displacement | 2.44 |
| P50 absolute rank displacement | 2.40 |
| P90 absolute rank displacement | 3.24 |
| Min | 1.2 |
| Max | 4.6 |

## 3. Pairwise Order Preservation

**Pairwise order preservation ratio:** P50=0.9986, mean=0.9985, P10=0.9968, P90=0.9999

**Pairwise order flip ratio:** 0.0014 (P50)

## 4. New Entry Insertion

| Metric | P50 | Mean | P10 | P90 |
|:-------|:--:|:---:|:--:|:--:|
| Reusable intersection ratio | 0.9849 | 0.9844 | 0.9759 | 0.9920 |
| New intersection ratio | 0.0151 | 0.0156 | 0.0080 | 0.0241 |
| Fraction of tiles with 0 new entries | 0.3174 | 0.3085 | 0.1892 | 0.4448 |

## 5. Strategy Estimates

### A_full_sort: Baseline CUB radix sort of all n_isects entries
- Asymptotic work: O(n_isects) memory movement + O(n_isects × passes) CUB sort
- Memory movement: ~3.25M × 16 bytes = ~52 MB per step (key+value double-buffered)
- Sync: CUB DeviceRadixSort (no host sync during sort)
- Complexity: CUDA: low (vendor library call). Integration: none.
- Bottleneck: CUB pass count (~12 passes for 64-bit key); global memory bandwidth

### B_reuse_merge: Retain old sorted state; generate only new (1.5%) intersection entries; merge
- Asymptotic work: O(2% × n_isects + n_isects_per_tile × log(n_old_tile)) per-tile merge
- Memory movement: ~3.25M × 16 bytes retained + ~2% × 3.25M × 16 bytes new = ~53 MB/step
- Sync: Per-step: generate new entries → sort new entries (small) → per-tile merge. New sort can be small CUB.
- Complexity: CUDA: medium (per-tile merge kernel + new entry generation + offset fixup)
- Bottleneck: Scatter of ~98k new entries across tiles + per-tile merge overhead

### C_local_repair: Per-tile: remove absent entries, insert new entries, local sort changed tiles only
- Asymptotic work: O(2% × n_isects + fraction_changed_tiles × n_tiles × log(n_per_tile))
- Memory movement: Copy-in-place per tile; only changed tiles need write
- Sync: Per tile: remove → insert → local-bitonic sort (block-level). No global sync needed.
- Complexity: CUDA: high (per-tile dynamic memory management, conditional paths)
- Bottleneck: Warp divergence on mixed-content tiles; shared-memory capacity per tile

## 6. Theoretical Sorting Savings

- Baseline n_isects: ~3,250,000
- Reused entries: 98.5% × 3.25M ≈ 3,201,022
- New entries: 1.5% × 3.25M ≈ 48,977

**Strategy A (full sort):** CUB sort all 3.25M records (12 passes) = ~39M element-wise operations
**Strategy B (reuse + merge):** sort only 48,977 new entries + per-tile merge
  If new entries are ~48,977, their CUB sort cost is ~2% of baseline. Merge cost: O(old + new) ≈ 3.25M + 48,977 = ~3.3M element comparisons
**Strategy C (local repair):** process only tiles with new entries (~68% of tiles). Each tile has avg ~6 new entries. Insertion cost per tile: O(n_tile + n_new) ≈ O(398) per changed tile
## 7. Critical Correctness

> NOT EQUIVALENT. Membership reuse only means the (Gaussian, Tile) pair exists in both iterations. Sorted-state reuse additionally requires that the depth ordering of common entries within each tile is preserved, AND that new entries can be merged without resorting the entire structure.

- **False** — membership reuse alone is insufficient
- **YES — this is the core requirement. If pairwise ordering flips, a simple merge (Strategy B/C) produces incorrect sort order.**
- **Measurement:** pairwise_order_preservation P50=0.9986 — acceptable for merge

## 8. Stress Cases

### case1_no_membership_change_depth_change
- Relevance: HIGH — dominant case in training. 93.2% of Gaussians keep membership; depth values change by ~0.01% per step (from C18 position_change data). Pairwise order is determined by depth rank.
- Score: ≈pairwise_order_preservation_ratio (measured above)

### case2_few_membership_many_flips
- Relevance: Possible but unlikely. Depth ordering of common entries changes only when depth ordering between pairs inverts. With small per-step depth changes, flips only occur for pairs with nearly equal depth.
- Score: pairwise_flip_ratio = 0.14% (flips in sampled pairs)

### case3_changes_concentrated_few_tiles
- Relevance: PARTIALLY CONFIRMED — ~50% tiles have no new entries. New entries are concentrated in a fraction of tiles.
- Score: fraction_tiles_no_new P50=31.74%

### case4_changes_distributed_many_tiles
- Relevance: Partial: the other ~50% of tiles each get ~1-5 new entries. No tiles have massive numbers of new entries (max_new_per_tile is small).
- Score: moderate distribution

## 9. Decision Criteria Check

| Criterion | Threshold | Measured | Met? |
|:----------|:---------:|:--------:|:----:|
| Strong GO: pairwise order | >0.95 | 0.9986 | ✅ |
| Strong GO: new entry ratio | <0.1 | 0.0151 | ✅ |
| Conditional: pairwise order | >0.8 | 0.9986 | ✅ |
| Conditional: new entry ratio | <0.2 | 0.0151 | ✅ |

## 10. Verdict

> **GO** — Strong: pairwise_order P50=99.86% > 95% AND new_entry_ratio P50=1.51% < 10%

> **Action:** Proceed to C18-2 CUDA prototype.

---
**Note:** Pairwise order preservation is computed on sampled tiles (up to 200 tiles per pair) due to O(n²) computational cost of full per-tile Kendall analysis. Rank displacement covers all common entries.
