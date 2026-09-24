# C17-2 Implementation Gate Review

**Date:** 2026-10-19  
**Reviewing:** Active Pixel Mask early-exit design  
**Status:** ❌ **GATE NOT PASSED — Design disproven**

---

## 1. Six Gate Questions

> The analysis below is based on the exact kernel code at `RasterizeToPixels3DGSFwd.cu:40-188`.

### Q1: Pixel from active → inactive 的精确定义是什么？

If we define `inactive` identically to `done`:

```
active = (inside == true) AND (T >= 1e-4f)
inactive trigger: next_T <= 1e-4f   (same line that sets done = true)
```

Corresponding CUDA:

```cuda
// During compositing — exact same line where baseline sets done = true
if (next_T <= 1e-4f) {
    done = true;        // baseline
    active = false;     // C17-2 addition
    atomicSub(active_count, 1);  // C17-2 addition
}
```

---

### Q2: Inactive 条件与现有 done 条件的严格区别

**不存在区别。** Both trigger from the identical condition:

| Aspect | `done` condition | `active` condition |
|--------|-----------------|-------------------|
| Trigger event | `next_T <= 1e-4f` | `next_T <= 1e-4f` |
| Out-of-bounds pixel | `done = true` initially | Would initialize `active = false` |
| Semantic | "pixel stops processing" | "pixel no longer needs processing" |
| Tile-level exit check | `__syncthreads_count(done) >= block_size` | `active_count == 0` |

Since `active_count` is simply `block_size - syncthreads_count(done)`:

```
active_count == 0  ⇔  __syncthreads_count(done) >= block_size
```

**The two conditions are logically equivalent.** Both check whether all `block_size` threads' pixels have saturated.

---

### Q3: 是否存在 `active == false` 但 `done == false` 的状态？

**不存在。**

Proof: Both `active = false` and `done = true` are set by the same code path:

```cuda
if (next_T <= 1e-4f) {
    done = true;     // ← same condition
    active = false;  // ← same condition
    // ...
}
```

There is no code path where one is set without the other. The `continue` case (alpha < ALPHA_THRESHOLD) modifies neither.

---

### Q4: `active_count == 0` 能否发生在 `all_done` 之前？

**No.**

From the equivalence proved in Q2:

```
active_count == 0   ⇒   every thread's done == true
                      ⇒   __syncthreads_count(done) >= 256
                      ⇒   baseline barrier also triggers tile exit
```

The `syncthreads_count(done)` is checked at the **beginning** of each batch iteration, before any loading or compositing for that batch. The `active_count == 0` would also be checked at the same point. Both see the state after the previous batch's compositing.

**No code path allows C17-2 to detect all-done before the baseline does.**

---

### Q5: Deterministic counterexample — 能否构造？

**无法构造。**

To prove C17-2 exits earlier, we need:

```
Baseline:  __syncthreads_count(done) < 256  (keeps running)
C17-2:     active_count == 0                 (exits)
```

But Q2 proves this is impossible: `active_count == 0 ⇒ syncthreads_count(done) == 256`.

#### Toy Example — Full Trace

```
Tile: 256 × 256 pixels (block_size = 256)
Gaussians in tile: 2000

Pixel 0 (background, sky):   T stays 1.0 forever — no Gaussian covers it
Pixel 1..255 (foreground):   T → 1e-4 after Gaussian #50

Batch 0 (Gaussian 0..255):
  After compositing: pixels 1..255 done=true, pixel 0 done=false
  Batch check: syncthreads_count(done) = 255 < 256 → continue
  active_count = 1 ≠ 0 → continue

Batch 1..7 (Gaussian 255..2000):
  After compositing: pixel 0 still T=1.0 (no covering Gaussian in any batch)
  Batch check: syncthreads_count(done) = 255 < 256 → continue every time
  active_count = 1 ≠ 0 → continue every time

Result: Neither exits early. Both process all 2000 Gaussians.
```

**结论：不存在能让 C17-2 比 baseline 提前退出的 workload。**

---

### Q6: 如果 active mask 只是让 all_done 检查更便宜，请明确指出

**确实如此。** 结论必须明确：

| | Baseline | C17-2 active_set | 区别 |
|--|----------|-----------------|------|
| Tile-level exit condition | `all(256 threads).done` | `all(256 threads).inactive` | **语义等价** |
| **Worker load entry** | `syncthreads_count(done) >= 256` | `active_count == 0` | **同时触发** |
| 硬件操作 | Full block barrier + warp vote reduction | Shared memory uint32 read + compare | C17-2 ≈ 2 cycles, baseline ≈ ~50 cycles |
| **Compositing workload** | Identical count of alpha/Gaussian evaluations | **Identical** | **0% reduction** |
| **Global memory traffic** | Identical number of Gaussian attribute loads | **Identical** | **0% reduction** |
| **Synchronization saving** | Each batch: ~50 cycles → ~2 cycles | ~48 cycles saved per batch | **~0.27μs per tile (negligible)** |

**收益仅仅是 `__syncthreads_count` barrier 的 cycle 代价优化**，不是 workload reduction。

计算上限：
- 假设 8 batches × ~48 cycles saved per batch = ~384 cycles per tile
- @1.5 GHz: ~0.256 μs per tile
- For 8,160 tiles: ~2.1 μs total
- Out of ~13ms rasterization time: **~0.016% savings**

**这不是一个可以被认可的优化。**

---

## 2. Gate Verdict

### ❌ GATE NOT PASSED

**Active Pixel Mask（C17-2 v1）被证明与 baseline 的 `done` condition 完全等价。它不提供任何 compositing workload 减少。提交的 C17-2 v1 设计被 FALSIFIED。**

根本原因：

> 当前 kernel 已经具备 `done` flag 和 `__syncthreads_count(done)` 的 tile-level early exit。两者使用相同的 `next_T <= 1e-4f` 条件。**只要有一个 pixel 的 T 永远不降到 1e-4 以下（背景像素永远不会被任何 Gaussian 覆盖），整个 tile 就无法提前退出。** 无论用 done 还是 active_set 都无法绕过这个问题。

```
FALSIFIED: Active Pixel Mask reduces compositing workload
EVIDENCE:  active_count == 0 ⇔ syncthreads_count(done) >= 256
           Both check the same condition at the same point in the code
```

---

## 3. Corrected C17-2 Design Proposal

Since the active mask approach is disproven, I propose C17-2 pivot to a mechanism that genuinely reduces workload.

### New Target: Two-Phase Sorting (from Phase 17A original design)

C17-2 (Two-Phase Sorting) addresses the **sort stage** (21% of forward time for bicycle t16), not the rasterization stage.

#### Baseline sort

```
Single global CUB radix sort on 31-bit key (C1 applied):
  depth_upper(16) | tile_id(14) | image_id(1)
  8 CUB radix passes
  Result: tile groups contiguous AND depth-ordered simultaneously
  Memory traffic: n_isects × 16B × 8 = 6.08M × 16 × 8 = 778 MB
```

#### C17-2: Two-Phase Sort

```
Phase A: Sort by (tile_id | image_id) only — 15-bit key
  begin_bit=0, end_bit=15
  CUB DeviceRadixSort::SortPairs → 4 CUB passes
  Result: all items for same (image, tile) are contiguous
  Item ordering WITHIN each tile is arbitrary (depth-unsorted)

Phase B: Per-tile depth sort in shared memory
  After Phase A: tile ranges are contiguous → offset kernel gives tile boundaries
  New kernel: for each tile, load isect_ids+flatten_ids range into shared memory
  Sort by depth_upper (16 bits) using block-level bitonic sort
  Write sorted range back to global memory
  One block per tile (8,160 blocks)
  Each block sorts mean ~746 items using bitonic sort: O(n log² n) comparators

Result: same as baseline (tile-contiguous, depth-sorted) but through different path
```

#### Workload comparison

| Metric | Baseline (C1) | C17-2 Phase A | C17-2 Phase B |
|--------|:-------------:|:-------------:|:-------------:|
| **Sort key width** | 31 bits | **15 bits** | N/A |
| **CUB passes** | **8** (31/4=8) | **4** (15/4=4) | 0 |
| **CUB traffic per item** | 8 × 16B = **128B** | 4 × 16B = **64B** | 0 |
| **Total CUB traffic** | **778 MB** | **389 MB** | 0 |
| **Shared memory sort** | None | None | Per-tile bitonic |
| **Per-tile comparisons** | N/A (global sort) | N/A | ~746 × log²(746) = 746 × 87 = ~65K |
| **Total per-tile sort** | N/A | N/A | 8,160 tiles × 65K ≈ 530M cmp (in shared mem, fast) |

**Expected: Phase A 3-4× faster than full CUB sort, Phase B adds minimal overhead (~0.1ms).**

#### Why this is different from C17-1

| Aspect | C17-1 (Tile-Local Queues) | C17-2 (Two-Phase Sort) |
|--------|--------------------------|----------------------|
| Data structures | Replaces isect_ids/flatten_ids entirely | **Keeps same data structures** |
| Memory | Large pre-allocated per-tile buffers | **Same as baseline** |
| CUB sort | Eliminated entirely | **Reduced to 4 passes from 8** |
| Offset kernel | Eliminated | **Unchanged** |
| Rasterization | Must consume new buffer format | **Unchanged** |
| Backward | Must verify compatibility | **Unchanged** |
| Implementation risk | High (new kernel, binding, consumption) | **Medium (add Phase B sort kernel)** |

C17-2 keeps the existing data structures and only changes how the **sort** step works — it's an incremental improvement to the existing sort pipeline. C17-1 would eliminate the sort entirely with a fundamentally different data structure.

#### Composability

C17-2 is compatible with both C1 (already applied) and C17-1 (if implemented):
- C1 + C17-2: C1 reduces sort from 12→8 passes; C17-2 further reduces to 4 passes → combined 67% CUB pass reduction
- C17-1 + C17-2: If C17-1 is implemented, C17-2 becomes unnecessary (C17-1 eliminates CUB sort entirely)
- C17-2 alone: Valid independent optimization if C17-1 is not implemented

---

## 4. Revised Roadmap

```
C17-2 v1 (FALSIFIED):  Active Pixel Mask
├── GATE: NOT PASSED
└── Evidence: semantically equivalent to existing done condition

C17-2 v2 (PROPOSED):   Two-Phase Sorting
├── GATE: PENDING REVIEW
├── Targets: sort stage (21% of forward time)
├── Mechanism: Phase A (15-bit CUB) + Phase B (per-tile bitonic sort)
├── Expected: ~50% sort time reduction → ~10% forward reduction
└── Preserves: all data structures, rasterization, backward
```

---

*Gate review complete. Awaiting instruction: proceed with C17-2 v2 (Two-Phase Sorting) design, or another candidate mechanism.*
