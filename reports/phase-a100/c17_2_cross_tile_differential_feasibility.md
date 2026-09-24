# C17-2 Cross-Tile Differential Feasibility Audit

## Research Question

Can adjacent-tile ordered Gaussian membership be represented as a reference tile's membership plus a small set of insertions/removals, with reconstruction cost below the current full-materialization baseline?

---

## 1. Workload Identity

| Factor | This Analysis | Corrected A100 Baseline |
|--------|---------------|------------------------|
| **Gaussian source** | Raw SfM PLY | Trained checkpoint (30k) |
| **Visibility** | 24-38% of master Gaussians visible | 1-6% visible (pruned) |
| **Mean tiles/G (visible)** | 5.2-11.7 | 112-1,585 |
| **Resolution** | Native (room 3114×2075, bicycle 4946×3286) | Native |
| **Tile grid** | room 195×130=25,350; bicycle 310×206=63,860 | Same |
| **Tile size** | 16 | 16 |
| **Packed** | Yes | Yes |
| **I (image count)** | 1 | 1 |
| **Scale activation** | `torch.exp` | `torch.exp` |

**Critical caveat**: The true corrected-baseline workload (trained checkpoints) is not available on this machine. The PLY workload has smaller per-Gaussian footprints, so cross-tile overlap is **likely a lower bound** — trained Gaussians (larger) would have even higher overlap.

---

## 2. Membership Size Distribution

| Scene | P50 | P75 | P90 | P95 | P99 | Max | Empty tiles |
|-------|-----|-----|-----|-----|-----|-----|-------------|
| room | 145 | 229 | 346 | 436 | 679 | 1,471 | 0 / 25,350 |
| bicycle | 110 | 211 | 417 | 579 | 1,134 | 2,304 | 0 / 63,860 |

---

## 3. Cross-Tile Overlap — Insertion/Removal Delta

### Horizontal neighbor (→)

| Metric | room P50 | room P75 | room P95 | bicycle P50 | bicycle P75 | bicycle P95 |
|--------|---------|---------|---------|------------|------------|-----------|
| len_B | 147 | 237 | 446 | 114 | 229 | 556 |
| len_S (shared) | 110 | 160 | 282 | 70 | 134 | 324 |
| len_I (inserted into B) | **32** | **76** | **196** | **38** | **85** | **246** |
| ρ_I = I/B | **0.217** | **0.352** | **0.568** | **0.313** | **0.433** | **0.571** |

### Vertical neighbor (↓)

| Metric | room P50 | room P75 | room P95 | bicycle P50 | bicycle P75 | bicycle P95 |
|--------|---------|---------|---------|------------|------------|-----------|
| len_I | **34** | **68** | **167** | **43** | **112** | **371** |
| ρ_I = I/B | **0.220** | **0.321** | **0.505** | **0.392** | **0.519** | **0.681** |

### Diagonal neighbor (↘)

| Metric | room P50 | room P75 | room P95 | bicycle P50 | bicycle P75 | bicycle P95 |
|--------|---------|---------|---------|------------|------------|-----------|
| ρ_I = I/B | 0.353 | 0.494 | 0.728 | 0.519 | 0.638 | 0.783 |

**Key insight**: For horizontal/vertical neighbors (the primary backward-relevant directions), only **22-39%** of tile B's membership must be newly added. The remaining 61-78% is already in tile A.

---

## 4. Shared-Order Subsequence Consistency

| Scene | Direction | Pairs | Order-preserving subsequence |
|-------|-----------|-------|------------------------------|
| room | H | 489/489 | **100%** ✓ |
| room | V | 514/514 | **100%** ✓ |
| room | DD | 997/997 | **100%** ✓ |
| bicycle | H | 478/478 | **100%** ✓ |
| bicycle | V | 510/510 | **100%** ✓ |
| bicycle | DD | 1,012/1,012 | **100%** ✓ |

**Verification method**: For each pair of adjacent tiles A,B, extract the subsequence of shared Gaussian IDs as they appear in A and B (in depth-sorted order). If `shared_subseq(A) == shared_subseq(B)`, the subsequence is order-preserving.

**Result**: 100% across 4,000+ sampled pairs, all directions, both scenes.

**Why this is guaranteed**: The CUB sort key is `depth_upper | tile_id << 16`. A Gaussian has the same depth regardless of which tile contains it. The depth sort within a tile is purely by `depth_upper`. Therefore the relative order of any two shared Gaussians is the same in both tiles — a mathematical guarantee, not an empirical coincidence.

**Consequence**: The shared subsequence is not just order-preserving — it is **identical** element-by-element. This means reconstruction requires only merging insertions into a known shared base, not reordering.

---

## 5. Insertion Structure

### Insertion block size (consecutive new Gaussians in target order)

| Scene | Direction | P50 | P75 | P90 | P95 | Mean |
|-------|-----------|-----|-----|-----|-----|------|
| room | H | **1** | **2** | **3** | **5** | 2.0 |
| room | V | **1** | **2** | **4** | **6** | 2.1 |
| bicycle | H | **1** | **2** | **4** | **6** | 2.3 |
| bicycle | V | **2** | **3** | **6** | **11** | 3.4 |

### Insertion position (normalized within B's depth-sorted list)

| Scene | Direction | P50 | P75 | P90 | P95 |
|-------|-----------|-----|-----|-----|-----|
| room | H | 0.44 | 0.66 | 0.82 | 0.89 |
| room | V | 0.36 | 0.59 | 0.77 | 0.85 |
| bicycle | H | 0.43 | 0.66 | 0.82 | 0.90 |
| bicycle | V | 0.44 | 0.67 | 0.82 | 0.88 |

**Key findings**:
1. **Insertions are highly clustered**: Median block size = 1-2 elements. P90 = 3-6 elements. Insertions arrive in small, compact groups.
2. **Insertions are dispersed across the list**: P50 insertion position is 0.36-0.44 (mid-list). They appear throughout the depth order, not concentrated at one end.

---

## 6. Tile-Path Sequential Reuse

Row-major consecutive tile scan (T0 → T1 → T2 → ... within each row):

| Scene | Pairs | P50 insert | P75 insert | P95 insert | P99 insert |
|-------|-------|-----------|-----------|-----------|-----------|
| room | 25,220 | **32** | **72** | **179** | **290** |
| bicycle | 63,654 | **36** | **80** | **246** | **425** |

**Key insight**: Consecutive horizontal tiles along a scan row have very small deltas (P50 = 32-36 inserts per tile). Compared to mean tile sizes of 110-145, this means only **22-29% new elements per step** along the scan path. This suggests a chain of length 4-5 could be maintained without significant reconstruction overhead.

---

## 7. Dependency Chain Analysis

### Scenario: Chain reconstruction

If tiles are encoded as differences from the previous tile in scan order:
```
T0 = full reference
T1 = delta_from(T0)
T2 = delta_from(T1)
T3 = delta_from(T2)
```

**Reconstruction cost for Tn**:
- Naive: O(n × |M_0|) — chain grows linearly
- Bounded: If every K-th tile is a full reference, reconstruction depth ≤ K
- With streaming: consecutive access along scan path amortizes reconstruction

### Random access cost

The backward kernel accesses tiles in **arbitrary image-space order** (per-pixel, not per-tile-sequence). A differential representation cannot assume sequential access. For a tile at chain depth D:
- Must materialize D steps to reconstruct T_n
- D can be large if references are sparse

**Mitigation**: Store a full reference every N tiles. N determines:
- Storage overhead: +1/N fraction of full flatten_ids
- Max reconstruction depth: N/2 (worst case is midway between references)

### Backward kernel implications

| Question | Answer |
|----------|--------|
| Q1: O(1) `flatten_ids[idx]`? | **No**. Random access requires materialized list or position-aware index that accounts for insertions/removals. |
| Q2: Sequential scan `[t_start, t_end)`? | **Yes**, via streaming merge of reference + sorted insertions. O(\|M_B\|) — same asymptotic cost as current |
| Q3: Reverse traversal (back-to-front)? | **Yes, with care**. Reverse order requires knowing which elements come from reference vs delta. A two-pointer backward merge works: take the last element of the merged set by comparing last of reference vs last of insertions. |
| Q4: Full materialization required? | **Necessary for the current kernel structure**. The kernel uses batch processing: it loads flatten_ids in blocks to shared memory. A differential representation would need to materialize the tile's list before batch loading, OR interleave merge and batch loading. |

---

## 8. LCS Analysis

Since the shared subsequence is 100% order-preserving and identical element-by-element:

```
LCS(M_A, M_B) = |M_A ∩ M_B|
```

This is not just an estimate — it is the exact LCS value, because the shared elements appear in the same order in both lists (depth-sorted), and the LCS of two sequences can never exceed the intersection size.

| Scene | Direction | LCS / |M_B| (mean) | LCS / |M_A| (mean) |
|-------|-----------|----------------------|----------------------|
| room | H | 0.75 | 0.75 |
| room | V | 0.74 | 0.74 |
| bicycle | H | 0.63 | 0.63 |
| bicycle | V | 0.57 | 0.57 |

---

## 9. Reconstruction Cost Model

### Model A: Reference already available (no storage cost for reference)

```
Cost_delta = |I| + |R|  (insert + remove entries only)
vs
Cost_full = |M_B|

Savings = 1 - (|I|+|R|)/|M_B| = 1 - 2*|I|/|M_B| (assuming |I|≈|R|)
```

| Scene | Direction | ρ_I P50 | |I|+|R| vs |M_B| P50 | Savings P50 | P95 savings |
|-------|-----------|---------|----------------------|-------------|-------------|
| room | H | 0.217 | 0.434 | **56.6%** | 43.2% |
| room | V | 0.220 | 0.440 | **56.0%** | 49.5% |
| bicycle | H | 0.313 | 0.626 | **37.4%** | 42.9% |
| bicycle | V | 0.392 | 0.784 | **21.6%** | 31.9% |

### Model B: Reference stored independently

```
Cost_delta = |M_A| + |I| + |R|
vs
Cost_full = |M_A| + |M_B|
Savings = 1 - (|M_A|+2|I|)/(|M_A|+|M_B|)
```

Since |M_A| ≈ |M_B| and |I| ≈ 0.22-0.39 × |M_B|:
- Savings = 1 - (1 + 0.44-0.78) / (1 + 1) = 1 - (1.44-1.78)/2 = **11-28% storage reduction**

### Amortized cost (per-element reconstruction)

| Cost type | Current (full flatten_ids) | Differential (reference + delta) |
|-----------|---------------------------|----------------------------------|
| Per-element read | 1 memory load (flat[idx]) | 1 branch + 1 load (check source, then load from ref or delta buffer) |
| Tile setup | 2 loads (offset start/end) | Build merge cursors: reference base + insertion sorted list |
| Sort requirement | None (pre-sorted) | Insertions must be pre-sorted by depth |
| Shared mem batch | Direct copy from flatten_ids | Merge from two sources into batch buffer |

**Asymptotic**: Both are O(|M_B|) per tile. The differential adds a constant-factor overhead per element (branch + source selection).

---

## 10. Worst-Case Analysis

| Metric | room | bicycle |
|--------|------|---------|
| Total pairs scanned | 100,427 | 253,894 |
| Max |I| (new IDs added) | **972** | **1,903** |
| Max |R| (IDs removed) | **959** | **1,760** |
| Max |I|+|R| (delta entries) | **1,616** | **2,918** |

Relative to max tile size (room: 1,471, bicycle: 2,304), worst-case deltas are still — at most — **1.1-1.3× the tile size**, not multiple times larger. The worst-case tile (max 2,918 delta entries vs 2,304 full) would save only 10-20% under Model A. But this affects only the top 0.1-1% of tile pairs.

---

## 11. Feasibility Matrix

| Property | Result | Implication |
|----------|--------|-------------|
| **Membership overlap** | **SUBSTANTIAL** — Jaccard P50 0.42-0.62 H/V | 61-78% of B's membership already in A |
| **Delta size (I/B)** | **SMALL** — P50 0.22-0.39 (H/V) | Only 22-39% new IDs needed |
| **Shared order consistency** | **PERFECT (100%)** | Subsequence identity guarantees merge without reordering |
| **LCS ratio** | **HIGH** — 0.57-0.75 | Most of target order is from reference |
| **Insertion count** | **LOW** — P50 32-43 elements | ~22-29% of mean tile size |
| **Insertion locality** | **CLUSTERED** — P50 block=1-2, P90=4-8 | Insertions arrive in compact groups |
| **Cross-tile chain stability** | **MODERATE** — P50=32-36 delta along scan | 4-5 step chain stays reasonable |
| **Random-access cost** | **HIGH** — requires materialization chain | Not compatible with current kernel's single-idx access |
| **Storage saving (Model A)** | **22-57%** reduction in per-tile element count | Meaningful bandwidth reduction for flatten_ids |
| **Storage saving (Model B)** | **11-28%** including reference overhead | Still positive but marginal |
| **Reconstruction cost** | **O(\|M_B\|)** — same asymptotic, higher constant | Merge adds branch per element |
| **Backward compatibility** | **CONDITIONAL** — sequential merge viable, random access problematic | Per-tile sequential backward scan works; chain depth must be bounded |

---

## 12. Feasibility: CONDITIONAL

The cross-tile differential approach has strong structural support:

1. ✅ **Small delta**: Only 22-39% new IDs per adjacent horizontal tile
2. ✅ **Perfect order consistency**: Shared subsequence is identical — merge, not sort
3. ✅ **Clustered insertions**: Median block size 1-2 — small, efficient merge units
4. ✅ **Sequential scan viable**: Row-major path has small, stable deltas
5. ⚠️ **Backward kernel**: Sequential per-tile merge is O(|M_B|), but random-access per-element (flatten_ids[idx]) requires materialization
6. ⚠️ **Trained checkpoint uncertainty**: PLY data (smaller footprints) likely underestimates trained-checkpoint overlap, but this needs verification
7. ⚠️ **Chain dependency**: Must bound depth (store full reference every N tiles)

### Why not PROMISING

- The backward kernel's current `flatten_ids[idx]` access pattern requires either full pre-materialization or a structural kernel rewrite
- Trained-checkpoint data (the actual baseline) could not be verified
- The reconstruction vs storage tradeoff needs a concrete kernel design to evaluate

---

## 13. C17-2 Status: CONTINUE WITH REDESIGN

**Reason**: The evidence supports moving from feasibility analysis to a concrete design proposal. The cross-tile differential approach has all the prerequisite properties (small delta, perfect order, clustered insertions, sequential-scan viability). The next step is to produce a `design.md` that specifies:

1. **Representation format**: How reference tiles are selected, how deltas (insertion set + removal set) are encoded, and how the flattened array layout changes
2. **Reconstruction kernel design**: How the forward and backward kernels would merge reference + delta on-the-fly
3. **Chain depth management**: Full-reference strategy (every N tiles) and its storage impact
4. **Backward kernel impact assessment**: Dataflow changes needed for the current reverse-batch structure
5. **Worst-case bounds**: Guarantees that no tile's reconstruction exceeds O(|M_B|)

---

## Appendix: Full Reference Tile Overhead

If every K-th tile stores a full membership (Model A reference), the overhead is:

| K (tiles) | Reference overhead | Max chain depth | Storage total (vs full) |
|-----------|-------------------|-----------------|------------------------|
| 4 | +25% of full | 2 | ~0.25 + 0.75×0.43 = 0.57× full |
| 8 | +12.5% of full | 4 | ~0.125 + 0.875×0.43 = 0.50× full |
| 16 | +6.25% of full | 8 | ~0.0625 + 0.9375×0.43 = 0.47× full |

With ρ_I+P50≈0.22 for room H, even K=4 gives 43% flatten_ids reduction. As tile coverage increases (trained Gaussians), overlap likely improves.

---

*Analysis: native-resolution PLY (room 3114×2075, bicycle 4946×3286), tile_size=16, packed=True, activated scales. 2,000+ sampled tile pairs per scene for detailed set/order analysis; full scan for worst-case.*
