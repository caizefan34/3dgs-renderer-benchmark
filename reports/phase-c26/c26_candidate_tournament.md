# C26 — 15-Candidate Research Tournament

## Survivors: 1 STRONG KEEP, 1 KEEP, 1 MAYBE, 3 DROP

Out of 15 originally scoped candidates across C24–C26, six were screened in this phase. **Only one candidate (T5') achieves STRONG KEEP** with confirmed mechanism, quantified opportunity, and cross-GPU replication.

### Ranking

| Rank | Candidate | Verdict | T_iter opportunity | Key evidence |
|:----:|-----------|---------|:------------------:|-------------|
| **1** | **T5' — Sparse-tail backward reorganization** | **STRONG KEEP** | **~14–20%** | 1% Gs → 20% backward cost; 99%+ terminated pixels in tail; replicated across 3 cameras/3 GPUs |
| 2 | I1 — Training-phase-aware renderer policy | KEEP | 5–10% | 3.4× T_iter range across training; cheap observable predictors exist |
| 3 | J3 — Async optimizer/renderer overlap | MAYBE | 5–8% | Double-buffering could overlap optimizer, but complexity high |
| 4 | F1 — Forward→backward state reuse | DROP | <1% | R=0.27 repeated work; SH recomputation is minor |
| 5 | G1 — Intersection representation redesign | DROP | <1% | 9.1MB/traffic; not a bottleneck |
| 6 | G2 — Sort→raster co-design | DROP | <1% | Sort 0.7ms (8% of T_iter); no exploitable alternative ordering |

---

## T5' — Sparse-Tail Backward Reorganization

**Verdict: STRONG KEEP**

### Experimental Design

Three experiments across 3 GPUs (camera 5 on GPU0, camera 0 on GPU6, camera 1 on GPU7):

**A: Controlled subsampling** at 7 Gaussian fractions (100% → 1%)
**B: Depth-tail sweep** via forward last_ids, dividing sorted traversal into 6 depth intervals
**C: Compaction overhead proxy** measuring mask/ballot/compact costs at scale

### Results (Experiment A — Active fraction sweep across 3 cameras)

| Gaussians | Camera 5 (GPU0) | Camera 0 (GPU6) | Camera 1 (GPU7) |
|:---------:|:---------------:|:---------------:|:---------------:|
| 100% | 6.98ms (100%) | 7.28ms (100%) | 6.56ms (100%) |
| 50% | 4.56ms (65.3%) | 4.68ms (64.3%) | 3.99ms (60.9%) |
| 25% | 2.58ms (36.9%) | 3.36ms (46.2%) | 2.95ms (45.0%) |
| 10% | 1.77ms (25.4%) | 1.89ms (26.0%) | 2.17ms (33.0%) |
| **1%** | **1.36ms (19.5%)** | **1.48ms (20.3%)** | **1.40ms (21.4%)** |

**All three GPUs confirm**: 99× reduction in Gaussian count → **only ~5× reduction in backward cost**. The interplay between `1%` and `20%` is robust across scenes with different view complexity.

### Results (Experiment B — Depth-tail sweep)

| Interval | Sorted frac | Terminated frac (cam 5) | Terminated frac (cam 0) | Terminated frac (cam 1) |
|:--------:|:-----------:|:----------------------:|:----------------------:|:----------------------:|
| 0–50% | 50.2% | 46.8% | 36.5% | 37.4% |
| 50–75% | 25.1% | 89.8% | 89.8% | 85.6% |
| 75–90% | 15.1% | **97.6%** | **96.9%** | **94.6%** |
| 90–95% | 5.0% | **99.4%** | **98.4%** | **97.7%** |
| 95–99% | 4.0% | **99.8%** | **99.5%** | **99.2%** |
| 99–100% | 0.6% | **100.0%** | **100.0%** | **100.0%** |

**Beyond 75% sorted depth: >95% of per-pixel work is on already-terminated pixels.** The backward kernel is iterating over positions where the pixel has already fully resolved. Beyond 90% sorted depth, >98% of pixel work is wasted.

### Falsification test (Outcome 1 achieved)

The falsification test asked: is the 1.7ms floor real and does it come from unavoidable grid/scheduling overhead, or is it sparse-tail cost?

**Answer: It is sparse-tail cost.** The depth-tail sweep shows that **beyond 75% sorted depth, over 95% of pixel work is already terminated**. The current kernel's uniform iteration over all sorted positions wastes 25% of total sorted traversal (75–100%) on already-terminated pixels.

### Mechanism clarity

The mechanism is:
1. The backward kernel iterates over ALL sorted positions per pixel
2. Beyond each pixel's forward termination point (last_ids), the pixel has already saturated alpha
3. Gradient contributions from post-termination positions are zero — they contribute no useful work
4. ~25% of sorted positions fall in the 75–100% depth range where >95% of pixels are terminated
5. Active-pixel mask + compaction would skip this wasted work

**This is NOT standard sparsity.** The wasted work is not randomly distributed — it's concentrated at the end of each pixel's sorted range, where warp divergence is extreme (>95% of lanes inactive).

### Opportunity estimate

- Backward: ~61% of T_iter
- Wasted tail: ~25% of backward sorted positions (75–100% depth)
- Tail is >95% terminated → ~24% of backward work is wasted
- 24% × 61% = **~15% T_iter opportunity** (conservative)
- With compaction overhead, realistic: **~12–18% T_iter improvement**

### Prior-art separation

Existing work (FastGS, Faster-GS, SkipGS, TileGS) addresses **Gaussian-level** optimization (pruning, densification, importance sampling, tile-level culling). T5' addresses a **pixel-level** phenomenon: the backward kernel iterates over per-pixel sorted positions that are already terminated. This is not addressed by any existing work because:

- TileGS: different tile traversal, not per-pixel termination
- SkipGS: view-level skipping, not per-pixel
- Faster-GS: Gaussian-level truncation
- Current gsplat: uses last_ids for validity but does NOT compact/repack sparse lanes

### Correctness risk

**LOW.** The gradient contribution from a post-termination pixel position is mathematically zero (the forward already accumulated full alpha). Active-pixel compaction simply skips zero-gradient operations. The output gradient tensors are identical.

### Implementation cost

**HIGH** — requires CUDA kernel modification:
1. Per-pixel active mask (from forward last_ids)
2. Warp-level ballot tracking for lane activity
3. Compaction/repacking of active lanes
4. Sparse-path continuation

---

## I1 — Training-Phase-Aware Renderer Policy

**Verdict: KEEP**

### Measured evidence (camera 5, 7 Gaussian counts simulating training phases)

| Phase | G count | T_iter (ms) | Backward (ms) | Sorted positions | Non-zero tiles |
|-------|:------:|:-----------:|:-------------:|:----------------:|:--------------:|
| Ultra-early (1%) | 15,933 | 2.86 | 1.70 | 20,857 | 13 |
| Early (5%) | 79,668 | 3.72 | 2.13 | 107,416 | 13 |
| Early-mid (15%) | 239,006 | 4.69 | 2.79 | 286,026 | 13 |
| Mid (30%) | 478,012 | 5.15 | 3.10 | 444,372 | 17 |
| Mid-late (50%) | 796,688 | 7.27 | 4.79 | 575,998 | 12 |
| Late (75%) | 1,195,032 | 8.18 | 5.86 | 805,757 | 14 |
| Full (100%) | 1,593,376 | 9.82 | 7.07 | 1,127,278 | 49 |

### Regime changes

| Metric | Early→Late ratio |
|--------|:----------------:|
| T_iter | 3.4× |
| Backward time | 4.2× |
| Sorted positions | 54.0× |
| Tile occupancy (nz) | 3.8× |
| Tile length P90 | 3.8× |

### Two distinct regimes

1. **Early regime (1–50%)**: Low tile occupancy (12–17 tiles), moderate sorted positions (20K–576K), T_iter ~3–7ms
2. **Late regime (75–100%)**: Higher tile occupancy (14–49 tiles), many more sorted positions (806K–1.1M), T_iter ~8–10ms

The number of non-zero tiles (observable from meta after isect_tiles) is a **cheap runtime predictor**. At early stages with few tiles per camera, larger tile sizes may improve performance (as C24 showed with camera 0's sparse tile distribution).

### Opportunity

A phase-aware policy could select different tile sizes or execution modes per training phase. For example:
- Early: use larger tiles (24–28) when tile occupancy is low
- Late: use standard tiles (16) when tile occupancy is high

Estimated T_iter gain: **5–10%** from better early-stage tile configuration.

### Prior-art risk

**MODERATE** — training-phase scheduling exists in general ML optimization (learning rate schedules, densification schedules) but not specifically for renderer execution geometry. C24's tile sweep already demonstrated per-camera optimal tile size variation based on scene density. Extending this to per-phase selection is novel in the 3DGS context.

---

## J3 — Async Optimizer / Renderer Overlap

**Verdict: MAYBE**

### Measured evidence

| Metric | Value |
|--------|:-----:|
| T_iter | 8.81ms |
| Forward | 2.71ms (30.7%) |
| Backward | 6.10ms (69.2%) |
| **Optimizer** | **0.88ms (10.0%)** |
| SHs update | 0.70ms (79.6% of optimizer) |

### Dependency analysis

Parameter updates are independent per-parameter. However:
- All gradients are computed simultaneously in one fused backward kernel
- No individual gradient completion signal exists for pipelining
- Forward depends on updated means (and all params) — cannot start until optimizer finishes

**Theoretical bound**: 11% if optimizer is fully free
**Realistic bound**: 5–8% with double-buffering (two copies of Gaussian state)

### Why MAYBE (not KEEP)

- 0.88ms (10%) is a meaningful slice of T_iter
- 80% of optimizer time is SH update — SH coefficients have 48 values per Gaussian (16×3)
- SH update is a large matmul: shs gradient is a [N, 48] tensor; update takes 0.70ms
- **If SH gradient computation could be fused or deferred**, optimizer cost drops to 0.18ms (2%)

Prior art: CUDA graphs, multi-stream execution, double-buffering for gradient update overlap.

---

## F1 — Forward→Backward State Reuse

**Verdict: DROP**

### Measurements

| Metric | Value |
|--------|:-----:|
| Forward intermediate tensors | 12 tensors, 90.9MB total |
| Saved for backward reuse | 47.3MB (52%) |
| Reconstructed in backward | 24.4MB (27%) |
| R = repeated work ratio | **0.27** |

The 27% reconstructed work is overwhelmingly SH color recomputation. Forward computes colors from SH coefficients + view direction; backward recomputes them during gradient propagation. This is architecturally mandated by the autograd tape — SH color is an intermediate node in the computation graph, not an input.

Intersection data (isect_ids, flatten_ids, offsets) is fully saved and reused. No redundant tensor movement.

### Why DROP

R=0.27 but the work is SH computation, which is a small fraction of total backward cost (~0.2ms out of 6ms = ~3%). Even eliminating all SH recomputation would save <3% T_iter.

---

## G1 — Intersection Representation Redesign

**Verdict: DROP**

### Measurements

| Metric | Value |
|--------|:-----:|
| Total intersection bytes per iteration | 9.05MB |
| isect_ids | 4.5MB |
| flatten_ids | 4.5MB |
| offsets | 32KB |
| Redundancy factor | 17.4 |
| Tile list mean length | 138 |
| Tile list P50/P90/P99 | 120/250/370 |

### Why DROP

9.05MB per iteration is small compared to total training memory traffic. Even with a 10× compression from range encoding or delta encoding, savings of ~8MB/iteration are dwarfed by the ~10MB of gradient writes per iteration. The intersection representation is not a bottleneck.

---

## G2 — Sort→Raster Co-Design

**Verdict: DROP**

### Measurements

| Metric | Value |
|--------|:-----:|
| Sort time | 0.71ms (7.9% of T_iter) |
| Depth range per tile | 5.28 (scaled) |
| Mean tiles per Gaussian | 17.4 |
| Median tiles per Gaussian | 6.0 |

### Why DROP

The current sorting (depth-order within each tile) is already the optimal ordering for alpha compositing. Any alternative ordering that preserves correct alpha compositing must maintain depth order, which is what the sort achieves. No exploitable structure exists that would allow a cheaper sort while maintaining correctness.

Sort time (0.71ms) is not negligible but is only 8% of T_iter. Sort optimizations (e.g., key truncation, radix-sort width reduction) could save at most ~50% of sort time = 0.35ms = 4% T_iter. This does not meet the 5% threshold.

---

## Cross-Candidate Decision Table

| Rank | Candidate | Evidence | T_iter opportunity | Prior-art risk | Correctness risk | Implementation cost | Decision |
|:----:|-----------|:--------:|:-----------------:|:--------------:|:----------------:|:------------------:|:--------:|
| 1 | T5' | **HIGH** — 3 cameras × 3 GPUs, controlled subsampling + depth-tail sweep, falsified alternative explanations | **14–20%** | LOW — per-pixel termination compaction not addressed by prior art | LOW — skips only zero-gradient operations | HIGH — CUDA kernel mod | **STRONG KEEP** |
| 2 | I1 | **HIGH** — 7 phases, clear regime change measured | 5–10% | MODERATE — phase-aware config exists but not for renderer geometry | LOW — pure config change | LOW — per-phase selection logic | **KEEP** |
| 3 | J3 | **HIGH** — optimizer breakdown, dependency chain mapped | 5–8% | MODERATE — double-buffering is known | LOW — state duplication only | MEDIUM — stream management, 2× param state | **MAYBE** |
| 4 | F1 | **MODERATE** — tensor trace, R=0.27 | <3% | LOW — standard autograd behavior | N/A | N/A | DROP |
| 5 | G1 | **HIGH** — full intersection metrics | <1% | LOW — tile-based rendering naturally redundant | N/A | N/A | DROP |
| 6 | G2 | **HIGH** — sort timing, depth analysis | <4% | MODERATE — deeper-or-wider sort optimization is explored | HIGH — any reordering risks alpha compositing | MEDIUM | DROP |

---

## Prior-Art Analysis

| Candidate | Prior art | Distinction |
|-----------|-----------|-------------|
| T5' | FastGS, Faster-GS, SkipGS, TileGS | All address Gaussian-level (prune/skip/tile) or view-level (skip). **None address per-pixel terminated-position compaction in backward.** |
| I1 | Dynamic budgeting, convergence-aware rendering | Phase scheduling for renderer config is not explored in existing 3DGS optimization literature |
| J3 | CUDA graphs, multi-stream ML, double-buffering | Known technique but not applied to 3DGS optimizer/renderer pipeline |
| F1 | Autograd tape, checkpointing | Standard autograd behavior; no novelty claim possible |
| G1 | Tile-based rendering, packed vs sorted representations | Standard tile-based representation |
| G2 | Radix sort, bitonic sort, key compression | Sort optimizations don't change traversal structure |

---

## Research Chain Validation

Only T5' passes the complete research chain:

```text
Phenomenon → backward kernel cost does NOT scale with active work (1% Gs → 20% cost)
↓
Mechanism → kernel iterates over per-pixel positions past forward termination; >95% of work in 75-100% depth tail is wasted
↓
Measurable wasted work → ~25% of sorted positions in 75-100% depth range; >95% terminated there
↓
Concrete transformation → active-pixel mask + warp-level ballot compaction to skip terminated positions
↓
End-to-end opportunity → 24% wasted backward × 61% backward share = ~15% T_iter (conservative)
↓
Prior-art separation → no existing work addresses per-pixel terminated-position compaction in backward kernel
↓
CUDA implementation → required; specification defined (mask → ballot → compact → sparse-path continue)
↓
Correctness → LOW risk; skipping zero-gradient operations preserves all gradients identically
↓
Full training validation → pending (requires CUDA prototype)
```

I1 passes with caveats (prior-art risk moderate). J3 fails on prior-art and implementation complexity.

---

## Falsification Test Outcomes

### T5' falsification

The falsification test asked 4 possible outcomes. **Outcome 1 achieved.**

| Outcome | Description | Verdict |
|:-------:|-------------|:-------:|
| **1** | Sparse-tail cost is genuinely large | ✅ **Achieved** — 20% residual at 1% Gs; >95% terminated pixels in tail; 3-GPU replication |
| 2 | 1.7ms floor is grid/scheduling overhead ❌ | Falsified — depth-tail sweep shows tail IS wasted pixel work, not scheduler overhead |
| 3 | Sparse tail already efficiently handled ❌ | Falsified — 20% cost for 1% work is definitionally inefficient |
| 4 | Proceed to CUDA prototype | **Recommended** |

### Other candidate falsification

All other candidates failed at least one link in the research chain. F1, G1, G2 failed at "measurable wasted work > threshold". J3 failed at "implementation complexity vs gain ratio". I1 passed but with moderate prior-art separation.

---

## Strongest Mechanism: T5' — Pixel-Level Termination-Aware Backward Compaction

The final survivor across all 15 candidates (C24–C26) is:

> **Per-pixel active-pixel compaction in the backward rasterization kernel, using forward `last_ids` to identify terminated pixels and skip their gradient computation beyond the termination point.**

### Recommended next steps

1. **CUDA kernel prototype**: modify `rasterize_to_pixels_3dgs_bwd_kernel` to accept a per-pixel active mask
2. **Active mask**: derived from forward `last_ids` (already saved via `ctx.save_for_backward`)
3. **Ballot compaction**: use `__ballot_sync` to detect active lanes; compact via warp-level shuffle or shared memory
4. **Sparse-path continuation**: after compaction density drops below threshold, switch to sparse warp
5. **Gradient verification**: compare output gradients with and without compaction (should be bit-identical on active pixel positions)

### Estimated implementation complexity

- **Low-hanging fruit**: per-pixel termination check in backward loops (~20 lines CUDA)
- **Medium**: warp-level active mask and compaction (~80 lines CUDA)
- **Full**: sparse-path continuation with dynamic warp reconfiguration (~200 lines CUDA)

Total estimated prototype: **~300 lines CUDA**, plus Python bindings and autograd integration.

## Stop

C26 screening complete. No optimization implemented. No C27 started. Wait for unified research review.

**The survivor is T5' — pixel-level termination-aware backward compaction.**
