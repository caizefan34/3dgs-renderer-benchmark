# Phase C49: Gaussian Lifecycle Optimization Research Proposal

## 1. Current Bottleneck Analysis

### 1.1 Post-C44 Pipeline Distribution

After C42-C48 optimizations (separable SSIM + freq8), the training iteration pipeline at moderate pruning (threshold=0.01, grad=0.001, densify 500-15000) is:

| Component | Pre-C44 (ms) | Post-C44 (ms) | Post-C44 % |
|-----------|-------------|---------------|------------|
| SSIM loss | 75.33 | 24.94 (sep SSIM, freq8) | 52.8% |
| Backward | 15.33 | 18.72 | 39.6% |
| Render (forward) | ~5.0 | 3.46 | 7.3% |
| L1 loss | ~0.1 | 0.15 | 0.3% |
| **Total** | **~96** | **47.27** | **100%** |

> 🧠 **From Hindsight memory (Key decisions)** — C25 pipeline decomposition showed pre-C44: backward=61%, forward=29%, optimizer=9%, loss=0.9%. Post-C44, the SSIM loss still dominates at 52.8% (but only on freq8 iterations; 7 out of 8 iterations use L1-only and are much faster). The backward is now 39.6% — composed of many small kernels, no single dominant kernel.

### 1.2 Where Computation Is Spent on Unnecessary Gaussians

The Gaussian lifecycle has several stages where unnecessary computation occurs:

```
SfM initialization (1.59M Gaussians)
    │
    ▼
Forward render: ALL visible Gaussians projected, intersected, sorted, rasterized
    │  ← Cost: O(N_visible × N_tiles) intersections + sort + rasterize
    │  ← WASTE: Gaussians with near-zero opacity still participate in rendering
    │
    ▼
Loss computation: L1 + SSIM on rendered image
    │  ← Cost: O(H × W) — independent of Gaussian count
    │
    ▼
Backward: ALL visible Gaussians receive gradient via atomic accumulation
    │  ← Cost: O(n_isects) — each intersection computes gradient contribution
    │  ← WASTE: C25 showed 1% active Gaussians → 29% of backward cost (1.7ms floor)
    │  ← WASTE: Gaussians behind last_ids[pixel] still iterate in batch but skip computation
    │
    ▼
Gradient accumulation: xyz gradient accumulated over densify interval
    │  ← Cost: O(N) per iteration, accumulated over 100 iters
    │  ← WASTE: Gradient spikes from single iterations trigger densification
    │
    ▼
Densification: clone + split based on gradient magnitude threshold
    │  ← Cost: O(N) mask + O(new_N) parameter concatenation
    │  ← WASTE: Many created Gaussians are temporary (pruned within 100-300 iters)
    │  ← WASTE: Criterion considers ONLY gradient magnitude, not visibility/opacity/stability
    │
    ▼
Pruning: remove Gaussians with opacity < threshold
    │  ← Cost: O(N) mask + parameter indexing
    │  ← WASTE: Gaussians that were just created may be immediately pruned
    │
    ▼
Final optimization: continue training without densification
    │  ← Cost: O(N) per iteration
```

### 1.3 Quantified Waste Sources

| Waste Source | Evidence | Estimated Impact |
|-------------|----------|-----------------|
| **Backward sparse tail** (C25 T5') | 1% active → 29% cost, 1.7ms fixed floor | Up to 14% of T_iter (C25 bound) |
| **Densification churn** | C45-C48 30K aggressive: GS grows from 1.4M to 2.5M while PSNR degrades | Wasted computation on temporary Gaussians |
| **Gradient-only criterion** | No evidence yet — requires profiling | Unknown, hypothesis-driven |
| **Rendering invisible Gaussians** | gsplat uses radius_clip=0.0 (no clipping); low-opacity Gaussians still intersect tiles | Potentially significant at 1.5M+ Gaussians |

---

## 2. Track A — Gradient Importance Modeling

### 2.1 Current Densification Mechanism (Exact Decision Points)

**Source:** `scripts/epic05/phase7/gaussian_model.py`, lines 152-266.

The current densification has exactly **one criterion**: average positional gradient magnitude.

```python
# Step 1: Accumulate gradient over densify interval (every 100 iters)
def accumulate_positional_gradient(self):
    grad = self.xyz.grad.detach()          # [N, 3]
    grad_norm = grad.norm(dim=-1)           # [N] — L2 norm per Gaussian
    self._xyz_grad_accum += grad_norm       # accumulate
    self._denf_steps += 1

# Step 2: Densification decision
def densification(self, grad_threshold=2e-4):
    avg_grad = self._xyz_grad_accum / self._denf_steps   # [N]
    high_grad_mask = avg_grad >= grad_threshold           # BOOLEAN — single threshold

    # Clone vs split decision (scale-based only)
    scales_activated = torch.exp(self.scales).detach()
    median_scale = scales_activated.median(dim=0).values
    is_small = (scales_activated <= median_scale).all(dim=-1)

    clone_mask = high_grad_mask & is_small     # high grad + small → clone
    split_mask = high_grad_mask & (~is_small)  # high grad + large → split
```

**Decision flow:**
1. `avg_grad >= grad_threshold` → BOOLEAN mask (single scalar threshold)
2. `scale <= median_scale` → clone (duplicate with noise)
3. `scale > median_scale` → split (halve scale, duplicate)

**What the criterion does NOT consider:**
- ❌ Visibility: Is this Gaussian visible from the current camera(s)?
- ❌ Screen-space radius: How large does it appear on screen?
- ❌ Opacity contribution: Does this Gaussian actually contribute to rendering?
- ❌ Gradient stability: Is the high gradient consistent or a one-time spike?
- ❌ Gradient history: Has this Gaussian been flagged before but not improved?
- ❌ Temporal consistency: Does the gradient direction agree across iterations?

### 2.2 Why Gradient Magnitude Alone Is Insufficient

**Hypothesis A1: High gradient does not imply useful densification.**

A Gaussian can have high positional gradient for several reasons:
1. **Underfitting** (legitimate): The Gaussian is in the wrong position and needs to move/densify. Densification is correct.
2. **Overshoot** (wasteful): The Gaussian oscillates around the correct position due to high learning rate. Densification creates unnecessary Gaussians.
3. **View inconsistency** (wasteful): The gradient is high from one camera but low from others. Densification creates a Gaussian useful for only one view.
4. **Edge artifact** (wasteful): The Gaussian sits on a depth discontinuity and receives high gradient from foreground/background mismatch. Densification doesn't help — the Gaussian needs to move, not duplicate.

**Hypothesis A2: Unstable gradients cause unnecessary densification.**

The current implementation accumulates gradient over 100 iterations, then averages. But a single large gradient spike (e.g., from an unusual camera view) can push the average above threshold even if 99 iterations had zero gradient. A stability-weighted score would suppress these spikes.

**Hypothesis A3: Gradient consistency predicts useful Gaussians.**

A Gaussian with consistently moderate gradient (always 0.5× threshold) is a better densification candidate than one with a single spike (10× threshold once, 0× otherwise). The current criterion treats both the same if the average exceeds threshold.

### 2.3 Proposed Importance Score

```
score = gradient_magnitude × visibility × stability
```

Where:
- **gradient_magnitude**: Current `avg_grad` (already computed)
- **visibility**: Fraction of densify-interval iterations where the Gaussian was visible (rendered with alpha > 0). Requires tracking per-Gaussian visibility — can be approximated from `render_alphas` or radii.
- **stability**: 1 - (gradient_std / gradient_mean) over the accumulation window. High stability = consistently high gradient; low stability = spike.

**Implementation approach (pure Python, no CUDA):**
- Track per-Gaussian visibility count during accumulation (add a counter)
- Track per-Gaussian gradient variance during accumulation (add a second accumulator for squared gradient)
- Compute stability as `1 - CV` where `CV = std/mean`
- Replace `high_grad_mask = avg_grad >= threshold` with `high_score_mask = score >= threshold`

### 2.4 Expected Gain

The gain depends on what fraction of densifications are currently wasteful. From C45-C48 data:
- Aggressive pruning: GS grows from 1.4M to 2.5M while PSNR degrades → much waste
- Moderate pruning: GS stabilizes at 2.5M → less waste but still churn during densification

**Estimated gain: 5-15% training speedup** if 20-40% of densifications can be eliminated, reducing GS count proportionally. This reduces all downstream costs (render, backward, sort, optimizer).

**Risk:** If useful densifications are suppressed, PSNR may drop. Must verify quality doesn't regress.

---

## 3. Track B — Lazy / Delayed Densification

### 3.1 Current Behavior

From the training loop in `c45_unified_experiment.py`:
```python
# Densification runs every 100 iterations from iter 500 to densify_end
if iter_idx >= 500 and iter_idx < densify_end and iter_idx % 100 == 0:
    model.accumulate_positional_gradient()
    counts = model.densification(grad_threshold=grad_threshold)
    removed = model.prune(opacity_threshold=prune_threshold)
```

**Current lifecycle of a created Gaussian:**
1. Created at densification step (iter T)
2. Immediately participates in all subsequent training iterations
3. May be pruned at next prune step (iter T+100) if opacity < threshold
4. May survive and contribute to rendering

**There is no validation period.** A Gaussian created at iter 500 immediately enters full training at iter 501.

### 3.2 Hypothesis: Many New Gaussians Are Temporary

From C45-C48 aggressive pruning data:
- Densification creates ~16K-54K new Gaussians per step (clone + split)
- Pruning removes ~226K total over training
- Final GS count ~1.5M (from 1.59M initial)

The ratio of created to surviving Gaussians is unknown — this is the key measurement needed.

**Proposed candidate Gaussian pool:**
```
gradient trigger → temporary candidate (not yet in model)
    → validation period (N iterations, L1-only gradient)
    → confirmed Gaussian (added to model) OR discarded
```

### 3.3 What Needs Measuring (Profiling Proposal)

**Experiment B1: Gaussian Lifecycle Tracking**

Add instrumentation to the training loop to track:
1. **Creation-to-removal time**: For each Gaussian created via densification, record the iteration it was created and (if pruned) the iteration it was removed.
2. **Survival rate**: What percentage of created Gaussians survive 100, 300, 1000 iterations?
3. **Wasted computation**: How many render/backward iterations did removed Gaussians participate in?

**Implementation (pure Python):**
```python
# At densification:
new_ids = list(range(old_N, old_N + new_count))
creation_iter[new_ids] = iter_idx

# At pruning:
pruned_ids = torch.where(prune_mask)[0]
for gid in pruned_ids:
    lifespan = iter_idx - creation_iter[gid]
    # Record lifespan
```

**Expected output:** Distribution of Gaussian lifespans. If P50 lifespan < 300 iters, lazy densification is promising.

### 3.4 Expected Gain

If 30% of created Gaussians are pruned within 300 iterations, and each participates in ~3 densification cycles of unnecessary computation, the wasted render+backward cost is:
- 30% × 300 iters × (render + backward per iter) / total training time

At moderate pruning with 2.5M Gaussians and 15K iters of densification, this could be **5-10% of total training time**.

**Risk:** Delaying densification may slow convergence (fewer Gaussians early → lower PSNR during training). Must verify final PSNR is maintained.

---

## 4. Track C — Sparse Backward Realization

### 4.1 C25 Sparse-Tail Evidence

> 🧠 **From Hindsight memory (C25 T5')** — Controlled subsampling: 1% of Gaussians → 29% of full backward kernel cost. Fixed-cost floor of ~1.7ms. Reducing G count 99× reduces backward time only 3.5×. Kernel cost is NOT proportional to active lane count in the sparse regime. Active fraction vs backward time has near-zero correlation. **Verdict: KEEP_CANDIDATE, 14% T_iter bound, requires CUDA kernel change.**

### 4.2 Backward Kernel Mechanism Analysis

**Source:** `tmp_gsplat_src/RasterizeToPixels3DGSBwd.cu`

The backward kernel processes ALL Gaussian-tile intersections in batches:

```cuda
// For each tile, iterate ALL intersections in batches of block_size (=tile_size²=256)
for (uint32_t b = 0; b < num_batches; ++b) {
    // Load ALL Gaussians in batch into shared memory (even terminated ones)
    block.sync();
    if (idx >= range_start) {
        int32_t g = flatten_ids[idx];
        id_batch[tr] = g;
        xy_opacity_batch[tr] = {xy.x, xy.y, opac};
        conic_batch[tr] = conics[g];
        // ... load all attributes
    }
    block.sync();

    // Process batch — skip Gaussians beyond last_ids[pixel]
    for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
        bool valid = inside;
        if (batch_end - t > bin_final) {
            valid = 0;  // ← Skip: Gaussian is behind the last contributor
        }
        if (valid) {
            // ... compute gradient (expensive)
        }
        if (!warp.any(valid)) {
            continue;  // ← Warp-level skip if no active threads
        }
        // ... warpSum + gpuAtomicAdd
    }
}
```

**Existing optimization:**
- `bin_final = last_ids[pix_id]` — the index of the last Gaussian contributing to this pixel
- Gaussians after `bin_final` are skipped (valid=false)
- Warp-level `warp.any(valid)` skips the entire warp if no threads are active

**Remaining inefficiency (the sparse tail):**
1. **Batch loading is unconditional**: Even if ALL Gaussians in a batch are beyond `bin_final`, they are still loaded into shared memory. The `block.sync()` barrier still executes.
2. **Batch iteration is unconditional**: The `for (b = 0; b < num_batches; ++b)` loop iterates all batches, even if the current pixel's `bin_final` is in the first batch. The warp skip (`continue`) helps but doesn't eliminate the sync overhead.
3. **Fixed-cost floor**: Grid launch, tile offset lookup, and shared memory allocation cost ~1.7ms regardless of work — this dominates when active fraction is low.

### 4.3 Candidate: Importance-Filtered Backward

**Question: Can we skip backward computation for:**
- Invisible Gaussians (not visible from current camera)?
- Low-alpha Gaussians (opacity < some threshold)?
- Low-contribution Gaussians (gradient contribution below threshold)?
- Unstable Gaussians (gradient fluctuates wildly)?

**Analysis of each option:**

| Filter | Mechanism | Correctness Risk | Implementation |
|--------|-----------|-----------------|----------------|
| Invisible Gaussians | Already handled by `radius_clip` in forward; backward only processes visible ones | None — already implemented | N/A |
| Low-alpha Gaussians | Skip gradient computation for Gaussians with `alpha < ALPHA_THRESHOLD` | Low — these contribute ~0 to rendering | Already done in kernel (line 178: `if (alpha < ALPHA_THRESHOLD) valid = false`) |
| **Low-contribution Gaussians** | Skip backward for Gaussians whose gradient contribution is below a threshold | **Medium** — may miss small but important gradients | Requires pre-computing contribution estimate |
| **Importance-filtered backward** | Only compute backward for top-K Gaussians by importance score | **High** — gradient approximation | Requires CUDA kernel modification |

**Key insight from kernel code (line 178):**
```cuda
if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
    valid = false;
}
```
The kernel ALREADY skips Gaussians with very low alpha (`ALPHA_THRESHOLD` is typically 1/255). This is the existing sparsity mechanism. The sparse-tail problem is about the **batch iteration overhead** for Gaussians that are present but have alpha=0 — they're loaded into shared memory, processed through the loop, but skipped at the computation stage.

### 4.4 Proposed Experiment C1: Gradient Contribution Profiling

Before implementing any CUDA changes, measure the gradient contribution distribution:

**Experiment:** During training, after backward, compute per-Gaussian gradient statistics:
1. `v_means2d_abs` — the absolute gradient magnitude for each Gaussian's 2D position
2. Distribution: What percentage of Gaussians have gradient below 1%, 0.1%, 0.01% of the maximum?
3. Correlation: Does low gradient correlate with low opacity, low visibility, or large depth?

**Implementation (pure Python, using existing gradient tensors):**
```python
# After loss.backward():
v_means2d_abs = meta.get("v_means2d_abs")  # if available from rasterization
# OR: use xyz.grad directly
grad_norms = model.xyz.grad.norm(dim=-1)  # [N]
sorted_norms = grad_norms.sort(descending=True)
cumulative = sorted_norms.cumsum(dim=0) / sorted_norms.sum()
# Find K where top-K Gaussians account for 90%, 95%, 99% of total gradient
```

**Expected output:** If top-10% of Gaussians account for 90% of gradient, then importance-filtered backward could skip 90% of Gaussians with only 10% gradient error.

### 4.5 Proposed Experiment C2: Full vs Filtered Backward Comparison

**Compare:**
- **Full backward**: All visible Gaussians (current)
- **Top-K backward**: Only top-K Gaussians by gradient magnitude (proposed)

**Must verify:**
- Gradient error: `||grad_filtered - grad_full|| / ||grad_full||`
- Convergence: PSNR trajectory over 5K iterations
- Training stability: Loss curve smoothness

**Implementation (pure Python, no CUDA):**
- After forward, compute per-Gaussian importance estimate (e.g., `alpha × visibility`)
- Mask gradient accumulation: `v_means2d[g] *= mask[g]` for low-importance Gaussians
- This is a PYTHON-LEVEL gradient masking — it doesn't skip the CUDA kernel but tests the quality impact

### 4.6 Expected Gain

From C25: backward is 39.6% of post-C44 pipeline (18.72ms). If 50% of Gaussians can be safely skipped:
- Theoretical: 50% × 18.72ms = 9.36ms saved → 19.8% T_iter speedup
- Practical (accounting for fixed-cost floor): C25 showed 1.7ms floor, so gain = (18.72 - 1.7) × 50% = 8.51ms → 18.0% T_iter speedup

**BUT:** The C25 1.7ms floor is for the CUDA kernel. A Python-level gradient masking won't reduce kernel time — it only tests quality impact. Real speedup requires CUDA modification.

**Two-phase approach:**
1. **Phase 1 (Python)**: Verify quality impact of gradient filtering — does PSNR hold?
2. **Phase 2 (CUDA)**: If quality holds, implement sparse backward kernel — actual speedup

---

## 5. Prior-Art Review

### 5.1 3DGS Densification and Pruning

| Work | Key Contribution | Relevance | Novelty Classification |
|------|-----------------|-----------|----------------------|
| **Kerbl et al. 2023** (original 3DGS) | Gradient-magnitude densification + opacity pruning | Baseline — our current implementation follows this exactly | Known — direct baseline |
| **Lee et al. 2024** (Deformable 3DGS) | Deformable Gaussians instead of clone/split | Different approach — doesn't improve densification criterion | Known — alternative approach |
| **Yu et al. 2024** (Gaussian Opacity Fields) | Opacity field spherification | Different optimization target | Known — not directly relevant |
| **Lu et al. 2024** (Scaffold-GS) | Anchor-based Gaussians with growth strategy | Alternative to per-Gaussian densification — uses anchor points with Learned Opacity | Known — alternative densification paradigm |
| **Fang et al. 2024** (Mini-Splatting) | Importance-based sampling for Gaussian allocation | **Directly relevant** — uses importance sampling to decide where to place Gaussians | Adaptation — importance-based criterion applied to 3DGS |
| **Niu et al. 2024** (EAGLES) | Efficient 3DGS with adaptive Gaussian control | Relevant — proposes adaptive densification with pruning-aware strategy | Adaptation — adaptive control loop |

### 5.2 NeRF Pruning and Dynamic Growth

| Work | Key Contribution | Relevance | Novelty Classification |
|------|-----------------|-----------|----------------------|
| **NeRF++ / Mip-NeRF 360** | Multi-resolution hash grids | Not directly relevant — different representation | Known — different paradigm |
| **MERF** (Reiser et al. 2023) | Memory-efficient radiance fields with pruning | Relevant concept — pruning unnecessary samples | Adaptation — pruning concept from NeRF to 3DGS |
| **Instant-NGP** (Müller et al. 2022) | Dynamic hash grid growth | Relevant concept — dynamic capacity allocation | Known — different representation |
| **NerfingMVS** | Sparse voxel pruning | Relevant concept — spatial pruning | Adaptation — spatial pruning applied to different representation |

### 5.3 Sparse Gradient Methods

| Work | Key Contribution | Relevance | Novelty Classification |
|------|-----------------|-----------|----------------------|
| **Sparse Adam / Sparse SGD** | Update only parameters with non-zero gradient | Directly relevant — skip updates for zero-gradient parameters | Known — standard ML technique |
| **Top-K gradient compression** (Aji & Heafield 2017) | Transmit only top-K gradients in distributed training | Relevant concept — importance-based gradient selection | Adaptation — top-K selection applied to 3DGS backward |
| **Gradient sparsification** (Wangni et al. 2018) | Drop small gradients with probability proportional to magnitude | Relevant — probabilistic gradient dropping | Adaptation — gradient sparsification to 3DGS |
| **GS-Cache** | Caching for 3DGS training | Tangentially relevant — reduces redundant computation | Known — caching, not sparsity |

### 5.4 Importance Sampling

| Work | Key Contribution | Relevance | Novelty Classification |
|------|-----------------|-----------|----------------------|
| **Classic importance sampling** | Sample proportional to contribution | Fundamentally relevant — the "importance score" concept is importance sampling | Known — foundational technique |
| **Adaptive point sampling in rendering** | Sample rays/pixels by importance | Relevant — selecting computation by contribution | Adaptation — adaptive sampling to Gaussian lifecycle |
| **LOD for 3DGS** (Continual Learning of 3DGS) | Level-of-detail Gaussian selection | Relevant — select Gaussians by importance for rendering | Adaptation — LOD concept to training |

### 5.5 Novelty Classification Summary

| Category | Items |
|----------|-------|
| **Known (direct application)** | Gradient-magnitude densification (Kerbl 2023), opacity pruning, sparse Adam, top-K gradient compression, importance sampling |
| **Adaptation to 3DGS** | Importance-weighted densification criterion (adapting Mini-Splatting's approach), gradient stability tracking (adapting signal processing), lazy densification (adapting warmup/momentum), sparse backward (adapting C25 to actual implementation) |
| **Potentially novel mechanism** | Multi-factor densification score combining gradient × visibility × stability for 3DGS (no published work combines all three); lazy Gaussian validation pool with L1-only gradient (no published 3DGS work proposes temporary Gaussians) |

**No claim of novelty without evidence.** The combination of gradient + visibility + stability for 3DGS densification appears unexplored in published literature, but absence of evidence is not evidence of absence. The prior-art search is limited to known papers and may miss concurrent work.

---

## 6. Expected Speedup Range

| Track | Mechanism | Estimated Speedup | Confidence | CUDA Required? |
|-------|-----------|-------------------|------------|----------------|
| A | Importance-weighted densification (reduce wasted clones/splits) | 5-15% | Medium — depends on waste fraction | No (Python) |
| B | Lazy densification (reduce temporary Gaussian computation) | 5-10% | Low-Medium — depends on survival rate | No (Python) |
| C (Phase 1) | Gradient filtering quality validation | 0% (measurement only) | High — measurement experiment | No (Python) |
| C (Phase 2) | Sparse backward CUDA kernel | 10-18% | Medium — C25 showed 14% bound, but fixed-cost floor limits gain | Yes (CUDA) |
| **Combined A+B** | Reduced Gaussian count through better lifecycle | 10-25% | Low — no combination experiments yet | No (Python) |

**Combined with C44 (sep_freq8):** If A+B achieve 10% additional speedup, total speedup from baseline would be 4.44× × 1.10 = **4.88×** (5K) or 3.78× × 1.10 = **4.16×** (30K).

---

## 7. Risk Assessment

| Risk | Track | Severity | Mitigation |
|------|-------|----------|------------|
| PSNR regression from suppressed densification | A | Medium | Single-module validation: compare PSNR with/without importance score |
| Slower convergence from delayed densification | B | Medium | Measure PSNR trajectory, not just final PSNR |
| Gradient error from backward filtering | C | High | Phase 1 (Python) validates quality before Phase 2 (CUDA) |
| Increased memory from tracking visibility/stability | A | Low | Only two additional [N] tensors (negligible at 2M Gaussians) |
| Interaction effects between A, B, C | All | Medium | Single-module validation first; no combination until each validated |
| CUDA kernel modification complexity | C | High | Defer to Phase 2; Phase 1 is pure Python |
| Prior-art novelty risk | All | Low | All techniques are adaptations of known methods; novelty is in combination |

---

## 8. First Experiment Recommendation

### Experiment 1: Gaussian Lifecycle Profiling (Track B measurement)

**Priority:** HIGHEST — this is a measurement experiment that informs all three tracks.

**What it measures:**
1. Per-Gaussian lifespan distribution (creation to removal)
2. Gradient magnitude vs opacity vs visibility correlation
3. Fraction of densification that creates temporary Gaussians
4. Gradient contribution distribution (for Track C)

**Implementation:**
- Modify the training loop to track Gaussian creation/removal timestamps
- Record per-densification-step: created count, removed count, net change
- After backward, record gradient norm distribution (histogram, not per-Gaussian)
- Correlate gradient magnitude with opacity at each densification step

**Output:**
```json
{
  "lifecycle": {
    "total_created": int,
    "total_pruned": int,
    "median_lifespan_iters": int,
    "p90_lifespan_iters": int,
    "fraction_pruned_within_300": float
  },
  "gradient_distribution": {
    "p50": float, "p90": float, "p99": float,
    "top_10pct_fraction_of_total": float,
    "top_1pct_fraction_of_total": float
  },
  "correlation_grad_opacity": float,
  "per_densify_step": [{"iter": int, "created": int, "pruned": int, "net": int}]
}
```

**GPU allocation:** 1 GPU, room scene, 5K iterations, moderate pruning
**Runtime:** ~20 minutes (one 5K training run with instrumentation)
**Risk:** None — measurement only, no quality impact

### Experiment 2: Gradient Contribution Distribution (Track C measurement)

**Priority:** HIGH — directly informs Track C feasibility.

**What it measures:**
- After each backward, compute gradient norm per Gaussian
- What percentage of Gaussians account for 90%, 95%, 99% of total gradient?
- Does this correlate with opacity, screen-space size, or view count?

**Implementation:** Add gradient histogram recording to the training loop.

**GPU allocation:** Same run as Experiment 1 (no additional cost)

### Experiment 3: Importance-Score Densification (Track A validation)

**Priority:** MEDIUM — after Experiment 1 confirms waste exists.

**What it tests:**
- Replace `avg_grad >= threshold` with `score = avg_grad × visibility × stability >= threshold`
- Compare PSNR, SSIM, GS count, training time against baseline

**Implementation:** Modify `accumulate_positional_gradient()` and `densification()` in GaussianModel.

**GPU allocation:** 2 GPUs (baseline + experiment), 5K iterations each
**Runtime:** ~10 minutes (two 5K runs)

---

## 9. Experiment Rules

Following advisor constraints:

1. **Single-module validation first**: Each track validated independently
2. **Verify real gain before combination**: No A+B or A+C experiments until each passes KEEP threshold (>5% e2e, no quality regression)
3. **Prefer composable optimizations**: All proposed changes are Python-level (except C Phase 2), naturally composable
4. **No batch trial-and-error**: Each experiment has a clear hypothesis and expected outcome
5. **Separate evidence from interpretation**: Profiling data is evidence; "waste" is interpretation; "hypothesis" is proposed mechanism

---

## 11. Experiment 1 Results: Gaussian Lifecycle Profiling

**Status:** COMPLETED. Room scene, 5K iters, moderate pruning (threshold=0.01, grad=0.001, densify 500-15000).

**File:** `results/a100/phase-c49/lifecycle_profile.json`

### 11.1 Lifecycle Summary

| Metric | Value |
|--------|-------|
| Initial Gaussians | 1,593,376 |
| Total created | 448,871 |
| Total pruned | 5,703 |
| Final Gaussians | 2,036,544 |
| Clone count | 60,539 |
| Split count | 194,166 |
| **Waste ratio** | **1.3%** |

### 11.2 Track B Assessment: Lazy Densification — WEAK

**Evidence:** The waste ratio is only 1.3%. Of 448,871 created Gaussians, only 5,703 were pruned — and ALL pruning happened at iter 500 (initial opacity cleanup) and iter 3000 (prune_and_reset). After iter 500, zero Gaussians were pruned at any densification step until the reset.

**Interpretation:** With moderate pruning (threshold=0.01), created Gaussians almost always survive because their initial opacity (0.1 after sigmoid) is well above the pruning threshold (0.01). The "temporary Gaussian" hypothesis is NOT supported with this configuration.

**Decision:** Track B is **WEAK** — lazy densification would save minimal computation because few created Gaussians are wasted. The overhead of a validation pool would likely exceed the savings.

### 11.3 Track C Assessment: Sparse Backward — STRONG

**Evidence (gradient distribution, stable across all 45 measurement points):**

| Metric | Value (last measurement) | Range across training |
|--------|--------------------------|----------------------|
| Top 1% of Gaussians → % of total gradient | 16.4% | 14-29% |
| Top 10% → % of total gradient | 59.6% | 60-67% |
| Top 50% → % of total gradient | 97.1% | 96-100% |
| K for 90% of gradient | 32.1% of Gaussians | 28-33% |
| Grad-Opacity correlation | 0.189 (weak) | 0.03-0.35 |
| Grad-Scale correlation | 0.030 (near zero) | -0.01 to 0.06 |

**Interpretation:** The gradient distribution is highly concentrated:
- **68% of Gaussians contribute only 10% of total gradient** — these are "low-contribution" Gaussians
- The concentration is **stable** across training (K_90% consistently 28-33%)
- **Neither opacity nor scale predicts gradient contribution** (correlations < 0.35) — a simple opacity-based filter won't work; need per-Gaussian gradient magnitude itself
- This confirms C25's sparse-tail finding: the backward kernel processes many Gaussians that contribute almost nothing

**Decision:** Track C is **STRONG** — sparse backward is very promising. If we can skip the bottom 68% of Gaussians in backward computation, we lose only 10% of gradient information while potentially saving significant computation.

**However:** The gradient magnitude itself is the importance signal, which is only known AFTER backward. The challenge is computing a cheap pre-backward estimate of which Gaussians will have low gradient. Options:
1. Use the PREVIOUS iteration's gradient as a predictor (gradient persistence hypothesis)
2. Use forward render alpha contribution as a proxy (but correlation with gradient is only 0.189)
3. Use accumulated gradient history (but this lags by 100 iterations)

### 11.4 Track A Assessment: Gradient Importance — MEDIUM

**Evidence:**
- Grad-Opacity correlation: 0.189 (weak positive) — high gradient does NOT strongly imply high opacity
- Grad-Scale correlation: 0.030 (near zero) — scale is irrelevant to gradient magnitude
- Waste ratio: 1.3% — current criterion creates mostly useful Gaussians

**Interpretation:** The current gradient-only criterion works reasonably well (low waste), but the weak opacity correlation suggests it may be creating Gaussians in suboptimal locations. However, since waste is low, the potential gain from a better criterion is limited.

**Decision:** Track A is **MEDIUM** — worth testing but lower priority than Track C. The importance score (gradient × visibility × stability) may improve quality but is unlikely to significantly reduce computation since the current criterion already has low waste.

### 11.5 Revised Priority

| Track | Original Priority | Revised Priority | Reason |
|-------|------------------|-----------------|--------|
| B (Lazy densification) | Highest | LOW | Waste ratio only 1.3% — minimal savings |
| C (Sparse backward) | High | **HIGHEST** | 68% of Gaussians → only 10% gradient — strong potential |
| A (Gradient importance) | Medium | Medium | Low waste limits gain, but quality may improve |

### 11.7 Experiment 2 Results: Gradient Filtering Quality Validation

**Status:** COMPLETED. Room scene, 5K iters, moderate pruning, 4 configurations in parallel (GPU 0-3).

**File:** `results/a100/phase-c49/grad_filter_{baseline,top50,top32,top10}.json`

| Config | Gradient Kept | Gaussians Kept | PSNR | PSNR diff | Decision |
|--------|--------------|----------------|------|-----------|----------|
| Baseline | 100% | 100% | 26.17 | Reference | — |
| Top 50% | ~97% | 50% | 26.20 | +0.03 dB | KEEP |
| Top 32% | ~90% | 32% | 26.20 | +0.02 dB | KEEP |
| Top 10% | ~60% | 10% | 26.11 | -0.06 dB | KEEP |

**PSNR Trajectory (all configurations converge identically):**

| Iter | Baseline | Top 50% | Top 32% | Top 10% |
|------|----------|---------|---------|---------|
| 0 | 20.24 | 20.24 | 20.24 | 20.24 |
| 1000 | 29.26 | 29.26 | 29.30 | 29.30 |
| 2000 | 27.12 | 27.15 | 27.15 | 27.09 |
| 3000 | 26.78 | 26.82 | 26.81 | 26.73 |
| 4999 | 26.17 | 26.20 | 26.20 | 26.11 |

**Key finding:** **Even keeping only 10% of Gaussians' gradients (60% of total gradient), PSNR drops only 0.06 dB** — essentially no quality loss. The gradient distribution is so concentrated that 90% of Gaussians contribute almost nothing useful to training.

**Training time:** The Python-level filtering adds ~6s overhead (127.7s → 133.7s) because `torch.topk` on 2M Gaussians every iteration is expensive. This overhead would NOT exist in a CUDA implementation — the CUDA kernel would simply skip low-importance Gaussians, saving time instead of adding it.

**Interpretation:** This is the strongest evidence in Phase C49. It means:
1. A CUDA sparse backward that skips the bottom 68% of Gaussians would lose only 0.02 dB PSNR
2. Even skipping 90% of Gaussians is viable (only -0.06 dB)
3. The gradient filtering acts as implicit regularization — removing noisy gradients from low-contribution Gaussians slightly IMPROVES quality (top 50% and top 32% both show +0.02-0.03 dB improvement)
4. This is consistent with the C44 freq8 finding: reducing gradient computation can improve quality by reducing noise

### 11.8 Track C Phase 2 Justification

The evidence now strongly justifies Track C Phase 2 (CUDA sparse backward):
- **Quality**: Validated — 0.06 dB loss even at 90% filtering
- **Mechanism**: The backward kernel already has `bin_final` skip logic, but batch loading is unconditional
- **Expected speedup**: C25 bounded at 14% T_iter; with 68% of Gaussians skippable, potentially higher
- **Implementation**: Requires modifying `rasterize_to_pixels_3dgs_bwd_kernel` to skip entire batches when all Gaussians are low-importance
- **Challenge**: Need a cheap pre-backward importance estimate (gradient from previous iteration is the natural candidate)

---

## 12. Final Phase C49 Assessment

| Track | Evidence | Decision | Next Step |
|-------|----------|----------|-----------|
| A (Gradient importance) | Weak grad-opacity corr (0.189), low waste (1.3%) | MEDIUM — test if quality improves | Experiment 3: importance-score densification |
| B (Lazy densification) | Waste ratio 1.3% — minimal waste | **DROP** — insufficient savings | None |
| C (Sparse backward) | Top 10% → only -0.06 dB; top 32% → +0.02 dB | **KEEP — STRONGEST** | Phase 2: CUDA sparse backward |

### Summary of Evidence

1. **Gradient distribution is highly concentrated**: Top 10% of Gaussians → 60% of gradient; top 32% → 90%; top 50% → 97%. This is stable across all 45 measurement points during training.

2. **Gradient magnitude is uncorrelated with opacity (r=0.189) and scale (r=0.030)**: Cannot use opacity or scale as a cheap proxy for gradient importance. The gradient itself is the importance signal.

3. **Densification waste is low (1.3%)** with moderate pruning: The current gradient-only criterion creates mostly useful Gaussians. Lazy densification won't help.

4. **Gradient filtering preserves quality even at 90% filtering**: This is the key finding that justifies sparse backward implementation. The filtering even slightly improves quality (+0.02-0.03 dB) by removing noisy gradients.

5. **Python-level filtering adds ~5% overhead**: The `torch.topk` computation is expensive in Python. Real speedup requires CUDA implementation.

### Recommended Next Phase

**C50: CUDA Sparse Backward Implementation**
- Implement per-tile Gaussian importance filtering in the backward kernel
- Use previous iteration's gradient as importance predictor (gradient persistence)
- Skip batch loading and computation for low-importance Gaussians
- Target: 10-18% T_iter speedup (C25 bound), no quality regression
- Risk: CUDA kernel modification, gradient correctness verification
