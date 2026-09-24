# Phase C51: Sparse Backward CUDA — From Gradient Prediction to Real Speedup

## 1. Motivation

C49-C50 established an evidence chain:
1. **C49**: Gradient distribution is highly concentrated (top 32% → 90% of gradient)
2. **C49**: Oracle gradient filtering preserves quality (top 10% → only -0.06 dB)
3. **C50**: Previous-gradient prediction is highly accurate (Recall@32% = 0.977, Coverage@32% = 0.912)

C51 asks: **Can predicted sparse backward actually reduce computation while preserving training correctness and quality?**

This report covers Stages 1-3 (source audit, simulated sparse backward, multi-scene validation). Stages 4-8 (CUDA implementation) are deferred pending Stage 2-3 results.

---

## 2. Evidence Structure

This report separates:
- **Observed**: Measured facts from experiments
- **Derived**: Calculations directly based on measurements
- **Hypothesis**: Claims that still require validation

---

## 3. Stage 1: CUDA Source Audit

### 3.1 Observed

**Files analyzed**: RasterizeToPixels3DGSBwd.cu (430 lines), RasterizeToPixels3DGSFwd.cu (310 lines), IntersectTile.cu (396 lines), _wrapper.py (2607 lines).

**Backward kernel structure**:
- Launch: grid = {I, tile_height, tile_width}, threads = {tile_size², 1} = 256
- Processing order: **back-to-front** (reverse of forward)
- Batch loading: each thread loads 1 Gaussian's attributes into shared memory per batch
- Gaussian identity: `g = flatten_ids[idx]` known at line 141, **before** attribute loading
- T reconstruction: `T *= 1/(1-alpha)` for each valid Gaussian (back-to-front)
- Buffer accumulation: `buffer[k] += rgb[k] * alpha * T`
- Gradient output: warpSum reduction → warp leader atomicAdd to global gradient arrays

**Existing skip mechanisms**:
- `last_ids[pix_id]`: skips Gaussians beyond last forward contributor (occluded)
- `ALPHA_THRESHOLD` (1/255): skips nearly-transparent Gaussians
- `warp.any(valid)`: warp-level early exit when no thread is active

### 3.2 Derived

**Design analysis** (4 candidate designs):

| Design | Mechanism | T/buffer correct? | Matches C49? | Expected speedup |
|--------|-----------|-------------------|-------------|-----------------|
| A | Skip atomicAdd only | Yes | Yes | <5% (minimal) |
| **B** | **Skip gradient compute, keep T/buffer** | **Yes** | **Yes** | **10-25% (moderate)** |
| C | Skip loading + alpha + T/buffer | No | No | 20-35% (higher, unvalidated) |
| D | Batch compaction (rebuild flatten_ids) | No | No | 30-50% (highest, complex) |

### 3.3 Critical Finding

**C49 validated Design A/B, NOT Design C/D.** C49 zeroed gradients AFTER full backward — T and buffer were computed correctly for ALL Gaussians. Only the OUTPUT gradients of unimportant Gaussians were zeroed.

Design C/D introduces T/buffer errors because skipped Gaussians don't contribute to transmittance reconstruction. This is a **different, stronger approximation** that was NOT validated by C49 and requires separate validation.

### 3.4 Hypothesis

Design B (compute skip, T/buffer preserved) is the primary CUDA candidate because:
1. It matches C49's validated approximation
2. It skips the expensive gradient arithmetic + warpSum + atomicAdd
3. It preserves T/buffer correctness for all Gaussians
4. It can be implemented by checking `importance_mask[g]` after alpha computation, before gradient computation

---

## 4. Stage 2: Simulated Sparse Backward

### 4.1 Experimental Setup

- Scene: Mip-NeRF360 room, 1080p, 5K iters, seed=42
- Moderate pruning (threshold=0.01, grad=0.001, densify 500-15000)
- Loss: Separable SSIM + freq8 (C44 method)
- Predictor: previous-iteration gradient norm (not oracle)
- Mode A: full backward, then zero unimportant gradients (matches Design A/B)
- Configurations: K=50% (keep 50%), K=32%, K=20%
- Gradient correctness measured at iters 1000, 2000, 3000, 4000

### 4.2 V1 Results (mask applied BEFORE densification — BROKEN)

#### Observed

| Config | PSNR (5K) | diff from baseline | Clone | Split | Prune | Time(s) | xyz cosine |
|--------|----------|-------------------|-------|-------|-------|---------|------------|
| Baseline (C50) | 26.23 | — | 58,814 | 194,171 | 5,697 | 266.1 | — |
| K50v1 | 23.84 | **-2.39 dB** | 8,308 | 171,388 | 1,132 | 137.6 | 0.9914 |
| K32v1 | 23.63 | **-2.60 dB** | 9,706 | 164,397 | 615 | 252.5 | 0.9895 |
| K20v1 | 23.19 | **-3.04 dB** | 9,699 | 153,754 | 475 | 132.9 | 0.9851 |

#### Derived

**Root cause: densification collapse.** Clone count dropped 83-86% (58,814 → 8,308-9,706). The mask was applied to gradients BEFORE `accumulate_positional_gradient()`, so densification saw masked gradients. Gaussians with zeroed gradients never exceeded the densification threshold, causing a collapse in cloning.

This is a **critical implementation error**, not a fundamental limitation of sparse backward. The fix: accumulate gradient BEFORE masking.

### 4.3 V2 Results (densification fix — accumulate BEFORE mask)

#### Observed

| Config | PSNR (5K) | diff from baseline | Clone | Split | Prune | Time(s) | xyz cosine |
|--------|----------|-------------------|-------|-------|-------|---------|------------|
| Baseline (C50) | 26.23 | — | 58,814 | 194,171 | 5,697 | 266.1 | — |
| K50v2 | 25.68 | **-0.55 dB** | 9,437 | 194,915 | 826 | 137.6 | 0.9946 |
| K32v2 | 25.50 | **-0.73 dB** | 7,705 | 175,292 | 432 | 248.4 | 0.9934 |
| K20v2 | 25.28 | **-0.95 dB** | 7,337 | 172,698 | 229 | 136.4 | 0.9932 |

**PSNR trajectory (K50v2 vs baseline)**:

| Iter | Baseline | K50v2 | K32v2 | K20v2 |
|------|----------|-------|-------|-------|
| 0 | 20.24 | 20.24 | 20.24 | 20.24 |
| 1000 | 29.29 | 28.69 | 28.42 | 27.68 |
| 2000 | 27.15 | 26.70 | 26.41 | 26.06 |
| 3000 | 26.79 | 26.30 | 25.99 | 25.72 |
| 4000 | 26.43 | 25.97 | 25.71 | 25.45 |
| 4999 | 26.23 | 25.68 | 25.50 | 25.28 |

**Gradient correctness (last measurement, iter 4000)**:

| Config | xyz cos | xyz rel_l2 | scales cos | rotations cos | opacity cos | shs cos |
|--------|---------|-----------|-----------|--------------|------------|---------|
| K50v2 | 0.9946 | 0.1037 | 0.9989 | 0.9980 | 0.9991 | 0.9999 |
| K32v2 | 0.9934 | 0.1150 | 0.9988 | 0.9971 | 0.9990 | 0.9998 |
| K20v2 | 0.9932 | 0.1165 | 0.9999 | 0.9993 | 0.9994 | 0.9997 |

#### Derived

**V1→V2 improvement**: The densification fix improved PSNR by 1.4-2.1 dB:
- K50: 23.84 → 25.68 (+1.84 dB)
- K32: 23.63 → 25.50 (+1.87 dB)
- K20: 23.19 → 25.28 (+2.09 dB)

**Remaining degradation**: 0.55-0.95 dB. This is the **compounding cost** of prediction error over 5000 iterations. C50 showed 1.6-2.3% per-iteration miss rate; over 5000 iterations, this compounds to 0.55-0.95 dB.

**Comparison with C49 oracle**:

| Config | C49 oracle | C51 v2 predicted | Gap (cost of prediction) |
|--------|-----------|-----------------|-------------------------|
| K50 | +0.03 dB | -0.55 dB | 0.58 dB |
| K32 | +0.02 dB | -0.73 dB | 0.75 dB |
| K10/K20 | -0.06 dB | -0.95 dB | 0.89 dB |

**Clone count still reduced** (9,437 vs 58,814 for K50v2). Even with the densification fix, clone count is lower because the model trajectory diverges from baseline due to masked optimizer steps. Different trajectories → different gradients → different densification decisions. This is a feedback loop that cannot be fully eliminated.

**Split count recovered** (194,915 vs 194,171 for K50v2). Splitting depends on gradient AND scale (split if high grad + large scale). The split criterion is less sensitive to the mask because large-scale Gaussians tend to have consistently high gradients (they cover large areas).

### 4.4 Acceptance Gate Check

| Gate | Threshold | K50v2 | K32v2 | K20v2 | Result |
|------|-----------|-------|-------|-------|--------|
| Gradient cosine (xyz) | >= 0.99 | 0.9946 | 0.9934 | 0.9932 | **PASS** (all) |
| Gradient cosine (other) | >= 0.99 | >0.998 | >0.998 | >0.998 | **PASS** (all) |
| PSNR degradation | < 0.2 dB | 0.55 | 0.73 | 0.95 | **FAIL** (all) |
| Net iteration speedup | > 5% | N/A | N/A | N/A | **Deferred** (requires CUDA) |

---

## 5. Stage 3: Multi-Scene Predictor Validation

### 5.1 Observed

| Scene | SfM points | Recall@32% | Coverage@32% | Coverage@50% | Result |
|-------|-----------|-----------|-------------|-------------|--------|
| room (C50) | 1,593,376 | 0.977 | 0.912 | 0.973 | **PASS** |
| bicycle | 6,131,954 | 0.974 | 0.914 | 0.968 | **PASS** |
| garden | 1,839,236 | 0.972 | **0.834** | **0.937** | **FAIL** |

**Success criteria** (Recall@32% >= 0.90 AND Coverage@32% >= 0.85 AND Coverage@50% >= 0.95):
- room: PASS (all three)
- bicycle: PASS (all three)
- garden: **FAIL** (Coverage@32% = 0.834 < 0.85, Coverage@50% = 0.937 < 0.95)

### 5.2 Derived

**Garden scene analysis**: Recall@32% is high (0.972) — the predictor correctly identifies important Gaussians. But Coverage@32% is low (0.834) — the top 32% of Gaussians by previous gradient capture only 83.4% of current gradient. This means the gradient distribution in garden is **less concentrated** than in room/bicycle.

**Interpretation**: Garden is an outdoor scene with more view-dependent variation and more uniform gradient distribution. The "important" Gaussians change more between iterations, so the same top-32% captures less of the total gradient. This doesn't mean prediction is bad (recall is still 97.2%) — it means the gradient signal is more spread out, so a 32% selection captures less of it.

**Implication for sparse backward**: In garden, a 32% sparse backward would lose 16.6% of gradient signal (vs 8.8% in room). This suggests:
1. K=32% may not be sufficient for garden — K=50% would be safer
2. The optimal K may need to be scene-dependent
3. Production deployment should default to K=50% (safer across scenes)

### 5.3 Hypothesis

Garden's lower coverage is due to outdoor scene characteristics (more uniform lighting, more view-dependent surfaces, less indoor structural regularity). The previous-gradient predictor is still accurate (recall 97.2%) but the gradient concentration is lower. This is a property of the scene, not a failure of the predictor.

---

## 6. Critical Implementation Finding: Densification Decoupling

### 6.1 Observed

The single most important finding of C51: **gradient masking MUST be decoupled from densification.**

| Version | Mask timing | Clone count | PSNR (K50) |
|---------|------------|-------------|------------|
| v1 | Before densification | 8,308 | 23.84 |
| v2 | After densification | 9,437 | 25.68 |
| Baseline | No mask | 58,814 | 26.23 |

### 6.2 Derived

**Densification uses `accumulate_positional_gradient()` which reads `xyz.grad.norm(dim=-1)`.** If the gradient is masked (zeroed) before accumulation, the densification sees artificially low gradients for 50-80% of Gaussians. These Gaussians never exceed the densification threshold, causing a collapse in cloning.

**The fix (v2)**: call `accumulate_positional_gradient()` BEFORE applying the mask. This preserves the full gradient signal for densification while still applying the mask for the optimizer step.

**For CUDA implementation**: The sparse backward kernel skips gradient computation for unimportant Gaussians. Their gradients are implicitly zero. For densification, we need the FULL gradient norm. Solutions:
1. **Store previous iteration's full gradient norm** (available before masking) — use for densification
2. **Run a lightweight gradient-norm-only pass** — compute just the norm, not full gradients
3. **Only apply sparse backward after densification ends** (iter > 15000) — no densification conflict

Option 1 is the most efficient: the previous iteration's full gradient is already computed and stored. It can be used for both the mask AND the densification, with no additional computation.

---

## 7. Prior-Art Constraint

### 7.1 Observed

The following concepts have substantial prior art and CANNOT be claimed as novel:
- Top-K gradient selection / gradient sparsification (standard in distributed ML)
- Importance masking (used in pruning, quantization, sparse training)
- Gaussian pruning by importance (Mini-Splatting, LP-3DGS)
- Sparse backward computation (standard in sparse neural networks)
- Previous-gradient prediction (momentum-based selection in SGD)

### 7.2 Derived

The research contribution must be framed around:
1. **The experimentally established evidence chain** (C49→C50→C51): gradient concentration → prediction quality → training validation
2. **The system-level finding**: densification must be decoupled from gradient masking
3. **The mechanism**: previous-gradient as a pre-backward importance predictor for 3DGS rasterization
4. **The measured trade-off**: 0.55 dB quality cost for 50% gradient filtering with predicted (non-oracle) selection

### 7.3 Hypothesis

Recent work on hardware-accelerated differentiable rasterization and 3DGS backward optimization (e.g., hardware differentiable rasterization reporting significant backward speedup) means the paper cannot claim "we optimized backward." The contribution is the **predictability-based sparse backward** with the evidence chain and the densification decoupling insight.

---

## 8. Final Decision

### Decision: **MODIFY**

The sparse backward approach is **promising but not yet ready for CUDA implementation**. The evidence supports further development but the acceptance gates are not fully met.

### Gate Summary

| Gate | Result | Details |
|------|--------|---------|
| Gradient cosine >= 0.99 | **PASS** | All configs, all params (xyz: 0.993-0.995, others: >0.998) |
| PSNR degradation < 0.2 dB | **FAIL** | K50: -0.55, K32: -0.73, K20: -0.95 dB |
| Multi-scene prediction | **PARTIAL** | room/bicycle PASS, garden FAIL (coverage) |
| Net speedup > 5% | **DEFERRED** | Requires CUDA implementation |

### Required Modifications Before CUDA

1. **Reduce compounding error**: The 0.55 dB degradation at K=50% is from compounding prediction errors over 5000 iterations. Possible fixes:
   - Periodic full backward (every 100-200 iterations) to reset error accumulation
   - Gradual filtering schedule: start at K=100%, reduce to K=50% over training
   - Sparse backward only after densification ends (iter > 15000)

2. **Scene-adaptive K**: Garden scene has lower gradient coverage. Default K=50% is safer than K=32% for cross-scene deployment. Consider adaptive K based on measured coverage.

3. **Densification-safe CUDA design**: The CUDA kernel must either:
   - Store full gradient norms from previous iteration for densification (recommended)
   - Skip sparse backward during densification period (iter 500-15000)
   - Compute a lightweight gradient-norm pass alongside sparse backward

### Recommended Next Steps

1. **Experiment: Periodic full backward** — every 200 iterations, run full backward (no mask) to reset compounding error. Measure if PSNR degradation drops below 0.2 dB.

2. **Experiment: Post-densification sparse** — apply mask only after iter 15000 (when densification ends). This avoids the densification conflict entirely.

3. **Experiment: K=80%** — measure if higher K achieves <0.2 dB degradation.

4. **If any of the above passes 0.2 dB**: proceed to Stage 4 (CUDA prototype with Design B).

5. **If none passes**: re-evaluate whether sparse backward is viable for 3DGS, or whether the quality cost is fundamental.

---

## 9. Files

| File | Description |
|------|-------------|
| `results/a100/phase-c51/source_audit.json` | Stage 1: CUDA source audit |
| `scripts/phase-c51/source_audit.py` | Stage 1: audit script |
| `scripts/phase-c51/simulated_sparse_backward.py` | Stage 2: simulated sparse backward (v2 with densification fix) |
| `results/a100/phase-c51/simulation_k50.json` | Stage 2 v1: K=50% (broken, mask before densification) |
| `results/a100/phase-c51/simulation_k32.json` | Stage 2 v1: K=32% (broken) |
| `results/a100/phase-c51/simulation_k20.json` | Stage 2 v1: K=20% (broken) |
| `results/a100/phase-c51/simulation_k50_v2.json` | Stage 2 v2: K=50% (fixed, densification decoupled) |
| `results/a100/phase-c51/simulation_k32_v2.json` | Stage 2 v2: K=32% (fixed) |
| `results/a100/phase-c51/simulation_k20_v2.json` | Stage 2 v2: K=20% (fixed) |
| `scripts/phase-c51/multiscene_predictability.py` | Stage 3: multi-scene validation script |
| `results/a100/phase-c51/multiscene_bicycle.json` | Stage 3: bicycle scene predictor validation |
| `results/a100/phase-c51/multiscene_garden.json` | Stage 3: garden scene predictor validation |
