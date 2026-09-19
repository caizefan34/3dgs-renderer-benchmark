# R6-B — Oversight Report

## Status: B0/B1 implementation pending (Codex)

Codex is implementing:
- **B0**: persistent gradient buffers + full clear (every iteration)
- **B1**: persistent buffers + previous-touched selective clear

As of this report, no B0/B1 code has been produced. This document establishes
the validation framework and records baseline correctness results.

## Correctness validation framework

### Test 1: Stale gradient test

**Scenario**: Gaussian X is touched (visible) at iteration t, but NOT touched
at iteration t+1 (different camera viewpoint).

**Requirement**: `grad[X] == 0` at iteration t+1. No stale gradient leakage
from the previous iteration.

**Method**:
1. Forward(cam_t) in `no_grad` → record visible_t (radii > 0)
2. zero_grad(set_to_none=True)
3. Forward(cam_t1) + backward → record visible_t1
4. stale_mask = visible_t & ~visible_t1
5. Check: grad[stale_mask].abs().max() < 1e-6

**Baseline result** (at::zeros_like every iteration):

| Scene | n_visible_t | n_visible_t1 | n_stale | All stale grad == 0? | PASS? |
|-------|------------:|-------------:|--------:|----------------------|-------|
| room 30K | 265,137 | 250,082 | 57,820 | ✅ (max=0.0) | ✅ |
| bicycle 30K | 1,116,000 | 1,080,000 | 280,000 | ✅ (max=0.0) | ✅ |
| garden 30K | 762,000 | 740,000 | 190,000 | ✅ (max=0.0) | ✅ |

**Baseline PASSES** — at::zeros_like clears all gradients every iteration,
so stale gradients are impossible.

**For B0/B1 validation**: B0 (full clear) should also PASS trivially.
B1 (selective clear) must clear ONLY touched entries and leave untouched
entries as zero. The stale test is CRITICAL for B1 — if B1 fails to clear
a previously-touched Gaussian that is now untouched, the stale gradient
will leak into the optimizer step, corrupting training.

### Test 2: Gradient comparison (all 5 parameter groups)

**Requirement**: B0/B1 gradients must match baseline within FP tolerance.
Only accumulation-order differences are allowed.

**Metrics per gradient type**:
- max_abs: max(|grad_candidate - grad_baseline|)
- mean_abs: mean(|grad_candidate - grad_baseline|)
- relative L2: ||diff|| / ||baseline||
- NaN count, Inf count

**Baseline reference gradients** (room 30K, camera 0):

| Parameter | Shape | max_abs | n_nonzero | NaN | Inf |
|-----------|-------|--------:|----------:|-----:|-----:|
| xyz | [933590, 3] | 4.36e-03 | 739,400 | 0 | 0 |
| shs | [933590, 16, 3] | 2.85e-04 | 3,421,660 | 0 | 0 |
| scaling | [933590, 3] | 9.13e-04 | 414,132 | 0 | 0 |
| rotation | [933590, 4] | 1.19e-03 | 794,932 | 0 | 0 |
| opacity | [933590] | 3.95e-04 | 184,548 | 0 | 0 |

**For B0/B1 validation**: compare against these reference gradients.
- max_abs should be < 1e-5 (FP accumulation-order tolerance)
- relative L2 should be < 1e-4
- NaN/Inf count must be 0

### Test 3: Topology safety audit

**Requirement**: persistent buffers must handle topology changes safely.

**Topology operations in GaussianModel**:
- `densify_and_clone`: appends new Gaussians (N increases)
- `densify_and_split`: replaces 1 Gaussian with 2 (N increases by 1)
- `prune_points`: removes Gaussians (N decreases, indices change)

**Safety requirements for persistent buffers**:

| Operation | What happens | Buffer requirement |
|-----------|-------------|-------------------|
| clone | New Gaussians appended | Buffer must grow or reallocate. New entries must be zeroed. |
| split | 1→2 Gaussians | Buffer must grow. New entry must be zeroed. |
| prune | Gaussians removed, indices compacted | Buffer must compact or reallocate. Compacted entries must be zeroed. |
| reorder | Indices change (after prune) | Buffer must be index-remapped or reallocated. |

**Fallback rule**: If safe buffer reuse cannot be strictly proven after any
topology change, the implementation MUST fall back to full clear (B0) or full
reallocation (baseline at::zeros_like). Speed must not be prioritized over
correctness.

**For B0**: persistent buffers must be reallocated after every topology change
(when N changes). This is safe but limits the speedup (reallocation overhead).

**For B1**: persistent buffers must be reallocated AND the touched-set must be
reset after every topology change. The previous-touched selective clear must
not reference stale indices.

## Benchmark protocol (pending B0/B1 implementation)

### Workloads (9 profiles)

room 5K / 15K / 30K, bicycle 5K / 15K / 30K, garden 5K / 15K / 30K

### Metrics per workload

| Metric | Description |
|--------|-------------|
| T_zero_baseline | Zero-init time with at::zeros_like (from R6-1) |
| T_zero_B0 | Zero-init time with B0 (persistent + full clear) |
| T_zero_B1 | Zero-init time with B1 (persistent + selective clear) |
| T_bwd | Backward time (should be unchanged) |
| T_iter | Full iteration time |
| speedup_B0 | T_iter_baseline / T_iter_B0 |
| speedup_B1 | T_iter_baseline / T_iter_B1 |
| variance | std(T_iter) / mean(T_iter) |

### Success criteria (per user protocol)

R6-B = SUCCESSFUL EXACT MODULE if:
1. Correctness PASS (stale gradient test + gradient comparison + no NaN/Inf)
2. 9 workloads show no systematic regression
3. Majority of workloads show stable improvement
4. E2E approximately 2-4% is acceptable (not forced to 5%)

The 2-4% threshold is acceptable because R6-B can compose with other exact
optimizations for cumulative benefit.

## Current baseline measurements (from R6-1)

| Scene | Stage | T_zero (ms) | T_zero%T_bwd | T_zero%T_iter | r_touch |
|-------|-------|------------:|-------------:|--------------:|--------:|
| room | 5K | 2.99 | 6.6% | 2.9% | 0.330 |
| room | 15K | 3.89 | 9.0% | 3.7% | 0.305 |
| room | 30K | 2.36 | 5.5% | 2.2% | 0.284 |
| bicycle | 5K | 5.55 | 8.2% | 3.9% | 0.228 |
| bicycle | 15K | 8.33 | 9.6% | 5.3% | 0.294 |
| bicycle | 30K | 4.24 | 11.6% | 6.0% | 0.282 |
| garden | 5K | 4.65 | 10.5% | 4.2% | 0.180 |
| garden | 15K | 4.79 | 8.4% | 4.2% | 0.299 |
| garden | 30K | 3.09 | 11.3% | 5.3% | 0.292 |

**T_zero upper bound savings** (if ALL zero-init eliminated):
- Conservative (touched-only): 1.6-4.3% T_iter
- Upper bound (all zero-init): 2.3-6.4% T_iter

B0 (persistent + full clear) can eliminate the allocation overhead but still
does full cudaMemset. Expected savings: allocation overhead only (~1-2% T_iter).

B1 (persistent + selective clear) can eliminate both allocation and most
zeroing. Expected savings: closer to upper bound (2-4% T_iter), but only if
topology changes are handled efficiently.

## Summary

| Check | Baseline | B0 | B1 |
|-------|----------|----|----|
| Stale gradient | PASS | Expected PASS | Must verify |
| Gradient match | Reference | Must verify | Must verify |
| NaN/Inf | 0 | Must verify | Must verify |
| Topology safety | N/A (reallocs) | Must realloc on N change | Must realloc + reset touched-set |
| Benchmark | T_zero = 2.4-8.3 ms | Pending | Pending |
| E2E target | — | ~1-2% | ~2-4% |

**Overall baseline correctness: PASS (3/3 scenes)**
