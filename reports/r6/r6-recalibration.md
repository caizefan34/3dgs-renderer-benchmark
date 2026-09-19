# R6 — Evidence Recalibration Report

## Executive summary

The original R6 screening contained two systematic evidence errors that inflated
the apparent opportunity for all three candidates (R6-A, R6-B, R6-C). After
direct measurement with higher-quality evidence, all three candidates are
DEFERRED or DROPPED. No candidate meets the 5% conservative E2E threshold.

| Candidate | Original verdict | Corrected verdict | Evidence upgrade |
|-----------|-----------------|-------------------|-----------------|
| R6-B | PASS (B-GATE 9/9) | **CLOSED/DROP** | torch.profiler → direct CUDA event |
| R6-A | PASS (A-GATE 2/9) | **DEFER** | footprint estimate → direct kernel counter |
| R6-C | PASS (C-GATE 8/9) | **DEFER** | broken oracle → bandwidth-limited model |
| True AccuTile | — | identity validation pending | — |

## Item 1: Root cause of the T_zero overestimate

### The claim

Original R6-1 reported T_zero = 2.36–8.33 ms = 5.5–11.6% of T_bwd, attributed
to "gradient buffer zero-initialization (at::zeros_like → cudaMemsetAsync)
for all gradient buffers during backward."

### The reality

Direct CUDA-event measurement (B0 T_clear) shows the actual gradient buffer
zero-fill costs only 0.031–0.137 ms = 0.14–0.36% of T_bwd. The original
estimate was **30–60× too high**.

### Root cause

The `r6_1_bwd_decompose.py` profiler (line 168) matched kernel names containing
`"memset" or "fill"`:

```python
elif "memset" in name_lower or "fill" in name_lower:
    kernel_categories["memset_zero"].append(cuda_self)
```

This `memset_zero` category captured **every** memset/fill operation during
the backward pass, not just the 9 gradient buffer zero-init calls. The
additional time came from:

1. **DSSIM (SepSSIM) loss backward intermediates** — 15-channel separable
   Gaussian conv2d allocates and zero-fills multiple intermediate tensors
2. **L1 loss backward intermediates** — elementwise gradient buffer allocation
3. **Autograd internal bookkeeping** — temporary buffers for backward graph
4. **Potential cross-pass contamination** — kernels queued in the forward
   pass but executed during the backward profiling window

The `r6-2-zero-init-oracle.md` report then misattributed the entire
`memset_zero` category to "at::zeros_like for all gradient buffers," listing
9 specific buffers totaling 21N–69N floats. **This attribution was wrong.**

### Evidence hierarchy resolution

| Source | Level | T_zero (ms) | Status |
|--------|-------|-------------|--------|
| B0 T_clear (CUDA event) | 1 (direct implementation) | 0.031–0.137 | **AUTHORITATIVE** |
| torch.profiler memset_zero | 3 (attribution) | 2.36–8.33 | **SUPERSEDED** |

Level 1 supersedes level 3. The old claim is not preserved.

---

## Item 2: Corrected touched definitions

### R6-B context

| Term | Definition | Source |
|------|-----------|--------|
| `flatten_ids [n_isects]` | Tile–Gaussian intersection list **with duplicates**. A Gaussian intersecting 5 tiles appears 5 times. | gsplat forward metadata |
| `N_intersected` | `len(torch.unique(flatten_ids))` — unique Gaussian IDs that appear in ≥1 tile intersection. Correct "touched" set for selective-clear. | r6_1_bwd_decompose.py line 204 |
| `r_touch` (R6-1) | `N_intersected / N_total` = 0.18–0.33 (single camera, metadata-derived) | r6-1 report |
| `r_touch` (R6-B benchmark) | ≈ 0.6–0.8 (measured from B1-v2 clear behavior — selective clear writes to most rows) | r6-b-benchmark.md |
| `tiles_per_gauss` | Per-[camera, Gaussian] predicate in rasterization `meta` dict. Available but NOT used by B1-v1. | gsplat 1.5.3 API |

**B1-v1 flaw**: `prev_touched_ = flatten_ids` → O(n_isects) gradient writes
(one clear per intersection, not per unique Gaussian). B1-v2 repaired this
with a scatter→scan→clear approach.

### R6-A context

| Term | Definition | Source |
|------|-----------|--------|
| `n_isects` | Total (tile, Gaussian) intersection pairs (with cross-tile duplicates). Equals atomic count after block aggregation (1 per pair). | forward metadata |
| `actual_atomic` | Actual warp-leader atomicAdd execution sets, counted by `__device__` counter in the kernel. | direct debug instrumentation |
| `R_atomic_direct` | `actual_atomic / n_isects` — within-tile warp multiplicity. If every pair is processed by exactly 1 warp, R=1. | direct measurement |
| `block_unique_atomic` | `n_isects` — the theoretical minimum after block aggregation (1 atomic per tile-Gaussian pair) | analytical |

---

## Item 3: Corrected evidence hierarchy

See `r6-evidence-hierarchy.md` for the full report. Summary:

| Level | Type | R6-B example | R6-A example |
|-------|------|-------------|-------------|
| 1 | Direct implementation CUDA event | B0 T_clear = 0.031–0.137 ms | T_bwd, T_iter (CUDA events) |
| 2 | Direct debug instrumentation | — | `__device__` atomic counter (pending results) |
| 3 | torch.profiler attribution | T_zero (SUPERSEDED) | T_raster (validated) |
| 4 | Hardware throughput model | — | Conservative E2E (depends on R_atomic) |
| 5 | Metadata-derived oracle | r_touch, grad buffer bytes | Footprint R_atomic (SUPERSEDED) |

**Principle**: When evidence levels conflict, the higher-quality (lower-numbered)
evidence wins. The old claim is not preserved merely because it motivated the
optimization.

---

## Item 4: R6-A direct atomic results

The instrumented kernel inserts `::atomicAdd(&r6a_debug_count, 1ULL)` inside
the `if (warp.thread_rank() == 0)` block of
`rasterize_to_pixels_3dgs_bwd_kernel`. This counts every actual warp-leader
atomicAdd execution set, using the kernel's real `ALPHA_THRESHOLD = 1/255`
(not the simulation's 0.01), on all tiles (not a 200-tile sample).

### Results (all 9 workloads, A100, 5 measured iterations)

| Scene | Stage | n_isects | actual_atomic | R_atomic_direct | E2E (%) | Gate |
|-------|-------|----------|--------------:|----------------:|--------:|------|
| room | 5K | 4,482,292 | 17,829,372 | 3.978 | 2.57 | ❌ |
| room | 15K | 4,312,157 | 14,239,098 | 3.302 | 1.96 | ❌ |
| room | 30K | 4,002,983 | 11,470,825 | 2.866 | 1.57 | ❌ |
| bicycle | 5K | 6,669,895 | 22,032,638 | 3.303 | 2.36 | ❌ |
| bicycle | 15K | 8,877,402 | 20,225,740 | 2.278 | 1.30 | ❌ |
| bicycle | 30K | 7,898,651 | 16,609,715 | 2.103 | 1.09 | ❌ |
| garden | 5K | 4,166,629 | 13,234,996 | 3.176 | 1.72 | ❌ |
| garden | 15K | 5,879,867 | 14,619,764 | 2.486 | 1.38 | ❌ |
| garden | 30K | 5,502,981 | 12,848,502 | 2.335 | 1.21 | ❌ |

**R_atomic_direct = 2.103–3.978** (all ≥ 2, condition 1 passes 9/9)
**E2E = 1.09–2.57%** (all < 5%, condition 2 fails 0/9)
**Counter std = 0.0** for all workloads (perfectly deterministic)

### Comparison across all three evidence levels

| Method | Evidence level | R_atomic range | vs direct |
|--------|---------------|----------------|-----------|
| Footprint estimate (r6_4) | 5 | 5.50–7.61 | 1.4–1.9× too high |
| Pixel-level simulation (r6_a_direct_atomic.py) | 2 (proxy) | 2.07–2.86 | 0.7–1.0× (slightly underestimated) |
| **Direct kernel counter** | **2** | **2.10–3.98** | **ground truth** |

The pixel simulation underestimated because it used alpha_threshold=0.01
(2.5× higher than the kernel's 1/255). The footprint estimate overestimated
because bounding-box radii are much larger than actual alpha coverage.

---

## Item 5: Corrected R6-A conservative E2E oracle

### Formula

```
T_atomic = 5 × R_atomic_direct × n_isects / 30e9 × 1000  (ms, 5 cache lines, 30 G/s)
T_atomic_ideal = 5 × n_isects / 30e9 × 1000  (ms, block-aggregated)
Savings = (T_atomic - T_atomic_ideal) × 0.70  (30% __syncthreads overhead)
E2E = Savings / T_iter × 100
```

### Results

| Scene | Stage | R_atomic | T_atomic (ms) | T_atomic_ideal (ms) | Savings (ms) | T_iter (ms) | E2E (%) |
|-------|-------|----------|--------------:|--------------------:|-------------:|------------:|--------:|
| room | 5K | 3.978 | 2.97 | 0.75 | 1.56 | 60.59 | 2.57 |
| room | 15K | 3.302 | 2.37 | 0.72 | 1.16 | 59.15 | 1.96 |
| room | 30K | 2.866 | 1.91 | 0.67 | 0.87 | 55.61 | 1.57 |
| bicycle | 5K | 3.303 | 3.67 | 1.11 | 1.79 | 76.04 | 2.36 |
| bicycle | 15K | 2.278 | 3.37 | 1.48 | 1.32 | 101.90 | 1.30 |
| bicycle | 30K | 2.103 | 2.77 | 1.32 | 1.02 | 93.63 | 1.09 |
| garden | 5K | 3.176 | 2.21 | 0.69 | 1.06 | 61.49 | 1.72 |
| garden | 15K | 2.486 | 2.44 | 0.98 | 1.02 | 73.77 | 1.38 |
| garden | 30K | 2.335 | 2.14 | 0.92 | 0.86 | 70.66 | 1.21 |

**Corrected E2E = 1.09–2.57%** — below 5% for all 9 workloads.

### Comparison across evidence levels

| Method | R_atomic | E2E range | A-GATE-3 pass |
|--------|----------|-----------|---------------|
| Footprint + HW model (v1) | 5.5–7.6 | 1.8–10.0% | 2/9 |
| Pixel sim + HW model (v2) | 2.1–2.9 | 0.68–1.58% | 0/9 |
| **Direct counter + HW model** | **2.1–4.0** | **1.09–2.57%** | **0/9** |

The v1 estimate was inflated by 1.4–1.9× in R_atomic, producing false-positive
gate passes for bicycle 30K (7.8% → 1.09%) and garden 30K (10.0% → 1.21%).

### Gate conditions (user-specified)

R6-A may advance to minimal CUDA implementation ONLY if ALL:
1. Direct reduction potential ≥ 2× (R_atomic_direct ≥ 2)
2. Conservative E2E ≥ 5%
3. At least 2 real workloads pass both conditions

---

## Item 6: Whether R6-A may enter minimal CUDA implementation

### Verdict: **NO — DEFER**

R6-A does NOT advance to minimal CUDA implementation.

### Gate results

| Condition | Result |
|-----------|--------|
| 1. Direct reduction potential ≥ 2× | **PASS 9/9** (R_atomic = 2.10–3.98) |
| 2. Conservative E2E ≥ 5% | **FAIL 0/9** (E2E = 1.09–2.57%) |
| 3. At least 2 real workloads pass | **FAIL** (0/9 pass both) |

### Rationale

The block-aggregation mechanism is sound and directly measured: the warp-leader
atomic count is 2.1–4.0× higher than the block-aggregated minimum. However, even
with the directly measured R_atomic (which is higher than the pixel simulation
predicted), the conservative E2E impact is only 1.09–2.57% — well below the 5%
threshold for all 9 workloads.

The atomic count is perfectly deterministic (std=0.0), confirming this is a
genuine but SMALL opportunity, not measurement noise.

### Why not DROP?

1. The mechanism is sound (block-level aggregation is correct and novel)
2. R_atomic ≥ 2 confirms real duplication (condition 1 passes 9/9)
3. The direct measurement is deterministic and reproducible
4. Future hardware (more atomics contention) or larger scenes may increase benefit
5. The opportunity may become significant in composition with other exact optimizations

### What would change the verdict?

- If ncu profiling (requires root) shows actual atomic stall fraction > 50% of
  T_raster (vs. our model's implied ~10–15%), the E2E could reach 5%
- If the __syncthreads overhead is < 10% (vs. our 30% assumption), savings increase
- If composed with other exact optimizations, the relative E2E could be larger

---

## R6-C status: DEFER (confirmed)

R6-C remains DEFER. The repaired oracle (r6-c-oracle-repair.md) counts only
physically removable gradient traffic (2 × grad_bytes). Conservative E2E =
0.48–2.21% for all 9 workloads. See `r6-c-confirmation.md`.

The T_zero recalibration does not affect R6-C because R6-C targets
gradient write/read traffic at the backward-optimizer boundary, not
gradient buffer zero-init.

---

## Updated portfolio

| Candidate | Status | Evidence quality | Conservative E2E | Correctness |
|-----------|--------|-----------------|-----------------|-------------|
| R6-B | **CLOSED/DROP** | Level 1 (direct) | <0.4% of T_bwd (measured) | PASS (but no benefit) |
| R6-A | **DEFER** | Level 2 (direct counter) | 1.09–2.57% (measured) | N/A (not implemented) |
| R6-C | **DEFER** | Level 4 (model) | 0.48–2.21% | N/A (not implemented) |
| True AccuTile | identity validation pending | — | — | — |

### R6-B: CLOSED/DROP

- B0 (persistent + full clear): mean Δiter = −0.11%, 7/9 negative. No benefit.
- B1-v2 (touched-mask selective clear): mean Δiter = −0.40%, 9/9 negative.
  Scatter costs MORE than the clear it eliminates.
- Correctness: PASS (stale gradient test, gradient comparison, topology safety)
- The gate was a false positive caused by profiler attribution error.
- T_clear = 0.031–0.137 ms = <0.4% of T_bwd (directly measured).

### R6-A: DEFER

- Direct kernel counter: R_atomic = 2.10–3.98 (passes ≥2× condition 9/9)
- Conservative E2E = 1.09–2.57% (fails ≥5% condition 0/9)
- Counter is perfectly deterministic (std=0.0)
- The block-aggregation mechanism is sound but the E2E impact is too small.
- See `r6-a-direct-kernel-results.md` for full details.

### R6-C: DEFER

- Conservative E2E = 0.48–2.21%, below 5% for all workloads.
- Counts only physically removable gradient traffic (2 × grad_bytes).
- See `r6-c-confirmation.md` for full details.

---

## What was NOT done

- No new optimization candidate was started
- No approximate gradients, gradient skipping, or loss/densification changes
- B2 was not implemented
- R6-A was not implemented (only instrumented for measurement)
- The old T_zero claim was not preserved merely because it motivated R6-B
