# R6-6 — Candidate Ranking

## Gate results summary

| Candidate | Gate-1 | Gate-2 | Gate-3 (5% E2E) | Profiles passing | Best E2E | Median E2E |
|-----------|--------|--------|-----------------|-------------------|----------|------------|
| R6-B (zero-init) | 9/9 ✅ | 7/9 ✅ | 0/9 ❌ | 9/9 (B-GATE-1) | 4.3% | 3.0% |
| R6-A (atomic agg) | 9/9 ✅ | 9/9 ✅ | 2/9 ✅ | 9/9 (A-GATE-1+2) | 10.0% | 3.1% |
| R6-C (fusion) | 8/9 ✅ | — | 8/9 ✅ | 8/9 (C-GATE) | 17.2% | 11.3% |

## Conservative E2E oracle comparison

| Scene | Stage | R6-B cons (%T_iter) | R6-A cons (%T_iter) | R6-C cons (%T_iter) |
|-------|-------|--------------------:|--------------------:|--------------------:|
| room | 5K | 1.9% | 3.3% | 4.8% |
| room | 15K | 2.6% | 3.0% | 7.2% |
| room | 30K | 1.6% | 2.6% | 7.5% |
| bicycle | 5K | 3.0% | 2.1% | 11.0% |
| bicycle | 15K | 3.7% | 1.8% | 16.5% |
| bicycle | 30K | 4.3% | 7.8% | 17.2% |
| garden | 5K | 3.4% | 3.5% | 11.3% |
| garden | 15K | 2.9% | 3.2% | 13.9% |
| garden | 30K | 3.8% | 10.0% | 14.2% |

## Ranking criteria

1. **Conservative E2E impact** (primary): must reach ≥5% T_iter under conservative
   assumptions to justify implementation effort
2. **Gate universality**: does the candidate pass across all scenes/stages?
3. **Implementation complexity** (secondary): lower complexity preferred
4. **Novelty** (secondary): must offer something not already in gsplat 1.5.3
5. **Correctness risk** (secondary): must preserve exact gradients

## Candidate assessment

### R6-C: Backward-Optimizer Fusion

**Score: STRONGEST gate, HIGHEST complexity**

- C-GATE: 8/9 pass (4.8-17.2% T_iter)
- Only room 5K narrowly misses (4.8% ≈ 5%)
- Scales with N — grows as training progresses
- Largest absolute savings for large scenes (bicycle 30K: 14.46ms saved/iter)
- **But**: implementation complexity is highest (modify rasterizer backward +
  embed Adam logic + handle absgrad separately + coordinate with densification)
- **Prior art**: fusion concept exists, but 3DGS-specific fusion is novel
- **Verdict**: DEFER to separate implementation phase. Gate potential is confirmed
  but scope exceeds "exact backward optimization" framing.

### R6-B: Touched-Only / Lazy-Zero Gradient Buffer Init

**Score: UNIVERSAL gate, MODERATE complexity, LOW conservative E2E**

- B-GATE-1: 9/9 pass (5.5-11.6% T_bwd)
- B-GATE-2: 7/9 pass (3% T_iter)
- Conservative E2E: 1.6-4.3% (below 5% threshold)
- Upper bound E2E: 2.3-6.4% (reaches 5%+ for bicycle/garden 30K)
- **Novel**: gsplat uses at::zeros_like (full zero); touched-only zero is new
- **Correctness**: untouched buffers MUST remain zero (Adam reads them). The
  optimization must zero only touched entries, keeping untouched as zero. This
  is exactly equivalent — untouched gradients are already zero in baseline.
- **Implementation**: replace at::zeros_like with (1) pre-allocated persistent
  buffer + (2) zero only touched entries after backward, OR (3) lazy zero-on-
  write in the rasterizer backward. Approach (3) is cleanest — initialize
  gradient to zero only when first writing, using a sentinel/flag.
- **Risk**: The conservative estimate (1.6-4.3%) assumes only untouched buffers
  are skipped. The upper bound (2.3-6.4%) assumes ALL allocation overhead is
  eliminated via buffer reuse. The truth is between these, likely 3-5%.
- **Verdict**: ADVANCE — gate passes universally, low implementation complexity,
  low correctness risk. Conservative E2E is below 5% but upper bound reaches
  5%+ for large scenes at 30K. Buffer reuse may push achievable savings higher.

### R6-A: Block-Level Gradient Aggregation

**Score: STRONG gates, MODERATE-HIGH complexity, WORKLOAD-DEPENDENT E2E**

- A-GATE-1: 9/9 pass (41-59% T_bwd is rasterizer)
- A-GATE-2: 9/9 pass (R_atomic = 5.5-7.6)
- A-GATE-3: 2/9 pass (≥5% only for bicycle 30K and garden 30K)
- Conservative E2E: 1.8-10.0% (highly workload-dependent)
- **Novel**: gsplat does within-warp aggregation; block-level is new
- **Correctness**: block-level reduction of warp partials → single atomicAdd.
  Sum of warp partials = full gradient. FP accumulation-order difference only.
- **Implementation**: shared-memory buffer + __syncthreads + block-leader
  atomicAdd. Need fast path for single-warp Gaussians (avoid barrier overhead).
- **Risk**: __syncthreads barrier adds latency. For Gaussians processed by only
  1 warp in a block (common), the barrier is pure overhead. Net benefit depends
  on the fraction of multi-warp-per-tile Gaussians. The conservative model
  already accounts for 30% overhead, but actual overhead may be higher.
- **Verdict**: ADVANCE (conditional) — gates pass, but A-GATE-3 only reaches 5%
  for large-scene 30K. The block-level mechanism is sound and novel. Conservative
  E2E is borderline — implementation should include a single-warp fast path to
  minimize barrier overhead.

## Ranking

| Rank | Candidate | Gate pass | Conservative E2E | Complexity | Novelty | Decision |
|------|-----------|-----------|------------------|------------|---------|----------|
| 1 | R6-B | 9/9 ✅ | 1.6-4.3% (UB: 2.3-6.4%) | LOW | YES | ADVANCE |
| 2 | R6-A | 9/9 ✅ (A-GATE-3: 2/9) | 1.8-10.0% | MOD-HIGH | YES | ADVANCE (conditional) |
| 3 | R6-C | 8/9 ✅ | 4.8-17.2% | HIGH | Partial | DEFER |

## Advancing candidates

Per protocol: at most 2 candidates advance to implementation.

- **R6-B** advances: universal gate pass, lowest implementation complexity, lowest
  correctness risk. Conservative E2E is below 5% but achievable savings may reach
  5% with buffer reuse (upper bound). The sparse-tail pattern (B-GATE-3) confirms
  the opportunity grows at later training stages.

- **R6-A** advances (conditional): strong gates (A-GATE-1, A-GATE-2 universal),
  novel block-level mechanism, but A-GATE-3 is workload-dependent. The
  conditional advance requires a single-warp fast path in the implementation to
  avoid barrier overhead for the common single-warp case. If the fast-path
  implementation shows net overhead exceeds savings, R6-A should be dropped.

- **R6-C** is deferred: strongest gate results but highest implementation
  complexity and scope exceeds "exact backward optimization." The fusion concept
  has prior art. R6-C should be re-evaluated as a separate initiative.
