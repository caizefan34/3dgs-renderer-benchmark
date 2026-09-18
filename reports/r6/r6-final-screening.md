# R6 — Exact Backward Cost Elimination: Final Screening Report

## Objective

Find exact / semantics-preserving backward optimizations that reduce real
training cost without intentionally removing gradient information.

## Hard rules (enforced throughout)

- NO gradient skipping
- NO approximate gradients
- NO loss/densification/pruning/optimizer/LR/rendering changes
- Require g_optimized ≈ g_baseline (only FP accumulation-order differences allowed)

## Candidates screened

| ID | Name | Mechanism |
|----|------|-----------|
| R6-B | Touched-only / lazy-zero gradient buffer init | Eliminate zeroing of untouched gradient buffers (67-82% of N never touched per frame) |
| R6-A | Block-level gradient aggregation | Reduce global atomicAdd by aggregating warp partials in shared memory before block-leader atomic |
| R6-C | Backward-optimizer fusion | Eliminate gradient write→read traffic by fusing Adam update into backward kernel |

## Profiling methodology

- **Hardware**: A100-PCIE-40GB × 8, CUDA 12.4, PyTorch 2.4.1, gsplat 1.5.3
- **Workloads**: 3 scenes (room, bicycle, garden) × 3 training stages (5K, 15K, 30K) = 9 profiles
- **Configuration**: ReferenceV1Config (seed=0, SH deg 3, 1080p, tile_size=16, packed=False, absgrad=True, Adam eps=1e-15)
- **Measurements**: CUDA Events (100 iterations, 20 warmup) + torch.profiler (30 backward-only iterations)
- **Profiler limitations**: ncu unavailable (ERR_NVGPUCTRPERM — GPU counters disabled for non-root; also ncu 2021.3 section files broken). nsys broken (GLIBC symbol error). Atomic stall fraction estimated via hardware throughput model (evidence level: MEDIUM).

## Key measured results

### R6-1: Backward decomposition (all 9 profiles)

| Metric | Range | Notes |
|--------|-------|-------|
| T_bwd / T_iter | 39-58% | Backward is primary cost |
| T_raster / T_bwd | 41-59% | Rasterizer is dominant backward kernel |
| T_zero / T_bwd | 5.5-11.6% | Zero-init is significant fixed cost |
| T_zero / T_iter | 2.2-6.0% | |
| r_touch | 0.18-0.33 | 67-82% of gradient buffers untouched |
| T_optimizer / T_iter | 4.6-16.3% | Scales with N, grows with training |

### R6-4: Warp duplicate (all 9 profiles)

| Metric | Range | Notes |
|--------|-------|-------|
| R_atomic | 5.50-7.61 | 5-8× redundant atomic ops per Gaussian |
| tiles/Gaussian | 7.4-25.1 | Cross-tile duplication (dominant) |
| warps/Gaussian | 3.8-6.6 | Total warp-level processing |
| Cross-tile fraction | 48-74% | Of R_atomic |

### R6-5: Fusion oracle (all 9 profiles)

| Metric | Range | Notes |
|--------|-------|-------|
| T_C_cons / T_iter | 4.8-17.2% | Conservative fusion speedup |
| Avoidable traffic | 309-2184 MB | Gradient read traffic eliminated |
| T_optimizer / N | 3.5-4.5 µs | Linear scaling with N |

## Gate evaluation

| Candidate | Gate | Condition | Result |
|-----------|------|-----------|--------|
| R6-B | B-GATE-1 | T_zero ≥ 5% T_bwd | **PASS 9/9** (5.5-11.6%) |
| R6-B | B-GATE-2 | T_zero ≥ 3% T_iter | PASS 7/9 (2.2-6.0%) |
| R6-B | B-GATE-3 | Sparse-tail (T_zero%T_bwd grows at 30K) | PASS (bicycle +21%, garden +35%) |
| R6-A | A-GATE-1 | Rasterizer ≥ 40% T_bwd | **PASS 9/9** (41-59%) |
| R6-A | A-GATE-2 | R_atomic ≥ 2 | **PASS 9/9** (5.5-7.6) |
| R6-A | A-GATE-3 | Conservative E2E ≥ 5% T_iter | PASS 2/9 (bicycle 30K: 7.8%, garden 30K: 10.0%) |
| R6-C | C-GATE | T_C_cons ≥ 5% T_iter | **PASS 8/9** (4.8-17.2%) |

## Conservative E2E oracle (all candidates, all profiles)

| Scene | Stage | R6-B | R6-A | R6-C |
|-------|-------|------|------|------|
| room | 5K | 1.9% | 3.3% | 4.8% |
| room | 15K | 2.6% | 3.0% | 7.2% |
| room | 30K | 1.6% | 2.6% | 7.5% |
| bicycle | 5K | 3.0% | 2.1% | 11.0% |
| bicycle | 15K | 3.7% | 1.8% | 16.5% |
| bicycle | 30K | 4.3% | 7.8% | 17.2% |
| garden | 5K | 3.4% | 3.5% | 11.3% |
| garden | 15K | 2.9% | 3.2% | 13.9% |
| garden | 30K | 3.8% | 10.0% | 14.2% |

## Final decisions

### ADVANCE: R6-B (touched-only / lazy-zero gradient buffer init)

**Rationale**:
- B-GATE-1 passes universally (5.5-11.6% T_bwd)
- Lowest implementation complexity (replace at::zeros_like with persistent buffer + selective zero)
- Lowest correctness risk (untouched gradients are already zero — no semantic change)
- Sparse-tail pattern confirmed: opportunity grows at later training stages
- Conservative E2E (1.6-4.3%) is below 5%, but upper bound (2.3-6.4%) reaches 5%+
  for bicycle/garden 30K. Buffer reuse may close the gap.
- Novel: gsplat 1.5.3 uses full at::zeros_like; touched-only zero is new

**Implementation path**: Persistent gradient buffers + zero only touched entries
(post-backward scatter-zero), or lazy zero-on-first-write in the rasterizer kernel.

### ADVANCE (conditional): R6-A (block-level gradient aggregation)

**Rationale**:
- A-GATE-1 + A-GATE-2 pass universally (9/9)
- Block-level reduction is novel (gsplat only does within-warp)
- A-GATE-3 passes for large-scene 30K (bicycle 7.8%, garden 10.0%)
- Conservative E2E is workload-dependent (1.8-10.0%)
- Correctness preserved: sum of warp partials = full gradient (FP order only)

**Condition**: Implementation MUST include a single-warp fast path to avoid
__syncthreads barrier overhead for the common case (1 warp per Gaussian per tile).
If the fast path cannot avoid net overhead, R6-A should be dropped.

**Implementation path**: Shared-memory gradient buffer + __syncthreads + block-
leader atomicAdd, with warp-count check for fast path.

### DEFER: R6-C (backward-optimizer fusion)

**Rationale**:
- C-GATE passes 8/9 with strongest results (4.8-17.2% T_iter)
- But implementation complexity is highest (embed Adam in backward kernel)
- Fusion concept has prior art (not novel as a concept; 3DGS-specific application is novel)
- Scope exceeds "exact backward optimization" — modifies optimizer behavior
- Should be re-evaluated as a separate initiative

## Stop condition evaluation

Per protocol: "NO hard implementation if none hit 5% conservative E2E."

- R6-A hits 5%+ conservative E2E for 2/9 profiles (bicycle 30K, garden 30K)
- R6-C hits 5%+ for 8/9 profiles but is deferred (complexity + scope)
- R6-B does not hit 5% conservative but reaches 5%+ upper bound for 4/9 profiles

Two candidates (R6-B, R6-A) advance to the implementation phase. The 5%
conservative threshold is met by R6-A (conditionally) and approached by R6-B
(with buffer reuse potential).

## Files produced

| File | Description |
|------|-------------|
| `reports/r6/r6-0-codepath-audit.md` | Backward call chain, 7 audit questions, zero-init budget |
| `reports/r6/r6-1-backward-cost-breakdown.md` | T_bwd decomposition (9 profiles) |
| `reports/r6/r6-2-zero-init-oracle.md` | R6-B gate evaluation + oracle |
| `reports/r6/r6-3-atomic-profile.md` | R6-A gate evaluation + atomic stall estimation |
| `reports/r6/r6-4-warp-duplicate-analysis.md` | R_atomic measurement + block-level reduction potential |
| `reports/r6/r6-5-fusion-oracle.md` | R6-C gate evaluation + traffic analysis |
| `reports/r6/r6-6-candidate-ranking.md` | Cross-candidate comparison + ranking |
| `reports/r6/r6-final-screening.md` | This report |
| `reports/r6/r6-profile-results.json` | Machine-readable results (all 9 profiles × 3 candidates) |
| `experiments/r6/r6_1_bwd_decompose.py` | Backward wall-time decomposition profiler |
| `experiments/r6/r6_3_ncu_target.py` | ncu target (single fwd+bwd) |
| `experiments/r6/r6_4_warp_duplicate.py` | Cross-tile/cross-warp atomic reduction potential |
| `experiments/r6/r6_5_fusion_oracle.py` | Optimizer traffic census + fusion oracle |
| `experiments/r6/summarize_results.py` | Results aggregator |
| `experiments/r6/collect_results.py` | Machine-readable JSON collector |

## Evidence level disclosure

- **R6-1 (backward decomposition)**: DIRECT measurement (CUDA Events + torch.profiler). Evidence level: HIGH.
- **R6-4 (warp duplicate)**: DIRECT measurement (forward intersection metadata). Evidence level: HIGH.
- **R6-5 (fusion oracle)**: DIRECT measurement (T_optimizer via CUDA Events) + analytical model (traffic census). Evidence level: HIGH for T_optimizer, MEDIUM for T_C model.
- **R6-3 (atomic stall)**: ncu unavailable. Estimated via hardware throughput model (A100 L2 atomic throughput). Evidence level: MEDIUM. Direct measurement would require root access to enable GPU performance counters.
