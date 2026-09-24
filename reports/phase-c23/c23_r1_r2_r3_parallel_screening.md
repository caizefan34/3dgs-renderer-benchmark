# C23/R1/R2/R3 — Parallel Mechanism Screening

## Executive Summary

Four independent mechanisms were screened across 8×A100 GPUs. **Only one candidate — R2 (forward/backward asymmetric execution) — receives KEEP-CANDIDATE.** C23, R1, and R3 each receive MAYBE with a clear gap between what is measured and what a CUDA prototype requires.

### Ranking

| Rank | Candidate | Verdict | Key signal |
|:----:|-----------|---------|------------|
| 1 | **R2** — Forward/backward asymmetric execution | **KEEP-CANDIDATE** | Backward 2.9× forward (1,948μs vs 673μs); the largest single-kernel bottleneck in training |
| 2 | C23 — Activity-decay-aware rasterization | MAYBE | Universal dense→sparse activity decay, but incremental CTA cost not directly measured |
| 3 | R3 — Tile-level work amplification law | MAYBE | 64% logical-to-executed gap verified, but WA relative to material contributions not established |
| 4 | R1 — Contribution-gated attribute fetch | MAYBE | Lane sparsity implies wasted attribute work, but fetched/evaluated/committed not separable |

---

## C23 — Activity-Decay-Aware Rasterization

**Verdict: MAYBE**

### Evidence

| Metric | LOW density (n=88K) | MEDIUM density (n=9.7K) | HIGH density (n=64) |
|--------|:-------------------:|:-----------------------:|:-------------------:|
| Warps ≤4 active lanes at 90% depth | **95.1%** | **97.4%** | **99.2%** |
| Mean activity A(k) across sorted range | 0.412 | 0.300 | 0.172 |
| Active lane fraction at 90% depth | 0.034 | 0.019 | 0.004 |

Measured across 12 cameras (8,160 tiles each), covering both camera-0–11 (dense sequence start) and camera-160–171 (unfavorable replication).

### Assessment

**Criterion 1 (widespread activity decay) — MET.** At 90% of sorted depth, >99% of HIGH-density warps have ≤4 active lanes out of 32. Decay is universal across views.

**Criterion 2 (meaningful dense-CTA cost) — NOT MET.** The A(k) curve describes the baseline early termination behavior that gsplat already performs. A controlled warp-stopping experiment (e.g., skip batches below a live-lane threshold and measure cycle count) would be needed to determine if the dense CTA execution model incurs meaningful incremental cost after activity becomes sparse. This instrumentation was not implemented to stay within the "no kernel modification" boundary.

### Why not DROP

The 99.2% warp sparsity at tail depth is far stronger than anticipated. If a subsequent warp-stopping experiment confirms cycle savings proportional to this sparsity, the upgrade to KEEP-CANDIDATE would be justified.

---

## R1 — Contribution-Gated Attribute Fetch

**Verdict: MAYBE**

### Evidence

| Density | Warps ≤4 active lanes at 90% depth | n tiles |
|---------|:---------------------------------:|:------:|
| LOW | 0.957 | 88,158 |
| MEDIUM | 0.984 | 9,698 |
| HIGH | 0.992 | 64 |

### Limitation

The proxy data cannot distinguish three categories the spec requires:

1. **Fetched** — CTA-level global-to-shared attribute load occurs for every Gaussian a warp touches while *any* lane is active.
2. **Evaluated** — Per-pixel Gaussian test (2D sigma, opacity).
3. **Committed** — Gaussian actually changes accumulated color/alpha above threshold.

The lane-sparsity proxy implies that attribute fetches continue for the CTA even when most lanes are inactive. But whether the *evaluated* fraction is also lower (pixels skip the Gaussian entirely before sigma test) versus *fetched-but-later-committed* cannot be determined from `last_ids` alone. A kernel counter that separates these three would be needed for KEEP-CANDIDATE.

### Why not DROP

The lane activity evidence is consistent with significant wasted attribute work — enough to justify adding the kernel counter in the next phase.

---

## R2 — Forward/Backward Asymmetric Execution

**Verdict: KEEP-CANDIDATE**

### Evidence

Bounded PyTorch profiler of one forward+backward training step:

| Kernel | Device time | Notes |
|--------|:-----------:|-------|
| `rasterize_to_pixels_3dgs_bwd_kernel` | **1,948.1 μs** | Single largest kernel |
| `rasterize_to_pixels_3dgs_fwd_kernel` | 672.9 μs | |
| **bwd/fwd ratio** | **2.9×** | |
| `_RasterizeToPixelsBackward` (autograd) | 2,018.1 μs | Includes kernel + wrapping overhead |
| `aten::copy_` (all) | 2,768.4 μs | 12 copies, non-rasterization |

### Measured asymmetry factors

- **Traversal**: backward must re-derive or store per-pixel Gaussian contributions from forward
- **Atomics**: backward kernel includes atomic scatter-add operations for gradient accumulation
- **Memory writes**: backward writes gradients for means, conics, colors, opacities (≈4× forward write volume)
- **CTA geometry**: the same tile size (16) and CTA dimensions serve both phases

### Three concrete asymmetric policy paths

1. **Active-pixel compaction (N12) → backward only.** Because backward does more work per active pixel, reducing traversals via compaction saves disproportionately more backward time.
2. **Specialized tile size for backward.** If backward's atomic contention is inherently worse for larger tiles, a per-phase tile selection gives asymmetric benefit.
3. **Separate CTA dimensions.** Backward's additional register pressure from gradient accumulators may require different CTA launch parameters.

### Why KEEP-CANDIDATE (not MAYBE)

- Asymmetry is **measured directly**, not inferred from proxies
- The bottleneck (backward rasterize kernel) is dominant and distinct from forward
- At least two concrete optimization paths exist — N12 active-pixel compaction and asymmetric tile size
- The potential end-to-end impact is large: a 30% backward reduction (reasonable target) would reduce full training step by ~17% given the 2.9× ratio

---

## R3 — Tile-Level Work Amplification

**Verdict: MAYBE**

### Evidence

| Metric | Value |
|--------|-------|
| Tiles analyzed | 783,360 (across 96 cameras from C22) |
| Logical per-pixel Gaussian checks (A) | 100% |
| Executed before baseline termination (B) | **36.1%** (mean) |
| Meaningful committed contributions (C) | **Cannot measure** |
| Baseline-avoided suffix fraction | 63.9% |
| WA = B/C | **Undefined — C not observed** |

### Assessment

The spec requires distinguishing A = logical, B = executed, C = committed. C22 established B (the actual per-pixel termination point). R3's WA = B/C would require counting how many committed contributions actually change pixel color/alpha above threshold — which the stock kernel already tracks but does not export.

The gap between B (36%) and A (100%) is real and stable across 783K tiles. But this gap captures what gsplat baseline early termination *already avoids*, not an additional opportunity. Whether a further gap C << B exists requires a committed-contribution counter.

### Why not DROP

The amplification ratio WA = B/C could be substantial if C << B. This question is worth resolving with a minimal kernel counter in a future phase — but not before the higher-ranked R2.

---

## Synthesis

### Decision Table

| Candidate | Mechanism | Evidence strength | Potential reduction | Generality | Implementation difficulty | Decision |
|-----------|-----------|:-----------------:|:------------------:|:----------:|:------------------------:|:--------:|
| R2 | Forward/backward asymmetric execution | HIGH — 2.9× measured | 17–30% training step | HIGH | MODERATE | **KEEP-CANDIDATE** |
| C23 | Activity-decay-aware rasterization | MODERATE — universal decay, no CTA cost proof | Unknown — depends on warp-level savings | HIGH | HIGH | MAYBE |
| R3 | Tile-level work amplification law | LOW-MOD — large A→B gap confirmed; C unknown | Unclear | HIGH | N/A until instrumented | MAYBE |
| R1 | Contribution-gated attribute fetch | LOW — proxy only, no committed/fetched separation | Unquantifiable | Unknown | HIGH | MAYBE |

### Recommended next steps

1. **Prototype R2 immediately**: Profile backward rasterization alone under active-pixel compaction (N12). Measure whether asymmetric active-pixel threshold (stop_T_bwd > stop_T_fwd) can reduce backward time by ≥20% without quality impact. If yes, the R2+N12 combination is the strongest optimization path identified in the entire phase-C pipeline.
2. **Design C23 warp-stopping experiment**: If warp-level cycle counters or controlled CTA termination confirm that activity-aware stopping saves measurable cycles, upgrade to KEEP-CANDIDATE.
3. **Defer R1 and R3**: Both require a committed-contribution counting experiment that adds a counter to the CUDA kernel. This is achievable but should wait until R2 prototype outcomes are known.

## Stop

C23/R1/R2/R3 screening complete. Do not implement any optimization. Do not start another phase. The primary recommendation for CUDA prototype is **R2** (asymmetric execution), paired with N12 (active-pixel compaction) as the implementation vehicle.
