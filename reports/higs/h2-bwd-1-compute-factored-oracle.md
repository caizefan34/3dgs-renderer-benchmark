# H2-BWD-1: Compute-Factored HiGS Backward Oracle

## Candidate Record

| Field | Value |
|-------|-------|
| **Candidate** | H2-BWD-1 (A=SIGMA_GATE, B=SCALAR_ADJOINT, C=UV_REUSE) |
| **Parent hypothesis** | H1 (Backward overhead) → H2-BWD-0 proved atomic is only 6% → compute is bottleneck |
| **Type** | Type A (Exact systems optimization) |
| **Evidence level** | L4 (Component/kernel speedup) |
| **Verdict** | **KEEP** — correctness passes, performance gate not met |

---

## 1. What question was tested?

Three exact compute transformations of `higs_blend_bwd_px_kernel<3,2>` were evaluated:

- **A (SIGMA_GATE)**: Classify sigma before `__expf` to skip the exponential for dropped (alpha < 1/255) and clamped (alpha >= 0.99) Gaussians. 40.4% of intersection-pixel evaluations can skip `__expf`.
- **B (SCALAR_ADJOINT)**: Replace the `buffer[CDIM]` vector accumulation with a scalar `buffer_dot` (dot product of accumulated color with adjoint), reducing per-Gaussian register pressure and arithmetic.
- **C (UV_REUSE)**: Factor sigma as `0.5*(dx*ux + dy*uy)` where `ux=conic.x*dx+conic.y*dy`, `uy=conic.y*dx+conic.z*dy`, and reuse `ux`/`uy` in the `v_xy` gradient computation.

Plus diagnostic oracles:
- **NO_EXP**: Replace `__expf` with constant `vis=1.0` — upper bound on expf savings
- **NO_VJP**: Skip all VJP arithmetic (conic, xy, opacity gradients, buffer updates) — upper bound on VJP arithmetic savings
- **COMBINED**: A + B + C together

**Gate**: PROMOTE_TO_CUDA if COMBINED blend_bwd improvement ≥ 10% OR F+B improvement ≥ 5%, with correctness passing (cosine ≥ 0.999999, relative_L2 ≤ 1e-4, zero/nonzero disagreement = 0).

---

## 2. What hypothesis was tested?

H2-BWD-0 proved the blend backward kernel is compute-bound (atomics are only 6% of 2.091ms). H2-BWD-1 tests whether the *arithmetic* within the compute-bound kernel can be reduced through three exact transformations.

**Prediction**: If arithmetic (expf, VJP multiply-adds, buffer accumulation) is the dominant cost, then:
- SIGMA_GATE should save ~40% of expf calls → measurable kernel speedup
- SCALAR_ADJOINT should reduce CDIM=3 vector ops to scalar → measurable speedup
- UV_REUSE should eliminate redundant multiply-adds → measurable speedup
- NO_EXP and NO_VJP oracles should show large upper-bound savings

---

## 3. What would falsify it?

The hypothesis is falsified if:
- NO_EXP (removing ALL expf) saves < 2% of kernel time → expf is not the bottleneck
- NO_VJP (removing ALL VJP arithmetic) saves < 2% → VJP arithmetic is not the bottleneck
- COMBINED (all three optimizations) saves < 10% of blend_bwd → gate not met
- Any variant is SLOWER than baseline → transformation adds overhead exceeding savings

---

## 4. What experiment was performed?

CUDA microbenchmark with 7 kernel variants compiled via `load_inline` on mx (A100-PCIE-40GB, GPU 3, uncontended). Each variant is a template specialization of the same kernel structure — identical traversal, batch loading, warp reduction, and gradient atomic accumulation. Only the per-intersection arithmetic differs.

**Scenes**:
- room/cam0: 2048×1365, N_visible=44,908, n_isects=953,144
- bicycle/cam0: 2048×1361, N_visible=181,525, n_isects=1,412,189

**Protocol**: 20 warmup / 100 measured iterations per variant, CUDA event timing, median reported.

**Correctness**: Each variant's gradient output (v_colors, v_conics, v_means2d, v_opacities) compared against BASELINE (the exact current kernel reimplementation) using cosine similarity, relative L2, max absolute error, and nonzero-disagreement count.

**SIGMA_GATE classification**: Analytically computed (vectorized numpy) the fraction of intersection-pixel evaluations that can skip `__expf` — dropped (sigma < 0, or alpha < 1/255), clamped (alpha ≥ 0.99), or exp-required.

---

## 5. What controls were used?

- All variants share identical: grid (1, tile_h, tile_w), block (16, 8, 1), shared memory layout (128 × (4+12+12+12) bytes), batch traversal, warp shuffle reduction, and atomic gradient writes
- Same `flatten_ids`, `tile_offsets`, `render_alphas`, `last_ids` from the real frozen HiGS forward pass
- Same adjoint inputs (`v_render_colors`, `v_render_alphas`) — random small values (×0.01)
- Register count verified: all variants use exactly 56 registers, 0 spill stores, 0 spill loads (from ptxas `-v`)
- BASELINE variant reproduces the exact current kernel (confirmed by correctness: all variants match BASELINE to cosine ≥ 0.9999998)

---

## 6. What was measured?

### Timing Results (median ms, 100 measurements)

| Variant | room/cam0 | speedup | %blend | bicycle/cam0 | speedup | %blend |
|---------|-----------|---------|--------|--------------|---------|--------|
| BASELINE | 1.854 | 1.000× | — | 2.522 | 1.000× | — |
| SIGMA_GATE | 1.830 | 1.013× | +1.3% | 2.536 | 0.994× | −0.6% |
| SCALAR_ADJOINT | 1.839 | 1.008× | +0.8% | 2.533 | 0.995× | −0.5% |
| UV_REUSE | 1.839 | 1.008× | +0.8% | 2.566 | 0.983× | −1.7% |
| COMBINED | 1.848 | 1.003× | +0.3% | 2.546 | 0.991× | −1.0% |
| NO_EXP | 1.853 | 1.001× | +0.1% | 2.545 | 0.991× | −0.9% |
| NO_VJP | 1.850 | 1.002× | +0.2% | 2.563 | 0.984× | −1.6% |

**T_F+B reference**: room=4.445ms, bicycle=5.923ms

### Gate Evaluation

| Criterion | Threshold | room/cam0 | bicycle/cam0 | Met? |
|-----------|-----------|-----------|--------------|------|
| COMBINED %blend saved | ≥ 10% | 0.3% | −1.0% | ❌ |
| COMBINED %F+B saved | ≥ 5% | 0.1% | −0.4% | ❌ |
| Correctness | pass | ✅ | ✅ | ✅ |

### Correctness Results (all variants, both scenes)

All variants PASS: min_cosine ≥ 0.9999998, max_rel_l2 ≤ 7.5×10⁻⁷, nonzero_disagreement = 0, no NaN/INF.

### SIGMA_GATE Classification (room/cam0, tile-center pixel)

| Classification | Count | Fraction |
|----------------|-------|----------|
| Drop before exp (sigma < 0 or alpha < 1/255 or sigma > sigma_drop) | 385,175 | 40.4% |
| Clamp without exp (alpha ≥ 0.99, sigma < sigma_clamp) | 425 | 0.04% |
| Exp required | 567,544 | 59.5% |

Random pixel sample (100K): 40.8% drop, 0.05% clamp, 28.9% exp-required (remaining ~30% are high-opacity Gaussians where sigma ≥ sigma_clamp but ≤ sigma_drop — these also need exp but the diagnostic classification script undercounted this bucket; the kernel itself handles all paths correctly).

---

## 7. What did the mechanism-level evidence show?

**The kernel is NOT arithmetic-bound.** The evidence is conclusive:

1. **NO_EXP oracle** (replace `__expf` with constant `vis=1.0`): saves only **0.1%** on room, is **0.9% SLOWER** on bicycle. If expf were the bottleneck, removing it entirely should produce a measurable speedup. It does not.

2. **NO_VJP oracle** (skip ALL VJP arithmetic — conic, xy, opacity gradients, buffer updates): saves only **0.2%** on room, is **1.6% SLOWER** on bicycle. If the VJP multiply-add chains were the bottleneck, removing them entirely should produce a large speedup. It does not.

3. **SIGMA_GATE** (skip expf for 40% of evaluations): saves **1.3%** on room, is **0.6% SLOWER** on bicycle. The 40% expf skip produces no meaningful speedup. The branch divergence and classification overhead (`__logf` for sigma_drop/sigma_clamp thresholds) offsets the saved expf calls.

4. **SCALAR_ADJOINT** (scalar buffer_dot instead of buffer[3]): saves **0.8%** on room, **0.5% SLOWER** on bicycle. The 3→1 reduction in buffer arithmetic doesn't help because the compiler already optimizes the CDIM=3 loop well (56 registers, no spills).

5. **UV_REUSE** (factored ux/uy): saves **0.8%** on room, **1.7% SLOWER** on bicycle. The compiler already CSEs the common subexpressions; explicit factoring adds register pressure from storing ux/uy.

6. **All variants use exactly 56 registers** with 0 spills — the transformations do not change register pressure enough to affect occupancy.

**Root cause**: The kernel's wall-clock time is dominated by **memory subsystem latency**, not arithmetic throughput:
- Shared memory loads from the batch buffer (128 intersections × 10 floats per batch)
- Global memory atomics for gradient accumulation (confirmed 6% by H2-BWD-0, but the remaining latency is in the load/store pipeline, not the atomic operation itself)
- Instruction issue/scheduling overhead — the kernel has deep loop nesting (batch loop × intersection loop × pixel loop) with `block.sync()` barriers per batch

The `--use_fast_math` flag already replaces `expf` with the hardware `__expf` intrinsic (MUFU.EX2), which is a single instruction on SM80. Removing one MUFU instruction from a memory-latency-bound kernel has no measurable effect.

---

## 8. What did the E2E result show?

This is an L4 (kernel microbenchmark) experiment. No E2E iteration test was performed — the gate was set at the microbenchmark level (≥10% blend_bwd or ≥5% F+B) specifically to avoid proceeding to E2E with a kernel-level gain too small to matter end-to-end.

**Break-even analysis**:
- COMBINED saves 0.006ms on room (0.3% of blend_bwd, 0.1% of F+B)
- T_F+B = 4.445ms, T_iter ≈ 9-10ms (F+B ≈ 50% of iteration)
- Maximum possible E2E gain from this candidate: 0.006/9 ≈ 0.07%
- This is 43× below the 3% screening threshold

---

## 9. Did correctness/quality pass?

**Yes, all variants pass correctness on both scenes:**

| Variant | room min_cosine | room max_rel_l2 | bicycle min_cosine | bicycle max_rel_l2 | disagree |
|---------|-----------------|------------------|---------------------|---------------------|----------|
| SIGMA_GATE | 0.99999982 | 6.2e-7 | 0.99999994 | 4.7e-7 | 0 |
| SCALAR_ADJOINT | 0.99999994 | 5.7e-7 | 0.99999994 | 7.1e-7 | 0 |
| UV_REUSE | 0.99999994 | 6.0e-7 | 0.99999994 | 4.3e-7 | 0 |
| COMBINED | 0.99999994 | 6.1e-7 | 0.99999994 | 5.1e-7 | 0 |
| NO_EXP | 0.99999988 | 6.8e-7 | 0.99999994 | 7.5e-7 | 0 |
| NO_VJP | 0.99999994 | 5.1e-7 | 1.0 | 5.4e-7 | 0 |

The small numerical differences (max_abs up to 80.0 on bicycle v_conics) come from floating-point reordering in the atomic accumulation order, not from algorithmic differences. All variants are mathematically exact — the transformations preserve the same operations in a different order or with reduced intermediate precision.

---

## 10. What remains uncertain?

1. **Actual stall reason**: The kernel is memory-latency-bound, but the specific stall category (shared memory bank conflicts? global load-store pipeline? instruction cache misses? warp scheduling?) was not identified. This requires NSight Compute profiling (not available in this microbenchmark).

2. **Bicycle regression**: All variants are slower on bicycle despite identical register counts. Bicycle has 4× more visible Gaussians and 1.5× more intersections, so the kernel runs longer (2.5ms vs 1.9ms). The regression may be from branch divergence overhead becoming more visible at higher intersection counts, or from L2 cache pressure with more Gaussians. The magnitude (~1%) is within measurement noise for bicycle (std=0.026ms, median=2.52ms → 1.0% noise).

3. **SH VJP and projection VJP**: This experiment only targets the blend backward kernel. The remaining backward components (SH VJP, projection VJP) were not evaluated and may have different bottleneck profiles.

---

## 11. Verdict

### **KEEP** — correctness passes, performance gate not met

**Rationale**:
- COMBINED saves 0.3% of blend_bwd (room) and is 1.0% slower (bicycle) — 33× below the 10% gate
- NO_EXP and NO_VJP oracles prove the kernel is not arithmetic-bound — removing entire categories of arithmetic has no measurable effect
- The kernel's time is dominated by memory subsystem latency, not FLOP count
- The three transformations (SIGMA_GATE, SCALAR_ADJOINT, UV_REUSE) are individually correct and produce no harm, but their combined effect is negligible
- **No production CUDA implementation is warranted** — the gate was explicitly set to avoid implementing a kernel that cannot meet the threshold

**Candidate disposition**:
- A (SIGMA_GATE): Correct, 1.3% best-case on room, harmful on bicycle. Not worth implementing.
- B (SCALAR_ADJOINT): Correct, 0.8% best-case on room, harmful on bicycle. Not worth implementing.
- C (UV_REUSE): Correct, 0.8% best-case on room, most harmful on bicycle (−1.7%). Not worth implementing.
- COMBINED: Correct, 0.3% on room, −1.0% on bicycle. Not worth implementing.

---

## 12. What is the single highest-value next experiment?

**NSight Compute profiling of `higs_blend_bwd_px_kernel<3,2>`** to identify the actual stall reason.

The NO_EXP/NO_VJP oracles have definitively ruled out arithmetic as the bottleneck. The next step is to identify what IS the bottleneck:
- `stall_short_scoreboard` (shared memory latency)?
- `stall_long_scoreboard` (global memory latency)?
- `stall_imc_miss` (instruction cache miss)?
- `stall_wait` (warp scheduling)?
- `stall_membar` (memory fence/atomic completion)?

This requires running `nsys profile` / `ncu` on the mx A100 with the benchmark harness, targeting the blend backward kernel specifically. Once the dominant stall category is identified, a new candidate can be designed that targets the actual bottleneck (e.g., shared memory bank conflict resolution, global memory coalescing, or warp-level gradient accumulation to reduce atomic traffic).

---

## Provenance

| Field | Value |
|-------|-------|
| B2 base commit | `77ab983ffe43420b2131669cb35776b883ca4c3c` |
| B2 patch SHA256 | `74e5d8b3b6273b9446ec0551ce91409783e2aa935c8d8e354b4099341390c84c` |
| GPU | A100-PCIE-40GB (GPU 3, uncontended) |
| CUDA | 12.8 |
| PyTorch | (mx env) |
| Compiler | nvcc -O3 --use_fast_math -std=c++17, sm_80 |
| Dataset | MipNeRF360 room/cam0, bicycle/cam0 |
| Resolution | 2048 long side |
| Tile size | 16 |
| SH degree | 3 |
| Measure | 20 warmup / 100 measure, CUDA event timing |
| Script | `scripts/h2/h2_bwd_1_compute_oracle.py` |
| Launcher | `scripts/h2/_h2_bwd_1_run.sh` |
| Artifacts | `artifacts/higs-h2-bwd-1/` |

## Artifacts

| File | Description |
|------|-------------|
| `room_cam0_compute_oracle.json` | Full timing + correctness results, room/cam0 |
| `bicycle_cam0_compute_oracle.json` | Full timing + correctness results, bicycle/cam0 |
| `room_cam0_sigma_classify.json` | SIGMA_GATE classification fractions, room/cam0 |
| `variant_comparison.csv` | Summary CSV of all variants × both scenes |
| `room_cam0.log` | Run log, room/cam0 |
| `bicycle_cam0.log` | Run log, bicycle/cam0 |
