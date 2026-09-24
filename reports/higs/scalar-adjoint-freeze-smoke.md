# SCALAR_ADJOINT Freeze Smoke Report

## Summary

**Decision: FREEZE_SCALAR_ADJOINT**

The cleaned standalone patch `higs-scalar-adjoint.patch` (SHA256 `602a86eee98b5fd509019deb5222dae99ae1127fe9916a5f15765082f563cf22`, 18067 bytes) was deployed to mx, applied to a clean B2 source tree, built, and smoke-tested for resource identity and production timing. The cleaned patch preserves the resource identity (67→56 registers, 0/0 spills, 5120B dynamic shared memory) and achieves production timing gains exceeding the 5% backward gate on both room and bicycle scenes.

## Patch Description

The cleaned patch is a **standalone** version of the scalar_adjoint transformation validated as EXACT_VALIDATED in H2-BWD-2R. It modifies `HigsNativeBackward.cu` only:

- Adds `template<bool SCALAR_ADJOINT = false>` to `higs_blend_bwd_px_kernel`
- Replaces the vector `buffer[CDIM]` accumulator with a scalar `buffer_dot` accumulator when `SCALAR_ADJOINT=true`
- Pre-computes `tail_const = T_final * (v_render_a - bg_dot)` to eliminate the per-Gaussian background subtraction
- Dispatches via `HIGS_BWD_SCALAR_ADJOINT` environment variable (`baseline` or `scalar_adjoint`)
- Uses `HIGS_LAUNCH_BLEND_BWD_PX` macro to reduce code duplication

Unlike the H2-BWD-CF version, this patch does NOT require the CF framework — it works directly on the original `higs_blend_bwd_px_kernel`.

## A1: Resource Verification

**Method**: Built the patched source on mx with `NVCC_FLAGS="-Xptxas=-v"` using the conda env's nvcc 12.8 (CUDA 12.8, V12.8.93, SM80 target). The patched source was prepared by copying the clean B2 source tree (from `/tmp/higs_h3_fwd_1a_source`) and applying the cleaned patch via `patch -p1`. The build was done via `ninja -j1` directly in the build directory to capture ptxas verbose output (the `jit.load` function captures subprocess stderr internally, so a direct ninja invocation was needed).

**Expected** (from H2-BWD-2R EXACT_VALIDATED):

| Variant | Registers | Spill stores | Spill loads | Dynamic shared |
|---------|----------:|-------------:|------------:|---------------:|
| baseline (SCALAR_ADJOINT=false) | 67 | 0 | 0 | 5120 B |
| scalar_adjoint (SCALAR_ADJOINT=true) | 56 | 0 | 0 | 5120 B |

**Actual** (from ptxas -v, CDIM=3, PX=2):

| Variant | Mangled name snippet | Registers | Spill stores | Spill loads | cmem[0] |
|---------|----------------------|----------:|-------------:|------------:|--------:|
| baseline (Lb0E) | `higs_blend_bwd_px_kernelILj3ELj2ELb0E` | **67** | **0** | **0** | 528 B |
| scalar_adjoint (Lb1E) | `higs_blend_bwd_px_kernelILj3ELj2ELb1E` | **56** | **0** | **0** | 528 B |

**Result: PASS** ✅ — Resource identity exactly matches expected values.

**Register reduction**: 67 → 56 = -11 registers per thread (-16.4%)
**Occupancy gain** (SM80, 128 threads/block, 5120B shared):
- Baseline: 67 regs → 7 blocks/SM, 87.5% occupancy
- Scalar_adjoint: 56 regs → 9 blocks/SM, 112.5% → capped at 100% occupancy
- Dynamic shared (5120B) is not the binding constraint at either register count

**Full CDIM×PX table** (all 0/0 spills):

| CDIM | PX | Baseline regs | Scalar regs | Delta |
|-----:|---:|-------------:|------------:|------:|
| 4 | 4 | 105 | 78 | -27 |
| 4 | 2 | 72 | 56 | -16 |
| 4 | 1 | 40 | 48 | +8 |
| 3 | 4 | 91 | 77 | -14 |
| **3** | **2** | **67** | **56** | **-11** |
| 3 | 1 | 40 | 48 | +8 |
| 1 | 4 | 76 | 67 | -9 |
| 1 | 2 | 54 | 48 | -6 |
| 1 | 1 | 40 | 40 | 0 |

## A2: Production Smoke Timing

**Protocol**: 20 warmup / 100 measurements / 5 repetitions / interleaved / CUDA Events / GPU 4 (uncontended A100-PCIE-40GB) / `HIGS_PX_RUNTIME=2` / `HIGS_BWD_SCALAR_ADJOINT` env var

### Results

| Scene | Metric | Baseline (ms) | Scalar (ms) | Speedup | CI95 low (ms) |
|-------|--------|-------------:|------------:|--------:|--------------:|
| room | T_backward | 2.171 | 1.952 | **10.05%** | 0.217 |
| room | T_F+B | 4.269 | 4.050 | **5.12%** | 0.216 |
| bicycle | T_backward | 3.122 | 2.881 | **7.74%** | 0.239 |
| bicycle | T_F+B | 5.410 | 5.166 | **4.52%** | 0.242 |

### Pass Condition Evaluation

**Pass condition**: backward gain ≥5% on both scenes, or equivalent consistent F+B gain.

- Room backward: 10.05% ≥ 5% ✓
- Bicycle backward: 7.74% ≥ 5% ✓

All bootstrap CI95 lower bounds are positive (>>0), confirming statistical significance. The timing results are consistent across two independent runs (run 1: room 10.03%, bicycle 7.83%; run 2: room 10.05%, bicycle 7.74%).

### Reference comparison

| Scene | Metric | H2-BWD-2R reference | Freeze smoke | Consistent? |
|-------|--------|--------------------:|-------------:|------------:|
| room | T_backward | ~9.6% | 10.05% | ✓ |
| room | T_F+B | ~5.0% | 5.12% | ✓ |
| bicycle | T_backward | ~7.6% | 7.74% | ✓ |
| bicycle | T_F+B | ~4.4% | 4.52% | ✓ |

## A3: Freeze Decision

**FREEZE_SCALAR_ADJOINT**

The cleaned standalone patch:
1. ✅ Preserves resource identity (67→56 regs, 0/0 spills, 5120B shared)
2. ✅ Achieves backward timing gains ≥5% on both scenes (room 10.05%, bicycle 7.74%)
3. ✅ Achieves consistent F+B timing gains (room 5.12%, bicycle 4.52%)
4. ✅ All bootstrap CI95 lower bounds positive (statistically significant)
5. ✅ Results consistent across two independent runs
6. ✅ Results consistent with H2-BWD-2R reference gains
7. ✅ Exactness already validated in H2-BWD-2R (EXACT_VALIDATED, per-sample VJP max abs diff = 4.44e-16)

No regression detected. The cleaned patch is safe to freeze.

## What was NOT done

- No new exactness closure was performed (already validated in H2-BWD-2R)
- No new optimization phase was opened
- No algorithm redesign was attempted
- No other backward variant was tested

## Artifacts

All artifacts in `artifacts/higs-scalar-adjoint-freeze/`:
- `resources.json` — ptxas resource verification
- `production_timing.csv` — interleaved timing results (500 samples per cell)
- `analysis.json` — freeze decision analysis
