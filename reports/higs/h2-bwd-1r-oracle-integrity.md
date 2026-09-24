# H2-BWD-1R: Oracle Integrity Repair

## Root Cause

The original H2-BWD-1 used `#if VARIANT == ...` preprocessor directives to select between kernel variants. **This is a C++ template/preprocessor gotcha**: `VARIANT` is a template parameter, not a preprocessor macro. The preprocessor evaluates undefined identifiers as 0, so `#if VARIANT == VAR_BASELINE` evaluates as `#if 0 == 0` → **true for ALL variants**. Every `#elif` branch (SIGMA_GATE, UV_REUSE, SCALAR_ADJOINT, NO_EXP, NO_VJP) was **never compiled**. All 7 variants produced identical machine code.

**Evidence of the bug** (from original H2-BWD-1):
- All 7 variants used exactly 56 registers, 0 spills → identical code
- NO_EXP (replace expf with constant) produced gradients identical to BASELINE (cosine=1.0) → impossible if expf were actually removed
- NO_VJP (skip VJP arithmetic) produced gradients identical to BASELINE → impossible if VJP were actually skipped
- SIGMA_GATE counters reported 0 → counter code was never compiled

**Fix**: Replaced all `#if VARIANT == ...` with `if constexpr (VARIANT == ...)` (C++17), which evaluates at template instantiation time, not preprocessing time.

---

## Integrity Verification Results

### 1. Were variants actually distinct? **YES**

| Variant | Template Int | Expected Sig | Actual Sig | Registers | Match? |
|---------|-------------|-------------|-----------|-----------|--------|
| BASELINE | 0 | 1001 | 1001 | 56 | ✓ |
| SIGMA_GATE | 1 | 1002 | 1002 | 56 | ✓ |
| SCALAR_ADJOINT | 2 | 1003 | 1003 | 48 | ✓ |
| UV_REUSE | 3 | 1004 | 1004 | 56 | ✓ |
| COMBINED | 4 | 1005 | 1005 | 48 | ✓ |
| NO_EXP | 5 | 1006 | 1006 | 56 | ✓ |
| NO_VJP | 6 | 1007 | 1007 | 32 | ✓ |
| BROKEN_VIS | 7 | 1008 | 1008 | 56 | ✓ |
| ZERO_VJP | 8 | 1009 | 1009 | 32 | ✓ |

Register counts now **differ** across variants (56, 48, 32), proving the compiler generates different code per template instantiation. All 9 kernel symbols are distinct in the binary.

### 2. Did BROKEN_VIS fail correctness? **YES**

BROKEN_VIS forces `vis=1.0, alpha=min(MAX_ALPHA, opac)` for all valid samples. This produces NaN/inf in gradient outputs (transmittance diverges without exponential decay).

| Metric | rel_l2 | cosine | var_norm |
|--------|--------|--------|----------|
| v_colors | inf | NaN | inf |
| v_conics | NaN | 0.0 | NaN |
| v_means2d | NaN | 0.0 | NaN |
| v_opacities | NaN | 0.0 | NaN |

### 3. Did ZERO_VJP fail correctness? **YES**

ZERO_VJP retains v_rgb computation but zeroes v_conic, v_xy, v_opacity. Color gradients match baseline; geometry gradients are exactly zero.

| Metric | rel_l2 | cosine | var_norm | Zero? |
|--------|--------|--------|----------|-------|
| v_colors | 5.9e-7 | 1.0 | 521.7 | No (matches) |
| v_conics | 1.0 | 0.0 | 0.0 | **Yes** |
| v_means2d | 1.0 | 0.0 | 0.0 | **Yes** |
| v_opacities | 1.0 | 0.0 | 0.0 | **Yes** |

### 4. Were output buffers independent? **YES**

All 9 variants use freshly allocated output tensors. No pointer reuse detected. The `buffer_identity.json` artifact records all pointer values.

### 5. Did CUDA sigma counters work? **YES**

| Classification | Count | Fraction |
|----------------|-------|----------|
| N_classified (warp-level) | 5,570,055 | — |
| N_drop | 2,437,652 | 43.8% |
| N_clamp | 977 | 0.02% |
| N_exp | 3,131,426 | 56.2% |
| **Sum check** (drop+clamp+exp=total) | | **PASS** |

The 43.8% drop fraction approximately agrees with the offline analysis (40.4% at tile-center pixel). The difference is expected: the CUDA kernel counts per-warp-per-pixel-slot evaluations across all pixel positions, while the offline analysis samples only the tile-center pixel.

### 6. Was MUFU.EX2 removed in NO_EXP? **YES**

SASS instruction counts from `cuobjdump`:

| Variant | Total | MUFU | FFMA | FADD | FMUL | Arith Total | LDG | RED | BAR |
|---------|-------|------|------|------|------|-------------|-----|-----|-----|
| BASELINE | 912 | **4** | 46 | 56 | 36 | **138** | 24 | 10 | 2 |
| NO_EXP | 896 | **2** | 42 | 60 | 32 | 134 | 24 | 10 | 2 |
| NO_VJP | 464 | **2** | 4 | 7 | 14 | **25** | 14 | 10 | 2 |
| SIGMA_GATE | 1408 | 10 | 46 | 56 | 50 | 152 | 34 | 30 | 2 |
| SCALAR_ADJOINT | 880 | 4 | 34 | 60 | 32 | 126 | 24 | 10 | 2 |
| UV_REUSE | 912 | 4 | 44 | 56 | 34 | 134 | 24 | 10 | 2 |
| COMBINED | 960 | 10 | 34 | 60 | 46 | 140 | 24 | 10 | 2 |
| ZERO_VJP | 624 | 4 | 10 | 26 | 18 | 54 | 22 | 10 | 2 |
| BROKEN_VIS | 896 | 2 | 42 | 60 | 32 | 134 | 24 | 10 | 2 |

- **NO_EXP**: 2 MUFU vs BASELINE 4 → **MUFU reduced by 50%** ✓
- **NO_VJP**: 25 arithmetic vs BASELINE 138 → **arithmetic reduced by 82%** ✓

### 7. Was VJP arithmetic materially removed in NO_VJP? **YES**

NO_VJP SASS: 464 total instructions (vs BASELINE 912), 25 FFMA+FADD+FMUL (vs 138), 14 LDG (vs 24). The VJP computation is substantially removed while preserving traversal, shared memory loads, warp reduction, and gradient scatter structure.

---

## Corrected Timing Results

### Protocol
- Interleaved randomized-block design: 5 reps × 100 measurements
- Within each measurement, all 7 variants run in random order (seed=42)
- Paired delta = baseline_time - variant_time (positive = variant faster)
- CUDA event timing on uncontended A100-PCIE-40GB (GPU 3)

### Results (median ms)

| Variant | Median | Mean | Std | Paired Δ | % of baseline | p90 |
|---------|--------|------|-----|----------|---------------|-----|
| BASELINE | 1.794 | 1.823 | 0.201 | — | — | — |
| **NO_VJP** | **0.721** | 0.733 | 0.081 | **+1.072** | **+59.8%** | +1.096 |
| SCALAR_ADJOINT | 1.713 | 1.742 | 0.191 | +0.081 | +4.5% | +0.108 |
| UV_REUSE | 1.779 | 1.811 | 0.208 | +0.016 | +0.9% | +0.037 |
| COMBINED | 1.790 | 1.820 | 0.199 | +0.004 | +0.2% | +0.027 |
| SIGMA_GATE | 1.969 | 2.002 | 0.220 | −0.176 | −9.8% | −0.153 |
| NO_EXP | 1.976 | 2.009 | 0.224 | −0.183 | −10.2% | −0.158 |

T_F+B reference: 4.445ms

### Correctness

| Variant | Type | min_cosine | max_rel_l2 | Pass? |
|---------|------|-----------|-----------|-------|
| SIGMA_GATE | Exact | 1.00000000 | 9.6e-7 | ✓ PASS |
| SCALAR_ADJOINT | Exact | 1.00000000 | 4.1e-7 | ✓ PASS |
| UV_REUSE | Exact | 1.00000000 | 1.1e-5 | ✓ PASS |
| COMBINED | Exact | 1.00000000 | 9.7e-7 | ✓ PASS |
| NO_EXP | Diagnostic | NaN | inf | ✗ FAIL (expected) |
| NO_VJP | Diagnostic | 0.0 | 1.0 | ✗ FAIL (expected) |
| BROKEN_VIS | Canary | NaN | inf | ✗ FAIL (expected) |
| ZERO_VJP | Canary | 0.0 | 1.0 | ✗ FAIL (expected) |

---

## Key Findings

### Finding 1: VJP arithmetic is the dominant kernel cost (59.8%)

**NO_VJP** (skip all VJP: v_alpha, v_sigma, v_conic, v_xy, v_opacity, buffer update) saves **1.072ms** — **59.8% of the baseline 1.794ms**. The SASS confirms: NO_VJP has only 25 arithmetic instructions vs BASELINE's 138 (82% reduction). The kernel is overwhelmingly compute-bound on VJP arithmetic, not memory-bound.

This **completely overturns** the original (invalid) H2-BWD-1 conclusion of "not arithmetic-bound". The original conclusion was based on comparing identical kernels.

### Finding 2: Removing expf makes the kernel SLOWER (−10.2%)

**NO_EXP** (vis=1.0 instead of exp(−σ)) is **0.183ms slower** than baseline. The SASS shows 2 fewer MUFU instructions, but the kernel is slower because:
- vis=1.0 → alpha = min(0.99, opac) is much higher than alpha = min(0.99, opac·exp(−σ))
- More Gaussians pass the ALPHA_THRESHOLD filter → more valid intersections
- Each newly-valid Gaussian requires full VJP computation
- The extra VJP work (the dominant cost per Finding 1) outweighs the saved MUFU instructions

**Insight**: `__expf` serves as a natural importance filter. By making distant/low-contribution Gaussians invisible (low alpha), it reduces the number of Gaussians entering the expensive VJP path. Removing this filter increases total work.

### Finding 3: SIGMA_GATE is SLOWER (−9.8%)

**SIGMA_GATE** (classify before expf, skip expf for 43.8% of evaluations) is **0.176ms slower**. The SASS shows 10 MUFU (vs 4 baseline) — the `__logf` classification thresholds and branch divergence add MORE instructions than they save. The 43.8% of evaluations that skip `__expf` would have been cheap (1 MUFU each), but the classification overhead (`__logf`, branches, atomicAdd counters) costs more than it saves.

### Finding 4: SCALAR_ADJOINT saves 4.5%

**SCALAR_ADJOINT** (scalar buffer_dot instead of buffer[CDIM]) saves **0.081ms (4.5%)**. The SASS shows 126 arithmetic instructions vs 138 (8.7% reduction) and 48 registers vs 56 (14% reduction). This is the only exact transformation that produces a measurable improvement.

### Finding 5: COMBINED is net-neutral (+0.2%)

**COMBINED** (SIGMA_GATE + SCALAR_ADJOINT + UV_REUSE) saves only 0.004ms. The SIGMA_GATE overhead (−9.8%) nearly cancels the SCALAR_ADJOINT benefit (+4.5%). UV_REUSE (+0.9%) is negligible.

---

## Classification

### **ARITHMETIC_MATERIAL**

**Evidence**:
- NO_VJP saves 59.8% (SASS-confirmed: 25 vs 138 arithmetic instructions) → VJP arithmetic is the dominant cost
- NO_EXP is 10.2% slower (SASS-confirmed: 2 vs 4 MUFU, but more valid Gaussians → more VJP work)
- Both oracles are SASS-verified, destructive canaries pass, dispatch verified, counters work

**Implication**: The `higs_blend_bwd_px_kernel<3,2>` is compute-bound on VJP arithmetic. The VJP computation (v_alpha loop over CDIM=3, v_sigma, v_conic×3, v_xy×2, v_opacity, buffer×3) accounts for ~60% of kernel time. Memory subsystem (shared memory loads, global atomics) accounts for the remaining ~40%.

**Important caveat**: This classification means arithmetic IS material, NOT that memory is irrelevant. The NO_VJP oracle removes BOTH arithmetic AND some memory traffic (buffer updates, 14 LDG vs 24). A separate NCU analysis is needed to attribute the 59.8% savings precisely between arithmetic and memory components. The user's instruction "Do NOT claim memory-bound merely because arithmetic is not dominant" is respected — the conclusion is that arithmetic IS dominant, which is the opposite claim.

---

## Comparison: Original (Invalid) vs Repaired

| Metric | H2-BWD-1 (Invalid) | H2-BWD-1R (Repaired) |
|--------|-------------------|---------------------|
| Root cause | `#if` on template param → all variants = BASELINE | `if constexpr` → variants truly distinct |
| Register counts | All 56 (identical) | 32-56 (varies by variant) |
| NO_EXP correctness | cos=1.0 (impossible) | cos=NaN, rel_l2=inf (correct failure) |
| NO_VJP correctness | cos=1.0 (impossible) | cos=0.0, rel_l2=1.0 (correct failure) |
| NO_EXP timing | 0.1% (same code) | −10.2% (slower, more VJP work) |
| NO_VJP timing | 0.2% (same code) | **+59.8%** (VJP is dominant) |
| SIGMA_GATE counters | 0 (never compiled) | 5.57M (43.8% drop, 56.2% exp) |
| Classification | KEEP (wrongly) | **ARITHMETIC_MATERIAL** |

---

## Known Limitations

1. **Bootstrap CI bug**: The 95% CI columns show [median, median] because `rng.sample()` (without replacement) was used instead of `rng.choices()` (with replacement). All bootstrap medians are identical. The median values themselves are correct; only the CI width is underestimated. For the NO_VJP delta of +1.072ms (59.8%), the conclusion is robust regardless of CI.

2. **Single scene**: Only room/cam0 was tested. Multi-scene validation (bicycle) is needed for confirmation.

3. **No NCU**: The user instructed not to run NCU. The 59.8% NO_VJP savings cannot be precisely attributed between arithmetic and memory components without NCU stall analysis.

4. **SIGMA_GATE counter overhead**: Fixed by adding `do_count` runtime flag. Timing without counters shows SIGMA_GATE is 9.8% slower (branch divergence + `__logf` overhead).

---

## Provenance

| Field | Value |
|-------|-------|
| run_id | 77cb0b1b-95e |
| timestamp | 20260920T142237 |
| GPU | NVIDIA A100-PCIE-40GB (GPU 3, uncontended) |
| cuda_source_sha256 | 4415dc8612e4533e... |
| binary_sha256 | ba505fbeeee4f50c... |
| fixture_hash | 9b42fc434518a278 |
| B2 base commit | 77ab983ffe43420b2131669cb35776b883ca4c3c |
| B2 patch SHA256 | 74e5d8b3b6273b9446ec0551ce91409783e2aa935c8d8e354b4099341390c84c |
| N_visible | 44,908 |
| n_isects | 953,144 |
| Timing | 5 reps × 100 measurements, interleaved randomized blocks, seed=42 |

## Artifacts

| File | Description |
|------|-------------|
| `variant_dispatch.json` | Variant → template int → signature mapping (all 9 verified) |
| `destructive_canary_correctness.json` | BROKEN_VIS + ZERO_VJP canary results (both fail as expected) |
| `buffer_identity.json` | Output tensor pointer verification (all independent) |
| `sigma_cuda_counts.json` | SIGMA_GATE CUDA counter results (non-zero, sum check passes) |
| `sass_instruction_counts.csv` | Per-variant SASS instruction counts from cuobjdump |
| `oracle_timings.json` | Full timing + correctness results |
| `paired_timing.csv` | Paired median delta, p50, p90, CI per variant |
| `provenance.json` | run_id, timestamps, hashes, GPU info |
| `generated_cuda.cu` | The corrected CUDA source with `if constexpr` |
| `room_cam0.log` | Full run log |
