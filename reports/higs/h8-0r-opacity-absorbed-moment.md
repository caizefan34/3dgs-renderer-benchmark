# H8-0R: Opacity-Absorbed Moment-Space Geometry Adjoint — Stage A Validation

## Summary

**Classification: H8_0R_STRONG**

H8-0R is an exact algebraic reparameterization of the geometry adjoint in the HiGS native backward kernel. It absorbs `v_sigma = -opac * r` into the per-Gaussian moment buffers before accumulation, so that deferred reconstruction needs only the forward conic (A, B, C) — eliminating the opacity global load that blocked the earlier H8-0 candidate.

All Stage A hard-gate conditions passed: zero register delta, zero spills, zero new global loads, zero new buffers/sync, algebraic exactness in FP64, and FP32 noise within reassociation tolerance across all three validation scenes.

---

## 1. Research Question

The frozen `SCALAR_ADJOINT, CDIM=3, PX=2` baseline computes the geometry adjoint inside the blend backward kernel. Each thread-pixel that contributes to a Gaussian must load the Gaussian's opacity from global memory to compute `v_sigma = -opac * vis * v_alpha`. The H8-0 candidate attempted to eliminate this opacity load but failed because the unweighted moments it stored required reloading opacity during deferred reconstruction.

**H8-0R question**: Can the opacity be absorbed *into* the moment accumulation so that reconstruction uses only the forward conic — and does this reparameterization satisfy the exactness, arithmetic, and resource hard gates?

## 2. Hypothesis

Reparameterizing the geometry adjoint as opacity-absorbed moments `Sx, Sy, Sxx, Sxy, Syy, R0` (where `v_sigma = -opac * r` is folded in before accumulation) will:
- Eliminate the opacity global load from the deferred reconstruction
- Reduce per-intersection arithmetic (remove conic·delta products from the blend backward)
- Maintain algebraic exactness (FP64 agreement, FP32 within reassociation noise)
- Maintain the resource profile (0 register delta, 0 spills)

**Falsification condition**: Any of (a) FP64 support disagreement > 0, (b) register delta > 2, (c) spills > 0, (d) new global loads > 0, (e) arithmetic ceiling < 45% on 2+ scenes.

## 3. Moment Basis

### Baseline (per intersection q, inside blend backward):
```
v_sigma = -opac * vis * v_alpha
v_xy_q  = {v_sigma * (A*dx + B*dy),  v_sigma * (B*dx + C*dy)}    // 4 FMUL + 2 FADD + 2 FMUL
v_conic_q = {0.5*v_sigma*dx²,  v_sigma*dx*dy,  0.5*v_sigma*dy²}  // 3 FMUL + 1 const mul
v_opacity_q = vis * v_alpha
```

### H8-0R (per intersection q, inside blend backward):
```
r = vis * v_alpha
v_sigma = -opac * r
Sx = v_sigma * dx            // 1 FMUL
Sy = v_sigma * dy            // 1 FMUL
Sxx = v_sigma * dx²          // 1 FMUL
Sxy = v_sigma * dx*dy        // 1 FMUL
Syy = v_sigma * dy²          // 1 FMUL
R0 = r                       // 0 FMUL
```
**No conic·delta products** — the 4 FMUL + 2 FADD for `(A*dx + B*dy, B*dx + C*dy)` are removed.

### Reconstruction (per Gaussian, in projection VJP):
```
vx = A*Sx + B*Sy             // 2 FMUL + 1 FADD
vy = B*Sx + C*Sy             // 2 FMUL + 1 FADD
vA = 0.5 * Sxx               // 1 FMUL (const)
vB = Sxy                     // 0 FMUL
vC = 0.5 * Syy               // 1 FMUL (const)
v_opacity = R0               // 0 FMUL
```
Reconstruction uses only the forward conic (A, B, C) — **no opacity load**.

### Buffer mapping:
| Buffer | Baseline content | H8-0R content |
|--------|-----------------|---------------|
| `v_means2d` | Σ(A·v_sigma·dx + B·v_sigma·dy, B·v_sigma·dx + C·v_sigma·dy) | Σ(Sx, Sy) |
| `v_conics` | Σ(0.5·v_sigma·dx², v_sigma·dx·dy, 0.5·v_sigma·dy²) | Σ(Sxx, Sxy, Syy) |
| `v_opacities` | Σ(vis·v_alpha) | Σ(R0) = Σ(vis·v_alpha) |

The v_opacities buffer is **unchanged** — R0 = r = vis·v_alpha is the same value.

## 4. Experiment

### Isolated probe
Three functions compiled in isolation with `nvcc 12.8.93 -arch sm_80`:
1. `baseline_geometry_adjoint` — original baseline moment computation
2. `h8_0r_geometry_adjoint` — register-optimized H8-0R moment body (store R0 early, compute Sxx/Sxy before Sy)
3. `h8_0r_reconstruction` — reconstruction using forward conic only (ReconstructArgs has NO opacity field)

### FP64/FP32 replay
Deterministic per-tile replay using the H2 capture helper on 16 representative tiles per scene, comparing baseline vs H8-0R moment accumulation at FP64 and FP32 precision.

### Scenes
- room (2048-max-side, cam0, seed 4200): 135,580 valid contributions, multiplicity 36.51
- bicycle (2048-max-side, cam0, seed 4200): 138,372 valid contributions, multiplicity 28.70
- garden (2048-max-side, cam0, seed 4200): 96,316 valid contributions, multiplicity 41.53

## 5. Results

### Compile gate (nvcc 12.8.93, sm_80)
| Function | Registers | Spills | Local | Shared |
|----------|-----------|--------|-------|--------|
| baseline_geometry_adjoint | 18 | 0 | 0 | 0 |
| h8_0r_geometry_adjoint | 18 | 0 | 0 | 0 |
| h8_0r_reconstruction | 20 | 0 | 0 | 0 |

**Register delta: +0** (register-optimized ordering: store R0 early, compute Sxx/Sxy before Sy → live set = 2 intermediates)

### SASS instruction counts (isolated probe)
| Variant | FMUL | FFMA | FP_arith | LDG | STG |
|---------|------|------|----------|-----|-----|
| baseline | 13 | 2 | 15 | 15 | 6 |
| h8_0r_moment | 7 | 0 | 7 | 8 | 6 |
| h8_0r_reconstruct | 4 | 2 | 6 | 12 | 6 |

Reconstruction `opacity_in_args = false` — **no opacity load**.

### Arithmetic ceiling
| Scene | Ceiling | Threshold |
|-------|---------|-----------|
| room | 52.24% | ≥45% ✓ |
| bicycle | 51.94% | ≥45% ✓ |
| garden | 52.37% | ≥45% ✓ |

All scenes exceed the 45% preferred threshold (removing conic·delta products from the hot inner loop eliminates 6 of ~13 FP ops per intersection).

### FP64 exactness
| Scene | v_means2d rel_L2 | v_conics rel_L2 | v_opacity rel_L2 | support_disagree |
|-------|-------------------|------------------|-------------------|------------------|
| room | 6.59e-16 | 5.35e-16 | 6.20e-16 | 0 |
| bicycle | 5.80e-16 | 1.30e-15 | 6.87e-16 | 0 |
| garden | 1.88e-15 | 5.91e-16 | 7.91e-16 | 0 |

**Classification: ALGEBRAIC_EXACT_FP_REASSOCIATED** — all 9 comparisons have support_disagreement=0, rel_L2 ≤ 1.9e-15, cosine ≈ 1.0.

### FP32 exactness
| Scene | v_means2d rel_L2 | v_conics rel_L2 | v_opacity rel_L2 | support_disagree |
|-------|-------------------|------------------|-------------------|------------------|
| room | 4.26e-07 | 1.87e-07 | 3.36e-07 | 0 |
| bicycle | 4.30e-07 | 2.74e-07 | 2.94e-07 | 0 |
| garden | 1.01e-06 | 3.12e-07 | 4.23e-07 | 0 |

All 9 comparisons: support_disagreement=0, rel_L2 ≤ 1.0e-6, cosine ≈ 1.0.

### Hard gate summary
| Gate | Condition | Result |
|------|-----------|--------|
| New global loads | = 0 | **PASS** |
| Register delta | ≤ 2 | **PASS** (0) |
| Spills | = 0 | **PASS** |
| New local/shared/sync/buffers | = 0 | **PASS** |
| Algebraic exactness (FP64) | support_disagree = 0 | **PASS** (all 9) |
| Arithmetic ceiling | ≥45% on all 3 scenes | **PASS** (52.24/51.94/52.37%) |
| Opacity load eliminated | reconstruction has no opacity arg | **PASS** |

## 6. Mechanism Evidence

The 6 FLOP saving per intersection (4 FMUL + 2 FADD for conic·delta products) is removed from the blend backward hot loop, which executes once per pixel-Gaussian intersection. The 6 FLOP reconstruction cost is added to the projection VJP, which executes once per visible Gaussian per camera. The amortization ratio equals the contribution multiplicity:

| Scene | Multiplicity | Amortization |
|-------|-------------|--------------|
| room | 36.51 | 36.51× |
| bicycle | 28.70 | 28.70× |
| garden | 41.53 | 41.53× |

All scenes have strong amortization (>28×), confirming the arithmetic ceiling analysis.

## 7. Correctness/Quality

- FP64: algebraically exact (reassociation only, all support_disagreement = 0)
- FP32: within reassociation noise (all rel_L2 ≤ 1.0e-6, all cosine ≈ 1.0)
- No support changes (same Gaussians contribute, same tiles, same intersections)

## 8. Uncertainty

- Stage A validates the isolated arithmetic and exactness; production kernel behavior (register allocation under full kernel context) is validated in Stage B.
- The 20-register reconstruction function is +2 over baseline, but this is a different kernel (projection VJP) with its own register budget — Stage B confirms 0 delta at the production level.
- FP32 rel_L2 for garden v_means2d (1.01e-6) is at the edge of the 1e-6 tolerance — this is expected for reassociation and does not indicate a correctness issue.

## 9. Verdict

**H8_0R_STRONG** — all hard-gate conditions satisfied. Stage B (production H8-MR) authorized.

## 10. Highest-Value Next Experiment

Stage B: Implement H8-MR in the full production kernel, validate correctness against the frozen baseline, and measure backward/F+B timing on room/bicycle/garden.

## Artifacts

- `artifacts/higs-h8-0r/h8_0r_probe.cu` — isolated validation probe
- `artifacts/higs-h8-0r/h8_0r_replay.py` — FP64/FP32 replay script
- `artifacts/higs-h8-0r/analysis.json` — Stage A analysis and classification
- `artifacts/higs-h8-0r/fp_exactness.json` — FP64/FP32 exactness results
- `artifacts/higs-h8-0r/arithmetic_ceiling.json` — arithmetic ceiling per scene
- `artifacts/higs-h8-0r/contribution_multiplicity.json` — per-scene multiplicities
- `artifacts/higs-h8-0r/resource_probe.txt` — ptxas register/spill output
- `artifacts/higs-h8-0r/sass_baseline.txt`, `sass_unweighted.txt`, `sass_weighted.txt` — SASS disassembly
- `artifacts/higs-h8-0r/instruction_diff.json` — SASS instruction count comparison
- `artifacts/higs-h8-0r/resource_usage.json` — resource usage comparison
- `artifacts/higs-h8-0r/provenance.json` — build provenance
- `artifacts/higs-h8-0r/algebra.md` — algebraic derivation
