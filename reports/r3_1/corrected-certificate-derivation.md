# R3.1 — Corrected Certificate Derivation

**Date:** 2026-09-17
**Scope:** R3.1-B (color factor repair) + §7 (opacity tail) + §8 (geometry)

---

## 1. Forward model

Alpha compositing for a single pixel:

```
C = Σ_i T_i α_i c_i         ( rendered color )
T_i = Π_{j<i} (1 - α_j)     ( transmittance )
α_i = α_raw_i (after sigmoid)    ( opacity )
c_i = SH(view_dir_i, shs_i)      ( SH-evaluated RGB, [3] )
```

The upstream gradient `q_p = ∂L/∂C[pixel]` enters the raster backward.

---

## 2. Opacity certificate — ∂C/∂α_i

### 2.1 Exact gradient

```
∂C/∂α_i = T_i · c_i  +  Σ_{j>i} (∂T_j/∂α_i) · α_j · c_j
```

Since `T_j = T_i · (1-α_i) · Π_{i<k<j}(1-α_k)`:

```
∂T_j/∂α_i = -T_i · Π_{i<k<j}(1-α_k) = -T_j / (1-α_i)
```

Therefore:

```
∂C/∂α_i = T_i · c_i  -  [ Σ_{j>i} T_j · α_j · c_j ] / (1-α_i)
```

This has **two terms**:
1. **Current contribution**: `T_i · c_i` (the direct effect of α_i on pixel i)
2. **Tail/downstream contribution**: `-[Σ_{j>i} T_j α_j c_j] / (1-α_i)` (the effect of α_i on all downstream Gaussians via transmittance)

### 2.2 Conservative upper bound

Taking norms:

```
|∂C/∂α_i| ≤ T_i · ||c_i||  +  [ Σ_{j>i} T_j · α_j · ||c_j|| ] / (1-α_i)
```

Upper-bounding `T_k ≤ 1`, `||c_j|| ≤ C_max_t`:

```
|∂C/∂α_i| ≤ ||c_i||  +  C_max_t · [ Σ_{j>i} α_j ] / (1-α_i)
           ≤ ||c_i||  +  C_max_t · N_downstream / (1-α_i)
```

where `N_downstream` = number of Gaussians behind i in this tile.

### 2.3 Code expression (repaired)

```python
# R3.1-B: c_norm = ||c_i||_2 from SH-evaluated colors_rgb
factor_op_v = (c_norm + C_max_t_val) * tile_Q_val
contrib_opacity = 1.0 * factor_op_v
```

This implements: `B_opacity_i = (||c_i|| + C_max_t) · Q_t`

**Assessment:** The `(||c_i|| + C_max_t)` factor corresponds to the bound
`||c_i|| + C_max_t · N_downstream / (1-α_i)` with `N_downstream = 1` and
`α_i → 0` (worst case simplification).  This is conservative **only when**
`N_downstream ≤ 1` or `α_i` is small.  For tiles with many downstream
Gaussians, the bound may be **under-conservative** (too tight), which would
produce false violations if the true tail contribution exceeds the bound.

**However**, the dominant error in R3 was not the tail factor but the
**conic-as-color proxy**.  After replacing conic norm with true color norm,
the bound becomes mathematically meaningful.  The tail simplification
(`N_downstream = 1`) is a separate, secondary concern that should be
validated empirically in the float64 forensics phase.

---

## 3. Color certificate — ∂C/∂c_i

```
∂C/∂c_i = T_i · α_i · q_p    (per pixel, summed over tile)
```

Bound:

```
|∂C/∂c_i| ≤ α_i · Σ_pixel |q_p| = α_i · Q_t
```

Code (unchanged, no conic proxy):
```python
A_coarse_v = clamp_max(o_j, 0.999)
contrib_color_coarse = A_coarse_v * tile_Q_val   # = α_i · Q_t
```

**Assessment:** Correct.  No color norm needed; no conic proxy.  This is why
color_coarse and color_tight had 0 violations.

---

## 4. Geometry certificates — ∂C/∂μ_i and ∂C/∂Σ_i

### 4.1 Mean2D gradient

```
∂C/∂μ_i = T_i · α_i · (∂c_i/∂μ_i) + T_i · α_i · c_i · (∂α_i/∂μ_i) + tail
```

The spatial derivative `∂c_i/∂μ_i` involves the Gaussian footprint's spatial
decay.  The code uses:

```python
base_geo_s = α_i · (||c_i|| + C_max_t) · Q_t
g_mean2d = base_geo_s · √λ_max · √(1/e)
```

where `λ_max` is the largest eigenvalue of the conic matrix.

The `√λ_max · √(1/e)` factor bounds the spatial decay: the maximum of
`exp(-σ)` occurs at the Gaussian center, but the **derivative** magnitude
`|Δx · exp(-σ)|` peaks at `σ = 0.5` (i.e., `exp(-0.5)`), giving `√(λ/e)`.

### 4.2 sigma_min concern (§8)

The user raises: does `sigma_min` bound `exp(-σ)` but fail to bound
`|Δx| · exp(-σ)` or `Δx² · exp(-σ)`?

**Analysis:** The code computes `s_min_v = sigma_min` (the minimum exponent
across the tile for each Gaussian).  The bound `g_mean2d = base_geo_s · mmu`
uses:

```python
mmu = where(s ≤ 0.5, √(λ_max/e), √(2·λ_max·s) · exp(-s))
```

This is the maximum of `|d/dx exp(-σ(x))|` = `|σ'(x) · exp(-σ(x))|` over the
tile, which **does** account for the spatial derivative factor.  The two
branches handle:
- `s ≤ 0.5`: derivative peaks inside the tile → `√(λ_max/e)`
- `s > 0.5`: derivative peaks at boundary → `√(2·λ_max·s) · exp(-s)`

**Assessment:** The sigma_min bound **does** include the spatial derivative
magnitude, not just `exp(-σ)`.  This is correct.  The primary fix needed
here is replacing the conic-norm color proxy with the true color norm (already
done in R3.1-B).

### 4.3 Conic gradient

```python
g_conic = base_geo_s · √1.5/e / λ_min
```

This bounds `|∂C/∂Σ_i|` using the conic eigenvalue `λ_min`.  The color factor
`base_geo_s` now uses the true color norm after R3.1-B.

---

## 5. Summary of changes

| Component | R3 (buggy) | R3.1 (repaired) |
|---|---|---|
| Color norm source | `conic_batch.norm(dim=-1)` | `colors_rgb[0, g_unique].norm(dim=-1)` |
| C_max_t source | `conic_norm_batch.max()` | `color_norm_batch.max()` |
| `compute_C_max_t` input | `conics` | `colors_rgb` |
| Function signature | `(opacities, conics, means2d, ...)` | `(opacities, conics, means2d, colors_rgb, ...)` |
| Opacity tail | `(‖c‖+C_max)·Q_t` (simplified) | Same formula, now with true ‖c‖ |
| Color certificate | `α·Q_t` (no proxy) | Unchanged (already correct) |

**What remains to validate:**
1. Whether the simplified tail bound `(‖c_i‖ + C_max_t) · Q_t` is sufficient
   or needs `N_downstream / (1-α_i)` → float64 forensics will answer this.
2. Whether the sigma_min spatial-derivative bound is tight enough → same.
3. Whether replacing conic norm with color norm eliminates the 7,494 violations
   or reveals new, smaller violations from the tail simplification.
