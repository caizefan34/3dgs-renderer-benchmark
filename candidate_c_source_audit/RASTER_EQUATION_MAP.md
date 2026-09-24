# Rasterization Equation Map (Candidate C Source Audit)

> **Purpose**: Map every source variable in the rasterization forward/backward CUDA kernels to its mathematical meaning, definition site, usage site, and exact source equation. No new math is derived — only what the source code expresses.

---

## Source Files Referenced

| File | Short Name |
|------|-----------|
| `canonical_source/raster_forward/rasterize_to_pixels_fwd.cu` | **fwd.cu** |
| `canonical_source/raster_backward/rasterize_to_pixels_bwd.cu` | **bwd.cu** |
| `canonical_source/gsplat_python/cuda/_wrapper.py` | **_wrapper.py** |
| `canonical_source/gsplat_python/rendering.py` | **rendering.py** |

---

## 1. Forward Pass Variables

### 1.1 `px`, `py` — Pixel center coordinates

```
source variable: px, py
  meaning:     Pixel-center coordinate; sub-pixel offset of +0.5 places the
               center at the center of pixel (j, i).
  defined in:  fwd.cu lines 60-61
  used in:     fwd.cu line 142 (delta computation)
  equation:    px = j + 0.5f    (j = tile.z * tile_size + thread.x)
               py = i + 0.5f    (i = tile.y * tile_size + thread.y)
```

### 1.2 `delta` — Pixel-to-Gaussian offset

```
source variable: delta (vec2<S>)
  meaning:     2D offset from pixel center to projected Gaussian mean
  defined in:  fwd.cu line 142
  used in:     fwd.cu lines 143-145 (sigma), bwd.cu lines 171-174 (sigma recompute),
               bwd.cu lines 223-231 (v_conic, v_means2d)
  equation:    delta = {xy.x - px, xy.y - py}
               where xy = means2d[g] — the projected 2D mean of Gaussian g
```

### 1.3 `sigma` — Gaussian exponent (quadratic form)

```
source variable: sigma (S)
  meaning:     Value of the 2D Gaussian exponent at this pixel, evaluated as
               the quadratic form ½·Δᵀ·Σ⁻¹·Δ where Σ⁻¹ is the inverse 2D
               covariance matrix (the "conic").
  defined in:  fwd.cu lines 143-145
  used in:     fwd.cu line 146 (alpha = min(0.999, opac * exp(-sigma))),
               fwd.cu line 147 (early-out guard: sigma < 0),
               bwd.cu lines 172-174 (recomputed), bwd.cu line 177 (guard),
               bwd.cu line 222 (v_sigma)
  equation:    sigma = 0.5f * (conic.x * delta.x * delta.x +
                               conic.z * delta.y * delta.y) +
                       conic.y * delta.x * delta.y

  Conic layout:  conic = (a, b, c) = upper triangle of the 2D precision matrix:
                   ⎡ a   b ⎤
                   ⎣ b   c ⎦
                 sigma = ½·a·Δx² + ½·c·Δy² + b·Δx·Δy
```

### 1.4 `alpha` — Gaussian alpha (opacity modulation)

```
source variable: alpha (S)
  meaning:     Effective per-pixel Gaussian alpha, clamped to [0, 0.999].
               Equals opac * exp(-sigma) when below the clamp.
  defined in:  fwd.cu line 146
  used in:     fwd.cu lines 147 (guard against too-small alpha),
               fwd.cu line 151 (T update), fwd.cu line 158 (vis),
               bwd.cu lines 161, 176 (recomputed), bwd.cu lines 194-210
  equation:    alpha = min(0.999f, opac * __expf(-sigma))
               Early-out: if sigma < 0 || alpha < 1/255 → continue (skip)
```

### 1.5 `vis` — Unmodulated exponential (forward) / Gaussian PDF (backward)

```
source variable: vis (S in forward as implicit expression; explicit in backward)
  meaning:     Forward: the value __expf(-sigma) is used directly in the alpha
               computation and not stored as "vis".  Backward: stored as `vis`
               after recomputing sigma.
  defined in:  fwd.cu line 146 (as part of alpha: opac * __expf(-sigma))
               bwd.cu line 175 (as vis = __expf(-sigma))
  used in:     fwd.cu line 146 (alpha), bwd.cu line 176 (alpha recompute),
               bwd.cu line 221 (clipping guard), bwd.cu line 222 (v_sigma),
               bwd.cu line 235 (v_opacity)
  equation:    vis = __expf(-sigma)
```

### 1.6 `T` — Transmittance (running, front-to-back)

```
source variable: T (S)
  meaning:     Running transmittance: the fraction of light that has NOT been
               absorbed by Gaussians processed so far (front-to-back traversal).
               Initialized to 1.0 and multiplied by (1 - alpha) at each step.
  defined in:  fwd.cu line 103: T = 1.0f
               fwd.cu line 151: next_T = T * (1.0f - alpha)
               fwd.cu line 166: T = next_T
               bwd.cu line 107: T = T_final (restored transmittance, reversed)
               bwd.cu line 195: T *= ra  (goes backwards: divides by (1-alpha))
  used in:     fwd.cu line 103 (init), fwd.cu line 151 (update),
               fwd.cu line 152 (early-out guard: next_T ≤ 1e-4),
               fwd.cu line 158 (vis = alpha * T), fwd.cu line 176 (render_alphas),
               fwd.cu lines 180-181 (background composite),
               bwd.cu lines 197-240 (v_rgb, v_alpha, buffer)
  equation:    Forward:  T₀ = 1.0
                         T_{n+1} = T_n * (1 - α_n)    [front → back]
                         T = T_final  after loop
               Backward: T = T_final (restored from forward output)
                         T ← T / (1 - α)  [back → front, reversing compositing]
```

### 1.7 `vis` (forward: `vis = alpha * T`) — Gaussian contribution weight

```
source variable: vis (S, distinct from bwd vis)
  meaning:     Contribution of this Gaussian to the pixel: the product of its
               per-pixel alpha and the current transmittance. This is the weight
               w_i = α_i * T_i used in the alpha-compositing sum.
  defined in:  fwd.cu line 158: const S vis = alpha * T
  used in:     fwd.cu line 162: pix_out[k] += c_ptr[k] * vis
  equation:    vis = alpha * T
               where T is the transmittance BEFORE processing this Gaussian
```

### 1.8 `pix_out` — Per-pixel RGB accumulator

```
source variable: pix_out (S[COLOR_DIM])
  meaning:     Per-pixel accumulator for the front-to-back alpha-compositing
               sum Σ color_i · α_i · T_i.  Does not include background.
  defined in:  fwd.cu line 112: S pix_out[COLOR_DIM] = {0.f}
  updated in:  fwd.cu lines 161-163: pix_out[k] += c_ptr[k] * vis
  used in:     fwd.cu lines 179-181 (final render_color assembly)
  equation:    pix_out[k] = Σ_i  colors_i[k] · α_i · T_i      [i over contributing Gaussians]
```

### 1.9 `render_colors` — Final per-pixel rendered color

```
source variable: render_colors (S*, global output)
  meaning:     Final per-pixel RGB (or N-D feature) output.  Equals the
               accumulated pix_out plus optional background * final_T.
  defined in:  fwd.cu kernel parameter (line 35), output tensor
  written at:  fwd.cu lines 176-182
  equation:    render_colors[pix_id * COLOR_DIM + k] =
                   backgrounds == nullptr
                       ? pix_out[k]
                       : pix_out[k] + T * backgrounds[k]
               where T here is the transmittance AFTER the last Gaussian.
```

### 1.10 `render_alphas` — Per-pixel alpha (opacity map)

```
source variable: render_alphas (S*, global output)
  meaning:     Per-pixel rendered alpha = 1 - final transmittance.
  defined in:  fwd.cu kernel parameter (line 36), output tensor
  written at:  fwd.cu line 176: render_alphas[pix_id] = 1.0f - T
  equation:    render_alphas[pix_id] = 1.0f - T   (T is final transmittance)
```

### 1.11 `last_ids` — Index of last contributing Gaussian

```
source variable: last_ids (int32_t*, global output)
  meaning:     Index (in the flattened / sorted intersection list) of the last
               Gaussian that contributed to this pixel.  Used by the backward
               pass to delimit the per-pixel Gaussian range.
  defined in:  fwd.cu kernel parameter (line 37), output tensor
  written at:  fwd.cu line 184: last_ids[pix_id] = static_cast<int32_t>(cur_idx)
  read at:     bwd.cu line 111: const int32_t bin_final = last_ids[pix_id]
  equation:    last_ids[pix_id] = flattened-index of the Gaussian whose alpha * T
               was the last contribution composited into this pixel.
```

### 1.12 `cur_idx` — Running index of most recent Gaussian

```
source variable: cur_idx (uint32_t)
  meaning:     Index in the flattened intersection array of the most recent
               Gaussian that contributed to this pixel's pix_out accumulator.
               (Not a kernel parameter — thread-local variable.)
  defined in:  fwd.cu line 105: uint32_t cur_idx = 0
  updated at:  fwd.cu line 164: cur_idx = batch_start + t
  written to:  fwd.cu line 184 (via last_ids)
  equation:    cur_idx = batch_start + t   (linear index within [range_start, range_end))
```

---

## 2. Backward Pass Variables

### 2.1 Gradients from the loss (kernel input parameters)

```
source variable: v_render_colors (S*)
  meaning:     Gradient of the loss w.r.t. render_colors (∂L / ∂render_colors).
  type:        Kernel parameter, bwd.cu lines 40-41
  read at:     bwd.cu lines 114-118 (→ local v_render_c)
  equation:    v_render_c[k] = v_render_colors[pix_id * COLOR_DIM + k]

source variable: v_render_alphas (S*)
  meaning:     Gradient of the loss w.r.t. render_alphas (∂L / ∂render_alphas).
  type:        Kernel parameter, bwd.cu line 42
  read at:     bwd.cu line 119: v_render_a = v_render_alphas[pix_id]
  equation:    v_render_a = v_render_alphas[pix_id]
```

### 2.2 Per-pixel backward state

```
source variable: T_final (S)
  meaning:     Transmittance AFTER the last Gaussian for this pixel.
               Reconstructed from render_alphas output.
  defined in:  bwd.cu line 106: S T_final = 1.0f - render_alphas[pix_id]
  used in:     bwd.cu line 107 (T init), bwd.cu lines 210, 218 (v_alpha)
  equation:    T_final = 1.0f - render_alphas[pix_id]

source variable: buffer (S[COLOR_DIM])
  meaning:     Accumulator holding Σ_{k>g} colors_k · α_k · T_k —
               the forward-composited colors of all Gaussians that are
               "further forward" (already processed in the backward pass).
               Starts at 0 and grows as we walk back-to-front.
  defined in:  bwd.cu line 109: S buffer[COLOR_DIM] = {0.f}
  updated at:  bwd.cu lines 238-240: buffer[k] += rgbs * (alpha * T)
  used in:     bwd.cu lines 205-207 (v_alpha computation)

source variable: bin_final (int32_t)
  meaning:     The index of the last Gaussian that contributed to this pixel.
               Derived from last_ids[fwd output].
  defined in:  bwd.cu line 111: bin_final = inside ? last_ids[pix_id] : 0
  used in:     bwd.cu lines 155-159 (valid-mask for loop bounds)
```

### 2.3 Local gradient variables (per-Gaussian, warp-level)

#### 2.3.1 `v_render_c` / `v_render_a`

```
source variable: v_render_c (S[COLOR_DIM]), v_render_a (S)
  meaning:     Per-pixel copies of the incoming gradient tensors.
  defined in:  bwd.cu lines 114-119
  equation:    v_render_c[k] = v_render_colors[pix_id * COLOR_DIM + k]
               v_render_a    = v_render_alphas[pix_id]
```

#### 2.3.2 `ra` — Reciprocal alpha complement

```
source variable: ra (S)
  meaning:     1 / (1 - alpha).  Used to reverse the transmittance
               multiplication from the forward pass:  T_{before} = T_{after} / (1 - α).
  defined in:  bwd.cu line 194: ra = 1.0f / (1.0f - alpha)
  used in:     bwd.cu line 195 (T *= ra), bwd.cu lines 206-218 (v_alpha)
  equation:    ra = 1 / (1 - alpha)
```

#### 2.3.3 `v_rgb_local` — Gradient w.r.t. Gaussian colors

```
source variable: v_rgb_local (S[COLOR_DIM])
  meaning:     ∂L / ∂colors[g] for the current Gaussian g.
               Forward: render_colors += Σ colors[i] · (α_i · T_i)
               So: ∂render_colors / ∂colors[g] = α_g · T_g
  defined in:  bwd.cu line 186: S v_rgb_local[COLOR_DIM] = {0.f}
  computed at: bwd.cu lines 197-201
  equation:    v_rgb_local[k] = alpha * T * v_render_c[k]
               where T is the transmittance BEFORE gaussian g
               (after the T *= ra reversal on line 195)
```

#### 2.3.4 `v_alpha` — Gradient w.r.t. Gaussian alpha

```
source variable: v_alpha (S)
  meaning:     ∂L / ∂alpha for the current gaussian g.  This is the core of
               the backward pass, encompassing three contributions:
               1.  The direct effect through w_g = α_g · T_g on rendered colors
               2.  The indirect effect through all further-forward weights
                   w_{k>g} = α_k · T_k, each of which depends on α_g via T_k
               3.  The effect on render_alphas (= 1 - T_final)
               4.  The effect through background composite (= T_final · bg)
  defined in:  bwd.cu line 203: S v_alpha = 0.f
  computed at: bwd.cu lines 204-219 (three terms)
  equation:    v_alpha =
                   ┌─ Color compositing effect ─────────────────┐
                   Σ_k ( rgb_g[k] * T  -  buffer[k] * ra ) * v_render_c[k]
                   ┌─ render_alphas effect ─────────────────────┐
                   +  T_final * ra * v_render_a
                   ┌─ Background composite effect ──────────────┐
                   +  -T_final * ra * Σ_k backgrounds[k] * v_render_c[k]
                   (background term present only when backgrounds ≠ nullptr)

               DERIVATION:
               - dw_g/dα_g          = T
               - dw_{k>g}/dα_g      = -w_k / (1-α_g) = -w_k * ra
               - d(render_alpha)/dα_g = T_final * ra    (since render_alpha=1-T_final)
               - d(background)/dα_g = -T_final * ra * background
```

#### 2.3.5 `v_sigma` — Gradient w.r.t. Gaussian exponent

```
source variable: v_sigma (S)
  meaning:     ∂L / ∂sigma.  Only computed when the alpha is not clamped
               (opac * vis <= 0.999f).
  computed at: bwd.cu line 222
  equation:    v_sigma = -opac * vis * v_alpha
               chain rule:  α = opac · exp(-σ) [no clamp]
                            ∂α/∂σ = -opac · exp(-σ) = -vis ... wait
               Actually: ∂α/∂σ = -opac · exp(-σ) = -opac · vis = -(alpha when unclamped)? No:
               α = opac · vis, and vis = exp(-σ), so dα/dσ = -opac · exp(-σ) = -opac · vis
               v_sigma = dα/dσ · v_alpha = (-opac · vis) · v_alpha
               Guard: only when opac * vis <= 0.999f (no clipping).
```

#### 2.3.6 `v_conic_local` — Gradient w.r.t. conic (inverse covariance)

```
source variable: v_conic_local (vec3<S>) = {v_a, v_b, v_c}
  meaning:     ∂L / ∂{conic.x, conic.y, conic.z} — the three upper-triangle
               values of the inverse 2D covariance.
               sigma = ½·a·Δx² + ½·c·Δy² + b·Δx·Δy
  computed at: bwd.cu lines 223-227
  equation:    v_conic_local = {
                   x: 0.5f * v_sigma * delta.x * delta.x,     // ∂σ/∂a = ½·Δx²
                   y: v_sigma * delta.x * delta.y,             // ∂σ/∂b = Δx·Δy
                   z: 0.5f * v_sigma * delta.y * delta.y       // ∂σ/∂c = ½·Δy²
               }
```

#### 2.3.7 `v_xy_local` — Gradient w.r.t. means2d

```
source variable: v_xy_local (vec2<S>) = {v_mx, v_my}
  meaning:     ∂L / ∂{means2d.x, means2d.y} — the projected 2D mean.
               Since delta = {mx - px, my - py}, the chain rule gives:
               ∂σ/∂mx = a·Δx + b·Δy
               ∂σ/∂my = b·Δx + c·Δy
  computed at: bwd.cu lines 228-231
  equation:    v_xy_local = {
                   x: v_sigma * (conic.x * delta.x + conic.y * delta.y),  // ∂σ/∂mx
                   y: v_sigma * (conic.y * delta.x + conic.z * delta.y)   // ∂σ/∂my
               }
```

#### 2.3.8 `v_xy_abs_local` — Absolute gradient for densification

```
source variable: v_xy_abs_local (vec2<S>)
  meaning:     |∂L / ∂means2d|, used by the AbsGS densification strategy.
  computed at: bwd.cu lines 232-234
  equation:    v_xy_abs_local = { abs(v_xy_local.x), abs(v_xy_local.y) }
               (only written when v_means2d_abs != nullptr, i.e. absgrad=True)
```

#### 2.3.9 `v_opacity_local` — Gradient w.r.t. opacity

```
source variable: v_opacity_local (S)
  meaning:     ∂L / ∂opacities[g].
               Since α = opac · vis (when not clamped), ∂α/∂opac = vis
  computed at: bwd.cu line 235
  equation:    v_opacity_local = vis * v_alpha
```

---

## 3. Kernel Output Gradients (write-back via atomic add)

| Variable | Meaning | Write site | Equation |
|----------|---------|-----------|----------|
| `v_colors` | ∂L / ∂colors[g] | bwd.cu lines 252-256 | `gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k])` |
| `v_conics` | ∂L / ∂conics[g] | bwd.cu lines 258-261 | `gpuAtomicAdd(..., v_conic_local.{x|y|z})` |
| `v_means2d` | ∂L / ∂means2d[g] | bwd.cu lines 263-265 | `gpuAtomicAdd(..., v_xy_local.{x|y})` |
| `v_means2d_abs` | \|∂L / ∂means2d[g]\| | bwd.cu lines 267-271 | `gpuAtomicAdd(..., v_xy_abs_local.{x|y})` |
| `v_opacities` | ∂L / ∂opacities[g] | bwd.cu line 273 | `gpuAtomicAdd(v_opacities + g, v_opacity_local)` |

All writes use `warpSum` reduction within each warp (lines 243-249), then lane 0 performs the atomic add.

---

## 4. Autograd Bridge (`_wrapper.py`)

### 4.1 `_RasterizeToPixels.backward` (lines 939-1010)

| Variable | Meaning | Equation (Python) | Source line |
|----------|---------|-------------------|-------------|
| `v_backgrounds` | ∂L / ∂backgrounds[c] | `(v_render_colors * (1.0 - render_alphas)).sum(dim=(1,2))` | 990-993 |

Since `render_alphas = 1 - T_final`, we have `1.0 - render_alphas = T_final`.
So `v_backgrounds[c] = Σ_{h,w} v_render_colors[h,w] · T_final[h,w]`,
which matches the chain rule: `∂render_colors/∂background = T_final`.

---

## 5. Forward Compositing — Row-by-Row Summary

| Step | Variable | Equation | Code Location |
|------|----------|----------|---------------|
| 1 | `T` | 1.0 | fwd.cu:103 |
| 2 | `delta` | means2d[g] - {px, py} | fwd.cu:142 |
| 3 | `sigma` | ½·a·Δx² + ½·c·Δy² + b·Δx·Δy | fwd.cu:143-145 |
| 4 | `alpha` | min(0.999, opac · exp(-sigma)) | fwd.cu:146 |
| 5 | *guard* | if sigma < 0 or alpha < 1/255 → skip | fwd.cu:147-148 |
| 6 | `next_T` | T · (1 - alpha) | fwd.cu:151 |
| 7 | *guard* | if next_T ≤ 1e-4 → done | fwd.cu:152-154 |
| 8 | `vis` | alpha · T | fwd.cu:158 |
| 9 | `pix_out` | += color · vis | fwd.cu:161-163 |
| 10 | `T` | = next_T | fwd.cu:166 |
| 11 | `render_alphas` | = 1 - T_final | fwd.cu:176 |
| 12 | `render_colors` | pix_out + T_final · background (if bg present) | fwd.cu:178-181 |
| 13 | `last_ids` | cur_idx | fwd.cu:184 |

---

## 6. Backward Compositing — Row-by-Row Summary

Processing in **reverse order** (back-to-front, undoing the forward compositing).

| Step | Variable | Equation | Code Location |
|------|----------|----------|---------------|
| 1 | `T_final` | 1.0 - render_alphas[pix_id] | bwd.cu:106 |
| 2 | `T` | = T_final | bwd.cu:107 |
| 3 | `buffer` | = {0} | bwd.cu:109 |
| 4 | `v_render_c` | = v_render_colors[pix_id] | bwd.cu:114-118 |
| 5 | `v_render_a` | = v_render_alphas[pix_id] | bwd.cu:119 |
| — | *→ For each Gaussian g (back-to-front):* | | |
| 6 | `ra` | 1 / (1 - alpha) | bwd.cu:194 |
| 7 | `T` | *= ra → T is now transmittance *before* g | bwd.cu:195 |
| 8 | `fac` | alpha · T | bwd.cu:197 |
| 9 | `v_rgb_local` | fac · v_render_c | bwd.cu:199-200 |
| 10 | `v_alpha` | = Σ(rgb_g·T - buffer·ra)·v_render_c | bwd.cu:204-206 |
| 11 | `v_alpha` | += T_final · ra · v_render_a | bwd.cu:210 |
| 12 | `v_alpha` | += -T_final · ra · Σ(bg·v_render_c) | bwd.cu:212-218 |
| 13 | `v_sigma` | = -opac · vis · v_alpha | bwd.cu:222 |
| 14 | `v_conic_local` | = {½·v_σ·Δx², v_σ·Δx·Δy, ½·v_σ·Δy²} | bwd.cu:223-227 |
| 15 | `v_xy_local` | = {v_σ(a·Δx+b·Δy), v_σ(b·Δx+c·Δy)} | bwd.cu:228-231 |
| 16 | `v_xy_abs_local` | = {\|v_xy.x\|, \|v_xy.y\|} | bwd.cu:232-234 |
| 17 | `v_opacity_local` | = vis · v_alpha | bwd.cu:235 |
| 18 | `buffer` | += rgb_g · fac | bwd.cu:238-240 |
| — | *warp sum, atomic add to global gradients* | | bwd.cu:243-273 |

---

## 7. Variable Provenance Summary

| Variable | Forward | Backward | Defined File | Lines |
|----------|---------|----------|-------------|-------|
| `px, py` | ✓ | ✓ | fwd.cu, bwd.cu | fwd:60-61, bwd:75-76 |
| `delta` | ✓ | ✓ | fwd.cu, bwd.cu | fwd:142, bwd:171 |
| `sigma` | ✓ | ✓ | fwd.cu, bwd.cu | fwd:143-145, bwd:172-174 |
| `vis` (exp) | ✓ | ✓ | fwd.cu, bwd.cu | fwd:146, bwd:175 |
| `alpha` | ✓ | ✓ | fwd.cu, bwd.cu | fwd:146, bwd:176 |
| `T` | ✓ | ✓ | fwd.cu, bwd.cu | fwd:103/151/166, bwd:107/195 |
| `T_final` | — | ✓ | bwd.cu | 106 |
| `vis` (weight) | ✓ | — | fwd.cu | 158 |
| `pix_out` | ✓ | — | fwd.cu | 112/161-163 |
| `render_colors` | ✓ | — | fwd.cu | 176-182 |
| `render_alphas` | ✓ | ✓ | fwd.cu, bwd.cu | fwd:176, bwd:106 |
| `last_ids` | ✓ | ✓ | fwd.cu, bwd.cu | fwd:184, bwd:111 |
| `cur_idx` | ✓ | — | fwd.cu | 105/164 |
| `buffer` | — | ✓ | bwd.cu | 109/238-240 |
| `ra` | — | ✓ | bwd.cu | 194 |
| `fac` | — | ✓ | bwd.cu | 197 |
| `v_render_c/a` | — | ✓ | bwd.cu | 114-119 |
| `v_rgb_local` | — | ✓ | bwd.cu | 199-200 |
| `v_alpha` | — | ✓ | bwd.cu | 203-219 |
| `v_sigma` | — | ✓ | bwd.cu | 222 |
| `v_conic_local` | — | ✓ | bwd.cu | 223-227 |
| `v_xy_local` | — | ✓ | bwd.cu | 228-231 |
| `v_xy_abs_local` | — | ✓ | bwd.cu | 232-234 |
| `v_opacity_local` | — | ✓ | bwd.cu | 235 |

---

## 8. Tensor Shapes

| Tensor | Shape (packed) | Shape (non-packed) | DType |
|--------|---------------|-------------------|-------|
| `means2d` | [nnz, 2] | [C, N, 2] | float32 |
| `conics` | [nnz, 3] | [C, N, 3] | float32 |
| `colors` | [nnz, D] | [C, N, D] | float32 |
| `opacities` | [nnz] | [C, N] | float32 |
| `backgrounds` | [C, D] | [C, D] (optional) | float32 |
| `render_colors` | [C, H, W, D] | same | float32 |
| `render_alphas` | [C, H, W, 1] | same | float32 |
| `last_ids` | [C, H, W] | same | int32 |
| `flatten_ids` | [n_isects] | same | int32 |
| `tile_offsets` | [C, tile_H, tile_W] | same | int32 |

---

## 9. Key Design Notes from Source Comments

1. **Float precision for T**: "transmittance is gonna be used in the backward pass which requires a high numerical precision so we use double for it. However double make bwd 1.5x slower so we stick with float for now." (fwd.cu:100-102, bwd.cu:171-175)

2. **Warp-optimized backward**: The backward pass uses `cg::tiled_partition<32>` warps and `warpSum` reduction to accumulate per-Gaussian gradients before writing to global memory (bwd.cu:124, 243-249).

3. **Skipped Gaussians**: A Gaussian whose `sigma < 0` or `alpha < 1/255` is skipped in both forward and backward (fwd.cu:147, bwd.cu:177).

4. **Early termination**: If `next_T ≤ 1e-4`, the pixel is considered fully opaque and remaining Gaussians are skipped (fwd.cu:152).

5. **Alpha clamping**: `alpha = min(0.999f, opac * exp(-sigma))` — the 0.999 clamp prevents numerical issues. In backward, when the clamp is active (`opac * vis > 0.999`), the gradient through sigma is not computed (bwd.cu:221 guard).

6. **Channel padding**: The Python wrapper pads color channels to the nearest supported dimension (1,2,3,4,5,8,9,16,17,…512,513) and strips the padding after the kernel call (`_wrapper.py`:483-523, 548-550).
