# A0-v2 / A0-P10 — Exact C0 Backward Coefficient Factorization

Frozen baseline: **P12/C0** (F9 + SCALAR_ADJOINT + H8_MR, `HIGS_PX_RUNTIME=2`,
`HIGS_BWD_ABSGRAD=1`, `eps2d=0.3`, `render_mode="RGB"`, `backgrounds=None` (black),
`sh_degree=3`, `tile_size=16`, `packed=False`).

Frozen sources audited (authoritative = `_moonshot/`, byte-identical to
`p18_p0/c0_source/` and to the remote frozen C0 worktree
`/root/epic05_frozen/higs_c0_final30k_worktree`; Stage 0 asserts the remote byte-identity):

| file | role |
|---|---|
| `RasterizeToPixels3DGSDevice.cuh` | per-hit forward **and** backward blend math (shared) |
| `HigsNativeBackward.cu` | native blend-bwd kernel (`higs_blend_bwd_px_kernel`, PX=2) + parameter-gradient stage |
| `Common.h` | float constants (`MAX_ALPHA`, `ALPHA_THRESHOLD`, `MIN_ONE_MINUS_ALPHA`, `TRANSMITTANCE_THRESHOLD`) |
| `gaussian_inference.py` | `_native_backward` wrapper (env flag resolution, state plumbing) |

This is the **exact algebra gate** required by `A0_V2_PREREG.md` §1 and
`DSH_TASK_A0V2.md` Stage 1. It shows the equations used to export the exact
`q_alpha` / `q_color` sufficient state, and the minimal exact state needed for
offline exact-VJP replay. **No proxy is used.**

---

## 1. Forward-captured state (consumed, never re-rasterized)

The C0 backward consumes the exact state captured by the differentiable forward
(same scene / order / visibility version as the forward; it never re-runs the
rasterization pipeline). Per the frozen `_native_backward` and the backward
kernel signature, the state is:

```
means2d      [N, 2]      per-Gaussian projected 2D center (visible subset, global ids)
conics       [N, 3]      inverse-covariance form (vec3 (x, y, z)); sigma = 0.5*(x·dx² + z·dy²) + y·dx·dy
colors_eval  [N, 3]      evaluated RGB used by the blend (render colors)
opacities    [N]         per-Gaussian opacity (the alpha pre-factor; in [0,1])
radii        [I, N, 2]
tile_offsets [I, tile_h, tile_w]   per-image, per-tile start index into flatten_ids
flatten_ids  [n_isects]            global Gaussian id per intersection (depth-sorted per tile)
render_alphas[I, H, W]             per-pixel accumulated alpha; T_final = 1 - render_alphas (transmittance reaching the background)
last_ids     [I, H, W]             global id of the front-most contributing Gaussian (per pixel)
```

`n_isects = len(flatten_ids)`. The per-tile sorted range is
`[tile_offsets[tile], tile_offsets[tile+1])` (last tile of an image ends at
`n_isects`). The per-pixel contributing range ends at `last_ids[pixel]` (the
front-most Gaussian that contributed to that pixel in the forward).

**Constant float values** (AUTHORITATIVE: `_moonshot/Common.h` == `p18_p0/c0_source/Common.h`,
lines 97–114; used verbatim by the walk):

```
ALPHA_THRESHOLD         = 1.f / 255.f     # = 1/255 ≈ 0.0039215686
MAX_ALPHA               = 0.99f
TRANSMITTANCE_THRESHOLD = 1e-4f
MIN_ONE_MINUS_ALPHA     = 1e-6f
```

> ⚠️ The v1 kit incorrectly used `1e-4` for `ALPHA_THRESHOLD`, `MAX_ALPHA` (as
> `1-1e-4`), and `MIN_ONE_MINUS_ALPHA`. Those were **wrong**; the authoritative
> values above are taken from the frozen `Common.h`.

### 1.1 Native bucket object (tile × batch) — NOT whole tiles

The native blend-bwd kernel is launched with a **block** per (tile, batch). For
the frozen PX=2 arm, `__launch_bounds__(256/PX)` ⇒ `block_size = 256/PX = 128`.
The kernel splits each tile's sorted intersection range into batches of width
`block_size = 128`, each processed back-to-front:

```
batch_end   = range_end - 1 - b·W          (W = block_size = 128)
batch_size  = min(W, batch_end + 1 - range_start)
g(t)        = flatten_ids[batch_end - t]    for local t in [0, batch_size)
```

A **native bucket object is the (tile_id, batch_index b)** pair; the walk's
analysis matrix for one bucket is `16×16` over the tile's pixels, and the column
count `m` (distinct accepted Gaussians) is **bounded by the real native bucket
width** (≤ 128). The P10 top-32 selection ranks native bucket objects by native
accepted-hit count only (no loss / gradient / rank).

---

## 2. Exact per-contribution coefficients

For a pixel `p` (absolute `(I,J)`) and a contributing Gaussian `i` (global id `g`,
local back-to-front index `t` within the bucket), with per-pixel running state `T`
and `buffer[k]` (accumulated color so far, `k ∈ {r,g,b}`), the frozen backward math
(`rasterize_to_pixels_3dgs_blend_bwd`) is:

```
# T is INITIALIZED TO T_final, NOT 1.0   (kernel: float T = T_final, line 404/149)
T        = 1.0 - render_alphas[p]         # transmittance reaching the background
buffer[k]= 0
bin_final= last_ids[p]                    # per-pixel validity floor

# per accepted hit (back-to-front), exact kernel lines 47-51, 126-154:
sigma    = 0.5*(conic.x·dx² + conic.z·dy²) + conic.y·dx·dy   # conic pairing
dx       = means2d.x - (J + 0.5)
dy       = means2d.y - (I + 0.5)
vis      = exp(-sigma)
alpha    = min(MAX_ALPHA, opacity·vis)
valid    = NOT (sigma < 0 OR alpha < ALPHA_THRESHOLD)        # acceptance gate (backward)

ra       = 1 / max(MIN_ONE_MINUS_ALPHA, 1 - alpha)
T        = T · ra                                            # transmittance AFTER this Gaussian
fac      = alpha · T
rgbs[k]  = colors_eval[g, k]
```

### 2.1 `q_color` (PRIMARY: exact per-channel color adjoint)

The exact per-channel color adjoint (the kernel's `v_rgb_local[k]`, line 134):

```
q_color[p, i, k]  =  fac · v_render_colors[p, k]   =  alpha·T · v_render_colors[p, k]
    (k = r, g, b)

    =>  dL/d(colors_eval[g, k])  +=  q_color[p, i, k]      (exact local color derivative)
```

`q_color` is the **REAL 3-channel** exact color adjoint coefficient — it includes
the per-channel image adjoint `v_render_colors[p,k]` (the gradient of the loss with
respect to the raw raster frame at pixel p). It is **not** a single scalar.

A **secondary** scalar `w_color[p,i] = fac` (the scalar color weight, broadcast) is
also exported for inspection, but it is **NOT** the primary exact color adjoint:
the primary `q_color` is the full per-channel tensor above. The Oracle-SVD joint
`Q_color` analysis uses the REAL 3-channel tensor `q_color` (reshaped to
`[256, m·3]`), not `w_color`.

### 2.2 `q_alpha` (exact scalar adjoint on `dL/dalpha`)

The alpha adjoint before the local alpha-derivative expansion (kernel lines 137–154):

```
q_alpha[p, i]  =  v_alpha
    =  Σ_k ( rgbs[k]·T - buffer[k]·ra ) · v_render_colors[p, k]
      +  T_final · ra · v_render_alphas[p]          # = 0 for render_mode="RGB"
      -  T_final · ra · Σ_k background[k] · v_render_colors[p, k]   # = 0 (backgrounds=None)
```

With `render_mode="RGB"` and `backgrounds=None` (the frozen C0 recipe), the last two
terms vanish, so:

```
q_alpha[p, i]  =  Σ_k ( rgbs[k]·T - buffer[k]·ra ) · v_render_colors[p, k]
```

`q_alpha` and `q_color` are **both directly exposed by the kernel** (no proxy).
Together with the exact local-derivative factors below they form a **clean exact
sufficient-state factorization**: the full contribution of this (pixel, Gaussian)
pair to *all* parameter gradients is `q_alpha · dalpha/dθ` + `q_color[k] · dcolor[k]/dθ`,
exactly.

### 2.3 Local alpha-derivative factors (documented; used by A1 offline replay)

Given `q_alpha`, the local `alpha = opacity · vis` derivative expands exactly to:

```
d alpha / d opacity   =  vis
d alpha / d sigma     = -opacity · vis
d sigma / d conic     = (x-deriv: dx²,  y-deriv: dx·dy,  z-deriv: dy²)   # d sigma/d conic
d sigma / d means2d   = ( -(x·dx + y·dy),  -(z·dy + y·dx) )              # from sigma = 0.5(x dx² + z dy²) + y dx dy
```

so (the C0 SCALAR_ADJOINT + H8_MR branch; `v_sigma_ref = -opac·vis·v_alpha` is shared):

```
v_opacity  =  vis · q_alpha
v_sigma    = -opacity · vis · q_alpha
v_conic    =  v_sigma · (dx², dx·dy, dy²)
v_xy       =  v_sigma · (dx, dy)         # H8_MR accumulates v_xy (moment-redundancy form)
```

> **H8_MR note.** The frozen C0 arm has `HIGS_BWD_H8_MR=1`. The H8-MR mode changes
> how the *local spatial derivative* `v_xy` is accumulated (a moment/redundancy
> reformulation of the same contraction, with `HIGS_BWD_ABSGRAD=1` also emitting
> the contracted per-pixel `|v_xy|`). **It does NOT change `q_alpha`, `q_color`, or
> the per-hit `v_sigma` / `v_conic` scalar factors** — those are computed identically
> in both H8_MR branches (`v_sigma_ref = -opac·vis·v_alpha` is the shared expression).
> Therefore the exported sufficient state `q_alpha`, `q_color`, plus the exact local
> derivative factors above, is **sufficient to reconstruct the exact native
> per-parameter gradient contribution for every Gaussian**, independent of the
> H8_MR branch. This is the identity the prereg §1 requires to document.

---

## 3. Minimal exact state for offline exact-VJP replay (A1)

To reconstruct the exact native **selected-bucket** gradient contribution offline
(without re-running CUDA), A1 needs, per accepted hit `(p, i)`:

```
q_alpha[p,i]            (float32)
q_color[p,i,3]          (float32, PRIMARY exact per-channel color adjoint = fac·v_render_colors)
w_color[p,i]            (float32, SECONDARY scalar weight = fac; broadcast)
g                      (global Gaussian id)
local_t                (back-to-front index within the bucket)
dx, dy                 (= means2d[g] - (pixel + 0.5))
conic[g]               (x, y, z)
opacity[g]             (the alpha pre-factor)
vis, alpha             (per-hit, for the dalpha/dtheta factors)
v_render_colors[p,3]   (the image adjoint at the pixel)   [the frozen raw-frame gradient]
```

From these, the exact native per-Gaussian gradient for the bucket is the sum over
its accepted hits of `q_alpha·(dalpha/dθ)` and `q_color[k]·(dcolor[k]/dθ)` per §2.3.
This is the **exact** selected-bucket gradient (no re-rasterization, no proxy).
The P10 preflight exports `q_alpha`, `q_color`, `w_color`, `g`, `local_t`, the
accepted mask, and the per-bucket accepted-hit census; the dx/dy/conic/opacity
factors are re-derived offline from the exported `col_gaussians` + the forward
state (or exported directly in A1).

### 3.1 The image adjoint `v_render_colors` (RAW-frame adjoint, same native graph)

`v_render_colors[p, k]` is the gradient of the loss with respect to the **raw
(pre-clamp) raster frame** at pixel p — the EXACT adjoint that enters the raster
backward. It is obtained from the **SAME native graph** as the render:

```
frame_raw = rasterize(...).frame                 # pre-clamp native output [H,W,3] (autograd graph)
loss      = 0.8·L1(frame_raw, gt) + 0.2·(1 - fused_ssim(frame_raw, gt, padding="same", train=True))
v_raw     = torch.autograd.grad(loss, frame_raw, retain_graph=True)[0]   # [H,W,3]
```

> ⚠️ The v1 kit re-derived the gradient on a *detached clamped* frame, which is NOT
> the adjoint that enters the backward. The corrected P10 takes the gradient of the
> loss with respect to the raw (pre-clamp) native frame directly, so the walk's
> `q_color` / `q_alpha` are computed against the same `v_render_colors` the native
> backward consumes.

---

## 4. Backward acceptance semantics (EXACT, differs from the forward early-stop)

The C0 **backward** does **not** re-check the forward saturation /
`TRANSMITTANCE_THRESHOLD` exclusion. `higs_blend_bwd_px_kernel` (PX=2), for one
native batch, walks, for each pixel, the local back-to-front indices
`t` in `[0, batch_size)` (where `batch_end = range_end-1-b·W`,
`batch_size = min(W, batch_end+1-range_start)`), and **accepts** a hit iff

```
accepted(p, i)  <=>  (pixel in image) AND (batch_end - t <= bin_final)
                      AND (sigma >= 0 AND alpha >= ALPHA_THRESHOLD)
```

with `bin_final = last_ids[pixel]`. The forward's early-stop (a Gaussian that
drops the pixel transmittance to `TRANSMITTANCE_THRESHOLD` is excluded) is a
**forward-only** optimization and is **not** reproduced in the backward walk. The
P10 walk in `lib.py` reproduces the backward acceptance rule exactly (no
`TRANSMITTANCE_THRESHOLD` test). This is why the walk's accepted-hit census matches
the native backward's per-hit work, and why it is the correct population for
"high-work" bucket ranking.

> Note on the per-pixel `bin_final` floor: the kernel iterates `t` with the
> per-pixel validity gated by `batch_end - t <= last_ids[pixel]`, so the effective
> per-pixel range is `t = 0` up to the front-most contributing Gaussian (clamped to
> `batch_end`). The P10 walk uses this per-pixel range, which is exactly the set of
> hits the native backward would accept for that pixel.

---

## 5. Numerical fidelity of the CPU walk (preflight-sanity envelope)

The P10 walk is a CPU float32 transcription of the CUDA fast-math:

- `sigma`, `vis`, `alpha`, `ra`, `T`, `fac`, `buffer`, `v_alpha` are computed in
  float32 with the same operation order as the kernel.
- `exp(-sigma)`: the kernel uses `__expf` (fast HW exp); the walk uses
  `float32(exp(-sigma))`. The residual is ~1 ulp (≤ 2^-23 relative) and is
  **absorbed by the "within the C0 numerical envelope" preflight-sanity gate** (the
  same envelope the frozen C0 backward itself uses vs a double-precision reference).
- The **acceptance decision** (validity) uses the same `ALPHA_THRESHOLD`/`sigma<0`
  test, so the accepted-hit *set* is exact; the tiny `vis` residual does not flip the
  acceptance (alpha is far from `ALPHA_THRESHOLD=1/255` for accepted hits by
  construction).

The P10 **preflight-sanity** gate (`A0_PREFLIGHT_SANITY.json`) — named *preflight
sanity* per the lead review; it does **NOT** claim full A0-v2 native-gradient
parity (that belongs to formal A0) — asserts:
1. the frozen-loss scalar is **reproducible** across two renders (abs diff < 1e-6);
2. the **native** parameter gradients are finite (native backward is the reference);
3. the walk's accepted-hit census is **deterministic** (re-run a bucket → identical
   hit sequence);
4. the exported `q_alpha`/`q_color` are finite and `m` ≤ native bucket width.

Per-parameter gradient *reconstruction* from the exported coefficients (the full
instrumented-vs-native reconstructed-VJP match across xyz/scale/quat/opacity/SH-DC/SH-rest)
is the **formal A0** item — it requires the frozen local-derivative factors and the
Adam-preconditioned comparison, which the P10 preflight deliberately does not
implement (per scope: "Do not implement DCT/VJP reconstruction").

---

## 6. Verdict

**Exact factorization AVAILABLE.** A single exact scalar `q_alpha[p,i]` and a REAL
3-channel exact color adjoint `q_color[p,i,3] = fac·v_render_colors[p,k]` exist under
the frozen C0 clamping / saturation / recurrence semantics (`T` initialized to
`T_final`, conic pairing `0.5(x·dx²+z·dy²)+y·dx·dy`), plus a documented exact
local-derivative factorization for offline replay. The prereg §1 gate **PASSES**:

```
A0_ALGEBRA_GATE = PASS   (q_alpha exact; q_color exact real 3-channel; no proxy;
                          T_init=T_final; conic pairing verified; H8_MR independence documented)
```

This authorizes proceeding to the P10 instrumented capture + preflight-sanity +
native-bucket + oracle-SVD preflight (the remainder of this kit).
