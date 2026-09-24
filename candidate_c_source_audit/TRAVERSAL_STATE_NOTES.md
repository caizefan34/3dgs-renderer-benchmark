# Traversal State: What's Stored vs. Reconstructed in the gsplat Rasterizer

> **Source files audited:**
> - `canonical_source/raster_forward/rasterize_to_pixels_fwd.cu`
> - `canonical_source/raster_backward/rasterize_to_pixels_bwd.cu`
> - `canonical_source/gsplat_python/cuda/_wrapper.py`

---

## 1. Overview of the Forward Traversal

The forward kernel processes Gaussians per-pixel in **front-to-back** order (`fwd.cu:82-167`). Each pixel thread walks the sorted Gaussian list for its tile and accumulates color:

```
T = 1.0f                                        (initial transmittance)
for each Gaussian g (front to back):
    sigma  = 0.5·(conic.x·Δx² + conic.z·Δy²) + conic.y·Δx·Δy
    alpha  = min(0.999, opac · exp(-sigma))
    if sigma<0 or alpha<1/255:  continue        (skip)
    next_T = T · (1 - alpha)
    if next_T ≤ 1e-4:  break                    (pixel opaque → done)
    vis    = alpha · T
    color += g.rgb · vis
    T      = next_T
```

After the loop, **three values are written to global memory** (`fwd.cu:170-185`):

| Output tensor | Shape | Computation | Code |
|---|---|---|---|
| `render_colors` | `[C, H, W, D]` | Accumulated pixel color (± background blend) | `fwd.cu:177-182` |
| `render_alphas` | `[C, H, W, 1]` | `1.0 - T` (final transmittance complement) | `fwd.cu:176` |
| `last_ids` | `[C, H, W]` | Index (into `flatten_ids`) of the **last Gaussian** that contributed to this pixel | `fwd.cu:184` |

---

## 2. What is Saved for the Backward Pass

The `_RasterizeToPixels` autograd function saves these tensors in `ctx.save_for_backward(...)` (`_wrapper.py:918-929`):

| Saved tensor | Source | Shape |
|---|---|---|
| `means2d` | Forward input | `[C, N, 2]` or `[nnz, 2]` |
| `conics` | Forward input | `[C, N, 3]` or `[nnz, 3]` |
| `colors` | Forward input | `[C, N, D]` or `[nnz, D]` |
| `opacities` | Forward input | `[C, N]` or `[nnz]` |
| `backgrounds` | Forward input (optional) | `[C, D]` or `None` |
| `masks` | Forward input (optional) | `[C, tile_h, tile_w]` |
| `isect_offsets` (a.k.a. `tile_offsets`) | Forward input | `[C, tile_h, tile_w]` |
| `flatten_ids` | Forward input | `[n_isects]` |
| `render_alphas` | **Forward output** | `[C, H, W, 1]` |
| `last_ids` | **Forward output** | `[C, H, W]` |

Note: `render_colors` is **not** saved for backward. The backward pass receives only its gradient (`v_render_colors`) from upstream.

---

## 3. Per-Quantity Classification

### 3.1 Per-Pixel After Forward

| Quantity | Classification | Details |
|---|---|---|
| **`render_colors`** (final pixel color) | **STORED** | Forward output tensor `[C, H, W, D]` returned to caller. Not saved in `ctx` (not needed), but available to any downstream consumer. |
| **alpha** (`1 - T`) | **STORED** | Written as `render_alphas[pix_id] = 1.0f - T` (`fwd.cu:176`). Saved in `ctx` as `render_alphas` (`_wrapper.py:927`). |
| **final T** (transmittance after last Gaussian) | **RECOMPUTED_IN_BACKWARD** | Backward computes `T_final = 1.0f - render_alphas[pix_id]` (`bwd.cu:106`). The final T is never stored directly; `render_alphas` = `1 - T_final` is the stored proxy. |
| **last_ids** (index of last contributing Gaussian) | **STORED** | Written as `last_ids[pix_id] = static_cast<int32_t>(cur_idx)` (`fwd.cu:184`). Saved in `ctx` (`_wrapper.py:928`). Used to clamp iteration in backward (`bwd.cu:111, 158-159`). |

### 3.2 Per Gaussian-Pixel (intermediate per-step values)

| Quantity | Classification | Details |
|---|---|---|
| **sigma** (exponent) | **RECOMPUTED_IN_BACKWARD** | Equation: `sigma = 0.5·(conic.x·Δx² + conic.z·Δy²) + conic.y·Δx·Δy` |
| | | Forward: `fwd.cu:143-145` |
| | | Backward: `bwd.cu:172-174` — identical computation from `conics` and recomputed `delta`. |
| **alpha** (per-Gaussian opacity) | **RECOMPUTED_IN_BACKWARD** | Equation: `alpha = min(0.999f, opac · exp(-sigma))` |
| | | Forward: `fwd.cu:146` |
| | | Backward: `bwd.cu:175-176` — identical computation from `opacities` and recomputed `sigma`. |
| **T_before** (transmittance *before* this Gaussian) | **RECOMPUTED_IN_BACKWARD** | **Key insight:** The backward kernel reconstructs T_before by walking Gaussians **back-to-front** and compounding the inverse of `(1-alpha)`. |
| | | Forward: T starts at 1.0, then `T = T · (1 - alpha)` after each step (`fwd.cu:103, 151, 166`). |
| | | Backward: starts from `T = T_final` (`bwd.cu:106-107`), then for each Gaussian (backwards): `ra = 1/(1-alpha)`; `T *= ra` (`bwd.cu:194-195`). After the multiply, T is T_before for that Gaussian. |
| **T_after** (transmittance *after* this Gaussian) | **RECOMPUTED_IN_BACKWARD** | This is T_before of the next Gaussain front-ward. In backward, it's the value of T **before** the `T *= ra` step. |
| **weight/vis** (`alpha · T_before`) | **RECOMPUTED_IN_BACKWARD** | Forward: `vis = alpha * T` (`fwd.cu:158`). Backward: `fac = alpha * T` (`bwd.cu:197`), where T is T_before after the `T *= ra` reconstruction. |
| **Gaussian-pixel contribution** (`g.rgb · vis`) | **RECOMPUTED_IN_BACKWARD** | Forward accumulates into `pix_out`. Backward re-reads colors from `rgbs_batch` (loaded from global `colors` at `bwd.cu:146-149`) and multiplies by the recomputed `fac`. The `buffer[]` accumulation in backward (`bwd.cu:239-241`) mirrors the forward `pix_out[]` accumulation but in reverse. |

### 3.3 Per-Gaussian (forward inputs, never reconstructed)

| Quantity | Classification | Details |
|---|---|---|
| **means2d** | **STORED** | Forward input, saved in `ctx` (`_wrapper.py:919`). Read directly in backward (`bwd.cu:142`). |
| **conics** | **STORED** | Forward input, saved in `ctx` (`_wrapper.py:920`). Read directly in backward (`bwd.cu:145`). |
| **colors** | **STORED** | Forward input, saved in `ctx` (`_wrapper.py:921`). Loaded into shared memory in backward (`bwd.cu:146-149`). |
| **opacities** | **STORED** | Forward input, saved in `ctx` (`_wrapper.py:922`). Read directly in backward (`bwd.cu:143`). |

### 3.4 Per-Tile (always available from forward inputs)

| Quantity | Classification | Details |
|---|---|---|
| **tile_offsets** (a.k.a. `isect_offsets`) | **STORED** | Forward input, saved in `ctx` (`_wrapper.py:926`). Defines `range_start` / `range_end` for each tile in both forward (`fwd.cu:82-86`) and backward (`bwd.cu:87-91`). |
| **flatten_ids** | **STORED** | Forward input, saved in `ctx` (`_wrapper.py:927`). Maps intersection-index → Gaussian index in both directions. |

---

## 4. Backward Pass Transmittance Reconstruction (the Core Design)

The most important insight for a Candidate C implementation:

```
# Forward pass (front-to-back)           # Backward pass (back-to-front)
T = 1.0                                   T = T_final  (= render_alphas[pix_id])
for g in [g0, g1, ..., gk]:               for g in [gk, ..., g1, g0]:
    alpha = compute_alpha(g)                   alpha = compute_alpha(g)   # recomputed
    vis = alpha * T                            ra = 1 / (1 - alpha)
    accumulate(g.rgb * vis)                    T *= ra                    # T is now T_before
    T *= (1 - alpha)      ← T becomes          fac = alpha * T            # fac = vis in forward
                            T_after             ...compute gradients...
                                                buffer += g.rgb * fac
```

**At the start of backward, T = T_final** (transmittance after the last Gaussian).
**After each backwards step, T = T_before** for that Gaussian (which equals T_after of the preceding Gaussian going forward).

This means the backward kernel reconstructs the **entire transmittance history** without storing per-Gaussian-pixel `T`, `sigma`, `alpha`, or `vis` — it trades recomputation for memory.

---

## 5. Summary Table

| Quantity | Category | Stored as | Shape | Source ref |
|---|---|---|---|---|
| `render_colors` | Per-pixel after-fwd | Output tensor | `[C, H, W, D]` | `fwd.cu:177-182` |
| `render_alphas` | Per-pixel after-fwd | `ctx` saved tensor | `[C, H, W, 1]` | `fwd.cu:176`; `_wrapper.py:927` |
| `last_ids` | Per-pixel after-fwd | `ctx` saved tensor | `[C, H, W]` | `fwd.cu:184`; `_wrapper.py:928` |
| final T | Per-pixel after-fwd | **Recomp:** `1 - render_alphas` | scalar/pixel | `bwd.cu:106` |
| `sigma` | Per Gaussian-pixel | **Recomp:** eqn from `conics,delta` | scalar | `fwd.cu:143-145`; `bwd.cu:172-174` |
| `alpha` | Per Gaussian-pixel | **Recomp:** `min(0.999, opac·exp(-sigma))` | scalar | `fwd.cu:146`; `bwd.cu:175-176` |
| `T_before` | Per Gaussian-pixel | **Recomp:** back-to-front `T *= 1/(1-alpha)` | scalar | `fwd.cu:103,166`; `bwd.cu:194-195` |
| `T_after` | Per Gaussian-pixel | **Recomp:** T_before of next Gaussian | scalar | Implicit in `bwd.cu:194-195` |
| `vis` / `fac` | Per Gaussian-pixel | **Recomp:** `alpha * T` (T=T_before) | scalar | `fwd.cu:158`; `bwd.cu:197` |
| contrib `g.rgb·vis` | Per Gaussian-pixel | **Recomp:** `rgbs_batch * fac` | `[D]` | `fwd.cu:161-163`; `bwd.cu:239-241` |
| `means2d` | Per-Gaussian | `ctx` saved fwd input | `[C,N,2]`/`[nnz,2]` | `_wrapper.py:919` |
| `conics` | Per-Gaussian | `ctx` saved fwd input | `[C,N,3]`/`[nnz,3]` | `_wrapper.py:920` |
| `colors` | Per-Gaussian | `ctx` saved fwd input | `[C,N,D]`/`[nnz,D]` | `_wrapper.py:921` |
| `opacities` | Per-Gaussian | `ctx` saved fwd input | `[C,N]`/`[nnz]` | `_wrapper.py:922` |
| `tile_offsets` | Per-tile | `ctx` saved fwd input | `[C,tile_h,tile_w]` | `_wrapper.py:926` |
| `flatten_ids` | Per-tile | `ctx` saved fwd input | `[n_isects]` | `_wrapper.py:927` |

---

## 6. Practical Implications for Candidate C

1. **No per-Gaussian-pixel state is stored.** Every intermediate value (`sigma`, `alpha`, `T_before`, `T_after`, `vis`) is recomputed in backward. A Candidate C implementation that stores a snapshot for each Gaussian-pixel would use **substantially more memory** than the original but could avoid recomputation.

2. **The backward shares the exact same tile traversal structure** as forward: same `tile_offsets`, same `flatten_ids`, same batching logic, same shared-memory layout (plus a `rgbs_batch` buffer).

3. **Backward adds one more shared-memory array** compared to forward: `rgbs_batch[block_size * COLOR_DIM]` (`bwd.cu:103`), because `colors[g]` must be reloaded and the backward accesses color data per-Gaussian-pixel.

4. **The transmittance reconstruction is numerically identical** (in infinite precision) to the forward pass — it's just the inverse operation. With float32, some compounding error may appear for very long sequences of near-transparent Gaussians.

5. **`last_ids` enables the workload cull**: backward skips Gaussians beyond `bin_final` (`bwd.cu:158-160`), matching the forward's early-exit at `next_T ≤ 1e-4` (`fwd.cu:152-154`). This means pixels that absorbed fully early in forward only process their contributing Gaussians in backward.
