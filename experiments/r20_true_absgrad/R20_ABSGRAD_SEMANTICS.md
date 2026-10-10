# R20 native AbsGrad semantics

Status: SOURCE_CONFIRMED; frozen MX numerical parity NOT_EXECUTED. Date: 2026-10-09.

Evidence is actual source, not function names. Paths below are relative to
`HIGS_SOURCE_HANDOFF_ACTUAL/source/` unless stated otherwise.

## Formula and order

`gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchBwd.cu:233-316`
computes, for a contributing Gaussian g and pixel p:

```text
delta = mean2d[g] - (pixel_x + 0.5, pixel_y + 0.5)
sigma = 0.5*(conic.x*dx^2 + conic.z*dy^2) + conic.y*dx*dy
vis = exp(-sigma)
alpha = min(MAX_ALPHA, opacity*vis)
ra = 1 / max(MIN_ONE_MINUS_ALPHA, 1-alpha)
T_i = running_T * ra
v_alpha = sum_channels((color_g*T_i - buffer*ra)*v_render_color)
          + T_final*ra*v_render_alpha
          - T_final*ra*dot(background, v_render_color) [if background exists]
v_sigma = -opacity*vis*v_alpha
g_pixel = v_sigma * (conic.x*dx + conic.y*dy,
                     conic.y*dx + conic.z*dy)
signed[g] = sum_pixels g_pixel
absgrad[g] = (sum_pixels abs(g_pixel.x), sum_pixels abs(g_pixel.y))
```

The derivatives of conic/position/opacity are zero when `opacity*vis > MAX_ALPHA`.
Invalid samples (`sigma < 0` or alpha below ALPHA_THRESHOLD), pixels outside the
image, and intersections after `last_ids[p]` contribute zero. Clamped equality
uses `<=`, exactly as the native source. Final transmittance and last IDs come
from forward; Gaussian traversal is back to front. The color-channel sum and
alpha/background contribution happen BEFORE the componentwise absolute value.

For each Gaussian: pixel local gradient -> componentwise absolute value ->
warpSum on separate signed and absolute vec2 values -> lane-0 atomic scatter.
Native uses `atomicAdd_system` for both absolute components. HiGS uses device
`atomicAdd`; its outputs are local CUDA tensors on the guarded device/current
stream, consumed after stream ordering. Neither scope makes FP32 sums deterministic.

`sum_p ||g_pixel||_2`, `||sum_p g_pixel||_2`, and
`||(sum_p |gx|, sum_p |gy|)||_2` are different signals. R19's claim that
`abs(signed)` is a faithful proxy is false: opposite contributions can cancel.
The native tensor has TWO components, not a scalar `|gx|+|gy|` accumulator.

## Layout, dtype, loss scaling

Native dense means2d/absgrad are FP32 `[..., C, N, 2]`; packed rasterization uses
`[nnz, 2]`. R20 targets the actual HiGS dense, single-scene-batch path (`B=1`):
internal projected rows `[I*N_visible, 2]`, `I=C`; Python master proxy and public
`.absgrad` `[C, N_master, 2]`, with exact zeros for invisible master rows.
No reduction across cameras is done in the CUDA absolute output.

Native CUDA consumes the upstream `v_render_colors` and `v_render_alphas` as
supplied. It does not divide by pixel count, camera count or resolution. A mean
loss therefore already supplies its factor through these cotangents. AMP loss
scaling would also scale this side channel; ordinary GradScaler parameter
unscaling does not automatically unscale a tensor attribute. R20 tests use FP32
without AMP, identical cotangents, and no extra normalization inside CUDA.

Positions are in PIXEL coordinates, as confirmed by pixel centers and delta.
`baseline/reference_v1/gaussian_model.py:260-294` clones the consumed gradient,
multiplies x by W/2 and y by H/2, then takes a vector norm and accumulates it for
visible rows. Mathematically this converts derivatives with respect to pixel
coordinates to derivatives with respect to normalized [-1,1] screen coordinates;
the trainer's comment describing the opposite direction is misleading.
`public_baselines/gsplat/gsplat/strategy/default.py:243-273` additionally multiplies
by camera count; that is a DIFFERENT consumer, not a factor to add in R20 CUDA.
The R19 reference trainer uses one camera, and its existing consumer/thresholds
are unchanged by the R20 adapter.

## Native attribute contract

`gsplat/cuda/_wrapper.py:1390-1408,1880-1950` attaches a forward holder and copies
the native backward absolute output into it. R20 exposes `.absgrad` on the
densification proxy after backward. The side channel does not replace `.grad`,
does not enter projection VJP, and has no autograd graph. Repeated backward
calls overwrite the side channel with that call's value, like native holder copy.

## Verified HiGS omission

Actual `HigsNativeBackward.cu:242-265,489-513` passes `compute_abs=false` to the
shared blend helper and aliases the unused absolute output to the signed scratch.
The PX variant accumulates signed pixel scratch across q before warpSum.
R20 therefore takes `fabsf(v_xy_q)` BEFORE q accumulation; simply taking abs at
warp output, or after PX sums, would preserve the R19 failure mechanism.

The exported native implementation contains the formula inline. The export is
missing the shared helper header, so its MX hash must be captured by preparation;
the local public header was inspected as supporting evidence only. R20 leaves
that helper and its `compute_abs=false` calls unchanged and derives absolute
components from the existing per-pixel output, avoiding an assumed helper ABI.
