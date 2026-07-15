# Benchmark Protocol v1.0

## Status and Scope

Protocol v1.0 defines the minimum conditions for a comparable 3D Gaussian
Splatting renderer result. A result is comparable only when its scene,
checkpoint, camera manifest, resolution, warmup count, measured-frame count,
repeat count, quality reference, and software environment are identical.

Synthetic stress results characterize renderer throughput and memory. They do
not establish reconstruction quality. Quality-bearing claims require paired
reference images whose provenance and hashes are recorded in the result JSON.

## Required Environment

Set both variables before importing PyTorch or any CUDA extension:

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
```

On PowerShell, use:

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
$env:OMP_NUM_THREADS = "1"
```

`CUDA_VISIBLE_DEVICES` must expose exactly one physical GPU to prevent
accidental multi-GPU placement and device-index ambiguity. The JSON report
must record the physical GPU model, driver version, PyTorch version, CUDA
runtime version, and renderer commit. `OMP_NUM_THREADS=1` prevents variable
CPU thread-pool scheduling from contaminating end-to-end latency. Runs using a
different thread count must record the value and form a separate cohort.

## Spherical Camera Trajectory

For (N) views, zero-based view index (i), base scene radius (R), and the
scene center at the origin, define:

\[
\theta_i = \frac{2\pi i}{N}, \qquad
\phi_i = \frac{\pi}{12}\sin(2\theta_i), \qquad
r_i = 0.8R + 0.2\sin(3\theta_i).
\]

The camera center is:

\[
\mathbf{c}_i =
\begin{bmatrix}
r_i\cos(\theta_i)\cos(\phi_i) \\
r_i\sin(\theta_i)\cos(\phi_i) \\
r_i\sin(\phi_i) + 0.5
\end{bmatrix}.
\]

Each camera uses world up ((0,0,1)), looks at the scene center, and uses the
positive camera-space (z) axis as its forward direction. The implementation
must reject a manifest for which the scene center has camera-space depth less
than or equal to zero. Protocol defaults are (R=5), horizontal field of view
(60^\circ), near plane 0.01, and far plane 100. The manifest stores float32
view, projection, and intrinsic matrices so every adapter receives identical
poses.

The committed `circle` preset is a separate planar diagnostic trajectory. A
result using that preset must identify it as `circle`, not `spherical-v1`.

## Resolution and Adapter Output

Resolution is fixed within one run but may differ between runs. The adapter
receives positive integer `height` and `width` fields. It must return float32
tensors on one device with these shapes:

| Field | Shape | Meaning |
| --- | --- | --- |
| `rgb` | `[H, W, 3]` | RGB renderer output constrained to `[0, 1]` |
| `depth` | `[H, W, 1]` | Camera-space depth |
| `alpha` | `[H, W, 1]` | Accumulated opacity constrained to `[0, 1]` |

All values must be finite. The benchmark calls `render_checked` before using
an adapter result for timing or quality evaluation.

## Timing Protocol

For each repeat, execute (W) warmup frames and discard them. Then measure
(M) frames in manifest order. Protocol defaults are (W=50), (M=200),
and three repeats. Synchronize after warmup and around each measured frame.
GPU kernel latency uses CUDA events. End-to-end latency uses a monotonic host
clock and is reported separately.

Concatenate the measured samples from all repeats in acquisition order:

\[
T = [t_{1,1}, \ldots, t_{1,M}, t_{2,1}, \ldots, t_{K,M}].
\]

P99 is the 99th percentile of this single fixed array. It is not a sliding
window statistic and is not averaged from per-repeat percentiles. Protocol
v1.0 uses NumPy's linear percentile interpolation: if
(h=(n-1)\times0.99), interpolate between sorted samples at
(\lfloor h\rfloor) and (\lceil h\rceil). Raw ordered samples must be
retained in JSON so the value can be recomputed.

Peak GPU memory is the maximum allocated process memory observed after reset
and across all repeats. Clear references, run Python garbage collection, and
call `torch.cuda.empty_cache()` between renderer test cases. Cache clearing is
isolation policy; it is never included in frame latency.

## Quality Gate

Compute PSNR, SSIM, and LPIPS for every paired view, then apply thresholds to
the arithmetic mean of each metric. Acceptance is inclusive:

\[
\overline{\mathrm{PSNR}} \ge P_{\min}, \qquad
\overline{\mathrm{SSIM}} \ge S_{\min}, \qquad
\overline{\mathrm{LPIPS}} \le L_{\max}.
\]

A failure of any condition rejects the speed result from the quality-gated
leaderboard. The raw speed result may remain in a synthetic or diagnostic
table when clearly labeled.

## Reproducibility Gate

Run the same synthetic checkpoint and camera twice without changing state.
For each run, compute PSNR against the same fixed reference image. The absolute
difference between the two PSNR values must be less than 0.01 dB. Record random
seeds, deterministic-algorithm settings, and any backend that cannot guarantee
determinism.
