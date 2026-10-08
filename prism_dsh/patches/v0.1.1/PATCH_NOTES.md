# PRISM-GS adapter patch v0.1.0 -> v0.1.1

Status: versioned interface/numerics fix for the M0 adapter gate. **No threshold
was changed** (see "Unchanged" below). The v0.1.0 tree and its raw BLOCKED_M0
output are preserved unmodified at `/root/epic05_prism/prism_gs/`.

## Observed failures (v0.1.0, bicycle 5K snapshot, cameras 163/28/6)

Raw output: `prism_gs/results/bicycle_5k_m0.json` (status BLOCKED_M0).

| sub-check | cam0 | cam1 | cam2 | fixed tolerance |
|---|---|---|---|---|
| forward re-render (`value_error`) | 0.0 | 0.0 | 0.0 | <= 2e-6 |
| zero displacement (`zero_error`) | 4.644e-4 | 1.40e-6 | 2.98e-7 | <= 2e-6 |
| quat->cov render (`cov_error`) | 6.204e-3 | 8.78e-3 | 7.74e-3 | <= 2e-5 |
| SH gradients (`sh0`/`shN` rel_l2) | 0.345 | 0.246 | 0.382 | <= 1e-3 |
| projection uv / shape | 1.69e-4 / 9.35e-7 | pass | pass | <= 2e-3 |

Diagnosis (all evidence in `outputs/diag_m0_v6b/v7/v7c/v7e_*.json`):

* The two fused-kernel branches (quats and covars) share all downstream code
  and the same device conversion function; build provenance shows source==build.
  No route-level kernel difference exists.
* The quats route's internal covariance is computed from `F.normalize(quats)`;
  the v0.1.0 adapter computed its comparison covariance from **raw** quats
  (snapshot norms 0.45-2.51). The two differ by ~1 ULP; the rasterizer amplifies
  ~1e-4 relative covariance changes into ~2e-3 image changes (measured sweep
  d=0: 6.2043e-3, d=1e-4: 8.52e-3, d=1e-3: 1.53e-2). Therefore no independent
  fp32 re-implementation can meet an image tolerance of 2e-5; only the
  renderer's own value transport can.
* The zero-displacement gap came from a conic->cov->log-Cholesky->cov->conic
  roundtrip inside `features()`/`draw()` (culled-row analysis shows visible-row
  means2d/depths/opacities are bitwise identical across routes; only the conic
  re-parameterization changed values).
* The SH gradient mismatch is the activation boundary: the kernel's gradient
  mask is strict (`> 0` at exactly 0), `clamp_min` is inclusive at 0.

## Changes

P1. `prism/geometry.py`, `prism/gsplat_backend.py`, `prism/oracle.py`,
   `tests/test_geometry.py`: the 5-DOF screen feature is now the renderer's
   native conic `[u,v,conic_a,conic_b,conic_c]`. `features()` returns the
   kernel's own conic values unchanged; `draw()` writes them back unchanged and
   recomputes radii through the same explicit 2x2 inverse the kernel uses
   (glm::inverse elementwise arithmetic). `project()` emits the same conic form
   (`cov_to_conic`), so `features()` and `project()` remain directly comparable.
   The log-Cholesky screen re-parameterization and its roundtrip are removed.

P2. `prism/gsplat_backend.py`: `colors_for` activation `clamp_min(0)` -> `relu()`
   to match the kernel's strict `> 0` gradient mask. Forward values are
   identical (max(x,0) in both cases); only the derivative at exactly 0 changes.

P3. `prism/geometry.py`: `covariance()` delegates on CUDA to the pinned
   renderer's own `quat_scale_to_covar_preci` with the same normalized
   quaternions the renderer receives, then mirrors the returned triu 6-vector
   into a full 3x3. The adapter no longer maintains a parallel fp32 conversion
   that cannot be bitwise-equal to the renderer's. The pure-torch expression
   remains as the CPU fallback (CPU has no renderer to match).

Test-side consequence of P1: `tests/test_geometry.py` keeps the same seven
checks; only line 17's involution assertion is expressed through the new pair
(`conic_to_cov(cov_to_conic(cov)) == cov`) instead of the removed log-Cholesky
roundtrip.

Version marker: `prism/__init__.py` `__version__ = "0.1.1"`.

## Observed results after the patch (same inputs, same thresholds)

E1 pre-check (`outputs/diag_m0_v7e_bicycle_5000.json`, pristine bundle, 3 cameras):
covars route with the renderer's own conversion is **bitwise-identical** to the
quats route on all three cameras (gap = 0.0; 100% of visible conics bitwise
equal; 0 radii diffs), while raw-quat conversions reproduce the recorded gaps
(c_raw_3: 6.204e-3/8.471e-3/7.736e-3; c_v0: 6.204e-3/8.782e-3/7.736e-3).
In-process reproduction of the v0.1.0 M0 numbers matched the recorded raw JSON
to all printed digits (zero: 4.644e-4/1.401e-6/2.980e-7; cov: 6.204e-3/8.782e-3/7.736e-3).

M0 rerun (`prism_gs_v0.1.1/results/bicycle_5k_m0.json`, adapter sha 5cc041e0...):

| sub-check | cam0 | cam1 | cam2 | tolerance |
|---|---|---|---|---|
| forward / repeat / zero / cov | 0.0 | 0.0 | 0.0 | 2e-6 / 2e-5 |
| proj uv / shape | 1.685e-4 / 4.92e-7 | 1.206e-4 / 3.89e-7 | 1.536e-4 / 1.84e-6 | 2e-3 |
| sh0/shN rel_l2 | 3.9e-7 / 3.9e-7 | 5.0e-7 / 5.1e-7 | 3.4e-7 / 3.4e-7 | 1e-3 |
| quats / log_scales rel_l2 | 1.73e-4 / 2.95e-5 | 3.24e-4 / 2.32e-5 | 3.85e-4 / 4.99e-5 | 1e-3 |

All three cameras `PASS_ADAPTER`. CPU: 7/7 tests pass; `cpu_probe` audit deltas
(PRISM 49.5% vs direct Adam 55.6% relative reduction) — the recorded CPU negative
result is unchanged in direction and is preserved in the delivery.

## Unchanged (explicitly)

* All M0/FIXED tolerances in `prism/run_gpu.py` (`2e-6`, `2e-5`, `2e-3`, `1e-3`).
* `prism/run_gpu.py` and `prism/cpu_probe.py` source (no threshold, camera rule,
  selection, or arm changes; the screen-arm scale vector stays `[1,1,.1,.1,.1]`).
* `PREREGISTRATION.json`, `DSH_EXECUTION.md`, `MANIFEST.json` (v0.1.0 record).
* Nothing under `/root/epic05_p8/` (renderer), `/root/epic05_grc3d/` (inputs) or
  `/root/epic05_p12/` (trainer/camera sequences) was touched.

## Rerun

```bash
cd /root/epic05_prism/prism_gs_v0.1.1
PRISM_GPU=2 /root/miniforge3/envs/gsplat/bin/python -m unittest discover -s tests -v
PRISM_GPU=2 /root/miniforge3/envs/gsplat/bin/python -m prism.run_gpu \
  --gx-path /root/epic05_grc3d/scripts --gpu 2 --scene bicycle \
  --snapshot /root/epic05_grc3d/inputs/snapshots/snap_bicycle_005000.pt \
  --camera-seq /root/epic05_p12/runs/p12a/p12a_gmainf_s42_bicycle/camera_sequence.npy \
  --m0-only --out results/bicycle_5k_m0.json
```

Note on the covars API: `gsplat.rasterization(covars=...)` accepts only
`[..., N, 3, 3]`; the Python wrapper itself converts it to the kernel's triu
6-vector. Passing `[N, 6]` silently mis-indexes to `[6]` and aborts.
