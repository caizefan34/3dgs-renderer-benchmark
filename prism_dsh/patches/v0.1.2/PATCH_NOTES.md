# PRISM-GS v0.1.2 patch notes

Parent: v0.1.1 (`PATCH_v0.1.1.diff`, `MANIFEST.v0.1.1.json` in this directory).
Base: the delivered v0.1.0 bundle (MANIFEST.json sha256 `6acd04e9…` zip).

## Observed failure that motivates v0.1.2 (raw record preserved)

Section-4 sweep under v0.1.1, 12 pre-registered units
(`{bicycle,garden} x {5K,14K} x seeds 20260927/28/29`, GPU 2):

| unit | result |
|---|---|
| bicycle 5K, seeds 27/28/29 | `PROBE_COMPLETE_NO_GO_VERDICT` (exit 0) |
| bicycle 14K, seeds 27/28/29 | `PROBE_COMPLETE_NO_GO_VERDICT` (exit 0) |
| garden 5K seed 27 | `PROBE_COMPLETE_NO_GO_VERDICT` (exit 0) |
| garden 5K seeds 28/29 | `ERROR` (exit 1) |
| garden 14K seeds 27/28/29 | `ERROR` (exit 1) |

Error text (5 units, same site each time):

```
File ".../prism/run_gpu.py", line 145, in main
    x0=pack_geometry(params['means'][ids],cov[ids])
File ".../prism/geometry.py", line 47, in pack_geometry
    l = torch.linalg.cholesky(cov)
torch._C._LinAlgError: linalg.cholesky: (Batch element 53): The factorization
could not be completed because the input is not positive-definite (the leading
minor of order 3 is not positive-definite).
```

The failing rows are not supplied by PRISM: `cov` comes from the pinned
renderer's own `quat_scale_to_covar_preci` on the same `F.normalize`d
quaternions the renderer receives (v0.1.1 P3). Measured with the same bootstrap
(`gx_common.boot_gmain`, GPU 2; `outputs/diag_pd_garden_*.json`):

| snapshot | N | non-PD rows | fraction | min eigenvalue of bad rows |
|---|---|---|---|---|
| garden 5K | 1,813,591 | 259 | 1.43e-4 | -1.16e-9 |
| garden 14K | 2,922,031 | 8,094 | 2.77e-3 | -1.23e-9 |

Bad-row statistics: `max_symmetry_error = 0.0`; log-scales down to -19.7
(scale ~2.7e-9, so `s_i*s_j` underflows towards 1e-18); raw quaternion norms
0.43–2.85 (no degenerate quaternion); the negative eigenvalue is tiny in
absolute terms (`|λ| ~ 1e-9`) while the row's largest eigenvalue can be as large
as 0.19. The kernel keeps returning exactly symmetric matrices, so this is
fp32 cancellation in the renderer's own covariance convention, not an
asymmetry or an adapter arithmetic error.

Consequence for the mechanism, stated plainly: the rasterizer never needs a
Cholesky factor, so the pinned renderer renders these rows without complaint;
only PRISM's 9-DOF diagnostic coordinate (`pack_geometry`) requires a positive
definite Σ. Sub-pixel-scale rows are also *enriched* in the common-visible
eligible set, because `radii = ceil(extend * sqrt(Σ2d_00))` saturates at 1 for
a vanishing ellipse, so they are never culled by the visibility rule.

## P4 — `pack_geometry` positive-definiteness repair

`prism/geometry.py`:

* symmetrize `Σ` (`(Σ+Σᵀ)/2`) before factorization;
* `torch.linalg.cholesky_ex`; rows reporting `info != 0` are retried with a
  **trace-relative diagonal jitter** `j * mean(diag Σ) * I` for
  `j ∈ {1e-6, 1e-4, 1e-2, 1e-1}`, escalating only for rows that still fail;
* a row that survives all four steps raises `RuntimeError` (never silently
  returns a non-finite coordinate);
* `stats=` (optional out-parameter) reports `rows`, `repaired_rows`,
  `max_relative_jitter`; `run_gpu.py` writes these to the probe JSON under
  `geometry_pack` so the repair is visible in every unit's raw output.

Properties verified locally (torch 2.13, CPU) and on the remote (torch 2.9.1):

* rows that are already positive definite take the untouched path — their
  packed 9-DOF vector is **bitwise identical** to the plain-Cholesky result
  (`test` output: `bitwise equal to plain cholesky: True`);
* a synthetic row with the recorded garden signature
  (`eig = [-5.0e-11, 2.0e-7, 4.0e-7]`) is repaired with `j = 1e-2`, changing
  `Σ` by at most 2.0e-9 (absolute) while the other 63 rows stay bitwise
  unchanged;
* the repaired coordinate reconstructs a symmetric `L Lᵀ` and finite
  parameters.

## Unchanged (explicitly)

* every M0 gate value and tolerance in `prism/run_gpu.py`
  (`2e-6`, `2e-6`, `2e-5`, `2e-3`, `2e-3`, `1e-3`, `1e-8`) and in
  `PREREGISTRATION.json`; the 5-camera rule, the seed list, the
  `max_active=512` cap, the arm definitions and the screen-arm scale vector
  `[1,1,.1,.1,.1]`;
* v0.1.1 P1/P2/P3 (native conic transport, strict-relu SH activation,
  renderer-delegated covariance);
* `MANIFEST.json`, `PREREGISTRATION.json`, `DSH_EXECUTION.md` of the base
  bundle; nothing under `/root/epic05_p8/`, `/root/epic05_grc3d/` or
  `/root/epic05_p12/` is touched;
* the v0.1.1 Section-4 outputs (including the five `ERROR` units) are kept
  byte-for-byte at `/root/epic05_prism/prism_gs_v0.1.1/results/`; v0.1.2 runs
  in a separate tree `/root/epic05_prism/prism_gs_v0.1.2/`;
* the four original Section-2 parity JSONs are restored byte-for-byte after the
  version-bound v0.1.2 parity run (sha256 verified), with the v0.1.2 outputs
  stored as `parity_*_v012.json`.

## Rerun (v0.1.2)

```bash
cd /root/epic05_prism/prism_gs_v0.1.2
/root/miniforge3/envs/gsplat/bin/python -m unittest discover -s tests -v
/root/miniforge3/envs/gsplat/bin/python -m prism.cpu_probe --out results/cpu_probe.json
PRISM_GPU=2 /root/miniforge3/envs/gsplat/bin/python -m prism.run_gpu \
  --gx-path /root/epic05_grc3d/scripts --gpu 2 --scene bicycle \
  --snapshot /root/epic05_grc3d/inputs/snapshots/snap_bicycle_005000.pt \
  --camera-seq /root/epic05_p12/runs/p12a/p12a_gmainf_s42_bicycle/camera_sequence.npy \
  --m0-only --out results/bicycle_5k_m0.json
bash /root/epic05_prism/scripts/run_section4_sweep_v012.sh   # 12 units
# Section-2 re-check bound to this tree (writes parity_*_v012.json):
bash /root/epic05_prism/scripts/v012_core.sh
```

## Reference: covars API (unchanged from v0.1.1)

`gsplat.rasterization(covars=…)` accepts only `[..., N, 3, 3]`; the Python
wrapper always applies `tri_indices` to build the 6-vector, so passing an
`[N, 6]` array is silently mis-indexed to shape `[6]` and raises. PRISM passes
the mirrored full matrix. `quat_scale_to_covar_preci(..., triu=True)` returns
the `[00,01,02,11,12,22]` layout; `triu=False` returns the full matrix.
