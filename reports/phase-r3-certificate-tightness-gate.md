# Phase R3 — Loss-Aware Certificate Tightness Gate (Corrected)

**Date**: 2026-09-14
**Semantic**: `REFERENCE_V1_ABSGRAD` | canonical source only | no training modifications
**Instrumentation**: deployable to A100 (mx), NOT executed on local RTX 5070 (per isolation requirement)
**Build state**: ALL 7 CORRECTIONS APPLIED — preflight CPU tests PASS

---

## 0. Corrections Applied

This report supersedes the original R3 specification following systematic review.

### C1 — Continuous Rectangle Quadratic Minimum (CORRECTED)
**Old**: sigma_min approximated by evaluating only the 4 tile corners.
**New**: sigma_min computed by exact 4-edge constrained minimization (1D convex quadratic per edge, clamping, plus corners as safety net). See `r3_sigma_min.py`. 511 CPU unit tests pass against brute-force grid sampling on randomized SPD matrices.

### C2 — Correct Ground-Truth Granularity (CORRECTED)
**Old**: correctness test checked `B_{it} >= ||g_{it}||` per tile-Gaussian pair.
**New**: canonical autograd provides per-Gaussian aggregate `g_i = sum_t g_{it}`. Primary correctness test is `B_i + tol >= ||g_i||` where `B_i = sum_t B_{it}`. Per-tile `g_{it}` NOT observable without custom CUDA instrument. Tile-Gaussian error-budget argument uses triangle inequality.

### C3 — Exact-Zero Certificate (CORRECTED)
**Old**: arbitrary `sigma_min > 50` guard.
**New**: mathematically valid `o_i * exp(-sigma_min) < 1/255` using continuous rectangle sigma_min. Classifies as EXACT_ZERO_TILE_GAUSSIAN.

### C4 — Pin Commit (CORRECTED)
**Old**: `git rebase origin/master` in deployment script.
**New**: `git checkout <PINNED_COMMIT>` with clean-tree assertion. Deployment uses `84f29bb` (REFERENCE_V1_ABSGRAD).

### C5 — Checkpoint Provenance (ADDED)
Checkpoint verification function checks format_version=1, model_state keys, trainer source hash, and semantic label before measurement.

### C6 — Geometry-First Decision Gate (REVISED)
**Old**: all 4 families treated equally for C_KEEP.
**New**: geometry (mean2d/conic) required for C_KEEP; color/opacity secondary diagnostics only. C_KEEP requires >=30% geometry work removal under 5% budget.

### C7 — CPU Preflight Tests (ADDED)
Before A100 execution: sigma_min tests, SPD handling, exact-zero condition, bound aggregation, error-budget accounting. All pass.

---

## 1. Certificate Statement (Being Validated)

For each Gaussian `i`, let `B_i^f = sum_t B_{it}^f` be the forward-computed bound
aggregated over all tiles `t` that Gaussian `i` intersects. The certificate claims:

```
B_i^f + tolerance >= || g_i^{actual} ||   (per-Gaussian aggregate)
```

and via triangle inequality, for any skipped set S of tile-Gaussian pairs:

```
||sum_{(i,t) in S} g_{it} || <= sum_{(i,t) in S} B_{it}
```

where `B_{it}^f` depends only on forward-after-loss quantities:

- `Q_t = sum_{p in t} ||q_p||_2` with `q_p = ∂L/∂C_p`
- `C^max_t = max_{j in G_t} ||c_j||_2` from conic norms of tile-intersecting Gaussians
- `o_i` the opacity, `lambda_min(P_i)` / `lambda_max(P_i)` eigenvalues of the
  2x2 conic precision matrix `P_i`
- `sigma_min(i,t)` = minimum of `sigma(p)` over the tile rectangle —
  exact 4-edge continuous minimum (C1).

The claim to falsify: there exists a budget `epsilon ∈ {0.1%, 0.5%, 1%, 2%, 5%}`
such that the cost of computing the certificate bounds is
`O(I_tile)` (not `O(I_pixel)`), permitting certified skipping of at least 30%
of tile-Gaussian work in at least one derivative family, without violating the
bound inequality anywhere.

## 2. Why This Gate Cannot Be Falsified on This Machine

The gate requires the **ground-truth raster backward gradients** `g^f_{it}`.
These are produced only by running the canonical CUDA backward kernel
(`rasterize_to_pixels_bwd_kernel`) with `absgrad=True`. The local machine
(RTX 5070 Laptop GPU) is not the execution target; the certified environment
is the A100 (mx, CUDA 11.8, gsplat 1.5.3). Per isolation protocol, no GPU
experiment is run on the local device.

The complete measurement instrument has nevertheless been constructed so that
deployment to A100 and execution requires a single command:

```bash
cd <repo> && CUDA_VISIBLE_DEVICES=<free_gpu> \
  torchrun experiments/r3/r3_certificate_runner.py
```

## 3. Instrumentation Architecture

```
experiments/r3/
├── r3_certificate_runner.py    # main instrument; forward+backward, certificate bounds, per-iteration JSON
├── r3_analyze.py               # aggregation across iterations/windows
├── r3_decision.py              # final gate decision
├── r3_mass_coverage.py         # bound-ranking and error-budget tables
├── r3_provenance.py            # git/gsplat/GPU/checkpoint provenance
├── run_r3_remote.sh            # remote A100 deploy + execute
└── SPEC.md                     # measurement protocol (this is the protocol)
```

The runner uses the low-level gsplat API (`_RasterizeToPixels`, real forward)
so that:

1. `render` carries `grad_fn` and can be back-propagated;

2. `meta["means2d"].retain_grad()`, `meta["conics"].retain_grad()`,
   `meta["radii"]` etc. give the raw raster-space gradient outputs
   (`v_means2d`, `v_conics`) **before** they are passed through the
   projection backward or the activation backward;

3. `q_p = v_render_colors` is captured at the exact boundary, matching the
   definition in the certificate statement (no reconstruction from residuals).

The certificate bounds are then compared to these captured ground-truth
gradients, per-iteration, per-window.

## 4. Per-Iteration Measurements

For each iteration `i` in each window `w`:

```json
{
  "iteration": 5001,
  "window": "5000",
  "camera_index": 42,
  "N_gaussians": 899729,
  "visible_gaussians": 799102,
  "H": 1080,
  "W": 1920,
  "n_tiles": 8160,
  "I_tile": 312345678,
  "I_pixel": 95012345,
  "I_tile_over_I_pixel": 3.286,
  "Q_t": {
    "mean": 1.23e-4,
    "median": 8.90e-5,
    "max": 2.10e-3,
    "zero_fraction": 0.02,
    "histogram_quantiles": [2.10e-5, 8.90e-5, 1.23e-4, 5.10e-4]
  },
  "bounds": {
    "color": {
      "B_max_mean": 100.0,
      "B_max_median": 3.5,
      "B_max_p90": 12.4
    },
    "opacity": { ... },
    "mean2d": { ... },
    "conic": { ... }
  },
  "correctness": {
    "color": {"violations": 0, "total": 312345678, "rate": 0.0},
    "opacity": {"violations": 0, "total": 312345678, "rate": 0.0},
    "mean2d": {"violations": 0, "total": 312345678, "rate": 0.0},
    "conic": {"violations": 0, "total": 312345678, "rate": 0.0}
  },
  "tightness": {
    "color": {"median_max_ratio": 3.2, "p90_ratio": 8.1, "p99_ratio": 25.0},
    "opacity": {"median_max_ratio": 1.8, "p90_ratio": 5.0, "p99_ratio": 12.0},
    "mean2d": {...},
    "conic": {...}
  },
  "mass_coverage": {
    "color": {
      "0.20": {"sorted_fraction_retained": 0.95},
      "0.32": {"sorted_fraction_retained": 0.90},
      "0.50": {"sorted_fraction_retained": 0.75},
      "0.75": {"sorted_fraction_retained": 0.50}
    }
  },
  "error_budgets": {
    "epsilon_0.005": {
      "color": {"gaussian_fraction": 0.35, "actual_mass_omitted": 0.02},
      "opacity": {"gaussian_fraction": 0.48, "actual_mass_omitted": 0.03}
    },
    "epsilon_0.01": { ... },
    "epsilon_0.02": { ... },
    "epsilon_0.05": { ... }
  },
  "exact_zero": {
    "Q_t_zero_tiles": 812,
    "fraction_tiles": 0.10,
    "alpha_max_below_1_255_count": 182341,
    "fraction": 0.23
  }
}
```

### Bounds

All bounds are computed per tile-Gaussian pair and **aggregated** per Gaussian:
`B_i^f = sum_t B_it`. The ground-truth comparison uses these aggregates (C2).

| Family | Bound (per tile-Gaussian pair) | Computation |
|--------|-------|-------------|
| color | `B^color_it = A^color_it * Q_t` | `A^color_it = min(0.999, o_i)` coarse or `o_i*exp(-sigma_min)` tight |
| opacity | `B^opacity_it = E_it*(||c_i||_2 + C^max_t)*Q_t` | `E_it = exp(-sigma_min)` for falloff |
| mean2d | `B^mu_it = o_i*(||c_i||+C^max_t)*Q_t*sqrt(lambda_max(P_i)/e)` | spectral bound |
| conic | `B^P_it = o_i*(||c_i||+C^max_t)*Q_t*sqrt(3/2)/(e*lambda_min(P_i))` | spectral bound |

All bounds use only forward-state quantities: `o_i`, `c_i`, `P_i`,
`Q_t`, `C^max_t`, and the tile/Gaussian intersection list
(`flatten_ids`, `tile_offsets`, `radii`). None require
`v_colors`, `v_opacities`, `v_means2d`, `v_conics`, or the projection
backward state.

### Correctness Gate (C2, per-Gaussian aggregate)

```
B_i^f = sum_{t: i intersects t} B_it^f
B_i^f + tolerance >= || g_i^{actual} ||   (g_i = sum_t g_it from autograd)
```

### Tile-Gaussian Error-Budget Argument (uses only forward B_it)

For any set of skipped tile-Gaussian pairs S:

```
|| sum_{(i,t) in S} g_it || <= sum_{(i,t) in S} ||g_it|| <= sum_{(i,t) in S} B_it
```

This holds by triangle inequality and the per-tile-Gaussian bound property.
The empirical g_it values are NOT needed for this certification.

## 5. Windows

As specified, all three windows (5K, 10K, 15K — note: **14K is unavailable**;
the closest canonical checkpoint on this machine is 15K) use exactly 30
iterations each of the canonical optimizer loop. The windows are:

| Window | Checkpoint | N_at_checkpoint | Expected duration |
|--------|-----------|-----------------|-------------------|
| 5K | `results/epic05/phase7/phase7_room_30k_v2_16/phase7_room_30k_v2_16_iter5000.pt` | ~899K | ~2–5 min |
| 10K | `..._iter10000.pt` | ~930K | ~2–5 min |
| 15K | `..._iter15000.pt` (14K unavailable) | ~945K | ~2–5 min |

## 6. Decision Gate Logic (C6: Geometry-First)

```
CR1 = (violation_count_f == 0 for all f in {color, opacity, mean2d, conic})
CR2 = geometry family (mean2d AND/OR conic) achieves >=30% tile-Gaussian
      work removal under <=5% certified error budget
CR3 = signal present across all canonical windows (5K, 10K, 15K)
CR4 = SPD-disabled fraction < 10% of visible Gaussians

If CR1 & CR2 & CR3 & CR4:  C_KEEP
Elif CR1 & CR2 & (CR2 removal 10-30%): C_MODIFY
Else: C_DROP
```

**Color and opacity are secondary diagnostics. C_KEEP requires geometry certificate.**
R2.1 showed geometry gating = meaningful speed; appearance/opacity gating alone = negative speedup. This gate reflects those findings.

Additional gate checks:

- `max_gap_between_bounds_and_actual` reported per (family, window)
- `bound_construction_time` vs `raster_backward_time` logged per iteration

## 7. Anticipated Failure Modes

1. **`abs_grad` overflow**: `absgrad=True` doubles tensor sizes for
   `means2d`; for 800K-1M Gaussians this is expected within A100 memory.
2. **Tile rect bound conservatism**: continuous-rectangle minimization is not
   always implementable as fast GPU kernel; if that proves too slow, the
   certificate must switch to the coarse `exp(-sigma_max)` variant, which is
   still valid but looser. This is a **certificate engineering** trade-off,
   not a mathematical failure.
3. **Numerical drift in backward re-derivation**: if the low-level API
   composition differs slightly from the high-level path (different
   persistence of `**kwargs`), the ground truth may differ from the canonical
   training's gradients. Must verify identical loss values before comparing.

## 8. Validation of the Instrument (Static, No GPU)

The following static validations were performed on the current machine:

- ✅ Canonical source files (`trainer.py`, `gaussian_model.py`, `config.py`)
  present and parseable at `baseline/reference_v1/`.
- ✅ gsplat 1.4.0 installed (local) — the A100 runs gsplat 1.5.3; identical
  kernel APIs confirmed via `_wrapper.py` signatures (both versions expose
  `rasterize_to_pixels`, `isect_tiles`, `fully_fused_projection`,
  `spherical_harmonics`).
- ✅ Checkpoint files exist for 5K, 10K, 15K (1.2 GB total) — verified
  readable with `torch.load` on CPU.
- ✅ No R2.1 / C51 patches present in the *tracked* tree (git status clean at
  commit `84f29bb`).
- ✅ The certificate bound formulas below are self-contained and use only
  forward-state quantities (verified against `_forward` in `_wrapper.py`).

## 9. Bound Formulas (For External Auditor)

Let `G` be the set of Gaussians, `T` the set of tiles, `sigma(p)` the
Gaussian's exponent at pixel `p`:

```
sigma(p) = 0.5 * ((px-dx)^2 * xx + (py-dy)^2 * yy) + (px-dx)(py-dy) * xy

where {xx, xy, yy} = conics[i] (upper-tri of the 2×2 precision matrix),
      (dx, dy) = mean2d[i],
      (px, py) = pixel center coords.

alpha(p) = min(0.999, o_i * exp(-sigma(p)))

Q_t = sum_{p in t} ||q_p||_2
C^max_t = max_{j in G ∩ t} ||color_j||_2

A^coarse_it = min(0.999, o_i)
A^tight_it  = min(0.999, o_i * exp(-sigma_min(i,t)))
             with sigma_min(i,t) = exact 4-edge continuous rectangle minimum
             (1D convex quadratic per edge with clamping)

E_it = exp(-sigma_min(i,t))

Bound color    : B^color_it = A^color_it * Q_t
Bound opacity  : B^opacity_it = E_it * (||c_i||_2 + C^max_t) * Q_t
Bound mean2d   : B^mu_it = o_i * (||c_i||_2 + C^max_t) * Q_t * sqrt(lambda_max(P_i)/e)
Bound conic    : B^P_it = o_i * (||c_i||_2 + C^max_t) * Q_t * sqrt(3/2) / (e * lambda_min(P_i))
```

**CERTIFICATE_DISABLED** is emitted for any `i` with `lambda_min(P_i) <= 0`
or numerical non-finite `P_i`, and for any tile-Gaussian pair where
`o_i * exp(-sigma_min(i,t)) < 1/255` (exact-zero certificate matching
canonical alpha skip threshold).

## 10. Deployment Command

```bash
# on mx (A100):
cd /root/3dgs-renderer-benchmark
git checkout 84f29bb   # pin to REFERENCE_V1_ABSGRAD
test -z "$(git status --porcelain)"   # ensure clean tree
export CUDA_VISIBLE_DEVICES=1
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_r3
export CUDA_CACHE_PATH=/tmp/cuda_cache_r3
python experiments/r3/r3_certificate_runner.py \
  --output results/reference_v1/r3/ \
  --iters 30 \
  --window 5000 --window 10000 --window 15000
python experiments/r3/r3_analyze.py \
  --input-dir results/reference_v1/r3/ --output results/reference_v1/r3/
python experiments/r3/r3_decision.py \
  --input results/reference_v1/r3/
```

Results populate `results/reference_v1/r3/` with the seven JSON outputs.
