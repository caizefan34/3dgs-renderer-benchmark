# Phase R3 — Certificate Tightness Gate (Corrected Version)

## 7 Corrections Applied

### C1 — Continuous Rectangle Quadratic Minimum
The constrained minimum of sigma(p) over a tile rectangle is **not** at a corner.
Correct algorithm:
1. If (mx, my) inside rectangle, sigma_min = 0.
2. Otherwise minimize independently on each of 4 edges (1D convex quadratic with clamping).
3. Take min of all edge evaluations. Corners are covered by clamping.
4. Non-SPD precision → CERTIFICATE_DISABLED.
See `r3_sigma_min.py` for full implementation + CPU unit tests.

### C2 — Correct Ground-Truth Granularity
Canonical autograd gives per-Gaussian aggregate gradients:
  g_i = sum_t g_{it}
NOT per-tile g_{it}.
Primary correctness test: B_i = sum_t B_it, then B_i + tol >= ||g_i||.
Per-tile comparison is NOT claimed without additional CUDA instrumentation.
Tile-Gaussian error-budget certification uses triangle inequality:
  ||sum_{(i,t) in S} g_{it}|| <= sum_{(i,t) in S} B_{it}
requiring only per-tile B_{it} (available from forward).

### C3 — Exact-Zero Tile-Gaussian Certificate
Replaced arbitrary sigma_max > 50 guard with mathematically valid condition:
  o_i * exp(-sigma_min) < 1/255
using continuous rectangle sigma_min (C1). This is a true source-semantic
certificate matching canonical raster skip logic.

### C4 — Pin Commit, Never Rebase
Removed `git rebase origin/master`. Changed to:
  git checkout <PINNED_COMMIT>
  test -z "$(git status --porcelain)"
No source movement after provenance verification.

### C5 — Checkpoint Provenance Verification
Before measurement, verify each checkpoint's format_version, model_state keys,
trainer source hash, and semantic label against Reference V1 expectations.
Do not substitute based on iteration number alone.

### C6 — Geometry-First Decision Gate
R2.1 showed geometry gating = meaningful speed; color/opacity alone = negative.
Revised gate:
  C_KEEP: zero violations + O(HW+I_tile) + geometry (mean2d/conic) >= 30% removal
          under 5% budget + signal across all windows
  C_MODIFY: geometry 10-30%
  C_DROP: geometry < 10% or invalid
Color/opacity remain secondary diagnostics only.

### C7 — CPU Preflight Tests
Before A100 execution, CPU tests MUST pass for:
  - Continuous rectangle sigma_min (vs brute-force sampling; 511 tests)
  - SPD eigenvalue handling
  - Exact-zero condition
  - Bound aggregation B_i = sum_t B_it
  - Error-budget cumulative bound accounting

---

## Objective

Measure, with canonical-source-only instrumentation, whether the **Loss-Aware
Pre-Backward Gradient Certificate** can bound raster backward gradient outputs
using only information available after forward + loss evaluation (before the
raster backward kernel runs).

The certificate claims: for each tile `t`, Gaussian `i`, and derivative family
`f ∈ {color, opacity, mean2d, conic}`, the bound `B^f_{it}` can be computed
from forward-only state, and satisfies:

```
B_i^f + tolerance >= || g_i^f_actual ||    (aggregate per-Gaussian)
||sum_{(i,t) in S} g_{it}|| <= sum_{(i,t) in S} B_{it}   (tile-Gaussian budget)
```

where `g_i^f_actual` is the per-Gaussian aggregate gradient from canonical autograd.
Per-tile `g_{it}` is NOT directly observable without a custom CUDA instrument.

## Tasks

| ID | Task |
|----|------|
| R3.1 | Capture ground-truth gradient tensors from canonical backward pass |
| R3.2 | Compute certificate bounds from forward-only state |
| R3.3 | Run correctness check: B + tolerance >= \|actual\| |
| R3.4 | Compute tightness ratios R = B / \|actual\| |
| R3.5 | Compute bound-based rankings (Section 15) |
| R3.6 | Compute certified error budgets (Section 16) |
| R3.7 | Compute tile-Gaussian certificate (Section 17) |
| R3.8 | Measure exact-zero statistics (Section 18) |
| R3.9 | Compute complexity accounting (Section 19) |

## Measurement Protocol

### Ground-Truth Capture

Use the **low-level gsplat API** (NOT the `rasterization()` high-level
wrapper) so each gradient output tensor is independently addressable:

```python
# cannonical forward (from low-level components)
proj = fully_forward_project(...)   # returns radii, means2d, depths, conics, compensation
colors = compute_sh(...)            # colors = f(W_dc, W_rest, dirs)
# raster forward
render, alpha = raster_forward(...)

# Loss
loss = ...

# Backward
loss.backward()

# Ground truth gradients:
# g_color    = colors.grad      [N, 3]
# g_opacity  = opacities.grad   [N]
# g_mean2d   = means2d.grad     [N, 2]
# g_conic    = conics.grad      [N, 3]
```

### Certificate Bounds

All bounds use only forward-time information:

- `Q_t = sum_{p in t} ||q_p||_2` where `q_p = v_render_colors[p]`
- `C_max_t = max_{j in G_t} ||c_j||_2` over Gaussians intersecting tile `t`
- `A^color_{it} = min(0.999, o_i)` (coarse) or `o_i * exp(-sigma_min(i,t))` (tight)
- `E_{it} = exp(-sigma_min(i,t))` for opacity falloff
- `lambda_min(P_i), lambda_max(P_i)` from conic precision matrix `P_i`

Bounds:

| Family | Bound formula |
|--------|--------------|
| color  | `B^color_it = A^color_{it} * Q_t` |
| opacity| `B^opacity_it = E_{it} * (||c_i||_2 + C_max_t) * Q_t` |
| mean2d | `B^mu_it = o_i * (||c_i||_2 + C_max_t) * Q_t * sqrt(lambda_max(P_i)/e)` |
| conic  | `B^P_it = o_i * (||c_i||_2 + C_max_t) * Q_t * sqrt(3/2)/(e*lambda_min(P_i))` |

### Correctness

For each visible Gaussian `i` intersecting tile `t`:

```
B^f_it + tolerance >= || g^f_actual_i ||   for all f
```

`violation_count_f` = number of (i,t) pairs with `B^f_it + tolerance < ||g_actual||`.

`violation_rate_f` = `violation_count_f / total_visible_intersections`.

### Tightness

For nonzero actual gradients:

```
R^f_i = B^f_i / || g^f_actual_i ||
```

Report median / p75 / p90 / p95 / p99 / max of `R^f` for each family and window.

### Bound-Based Ranking

Sort visible Gaussians by `B^f_i` ascending. For quantiles 20%, 32%, 50%,
75%, report:

```
retained = sum(||g^f_actual_i|| for i in top q percentile) / sum(all ||g^f_actual_i||)
```

### Certified Error Budgets

For each epsilon in {0.1%, 0.5%, 1%, 2%, 5%}:

```
sum(B^f_it for i,t in skipped) <= epsilon * sum(B^f_it for all i,t)
```

Skipped set = sorted by B^f_it ascending, taking lowest.

### Tile-Gaussian Certificate

For each family, report per-tile counts:
- `T^f_skipable(t)` = number of Gaussians in tile t with B^f_it <= epsilon*median(B^f)
- `T^f_total(t)` = number of Gaussians intersecting tile t
- fraction `T^f_skipable(t) / T^f_total(t)` per tile

The tile-Gaussian view is the **core finding**: the certificate's natural unit
of execution is `(tile, Gaussian, family)`, and the fraction of such units with
certified-zero work under a budget reveals the upper bound on speedup.

## Outputs

All outputs under `results/reference_r1/v3/r3/`:

| File | Contents |
|------|----------|
| `certificate_correctness.json` | Correctness test results |
| `certificate_tightness.json` | Tightness ratio distributions |
| `gradient_mass_coverage.json` | Bound-based ranking retained-mass curves |
| `certified_error_budgets.json` | Error budget tables |
| `tile_gaussian_certificate.json` | Per-tile × Gaussian certificate counts |
| `exact_zero_statistics.json` | Zero-gradient and alpha_max statistics |
| `complexity_accounting.json` | I_tile, I_pixel, ratios |
| `provenance.json` | Git, env, GPU, checkpoint hashes |

### Provenance

Every output file must contain:

```json
{
  "commit": "<master HEAD>",
  "git_dirty": false,
  "gsplat_version": "1.5.3",
  "gsplat_source_sha256": "<hash of rasterize_to_pixels_bwd.cu>",
  "checkpoint": "<checkpoint SHA256>",
  "camera": "<camera index>",
  "gpu": "A100",
  "R2_1_PATCH_LOADED": false,
  "C51_PATCH_LOADED": false
}
```

## Decision Criteria

| Criterion | Threshold |
|-----------|-----------|
| CR1: Certificates are correct | 0 genuine violations |
| CR2: Certificate is useful | >=1 family with >=30% certified removal @ epsilon=5% budget |
| CR3: Signal persists across windows | removal fraction >0 at 5K, 10K, 15K |
| CR4: Bounds remain finite / SPD | zero invalid certificates (lambda_min > 0 required) |

Final decision:

```
C_KEEP if CR1 and CR2 and CR3
C_MODIFY if CR1 and (CR2 or CR3)
C_DROP otherwise
```

---

## Red-Team Addendum (Pre-A100)

### R3.10 — SIGMAMIN_TIGHT Geometry Bounds

The global worst-case geometry bound factor is unnecessarily conservative for high-sigma (distant) tile-Gaussian pairs.  A simple analytic refinement yields a strict tightening.

Define `s = sigma_min(i,t)`.  The per-pair worst-case `exp(-sigma) * raster_factor(delta)` can be
solved in closed form as a function of `s` and the conic eigenvalues.

**Mean2D:**
```
M_mu(s) = 
    sqrt(lambda_max / e),                for s ≤ 0.5
    sqrt(2 * lambda_max * s) * exp(-s),  for s > 0.5

B^mu_it (sigmamin_tight) = o_i * (||c_i|| + C_max_t) * Q_t * M_mu(s)
```

The OLD global worst-case `sqrt(lambda_max / e)` always equals `M_mu(s ≤ 0.5)`.  For `s > 0.5` the
tight factor decays as `exp(-s)`, offering exponentially tighter bounds for high-sigma pairs.

**Conic:**
```
M_P(s) = (sqrt(3/2) / lambda_min) *
    (1/e),              for s ≤ 1
    s * exp(-s),        for s > 1

B^P_it (sigmamin_tight) = o_i * (||c_i|| + C_max_t) * Q_t * M_P(s)
```

Both OLD and SIGMAMIN_TIGHT bounds are reported for every iteration.  The SIGMAMIN_TIGHT variant
is used for the JOINT skip-set and gate decision because it is the tighter bound.

### R3.11 — JOINT Tile-Gaussian Skip-Set

R2.1A invalidated geometry-only gating: skipping only mean2d/conic while still computing
color and opacity saves no real work because the forward telemetry (alpha/T/buffer traversal)
is shared by all four families.

**Joint skip rule:** ONE interaction bit per (tile, Gaussian) pair suppresses ALL four
derivative families (color, opacity, mean2d, conic) simultaneously.

**Budget condition for JOINT skip:** A pair is skippable when EVERY family meets:

```
B^fam_it <= epsilon * total_B^fam
```

where `total_B^fam` = sum over all pairs of `B^fam_it`.

Greedy selection: sort pairs by max-normalized contribution (across all 4 families), skip
from smallest upward until any family's budget is exhausted.

### R3.12 — Weighted Work Fraction (W_it)

Raw pair count (`JOINT_SKIP_PAIR_FRACTION`) is misleading because a pair with one pixel lane
costs less than a pair with 16 pixel lanes.

**Definition:** `W_it` = number of pixel lanes in tile `t` for Gaussian `i` on which derivative
work executes.  This is observable from the forward pass's tile-map data (isect buckets).

**Primary gate metric:**
```
JOINT_SKIP_WEIGHTED_WORK_FRACTION 
    = sum(W_it for skipped pairs) / sum(W_it for all pairs)
```

This offline O(I_pixel) instrumentation is ONLY for evaluating the opportunity and is NOT
part of the proposed runtime method.

### R3.13 — Separate Prior-Art Exact Zero from Loss-Conditioned Culling

Exact-zero pairs (`o_i * exp(-sigma_min) < 1/255`, alpha < 1/255 per canonical raster skip)
are already removable by Speedy-Splat / AccuTile-style occlusion culling.  Candidate C must
not claim these as novelty.

**Two categories:**

| Type | Condition | Prior art? |
|------|-----------|-----------|
| `EXACT_ZERO_SUPPORT_CULLING` | `alpha_max < 1/255` | Yes (Speedy-Splat, AccuTile) |
| `LOSS_CONDITIONED_NONZERO_SUPPORT_CULLING` | Inside ordinary support but ALL 4 families meet budget | Candidate C novelty |

Both categories are reported separately for every budget epsilon.  The C_KEEP gate
requires the loss-conditioned fraction to be the primary signal.

### R3.14 — Revised C_KEEP Gate

```
C_KEEP:
  1. CR1: zero aggregate certificate violations (all 4 families, all windows)
  2. CR2: JOINT weighted work removal >= 30% under 5% budget
     where loss-conditioned culling is the primary signal
  3. CR3: signal at all canonical windows (5K, 10K, 15K)
  4. CR4: < 10% SPD-disabled
  5. O(HW + I_tile) complexity confirmed

C_MODIFY: JOINT removal 10–30%
C_DROP: JOINT removal < 10%, OR any CR violation, OR structurally too expensive
```

### R3.15 — Red-Team File Changes

| File | Change |
|------|--------|
| `r3_sigma_min.py` | Added `compute_mean2d_sigmamin_factor()`, `compute_conic_sigmamin_factor()` |
| `r3_certificate_runner.py` | `_accumulate_tile_bounds` extended: SIGMAMIN_TIGHT bounds, per-pair `pair_data` dict with `W_it`, JOINT greedy skip analysis, EXACT_ZERO/LOSS_CONDITIONED separation |
| `r3_decision.py` | Reads `tile_gaussian_certificate.json`, uses weighted work fraction as primary gate |
| `r3_joint_skip.py` | Standalone offline JOINT re-analysis from `pair_records.npz` |
| `r3_mass_coverage.py` | Added `mean2d_sigmamin`, `conic_sigmamin` families |
| `r3_analyze.py` | `aggregate_joint_skip()` plus JOINT summary printing |
| `r3_check.py` | Unchanged (no new correction codified) |
