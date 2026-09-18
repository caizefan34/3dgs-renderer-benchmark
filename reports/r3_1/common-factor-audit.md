# R3.1-A — Common Appearance-Factor Audit

**Date:** 2026-09-17
**Source file:** `experiments/r3/r3_certificate_runner.py`
**Frozen commit:** `32ab80e773f74f4d8e40ff4e338c86b29c7957f5`
**Source SHA256:** `98eec016ef6437d465e5fd0700c33890edf57b4417072c4d747ac8c4315da0c2`

---

## 1. Finding

The certificate runner uses **`||conic_i||_2` (the L2 norm of the 2D conic /
covariance parameter) as a proxy for `||color_i||_2` (the L2 norm of the
SH-evaluated per-Gaussian RGB)** in **six of eight** certificate families.

This is explicitly documented in the source code (lines 221–228):

```python
def compute_C_max_t(conics, tile_offsets, flatten_ids, tile_h, tile_w):
    """
    C_max_t = max_{j in G_t} ||c_j||_2 over Gaussian conic norms per tile.

    NOTE: This uses conic norms as proxy for Gaussian color norms.
    The proper C_max_t would use ||color_j|| from SH output, but the certificate
    derivation holds for any norm-bounded quantity. We record which norm is used.
    """
```

The claim that "the certificate derivation holds for any norm-bounded quantity"
is **false** when the certificate bounds a gradient that contains `c_i`
(the color) but the proxy uses `||conic_i||` (an unrelated geometric quantity).
The conic norm has **no upper-bound relationship** to the color norm; they live
in completely different spaces.

---

## 2. Mechanism

Inside `_accumulate_tile_bounds` (line 361), for each tile `t`:

```python
# Line 433: conic norm computed as the "color" proxy
conic_norm_batch = conic_batch.norm(dim=-1)          # ||conic_i||_2

# Line 445: C_max_t = max of conic norms in this tile
C_max_t_val = float(conic_norm_batch.max().item())

# Line 451: alias
c_norm = conic_norm_batch                             # "c" stands for color, but is conic

# Line 462: common opacity/geometry factor
factor_op_v = (c_norm + C_max_t_val) * tile_Q_val     # (||conic_i|| + max||conic||) * Q_t

# Line 490: geometry base factor
base_geo_s = spd_o * (spd_c + C_max_t_val) * tile_Q_val
#            = α_i * (||conic_i|| + max||conic||) * Q_t
```

The forward pass (line 1016) **does** compute the real per-Gaussian color:

```python
colors_rgb = spherical_harmonics(model.active_sh_degree, dirs, shs.unsqueeze(0))
#  shape [1, G, 3] — the true c_i
```

But `colors_rgb` is **never passed** to `_accumulate_tile_bounds` (line 1076):

```python
bounds = _accumulate_tile_bounds(
    opacities.detach(), conics.detach(), means2d.detach(),
    # ^^^ colors_rgb.detach() is MISSING
    tile_offsets, flatten_ids, Q_t, tile_h, tile_w, tile_size,
    H=H, W=W
)
```

---

## 3. Per-family audit

| Family | Math bound (intended) | Code expression (line) | Uses `||conic||` as color proxy? | Violations (R3) |
|---|---|---|---|---|
| **color_coarse** | `B = clamp(α_i, 0.999) · Q_t` | `A_coarse_v * tile_Q_val` (458) | **No** — uses opacity only | **0** |
| **color_tight** | `B = clamp(α_i · e^{-σ_min}, 0.999) · Q_t` | `A_tight_v * tile_Q_val` (459) | **No** — uses opacity only | **0** |
| **opacity** | `B = (‖c_i‖ + C_max_t) · Q_t` | `(c_norm + C_max_t_val) * tile_Q_val` (462–463) | **YES** — `c_norm = ‖conic_i‖` | **260** |
| **opacity_tight** | `B = e^{-σ_min} · (‖c_i‖ + C_max_t) · Q_t` | `E_tight_v * (c_norm + C_max_t_val) * tile_Q_val` (464) | **YES** — `c_norm = ‖conic_i‖` | **5602** |
| **mean2d** | `B = α_i · (‖c_i‖ + C_max_t) · Q_t · √λ_max · √(1/e)` | `base_geo_s * sqrt(lam_max) * SQRT_E_INV` (495) | **YES** — `spd_c = ‖conic_i‖` | **8** |
| **mean2d_sigmamin** | `B = α_i · (‖c_i‖ + C_max_t) · Q_t · μ(s,λ_max)` | `bg_s * mmu` (526) | **YES** — `spd_c = ‖conic_i‖` | **19** |
| **conic** | `B = α_i · (‖c_i‖ + C_max_t) · Q_t · √1.5/e / λ_min` | `base_geo_s * const_conic_factor / lam_min` (496) | **YES** — `spd_c = ‖conic_i‖` | **210** |
| **conic_sigmamin** | `B = α_i · (‖c_i‖ + C_max_t) · Q_t · m_p(s,λ_min)` | `bg_s * mp` (527) | **YES** — `spd_c = ‖conic_i‖` | **1395** |

---

## 4. Correlation with violation distribution

```
Uses conic-as-color-proxy:  YES  YES  YES  YES  YES  YES  │  NO   NO
Family:                     op  op_t mean2d m2d_ conic conic │ col_c col_t
                                 ight  sigmin       sigmin │ oarse ight
Violations:                 260 5602  8    19   210  1395 │   0     0
```

**Every family that uses the conic-as-color proxy has non-zero violations.
Every family that does NOT use the proxy has zero violations.**

This is a perfect binary partition — strong evidence that the conic-as-color
proxy is the dominant root cause of the 7,494 violations.

---

## 5. Why conic norm cannot bound color norm

The conic parameter `(Σ_xx, Σ_xy, Σ_yy)` describes the 2D spatial spread of
a Gaussian splat.  The color `c_i ∈ ℝ³` is the SH-evaluated RGB at the current
viewpoint.  These are **independent quantities**:

* A Gaussian can have a tiny conic (sharp splat) but a large color (bright RGB).
* A Gaussian can have a huge conic (diffuse splat) but a near-zero color (dark).

There is no inequality of the form `‖c_i‖ ≤ f(‖conic_i‖)` that holds for all
Gaussians.  Therefore, using `‖conic_i‖` as a proxy for `‖c_i‖` in an upper
bound produces a bound that is **sometimes smaller than the true gradient
magnitude** — which is exactly what a "violation" means.

---

## 6. Required fix (R3.1-B)

Replace `conic_norm_batch` with the true color norm from the forward pass:

```python
# BEFORE (buggy):
conic_norm_batch = conic_batch.norm(dim=-1)
c_norm = conic_norm_batch
C_max_t_val = float(conic_norm_batch.max().item())

# AFTER (repaired):
color_norm_batch = colors_rgb[0, g_unique].norm(dim=-1)   # ||c_i||_2 from SH
c_norm = color_norm_batch
C_max_t_val = float(color_norm_batch.max().item())
```

This requires passing `colors_rgb` (already computed at line 1016) into
`_accumulate_tile_bounds` as a new argument.

Additionally, `compute_C_max_t` (line 221) must be updated to accept `colors`
instead of `conics`, or a parallel `compute_color_C_max_t` must be created.

---

## 7. Summary

| Question | Answer |
|---|---|
| conic_as_color_proxy_present | **YES** |
| families_affected | opacity, opacity_tight, mean2d, mean2d_sigmamin, conic, conic_sigmamin |
| families_unaffected | color_coarse, color_tight (use opacity only, no conic proxy) |
| derivation_issue_confirmed | **YES** — the proxy has no mathematical upper-bound guarantee |
| correlation_with_violations | Perfect: all 6 proxy-using families violate; 2 non-proxy families pass |
