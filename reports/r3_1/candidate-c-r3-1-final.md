# Candidate C — R3.1 Certificate Repair Final Report

**Date:** 2025-09-18
**Audit package:** `audit_packages/candidate_c_r3_1_final/`
**Pinned commit:** `32ab80e`
**Scene:** Mip-NeRF 360 `room`
**Windows:** 5K snapshot, 15K snapshot, 30K mature snapshot (30 cameras each = 90 measurements)

---

## 1. Executive Summary

R3 identified 7,494 certificate violations across 6 of 8 gradient families.
R3.1 traced the root cause to a single shared defect — **using `||conic_i||`
(geometric conic norm) as a proxy for `||c_i||` (appearance color norm)** in
all appearance-factor certificate bounds — and repaired it by substituting
the true SH-evaluated RGB color norm.

After repair, all 90 measurements across 3 windows × 30 cameras produce
**zero violations** in all 8 families, with the worst-case ratio
`max(|g|/B) = 0.9998` (margin = 0.0002), confirmed stable in float64.
The final decision is **C_KEEP** — all four criteria (CR1-CR4) pass.

---

## 2. Root Cause Analysis (R3.1-A)

### 2.1 The Conic-as-Color Proxy

In `r3_certificate_runner.py` (frozen commit `32ab80e`), the function
`_accumulate_tile_bounds` computed appearance-factor bounds using:

```python
# Line 433 (OLD — BUGGY)
conic_norm_batch = conic_batch.norm(dim=-1)  # ||conic_i||₂

# Line 445
C_max_t = conic_norm_batch.max()  # max ||conic|| in tile

# Line 451
c_norm = conic_norm_batch  # used as color magnitude proxy

# Line 462
factor_op_v = (c_norm + C_max_t_val) * tile_Q_val  # opacity factor
```

The opacity certificate bound requires `||c_i||` (the actual SH-evaluated
RGB color magnitude) and `C_max_t = max_j ||c_j||` (the maximum color
magnitude in the tile). Using `||conic_i||` instead is **not a valid
conservative upper bound** — conic norm is a geometric quantity (covariance
matrix norm) that has no monotonic relationship with color magnitude.

### 2.2 Evidence of the Proxy

| Family | Uses conic proxy? | R3 violations | R3.1 violations |
|--------|------------------|---------------|-----------------|
| opacity | YES | 260 | 0 |
| opacity_tight | YES | 5602 | 0 |
| mean2d | YES | 8 | 0 |
| mean2d_sigmamin | YES | 19 | 0 |
| conic | YES | 210 | 0 |
| conic_sigmamin | YES | 1395 | 0 |
| color_coarse | NO | 0 | 0 |
| color_tight | NO | 0 | 0 |

The perfect correlation between proxy usage and violations (6/6 families
with proxy had violations; 2/2 without proxy had zero) confirms the root
cause.

---

## 3. Certificate Repair (R3.1-B)

### 3.1 The Fix

Replaced conic norm with true SH-evaluated color norm:

```python
# R3.1 REPAIR
colors_camera_specific = spherical_harmonics(active_sh_degree, dirs, shs)
# Already computed at line 1016 for forward rendering

# In _accumulate_tile_bounds:
color_basis = colors_rgb[0, gaussian_ids]  # actual RGB colors
c_norm = color_basis.norm(dim=-1)           # ||c_i||₂ (TRUE color norm)

# C_max_t from colors, not conics
C_max_t = compute_camera_specific_C_max_t(colors_rgb, ...)
```

### 3.2 Mathematical Validity

The repaired bound uses the actual appearance quantity `c_i = SH(view_dir, shs)`
that appears in the forward pass `C = Σ T_i α_i c_i`. The gradient bound:

```
∂C/∂α_i = T_i · c_i − [Σ_{j>i} T_j α_j c_j] / (1-α_i)
```

requires `||c_i||` for the current term and `max_j ||c_j||` for the tail
upper bound. Using the true color norm provides a **legitimate conservative
upper bound**.

---

## 4. Results

### 4.1 CR1 — Correctness (Zero Violations)

| Window | color_coarse | color_tight | opacity | opacity_tight | mean2d | mean2d_sigmamin | conic | conic_sigmamin | TOTAL |
|--------|-------------|------------|---------|--------------|--------|-----------------|-------|---------------|-------|
| 5K | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 15K | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 30K | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **ALL** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |

**CR1: PASS** ✅

### 4.2 Float64 Forensics

**Pre-declared constants:** tolerance = 1e-6, borderline threshold = 1%.

Full forensics on all 90 measurements (5K+15K+30K windows):

| Metric | float32 | float64 |
|--------|---------|---------|
| Total violations | 0 | 0 |
| Worst ratio max(|g|/B) | 0.9998 | 0.9998 |
| Borderline cases | 2 | 2 |

The worst-case ratio of 0.9998 means margin = 0.0002 — the bound holds with
0.02% headroom. While extremely tight, the float32 and float64 results are
identical, confirming this is a genuine margin and not a rounding artifact.
The 2 borderline cases (ratio > 0.99 but < 1.0) are flagged for transparency
but do not constitute violations.

**Float64 verification: PASS** ✅ (90 measurements confirmed)

### 4.3 CR2 — JOINT Weighted-Work Removal

Aggregated results from all 90 measurements (3 windows × 30 cameras):

| Budget | Mean pair fraction | Mean weighted-work fraction | Mean loss-cond fraction |
|--------|-------------------|---------------------------|----------------------|
| 0.1% | 65.12% | 49.89% | 51.54% |
| 0.5% | 66.89% | 52.40% | 53.51% |
| 1% | 67.78% | 54.24% | 54.83% |
| 2% | 68.85% | 56.93% | 56.71% |
| 5% | 70.75% | 62.51% | 60.52% |

Per-window CR2 (5% budget):
- 5K window: weighted work = 68.6%, loss-cond = 65.9% → **PASS** (≥30%)
- 15K window: weighted work = 59.1%, loss-cond = 58.1% → **PASS** (≥30%)
- 30K window: weighted work = 59.9%, loss-cond = 57.5% → **PASS** (≥30%)
- Cross-window: weighted work = 62.5%, loss-cond = 60.5% → **PASS** (≥30%)

**CR2: PASS** ✅ (62.5% at 5% budget, ≥30% threshold)

### 4.4 CR3 — Signal Coverage

Windows tested: 5K, 15K, 30K (3 windows)

**CR3: PASS** ✅

### 4.5 CR4 — SPD Certifiability

SPD-disabled fraction: 0.0% (threshold: <10%)

**CR4: PASS** ✅ (all 3 windows, 0% SPD-disabled)

---

## 5. R3.1-VEC Gate (Semantic Equivalence)

200 tile-Gaussian pair samples: scalar reference vs vectorized implementation
(synthetic data, 5096 Gaussians compared across 200 tiles).

| Family | max_abs_diff | max_rel_diff | Pass |
|--------|-------------|-------------|------|
| color_coarse | 2.96e-08 | 5.90e-08 | ✅ |
| color_tight | 1.03e-08 | 3.53e-07 | ✅ |
| opacity | 1.28e-07 | 1.11e-07 | ✅ |
| opacity_tight | 6.65e-08 | 4.71e-07 | ✅ |
| mean2d | 2.09e-06 | 6.25e-06 | ✅ |
| mean2d_sigmamin | 5.38e-09 | 3.62e-07 | ✅ |
| conic | 1.13e-07 | 2.42e-07 | ✅ |
| conic_sigmamin | 2.69e-09 | 4.31e-07 | ✅ |
| **Overall** | **2.09e-06** | **6.25e-06** | **✅** |

**Pass threshold:** max_rel_diff < 1e-5

**VEC Gate: PASS** ✅

---

## 6. Final Decision

```
CR1 (correctness):     PASS ✅ (0 violations, all 8 families, all 90 measurements)
CR2 (weighted work):   PASS ✅ (62.5% at 5% budget, ≥30% threshold)
CR3 (signal coverage): PASS ✅ (3 windows: 5K, 15K, 30K, 90 iterations)
CR4 (SPD):             PASS ✅ (SPD-disabled fraction = 0.0%)
```

**Final Decision: C_KEEP (KEEP)** ✅

**R3 → R3.1 transformation:**
- R3: C_DROP (CR1 FAIL: 7,494 violations across 6 families)
- R3.1: C_KEEP (CR1 PASS: 0 violations, CR2 PASS: 62.5% weighted work)

The root cause was a single shared defect — using conic norm as a proxy for
color norm in appearance-factor certificate bounds. Replacing it with the
true SH-evaluated RGB color norm eliminated all 7,494 violations while
preserving the opportunity signal (62.5% JOINT weighted-work removal at 5%
budget).

---

## 7. Audit Package

```
audit_packages/candidate_c_r3_1_final/
├── README.md
├── SHA256SUMS.txt         (66 entries, 100% verified)
├── manifest.csv
├── NPZ_REFERENCES.md      (2 NPZ files referenced by path + SHA256)
├── analysis/              (aggregated_verification, float64_forensics, decisions)
├── archive/               (portable tar.gz, SHA256: da8858d2...)
├── checkpoints/           (checkpoint manifest)
├── docs/                  (derivation, decision protocol, common-factor audit)
├── raw/{5000,15000,30000}/  (per-snapshot certificate JSONs + logs)
└── source/                (R3.1 experiment scripts)
```

---

## 8. Key Constraints Honored

1. ✅ Correctness first, opportunity second
2. ✅ No magic constants or tolerance relaxation
3. ✅ No custom loss functions to force PASS
4. ✅ Float64 forensics with pre-declared constants
5. ✅ Frozen R3 at commit 32ab80e (not modified)
6. ✅ Decision protocol from r3_decision.py source code (CR1-CR4)
