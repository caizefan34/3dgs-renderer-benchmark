# S1A-R1 Variance Erratum — HT Gradient Estimator Formula Audit

**Round**: S1A-R1 (continuation of S1A). **Node**: MX `bms-39468022-001`, A100 (cuda:6), Torch 2.9.1+cu128, CUDA 12.8. **Authorization**: S1A-R1 only; no S1B start, no new training, no CUDA rebuild, no env reinstall.

---

## 1. The Error (OBSERVED)

The original S1A-1 script (`s1a_gradient_value.py:127`) computed the analytic HT gradient variance as:

```
Var_wrong(g_hat) = sum_j [(1 - p_j) / p_j^2] ||g_j||^2      # WRONG
```

**Correct formula** for independent Bernoulli tile inclusion with a fixed additive objective:

```
E||g_hat - g||^2 = sum_j [(1 - p_j) / p_j] ||g_j||^2          # CORRECT
```

The original carried an extra `1/p_j` factor. Derivation: with `g_hat = sum_j (z_j/p_j) g_j`, `z_j ~ Ber(p_j)`,
`Var((z_j/p_j) g_j) = (1/p_j^2) Var(z_j) ||g_j||^2 = (1/p_j^2) p_j(1-p_j) ||g_j||^2 = (1-p_j)/p_j ||g_j||^2`.

The wrong `(1-p)/p^2` over-weights low-probability tiles. For uniform `p_j=p` it is a constant `1/p` factor (so U-vs-U ratios were preserved), but for non-uniform E/G/O it **distorts the strategy ranking**.

## 2. Validation Against Monte Carlo (OBSERVED)

Cached per-tile gradients (bicycle 5K, cam 0, 64 tiles, N=120,000). MC variance (300 trials, means group) vs analytic:

| p | Strategy | Analytic (wrong `(1-p)/p²`) | Analytic (correct `(1-p)/p`) | MC variance |
|---|---|---|---|---|
| 0.25 | U | 13.84·‖g‖² | 3.46·‖g‖² | 7.61e-3 |
| 0.25 | G | 8.57·‖g‖² | 1.70·‖g‖² | 3.76e-3 |
| 0.40 | U | 4.32·‖g‖² | 1.73·‖g‖² | 3.82e-3 |
| 0.40 | G | 2.72·‖g‖² | 0.69·‖g‖² | 1.53e-3 |

The **correct** formula matches MC to 3 significant figures (U: 7.627e-3 vs 7.609e-3; G: 3.758e-3 vs 3.764e-3). The wrong formula does NOT. **Math-correctness audit: PASS.**

## 3. Corrected NMSE Results (DERIVED)

Per-tile gradient concentration + correct analytic variance. Means-group NMSE = Var/‖g_means‖².

| p | Strategy | NMSE (wrong) | NMSE (correct) | Δ vs U (correct) |
|---|---|---|---|---|
| 0.25 | U | 13.84 | 3.46 | — |
| 0.25 | E | 23.51 | 4.41 | **−27% (worse)** |
| 0.25 | G | 8.57 | 1.70 | **+50.7%** |
| 0.25 | O | 8.57 | 1.70 | **+50.7%** |
| 0.40 | U | 4.32 | 1.73 | — |
| 0.40 | E | 7.88 | 2.32 | **−34% (worse)** |
| 0.40 | G | 2.72 | 0.69 | **+59.9%** |
| 0.40 | O | 2.72 | 0.69 | **+59.9%** |

## 4. Does the Oracle Advantage Still Hold? (CONCLUSION)

**YES — and it is STRONGER with the correct formula.** G/O beat U by **+50.7% (p=0.25)** and **+59.9% (p=0.40)** with the correct formula, versus +38%/+37% with the wrong formula. The wrong formula **understated** the advantage (because the extra `1/p_j` factor penalizes the concentrated high-`p_j` allocations that G/O use).

## 5. Is There a Real Tile-Dependent Cost? (CONCLUSION)

**NO.** O (cost-aware, `p_j ∝ ‖g_j‖/√c_j`) is **identical** to G (gradient-only, `p_j ∝ ‖g_j‖`) in the single-camera audit, and in the 2-state×2-camera Phase 3 run O never beats G (mean O-vs-G = −2.6%). Reason: the S1A-2 cost calibration shows **84% of GPU cost is fixed overhead** (projection + sort setup + memory allocation), so the per-tile `√c_j` term is negligible and provides no differentiation. **The cost-aware oracle adds no advantage over pure gradient-magnitude sampling at this scale.**

## 6. Preserved Old Data

Original wrong-formula results are preserved in `raw/S1A_GRADIENT_VALUE.json` (S1A round) and `S1A_REPORT.md`. This erratum does **not** overwrite them. Corrected results are in `S1A_R1_VARIANCE_ERRATUM.json` and `S1A_R1_GRADIENT_VALUE.json`.

## 7. Labels

- **OBSERVED**: formula error location, MC-vs-analytic validation, corrected NMSE table.
- **DERIVED**: cost model (84% fixed overhead), O==G identity, gradient concentration.
- **CONCLUSION**: correct formula confirmed; Oracle advantage holds and is larger; no real tile-dependent cost benefit; cost-awareness adds nothing over gradient-only.
