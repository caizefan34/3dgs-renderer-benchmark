# S7 — Candidate C49: Attribute-Decoupled Backward

**Digest for summer research evidence package.**
**Scope:** C49 ("attribute-decoupled backward") — decoupling the gradients of appearance
attributes (spherical harmonics, opacity) from geometry attributes (xyz, scale, rotation)
so that low-utility derivative paths can be skipped per Gaussian. Covers the utility
concentration analysis, the signed-utility correction, the backward cost split, and the
final ~1.13% end-to-end (E2E) opportunity verdict.
**Baseline cited in all reports:** git `32ab80e` (tag `baseline/reference-v1-absgrad`),
gsplat 1.5.3 absgrad mode, scene `mipnerf360/room`, checkpoint `iter_10000` (N=780,884).
**Provenance caveat:** see §8 — the R-series reports and the C49 report are untracked
working-tree files, `32ab80e` is not resolvable in this checkout, and most R2/R2.1 JSON
deliverables are absent on disk. Numbers below are taken from the report Markdown unless a
JSON is explicitly cited.

---

## 1. C49 hypothesis: what attribute-decoupled backward is

C49 originated as a three-track Gaussian-lifecycle optimization proposal in
`reports/phase_c49_gaussian_lifecycle_research.md`:

- **Track A** — gradient-importance modeling (a multi-factor densification score).
- **Track B** — lazy / delayed densification (a validation pool for newly created Gaussians).
- **Track C** — sparse backward (skip backward computation for low-contribution Gaussians).

The lifecycle profiling in that report (Experiment 1, `results/a100/phase-c49/lifecycle_profile.json`)
killed Track B (waste ratio only **1.3%** — 5,703 of 448,871 created Gaussians were ever
pruned, all at iter 500/3000) and promoted Track C as the strongest track, because the
gradient distribution was measured to be highly concentrated (top 10% of Gaussians → ~60%
of gradient; top 32% → ~90%; stable across all 45 measurement points).

"Attribute-decoupled backward" is the refinement of Track C. Rather than a single
all-or-nothing per-Gaussian backward mask, it proposes **independent active sets per
attribute family** so that, e.g., a Gaussian can keep its SH-coefficient gradient while
dropping its geometry (xyz/scale/rotation) gradient, or vice versa. The attribute
families, as fixed in the R-series gate work (`reports/phase-r2-attribute-decoupled-backward-gate.md`,
`reports/phase-r1-optimizer-space-utility.md`), are:

| Family | Parameters | Derivative path |
|---|---|---|
| Appearance (SH) | `shs` [N, K, 3] (K=48 at SH degree 3) | `spherical_harmonics_bwd` → `v_coeffs` (dL/dSH) |
| Geometry | `xyz` [N,3], `scaling` [N,3], `rotation` [N,4] | `projection_ewa_3dgs_fused_bwd` + SH's `v_dirs` → dL/dxyz, dL/dscale, dL/drot |
| Opacity | `opacity` [N] | `v_opacities` within the raster kernel |

**What was expected to be saved:** the exclusive derivative kernels downstream of the
raster backward — chiefly `spherical_harmonics_bwd` (appearance) and
`projection_ewa_3dgs_fused_bwd` (geometry) — plus, in the later R2.1 extension, the
per-branch atomicAdd accumulations *inside* the shared raster kernel. The upstream C49
report's optimistic bound was 10–18% of `T_iter` via a CUDA sparse backward
(`reports/phase_c49_gaussian_lifecycle_research.md` §4.6), later cut to ~1% by the gate work.

---

## 2. Utility concentration analysis

### Utility definition

"Utility" was defined not as raw gradient magnitude but as the **first-order loss decrease
the Adam update actually achieves** (`reports/phase-r1-optimizer-space-utility.md`, Part 4):

- **U_step** (per-Gaussian, per-param): `‖Δθ_{i,p}‖₂` — the L2 norm of the Adam update step.
- **U_loss** (per-Gaussian, per-param): `U_loss = -g_{i,p}^T Δθ_{i,p}` — the first-order
  loss decrease from that update.
- **U_loss_total** (per-Gaussian): `U_loss_total = Σ_p U_loss_{i,p}`, summed over all
  parameter families.

The Adam update reconstructed exactly (max reconstruction error ≤ 2.33e-7 vs the real
`optimizer.step()`) is:

```
m_t = β1·m_{t-1} + (1-β1)·g_t
v_t = β2·v_{t-1} + (1-β2)·g_t²
m̂_t = m_t/(1-β1^t),  v̂_t = v_t/(1-β2^t)
Δθ = -lr · m̂_t / (√v̂_t + ε)
```

A pre-backward predictor was also defined: `S_state = lr · ‖m̂_{t-1} / (√v̂_{t-1} + ε)‖₂`,
using only stored optimizer state available before `loss.backward()`
(`reports/phase-r1-optimizer-space-utility.md`, Part 7).

### Measured concentration (uncorrected, R1)

`reports/phase-r1-optimizer-space-utility.md`, Part 6, found **U_loss is far more
concentrated than raw gradient**: Top-10% of visible Gaussians hold **82.0%** of total
`U_loss` mass (vs ~62% for raw `G_opt`), Top-50% hold ~100.8%. This super-additivity arises
because the same Gaussians dominate utility across multiple parameter families. Per-family
`U_step` concentration: SH is the most concentrated (Top-50% = 89.2%); xyz/scale/rotation
are moderately concentrated (Top-50% = 76–89%).

### Concentration after signed-utility correction (R2)

`reports/phase-r2-attribute-decoupled-backward-gate.md`, Part 1, recomputed concentration
with positive-only (`C_K+`) and absolute (`C_K^abs`) mass because of negative `U_loss`
(see §3). Corrected values:

| Family | Neg fraction | C10_pos | C50_pos | C10_abs |
|---|:--:|:--:|:--:|:--:|
| Total | 14.0% | 0.758 | 0.988 | 0.738 |
| Geometry | 15.3% | 0.724 | 0.988 | 0.709 |
| **SH** | 10.1% | **0.898** | **1.000** | 0.882 |
| Opacity | 26.5% | 0.818 | 0.998 | 0.775 |

### Top-k attribute claim

The exact "top-k attributes" claim, from `reports/phase-r2-attribute-decoupled-backward-gate.md`
Parts 10–11: **SH utility is the most concentrated family — only 20% of visible Gaussians
are needed to capture 95% of positive SH update utility; opacity needs 32%; geometry is the
least concentrated, needing 50%.** The structural separation (Geo×App Top-50% Jaccard =
0.436 < 0.5; K=20% Jaccard = 0.294, from `reports/phase-r1-optimizer-space-utility.md`
Part 10) survives the signed-utility correction, so the *basis* for attribute decoupling is
real — the question is whether it is *computationally* exploitable (§6).

**Source files:** `reports/phase-r1-optimizer-space-utility.md` (Parts 4–10),
`reports/phase-r2-attribute-decoupled-backward-gate.md` (Part 1, Parts 10–11),
`results/a100/phase-c49/lifecycle_profile.json` (Track C gradient distributions).

---

## 3. Signed-utility correction

### The problem

`reports/phase-r1-optimizer-space-utility.md`, Part 4 / Appendix C, found that
**15.4% of visible Gaussians have negative `U_loss`** (median across iterations; ranges 0%
early to ~28% mid-training). Negative `U_loss` means `g^T Δθ > 0`, i.e. the Adam update
*increases* the first-order loss estimate. This happens when the accumulated momentum `m̂`
is **anti-aligned** with the current gradient `g` (momentum reversal / adaptive-scaling
mismatch after a gradient sign flip). Because `U_loss` is signed, the **raw Gini coefficient
is invalid** as a concentration measure, and the uncorrected Top-10% mass of 82.0% is
confounded by the negative tail (R1 already flagged this, noting "Gini is unreliable for
`U_loss`").

### The correction

`reports/phase-r2-attribute-decoupled-backward-gate.md`, Part 1, replaced the invalid
signed concentration with two corrected measures:

- **Positive-only concentration `C_K+`** — the fraction of *positive* utility mass held by
  the top-k Gaussians by positive utility.
- **Absolute concentration `C_K^abs`** — using `|U_loss|`.

### Numbers after correction

Corrected Total `C10_pos = 0.758` (down from R1's uncorrected Top-10% mass of 0.820); neg
fraction 14.0%; neg/pos ratio 1.3%. Per family: SH `C10_pos = 0.898` (still the most
concentrated), Geometry `C10_pos = 0.724`, Opacity `C10_pos = 0.818`
(`reports/phase-r2-attribute-decoupled-backward-gate.md` Part 1 table).

### Consequence for C49

Because the corrected positive-utility concentration (75.8%) is *lower* than R1's
uncorrected 82.0%, and because the 14% negative fraction plus the 1.3% neg/pos ratio
introduce confounding, R2 **reclassified C49 from `C49_UPDATE_STRONGER` (R1) to
`C49_DOWNGRADE`**. C49's gradient concentration (Gini 0.75) is a gradient-distribution
observation that only *partially* translates to update utility
(`reports/phase-r2-attribute-decoupled-backward-gate.md` Part 16).

---

## 4. Geometry vs appearance backward cost split

The decisive timing was measured at `iter_10000` (N=780,884), 5 cameras, 20 warmup + 100
measured iterations, CUDA-event timing with synchronized barriers
(`reports/phase-r2-attribute-decoupled-backward-gate.md`, Parts 6–9):

| Component | Median (ms) | % of backward | % of E2E | Class |
|---|:--:|:--:|:--:|---|
| `rasterize_to_pixels_3dgs_bwd` | 11.26 | 50.4% | **41.4%** | SHARED (all 4 gradient paths in one loop) |
| `spherical_harmonics_bwd` | 0.28 | 1.3% | **1.0%** | COUPLED (SH + xyz via `v_dirs`) |
| `projection_ewa_3dgs_fused_bwd` | 0.17 | 0.8% | **0.6%** | GEOMETRY_ONLY (separable in principle) |
| Other (loss + autograd) | 10.63 | 47.6% | 39.1% | SHARED |
| **Total backward** | **22.34** | 100% | 82.1% | |
| **E2E** | **27.18** | — | 100% | |

Inside the shared raster kernel, the per-branch atomicAdd cost split
(`reports/phase-r2.1-shared-raster-attribute-gating.md` Part 8) is:

| Branch | atomicAdds | % of atomicAdd cost |
|---|:--:|:--:|
| Geometry (`v_means2d` + `v_conics`) | 5 | **~63%** |
| Appearance (`v_colors`) | 3 | ~25% |
| Opacity (`v_opacities`) | 1 | ~12% |

Geometry accumulations dominate because `v_means2d`/`v_conics` require computing `v_sigma`
(the exponential footprint derivative) before each atomicAdd, whereas `v_colors` and
`v_opacities` are simple multiplies.

---

## 5. Final verdict and the 1.13% number

### What C49 became

C49 was **scoped down and dropped as an active optimization**, not folded into C51 or C53.
The progression across the R-series:

1. **R1** (`reports/phase-r1-optimizer-space-utility.md`): `C49_UPDATE_STRONGER` and
   `Candidate B = B_KEEP` (attribute decoupling has structural basis — utility
   concentrated, Geo×App Jaccard < 0.5). Justified a cost-profiled investigation.
2. **R2** (`reports/phase-r2-attribute-decoupled-backward-gate.md`): `C49_DOWNGRADE`,
   `Candidate B = B_DROP`, `Candidate A = DROP` (SH state predictor coverage 70.3% < 80%
   threshold). Max realistic E2E opportunity = **1.13%**.
3. **R2.1** (`reports/phase-r2.1-shared-raster-attribute-gating.md`): reopened by moving
   *inside* the shared raster kernel with per-branch masks; claimed `B_ARCH_STRONG` —
   15.6% raster speedup / 7.7% E2E gain with cross-camera oracle masks.
4. **R2.1A** (`reports/phase-r2.1a-geometry-focused-oracle.md`): **invalidated R2.1**.
   With current-camera oracle masks at exactly 95% positive utility, E2E gain fell to
   **1.0%** (FULL_ATTRIBUTE_95 = +0.97% E2E; CURRENT_CAMERA_ORACLE = +1.6% E2E pooled).
   Verdict: `ARCH_INVALIDATED`, dominant design `FULL_ATTRIBUTE_REQUIRED`.

The defensible, surviving number is therefore in the **~1.0–1.13% E2E** band, with the
1.13% figure being R2's theoretical ceiling and ~1.0% being R2.1A's measured ceiling.

### Derivation of 1.13%

`reports/phase-r2-attribute-decoupled-backward-gate.md`, Part 12 ("Attribute-Decoupled
Oracle Combination"), derives 1.13% from the **exclusive downstream kernels only**, with
perfect oracle attribute masks:

- Skip SH backward for 80% of Gaussians (keep 20%): `0.28 ms × 0.8 = 0.22 ms = 0.8% E2E`
- Skip projection backward for 50% of Gaussians (keep 50%): `0.17 ms × 0.5 = 0.09 ms = 0.3% E2E`
- **Combined ≈ 1.1% E2E** (reported as **1.13%** in the Final Answer block and the
  Candidate B gate table, Part 15).

The Candidate B gate (Part 15) fails on every computational criterion: max exclusive E2E
per family = 1.0% (SH) < 5%; combined oracle = 1.1% < the 5% `B_SYSTEMS_KEEP` and the 10%
`B_STRONG_KEEP` thresholds. The shared raster kernel (41.4% E2E) cannot be
attribute-decomposed without a full CUDA rewrite, so it is not counted.

### Where it is documented

- `reports/phase-r2-attribute-decoupled-backward-gate.md` — Final Answer, Parts 9, 12, 15, 16.
- `reports/phase-r2.1a-geometry-focused-oracle.md` — Decision block + Part 7 (1.0% measured).
- `results/reference_v1/r2.1a/r21a_benchmark.json` — the only surviving JSON:
  `full_attribute_95.e2e_gain_pct = 0.9706`; `current_camera_oracle.e2e_gain_pct = 1.6083`;
  `decisions.architecture_gate = "ARCH_INVALIDATED"`.

---

## 6. Why utility sparsity did NOT imply compute sparsity

This is the central reasoning chain, drawn from `reports/phase-r2-attribute-decoupled-backward-gate.md`
(Appendix) and `reports/phase-r2.1a-geometry-focused-oracle.md` (Parts 5–6):

1. **The exclusive per-attribute kernels are tiny.** SH backward (0.28 ms) + projection
   backward (0.17 ms) together are only **1.6% of E2E**. Even with perfect oracle masks
   (skip 80% of SH, 50% of geometry), the saving is ~1.1% E2E. The Amdahl ceiling is set by
   these tiny exclusive costs, not by the 82% utility concentration.

2. **The real cost is in the shared, monolithic raster kernel.** `rasterize_to_pixels_3dgs_bwd`
   (41.4% E2E) computes all four gradient paths (`v_colors`, `v_opacities`, `v_means2d`,
   `v_conics`) in a single per-pixel, per-Gaussian loop. The transmittance `T` accumulation
   is shared and must process every Gaussian in depth order regardless of which attribute
   paths are active. The kernel cannot be attribute-decomposed without a full rewrite, and
   a rewrite into separate passes would likely *increase* overhead.

3. **SH is coupled to xyz, not attribute-exclusive.** `spherical_harmonics_bwd` computes
   *both* `v_coeffs` (dL/dSH) and `v_dirs` (dL/dxyz via `dirs = means − campos`) from the
   same SH basis evaluation (`gsplat/cuda/_wrapper.py:1831–1845`, cited in R2 Part 2).
   Skipping SH backward for a Gaussian breaks its geometry gradient. The "exclusive" SH
   kernel is therefore not truly attribute-exclusive — classed `SH_PATH_COUPLED`.

4. **Utility and compute cost are positively correlated.** Oracle masks preferentially
   keep the **highest-utility Gaussians, which also have the most pixel intersections**
   (highest compute cost). Suppressing the low-utility remainder saves disproportionately
   little time. R2.1A Part 4 quantifies this: random synthetic masks (which also suppress
   high-cost Gaussians) save more per fraction suppressed than oracle masks. Suppressing
   50% of Gaussians by utility rank saves far less time than suppressing 50% randomly.

5. **Mask mechanism overhead always dominates the cheap branches.** The per-branch mask
   design adds 3 shared-memory loads + 3 conditional branches per Gaussian — **1.72 ms
   (7.0% of raster backward)** incurred regardless of mask values
   (`results/reference_v1/r2.1a/r21a_benchmark.json`, `mask_overhead`). The appearance and
   opacity atomicAdd branches are too cheap to benefit from individual masking: R2.1 Part 4
   shows APP and OPA suppression alone produce **negative** speedups (−1.6% / −4.7%). Only
   when all three branches are suppressed (FULL_ATTRIBUTE_95) do combined savings barely
   exceed the fixed overhead (R2.1A Part 3: GEO_ONLY_95 and GEO_OPACITY_95 are both
   *slower* than FULL).

6. **Cross-camera masks understate keep fractions; current-camera masks overstate them.**
   R2.1's 15.6% raster speedup used one camera's utility to mask five cameras, so effective
   utility was only 89–93% with aggressive keep fractions (geo=50%, app=20%, opa=32%). R2.1A
   used each camera's own utility at exact 95%, requiring higher keep fractions (geo=54.2%,
   app=24.5%, opa=44.2%) — and the speedup collapsed to 1.0% E2E (R2.1A Parts 2, 6).

**Bottom line:** utility sparsity is real and survives correction, but it lives in the
wrong place — the concentrated utility is concentrated among the *high-intersection*
Gaussians whose compute cost cannot be skipped, and the derivative paths that *could* be
skipped per-attribute are too small (and partly coupled) to matter.

---

## 7. Related candidates that came out of C49

- **C51 — selective backward.** `reports/phase-r0.1-evidence-correction.md` separated the
  two gradient signals C49 had conflated: `G_dens` (densification gradient,
  `means2d.absgrad`, what R0's C49 actually measured) vs `G_opt` (per-Gaussian parameter
  gradients from backward, relevant to skipping). `G_opt` concentration is strong (Top-50%
  mass = 95.4%, Gini = 0.751, Top-10% = 62.1%) — confirming C49's concentration for the
  *optimization* gradient. But lag-1 predictability is insufficient (Top-50% Jaccard
  median = 0.648 < 0.80; Pearson median = 0.033 ≈ zero). Verdict: **`C51_MODIFY`** —
  previous-gradient masking unsupported; only instantaneous/current-iteration identification
  is viable. The CUDA sparse-backward implementation and gradient-correctness verification
  are retained as systems assets.

- **C50 — gradient temporal predictability.** R0.1 corrects the protocol (true lag-1 over
  800 consecutive pairs, not distant checkpoints). Verdict: **does not survive** — near-zero
  Pearson, Top-10% Jaccard 0.182. Topology events (clone/split) further degrade
  predictability (Pearson 0.076 → 0.040).

- **C53 — workload persistence.** R0.1 reclassifies it as **view-driven, not view-robust**:
  similar-camera Pearson 0.391 vs dissimilar 0.188. C53 discovery/validation reports live at
  `reports/phase-c53-discovery.md`, `reports/phase-c53-validation.md`,
  `reports/phase-c53-validation2.md`.

- **R2.1 / R2.1A — shared-raster attribute gating architecture.** A concrete CUDA kernel
  modification (per-branch uint8 masks in `rasterize_to_pixels_3dgs_bwd_kernel`, 7 files
  patched, `reports/phase-r2.1-shared-raster-attribute-gating.md` Appendix). Gradient
  correctness verified (enabled branches match FULL within fp32, max abs error 1.16e-10).
  The architecture is sound but the opportunity is too small to justify — retained as an
  engineering asset, not a research contribution.

- **Post-densification absgrad micro-bench** (R1 Part 15): an exact 3.3% E2E engineering
  win from switching `absgrad=False` after densification ends, with no quality loss —
  `KEEP_SYSTEMS`, a byproduct of the same investigation.

---

## 8. Contradictions and gaps

1. **The headline 1.13% vs the measured 1.0%.** The task's "~1.13% max E2E opportunity" is
   R2's *theoretical ceiling* on the exclusive downstream kernels with perfect oracle masks.
   R2.1A's *measured* ceiling with correct current-camera oracle masks is **~1.0% E2E**
   (FULL_ATTRIBUTE_95 = +0.97%; CURRENT_CAMERA_ORACLE = +1.6% pooled). The two numbers are
   consistent (both ~1%) but measure different things; the more rigorous, defensible figure
   is R2.1A's 1.0%.

2. **R1's 82% vs R2's 75.8%.** R1 reported Top-10% `U_loss` mass = 82.0% and classified
   C49 as `UPDATE_STRONGER`. R2's signed-utility correction lowered this to `C10_pos =
   0.758` and reclassified to `C49_DOWNGRADE`. Both are reported; the corrected value is
   authoritative.

3. **R2.1's 7.7% E2E vs R2.1A's 1.0%.** R2.1 claimed `B_ARCH_STRONG` (15.6% raster / 7.7%
   E2E) using cross-camera oracle masks that delivered only 89–93% utility. R2.1A
   invalidated this with current-camera masks at exact 95% utility, collapsing the gain to
   1.0%. R2.1A is the later, correcting study.

4. **Provenance is weak.** (a) `reports/phase_c49_gaussian_lifecycle_research.md`,
   `reports/phase-r0.1-evidence-correction.md`, `reports/phase-r1-optimizer-space-utility.md`,
   `reports/phase-r2-attribute-decoupled-backward-gate.md`,
   `reports/phase-r2.1-shared-raster-attribute-gating.md`, and
   `reports/phase-r2.1a-geometry-focused-oracle.md` are all **untracked** working-tree files
   (`git ls-files --error-unmatch` fails for each). (b) Commit `32ab80e`, cited as the
   baseline tag `baseline/reference-v1-absgrad` in every R-report header and in
   `results/reference_v1/r2.1a/r21a_benchmark.json` (`git_commit: "32ab80e"`), is **not
   resolvable** in this checkout (`git rev-parse --verify 32ab80e` → "fatal: Needed a single
   revision"; no matching tag exists). (c) The R2 and R2.1 result directories
   (`results/reference_v1/r2/`, `results/reference_v1/r2.1/`) are **absent** on disk; only
   `results/reference_v1/r2.1a/r21a_benchmark.json` survives. The R2/R2.1 numbers therefore
   rest on the Markdown reports, not on independently verifiable JSONs in this checkout.

5. **C49's original optimistic bound (10–18%) was never realized.** `reports/phase_c49_gaussian_lifecycle_research.md`
   §4.6 projected a 10–18% `T_iter` speedup from a CUDA sparse backward, premised on
   skipping 50% of Gaussians and crediting the C25 1.7 ms fixed-cost floor. The gate work
   showed this conflated gradient sparsity with compute sparsity and ignored the monolithic
   shared raster kernel; the realized ceiling is ~1%.

6. **G_opt vs G_dens conflation.** R0.1 (`reports/phase-r0.1-evidence-correction.md` §2.3,
   §7) explicitly flags that R0's C49 measured `G_dens` (densification gradient), not
   `G_opt` (optimization gradient). The concentration survives for `G_opt` (Gini 0.751,
   even more concentrated), but the two are correlated not identical (Pearson 0.904 on
   Gini). Claims about C49 concentration should specify which signal.

---

*End of digest.*
