# WV-HIER-R1-MX — Stage A Report
## Hierarchical Work-Constrained Sampling (HWCS): Scientific Feasibility & Cost Modeling

**Round:** R1-MX · **Stage:** A (analytic + simulation only) · **Date:** 2026-10-08
**Evidence package:** `WV0_RAW_AUDIT_20261008_evidence.zip` (SHA256 `CB4181FE…6AF42`, 2798 entries, 8/8 priority files + CUDA binary verified)
**Prior verdict (WV0-ORACLE0-MX):** `BLOCKED_RAW_EVIDENCE / STRUCTURAL_ORACLE_ONLY` — **NOT reopened. No old GO restored.**

> **CORRECTION NOTICE 20261008:** This report predates the controlling Stage B protocol
> `WV_HIER_R1_MX_STAGE_B_REVISED_20261008`. Two framings here are superseded:
> (1) the child-refine efficiency `κ` is a **cost-model operation-cost parameter, NOT
> identifiable from intersection counts** — the "break-even κ\*" and "κ ≤ 0.5" statements are
> cost-model break-evens, not measurable thresholds, and are **not** a CUDA GO gate;
> (2) all Native-G4 floor / work-saving / segment claims are **N1 (dense-bin-then-mask) +
> assumed-cost-model conditional only** — they do not hold for N2 (mask-then-bin) and are
> intersection-count proxies, not GPU time. The variance-ordering `G4≤HWCS≤G16` is
> **scene-dependent** (signed within-parent cross-terms can be negative); only the convex-blend
> identity is universal. See `STAGE_A_CORRECTIONS.md` and the corrected `STAGE_A_VERDICT.json`.
> The verdict remains `CONDITIONAL_RESEARCH`.

---

## 0. Evidence access statement (read first)

| Item | Status |
|---|---|
| MX live filesystem `/mnt/storage_pool/...` | **NOT REACHABLE** from this host (absent). |
| Uploaded evidence zip | **PRESENT, VERIFIED** — all analysis below uses it. |
| This host | Windows laptop, **RTX 5070 Laptop (Blackwell, sm_120)**, 8 GiB, **no torch/CUDA Python env**. |
| MX target GPU | 8× **A100-PCIE-40GB** (sm_80); WV0 `gsplat_cuda.so` built for sm_80. |
| Checkpoints | Reside on MX; **not present locally**. |

**Consequence:** Stage A is executed **entirely from the verified static evidence package** (read-only science audit). **No GPU compute, no CUDA build, no training, no environment mutation were performed this round** (0 GPU-second, 0 CUDA edit, 0 training step — consistent with the frozen-evidence constraint and the TEMPORAL_COVERAGE-R1 priority). Anything requiring the MX A100 is **specified, not run** (see `STAGE_B_SPEC.md`).

---

## 1. Question tested

> When a 16×16 **parent** tile's Gaussian list and sort order are shared, and sampling is done **inside** 4×4 **child** tiles (HWCS), can we obtain **better gradient-estimation quality than G16 at lower real GPU work** — and specifically, can it undercut the high cost of **Native G4**?

This is the exact mechanism hypothesis the WV0 audit left open (`WV0_HIERARCHICAL_COST_MODEL.md`: "共享父列表有减少 native G4 重复 bin/sort 的机制依据，尚无证据证明总收益").

## 2. Hypothesis tested

**H-HWCS:** Two-stage hierarchical sampling (parent active w.p. `q`; child active w.p. `r` given parent; marginal child inclusion `p = q·r`; HT weight `1/p`) produces an **unbiased** gradient estimator whose (work, NMSE) operating point lies **below** the G16↔Native-G4 trade-off segment — i.e., it dominates both pure-granularity baselines at matched absolute structural cost.

## 3. What would falsify it

1. The hierarchical variance is **not** a convex blend of G4/G16 variance (i.e., the within-parent covariance term is mishandled). → *Tested: Monte Carlo (§5). NOT falsified.*
2. HWCS's total structural work is **≥** Native G4's minimum (dense-binning) work for realistic child-refine cost. → *Tested: break-even κ* (§7). HWCS-1 **undercuts the floor unconditionally**; HWCS-2 only conditionally.
3. At matched work, a **G16/G4 workload mixture** achieves lower NMSE than HWCS. → *Tested: segment sweep* (§8). HWCS loses to the mixture **only** when child-refine is fully conservative (κ=1.0).
4. The realized (measured) GPU-time advantage is absent. → **BLOCKED** (no Native-G4 or HWCS CUDA exists; all timing `NOT_MEASURED`).

## 4. Mechanism derivation (the math the task requires)

Per gradient component, let child `t` carry contribution `a_t`, parent `T` carry `b_T = Σ_{t∈T} a_t`, dense `g = Σ_t a_t = Σ_T b_T`.

**HWCS estimator:** parent active `M_T ~ Bern(q)`; child active `m_{t|T} ~ Bern(r)`; child indicator `ξ_t = M_T · m_{t|T}`; marginal `P(ξ_t=1) = q·r = p`; HT weight `1/p`; `ĝ = Σ_t (ξ_t/p)·a_t`.

**Unbiasedness:** `E[ĝ] = Σ_t (q·r/p)·a_t = Σ_t a_t = g`. ✔ (HT/IPW; holds even with boundary parents because per-pixel inclusion stays exactly `p`.)

**Joint inclusion (two children, same parent):** `E[ξ_t ξ_t'] = P(parent active)·P(both children | active) = q·r²`. Covariance `Cov(ξ_t,ξ_t') = q·r² − (qr)² = p·r·(1−q)`. This is the **explicit within-parent covariance term** the task mandates — HWCS is **not** independent-Bernoulli.

**Exact variance (derived, then MC-verified):**
```
Var(ĝ)_HWCS = (1−p)/p · S1  +  r(1−q)/p · (S2 − S1)
            = (1−α)·Var_G4 + α·Var_G16 ,
α = r(1−q)/(1−p),  S1 = Σ_t a_t²,  S2 = Σ_T b_T²
```
- `Var_G4 = (1−p)/p·S1` (iid Bernoulli over children), `Var_G16 = (1−p)/p·S2` (iid over parents).
- **HWCS variance is an exact convex blend** of Native-G4 and G16 variance. The covariance term is `(S2−S1) = Σ_T Σ_{t≠t'} a_t a_t'`.
- For same-sign local contributions (realistic for 3DGS residuals inside one parent), `S2 ≥ S1`, hence `Var_G4 ≤ Var_HWCS ≤ Var_G16`.
- **Endpoints:** `q=p, r=1 ⇒ HWCS = G16`; `q=1, r=p ⇒ HWCS = Native G4`. HWCS is a one-parameter interpolation between the two baselines in **both** variance and (§7) work.

**HWCS-1 (q=.5, r=.5):** α = 1/3 → `Var = (2/3)Var_G4 + (1/3)Var_G16`.
**HWCS-2 (q=.75, r=1/3):** α = 1/9 → `Var = (8/9)Var_G4 + (1/9)Var_G16`.

## 5. Monte Carlo validation (`stage_a_variance.py`)

Synthetic scene (40 parents × 16 children, parent-biased same-sign contributions), 200k draws:

| Config | α | Var ANALYTIC | Var MONTECARLO | rel err |
|---|---:|---:|---:|---:|
| HWCS-1 | 0.3333 | 9909.90 | 9931.37 | 0.22% |
| HWCS-2 | 0.1111 | 4487.63 | 4473.31 | 0.32% |
| (G4 ref) | — | 1776.49 | 1785.63 | 0.52% |
| (G16 ref) | — | 26176.71 | 26107.72 | 0.26% |

Blend identity `compact ≡ expanded` holds to 0.00e+00. **The hierarchical variance derivation — including the covariance term — is correct.**

## 6. NMSE prediction (DERIVED — never measured)

Because `NMSE = Var/‖g‖²` and the blend is linear, at matched marginal `p`:
```
NMSE_HWCS(p) = (1−α)·NMSE_G4(p) + α·NMSE_G16(p)
```
This is computed from **OBSERVED** G16/G4 NMSE (8-camera geometry macro). No raw HWCS gradient exists; these are **model predictions with an uncertainty band**, not measurements.

| state | ρ=I_G4/I_G16 | NMSE G16 | NMSE HWCS-2 | NMSE HWCS-1 | NMSE G4 |
|---|---:|---:|---:|---:|---:|
| bicycle__iter15000 | 6.01 | 1.9651 | 1.2445 | 1.4247 | 1.1544 |
| bicycle__iter30000 | 5.85 | 1.8116 | 1.1277 | 1.2986 | 1.0422 |
| bicycle__iter5000 | 7.96 | 1.8545 | 0.9655 | 1.1877 | 0.8543 |
| garden__iter15000 | 4.83 | 2.3535 | 1.7208 | 1.8789 | 1.6417 |
| garden__iter30000 | 4.83 | 1.9028 | 1.4074 | 1.5312 | 1.3455 |
| garden__iter5000 | 4.66 | 2.4629 | 1.7040 | 1.8937 | 1.6091 |
| room__iter15000 | 7.96 | 1.8124 | 1.0114 | 1.2117 | 0.9113 |
| room__iter30000 | 8.18 | 1.4503 | 0.7702 | 0.9402 | 0.6851 |
| room__iter5000 | 9.24 | 1.8001 | 0.8640 | 1.0980 | 0.7470 |

**Ordering G4 ≤ HWCS-2 ≤ HWCS-1 ≤ G16 holds in 9/9 states.** HWCS beats G16 on gradient quality at matched `p` (as the original WV0 GO claimed), but sits **between** G4 and G16 — it does **not** beat Native G4 on quality at matched `p`.

## 7. Absolute cost model (DERIVED structural proxy — NOT measured CUDA time)

Work units = `I_G16` dense parent intersections; raster weight `Γ=1`. `ρ = I_G4/I_G16 ∈ [4.66, 9.24]` (OBSERVED bbox proxies, 9/9 states > 4).
```
G16       : 1 + Γp                       (dense parent bin+sort)
Native G4 : ρ + Γp·ρ                     (dense child bin+sort)   ← work FLOOR = ρ
HWCS(κ)   : 1 + [p·ρ + κ(q−p)·ρ] + Γp·ρ  (parent bin+sort + child refine in active parents)
κ = child-refine efficiency ∈ [0,1]:  κ=1 refine ALL children in active parents (=q·ρ);  κ=0 touch ONLY selected children (=p·ρ, free parent-order filter)
```
HWCS's **raster** equals Native G4's (same marginal `p`); the entire difference is binning/sorting structure.

**Work saving vs Native G4 @ p=.25 (conservative κ=1):**
- **HWCS-1: 27.1% mean (22.8–31.3%)** · HWCS-2: 7.1% mean (2.8–11.3%).

**Native-G4 work-floor undercut (robust result):** HWCS-1 total work `1 + 0.75ρ` < G4 floor `ρ` iff `ρ > 4`. **All 9 states have ρ>4 ⇒ HWCS-1 undercuts Native G4's minimum possible work for ANY κ∈[0,1]** (break-even κ\* = 1.14–1.57 > 1). HWCS-2 (κ\* = 0.57–0.78) undercuts the floor only if κ < ~0.6–0.8.

## 8. Equal-cost dominance vs a G16/G4 workload mixture

The decisive fairness test: at fixed work, does HWCS sit **below** the straight segment joining G16 (low work, high NMSE) and Native G4 (high work, low NMSE)? Below ⇒ genuine advantage a pure-granularity split cannot reach; Above ⇒ dominated by a simple mixture.

Sweeping κ (states BELOW/9):

| κ (child-refine) | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---:|---:|---:|---:|---:|
| HWCS-1 BELOW | 9 | 9 | 9 | 9 | **4** |
| HWCS-2 BELOW | 9 | 9 | 9 | 9 | **4** |

**Reading:** if child-refine costs ≤ ~75% of the conservative estimate (κ ≤ 0.75), HWCS is below the mixture in **9/9** states for both configs. Only at fully-conservative κ=1.0 does it flip to dominated in 5/9. **The verdict hinges on the single unmeasured variable κ.**

**Audit reproduction (equal absolute proxy cost, G4 vs G16):** ratio **3.84–5.17** (audit reported 3.84–6.35; same low end, per-state ρ slightly compresses the high end). Confirms the audit's finite-model: at matched cost, pure G4 beats pure G16 — but that is a **pure-granularity** statement and does **not** by itself rule out a **hybrid** (HWCS) advantage.

## 9. Matched-work comparison vs Native G4 (the directly relevant baseline)

| Method | Result @ matched structural work (conservative) |
|---|---|
| **HWCS-1** | Below G4's work floor in **9/9** states — Native G4 cannot match its work at any `p`. |
| **HWCS-2** | Lower NMSE than G4-at-matched-work in **9/9** states (G4 must drop to low `p` to match, inflating NMSE). |

So **in the structural model, both HWCS configs undercut Native G4 at matched absolute cost** — answering the task's core question affirmatively **at the proxy level**.

## 10. Pareto re-check (OBSERVED/RECOMPUTED)

Fixed `p`: 72/72 points non-dominated (finer granularity ⇒ both lower NMSE and higher proxy cost). Free `p`: G4-p=.25 is dominated by cheaper configs in **9/9** states (`WV0_PARETO_RECHECK.csv`). HWCS inserts a **new intermediate point** between G16 and G4 that the original 4-granularity sweep did not have — its value is precisely that it may sit below the G16↔G4 chord (§8), which the original design could not test.

## 11. What is NOT established (honest BLOCKED list)

1. **Real HWCS NMSE** — predicted via blend, never measured.
2. **κ (child-refine efficiency)** — the decisive variable; the entire GO direction depends on it.
3. **GPU-time realization** — no Native-G4 or HWCS CUDA exists; all `kernel_ms`/`F+B_ms` = `NOT_MEASURED`; bbox proxy uses floor/closed/clip(tw−1), CUDA uses ceil/half-open/clip(tw).
4. **Native-G4 baseline design** — dense-binning (N1, HWCS-favorable) vs mask-then-bin (N2, HWCS-neutral). If the field's "Native G4" is N2, HWCS has **no** structural edge.
5. **Time-to-quality / training impact** — frozen-gradient NMSE ≠ convergence; break-even `steps_B2/steps_B0 < time_B0/time_B2` has no time ratio to test.
6. **Camera heterogeneity** — cost axis = camera 0, quality axis = 8-camera mean (mismatched).
7. **Statistical significance / paired CIs** — no raw per-seed gradient vectors (the original BLOCKED item, unchanged).

## 12. Verdict

**`CONDITIONAL_RESEARCH`** (Stage C scale: see `STAGE_A_VERDICT.json`).

- **Not CODEX_FEASIBILITY_GO:** every quality and cost number is a proxy or a prediction; nothing is measured on GPU; the mechanism's value rests on one unmeasured variable (κ) and an unconfirmed baseline design (N1).
- **Not DROP_HWCS_CURRENT_DESIGN:** basic feasibility is **not** falsified — the math is sound (MC-validated <0.3%), and HWCS-1 unconditionally undercuts Native G4's work floor in the structural model while beating G16 on quality. The mechanism is plausible and worth one cheap measurement.
- **Not BLOCKED_EVIDENCE (in the WV0 sense):** unlike WV0, we have a validated analytic model and reproducible proxies, not missing raw data. But the *realization* is blocked on measurement.
- **The single blocking unknown** between CONDITIONAL_RESEARCH and a Codex GO is: *does HWCS-1's structural advantage survive as measured GPU work, at a favorable κ, against a dense-binning Native G4?*

## 13. Single highest-value next experiment

**Stage B-min** (PyTorch oracle, **no CUDA build**, full-render + matched `1/p` masked SSE — identical protocol to the original 9-state data, so directly comparable). Full spec, code patch, matrix, and cost in **`STAGE_B_SPEC.md`**. It is a **falsifiable** pilot of the two decisive unknowns (real HWCS NMSE + real κ). Gate: run only on an **idle** MX A100 that is **not** the TEMPORAL_COVERAGE-R1 GPU; ≈10–30 min, 0 training steps.
