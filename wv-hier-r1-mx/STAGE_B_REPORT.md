# WV-HIER-R1-MX — Stage B Report
## Stage B-min under the controlling protocol `WV_HIER_R1_MX_STAGE_B_REVISED`

**Round:** WV-HIER-R1-MX · **Stage:** B-min · **Date:** 2026-10-08
**Controlling protocol:** `WV_HIER_R1_MX_STAGE_B_REVISED_20261008` (SHA256 `2C3AC529…C4B55`)
**Prior verdict (Stage A):** `CONDITIONAL_RESEARCH` (revised; conflicts resolved 20261008)

---

> **Operator?** Use the single-source runbook: **`STAGE_B_MIN_HANDOFF.md`** (pre-flight, files, execute command, decision rules, verdict vocabulary, fallback). This report is the analytical record; the handoff is the operational checklist.

## 0. Execution status (read first) — HONEST

| Item | Status |
|---|---|
| MX live A100 reachable from this host | **NO** — `/mnt` absent, no torch/gsplat, local GPU is RTX 5070 (sm_120), checkpoints on MX |
| Stage B-min GPU gradient oracle | **NOT_EXECUTED** (cannot run here) |
| Stage A verdict-conflict fixes (5 goal items) | **DONE** — see `STAGE_A_VERDICT.json` `residual_conflicts_resolved_20261008` |
| N1/N2 structural work audit | **DONE** (derived proxy) — see §4 |
| Variance-identity MC check | **DONE** (CPU, n=300k) — see §5 |
| Stage B-min runner (controlling protocol) | **BUILT + CPU self-test PASS** — see §6 |
| Raw per-sample GPU results | **NOT_PRODUCED** (requires MX dispatch) |

**Therefore no `ORACLE_PASS` / `ORACLE_INCONCLUSIVE` / `MODEL_OR_IMPLEMENTATION_MISMATCH` can be claimed from GPU data.** This report delivers the executable protocol, the two analyses computable without GPU (N1/N2 audit, variance identity), and the next-stage structural verdict. The GPU gradient oracle is a dispatch task for an idle MX A100.

---

## 1. What Stage B-min is (per revised controlling protocol)

A **supervision oracle**, not a performance benchmark. It validates the HWCS gradient *estimator math* on the exact original cameras/seeds and audits structural *counts* for two Native-G4 baselines (N1, N2). It does **not** measure sparse-CUDA F+B time and does **not** train.

- Checkpoint: `bicycle__iter15000` (N=3,903,754; SHA `a2c82199…37329`)
- Cameras: IDs 0, 8 (bicycle, 540×813)
- Methods: G16, Native-G4 **N1**, Native-G4 **N2**, HWCS-1 (q=.5,r=.5), HWCS-2 (q=.75,r=1/3)
- Marginal p = 0.25 (all methods matched)
- 32 predetermined paired seeds; 2 dense references (one per camera)
- 0 training steps; no CUDA edit; no compile; no scene expansion
- Common random numbers: shared `(U_parent, U_child)` per camera/seed across all methods
- All full-render + masked-loss timing labeled `SUPERVISION_ORACLE` (never sparse CUDA)

## 2. What Stage B-min measures (and does not)

**Measures (on MX):** per camera/seed/group NMSE, cosine, rel-L2, dense/est gradient norms, dot; sampling mask bits + SHA; actual per-pixel inclusion frequency; parent/child tile counts; **real projected Gaussian statistics** (camera-level, not just cam0 bbox proxy); full input/source/`.so` hashes.

**Does NOT measure:** the operation-cost parameter κ (not count-identifiable); sparse-CUDA kernel/F+B time (no sparse renderer exists); time-to-quality / convergence (0 training steps).

## 3. Falsification protocol (per-seed paired CI, NOT hard bands)

For each camera, within the same 32-seed replay:
```
d_1(s) = NMSE_HWCS1(s) − [ (2/3)·NMSE_G4(s) + (1/3)·NMSE_G16(s) ]   (α=1/3)
d_2(s) = NMSE_HWCS2(s) − [ (8/9)·NMSE_G4(s) + (1/9)·NMSE_G16(s) ]   (α=1/9)
```
E[d]=0 under the exact estimator model. Report per-camera paired bootstrap 95% CI (resampling paired seed vectors); pool across cameras only if clearly labeled and stratified. Quality gate: `NMSE_G16 − NMSE_HWCS1` with paired CI + effect size. **A single-seed deviation is NOT a formula failure.** Insufficient power ⇒ `ORACLE_INCONCLUSIVE`, never auto-DROP.

## 4. N1 / N2 structural work audit (DONE — derived proxy, not GPU time)

Two Native-G4 baselines audited independently (units of `I_G16`; `GAM=1`):

- **N1 (dense-bin-then-mask):** bin+sort ALL `I_G4` child entries, then mask. `W_N1 = ρ(1+GAM·p)`.
- **N2 (mask-then-bin):** choose child tiles (mask at p), bin+sort only selected. `W_N2 = p·ρ(2+GAM)`.
- **HWCS(κ):** parent bin+sort (1) + child refine `[p + κ(q−p)]ρ` + raster `GAM·p·ρ`.

`ρ = I_G4/I_G16` per state (OBSERVED bbox proxies). Full table: `stage_b_n1_n2_audit.csv`; verdict: `stage_b_n1_n2_verdict.json`.

| κ (child-refine) | HWCS beats N1 | HWCS beats N2 |
|---|---|---|
| 0.0 (optimistic) | 18/18 (mean save 47%) | 18/18 (mean save **12%**) |
| 0.5 | 18/18 (mean save 32%) | **2/18** (mean save −13%) |
| 1.0 (conservative) | 18/18 (mean save 17%) | **0/18** (mean save **−38%**) |

**N2 vs N1 (strong baseline already saves):** mean **40%**.

**Finding (decisive for honesty):** HWCS's structural edge over N1 is modest and cost-model-conditional. Against the **strong N2 baseline**, HWCS does **not** reliably win: at κ=1 it loses in **all 18** state-method combinations; only in the optimistic κ→0 limit does it edge N2, and then only ~12% (4.7%–18.9% per state, and only where ρ>4). **No HWCS structural dominance may be claimed off N1 alone.** Because κ is not count-identifiable and N2 is a legitimate sparse baseline, the structural-potential case is **weak/N2-conditional**.

> Honest restatement: the earlier "HWCS undercuts Native-G4 floor by 27%" was an **N1-only** statement. Under N2 the advantage largely disappears. This is the single most important correction of this round.

**Round-2 robustness (goal item #4 — "N1 + current structural cost model"):** the round-1 audit used `GAM=1`. A natural challenge is whether the N2-loss conclusion is a raster-weight artifact. **It is not.** `GAM·p·ρ` appears on *both* sides of the HWCS-vs-N1 and HWCS-vs-N2 comparisons and **cancels algebraically**, so the dominance conclusions depend only on `ρ`, not `GAM`. The analytic break-even `κ*` matches a numeric root-find to 4 decimals at both `GAM=1` and `GAM=3`:

- HWCS beats N1 iff `κ < (1−p−1/ρ)/(q−p)` = **1.07 … 2.57** (all >1 ⇒ HWCS beats N1 at κ=1, GAM-invariant).
- HWCS beats N2 iff `κ < (p−1/ρ)/(q−p)` = **0.07 … 0.57** (all <1 ⇒ HWCS loses to N2 at κ=1 in all 18 states, GAM-invariant).

Artifacts: `stage_b_n1_n2_gam_robustness.csv` (18 states) + `stage_b_n1_n2_gam_robustness.json`. The N2-conditional weakness is therefore **structural, not a `GAM=1` artifact** — but it remains κ-conditional and N2-conditional (it vanishes only if child-refine is nearly free, κ<~0.5).

### 4.1 Evidence provenance (goal item: "区分真实观测数量与 GPU 时间")

Every N1/N2 cost-model input is classified OBSERVED / DERIVED / NOT_MEASURED (`stage_b_evidence_provenance.json` + `stage_b_rho_provenance.csv`):

| quantity | class | role |
|---|---|---|
| active-pixel fraction | **OBSERVED** | 0.2500 vs target 0.25 (max dev 0.0012); 0.3996 vs 0.40 — validates inclusion=p on real MX-rendered masks |
| selected pixels | **OBSERVED** | 109475 vs derived 109755 (0.26% rel err) |
| `dense_intersection_proxy` (I_G16, I_G4, ρ) | **DERIVED** | analytic `drawn·π·r²·G²/(H·W)`; the ρ basis of N1/N2 |
| `physical_cuda_intersections` | **NOT_MEASURED** | no kernel-visible counter — ρ cannot be validated against a true count |
| `kernel_ms`, `forward_backward_ms` | **NOT_MEASURED** | no GPU timing exists |

**What this means:** the OBSERVED data confirms the **sampling design is correct** (inclusion probability = p, on real rendered masks). The N1/N2 **work comparison rests entirely on the DERIVED ρ proxy** — GAM-invariant (§4) but **not** validated against any OBSERVED intersection count, because `physical_cuda_intersections` was never instrumented. No GPU-time claim is possible. This is consistent with the report's standing caveat: the result is a structural proxy, not measured GPU work.

### 4.2 Cost-model sensitivity to N2 overhead and parent-binning cost (Round 4)

§4–4.1 fixed `GAM` and `ρ`. This sweep relaxes two more cost-model assumptions: parent-binning cost `c_par ∈ [0,3]` (in I_G16 units) and an **N2 implementation-overhead multiplier `o_N2 ∈ [1,2]`** (1 = ideal mask-then-bin; >1 = two-pass / load-imbalance / extra-launch cost). At κ=1:

- **Robust:** the "HWCS loses to N2" result holds for all `c_par` when `o_N2=1` (ideal N2) — consistent with the round-2 GAM-invariance.
- **Not robust to `o_N2`:** HWCS-1 beats N2 at κ=1 **only if `o_N2` exceeds a per-state critical threshold of 1.14–1.29×**. Below that, HWCS-1 loses to N2 in all 9 states.

| o_N2 (N2 overhead) | 1.00 | 1.10 | 1.25 | 1.50 |
|---|---|---|---|---|
| P( HWCS-1 beats N2 ), c_par=1 | 0.00 | 0.00 | 0.67 | 1.00 |

**Key unmeasured quantity identified:** real N2 CUDA overhead `o_N2`. If N2's mask-then-bin structure incurs >~1.2× overhead, HWCS can beat N2 at κ=1; if N2 is near-ideal, it cannot. This is precisely what Stage B-min structural counts and Stage C timing must pin down. **It does NOT restore a CODEX_GO** — `o_N2` is itself unmeasured, and a count-based proxy cannot establish it. Artifacts: `stage_b_cost_model_sensitivity.csv` + `.json`.

## 5. Variance-identity MC check (DONE — CPU, n=300k draws/config)

Artifact: `variance_identity_check.csv` (36 rows: 3 contribution regimes × 3 grid sizes × 4 methods). Validates `V_HWCS = (1−α)V_G4 + α·V_G16`.

| method | mean rel-err | max rel-err |
|---|---|---|
| G16 | 0.23% | 0.47% |
| G4 | 0.22% | 0.56% |
| HWCS-1 | 0.16% | 0.28% |
| HWCS-2 | 0.11% | 0.29% |

All < 0.6% (threshold 5%). The identity is **exact across positive, mixed, and signed/anti-aligned contributions**.

**Ordering scene-dependence (documented, not assumed):** `s2−s1` (signed within-parent cross-term) is **+19390** (positive regime → G4≤HWCS≤G16), **−23.8** (mixed), **−143** (anti-aligned → ordering reverses). Only the convex-blend identity is universal; the variance ordering is not. Stage B-min must confirm the identity on **real** gradients, not assume the ordering.

### 5.1 OBSERVED mask-bit decode validation (Round 5 — real rendered masks)

Beyond the aggregate fraction in the cost CSV, I decoded the actual `reconstructed_masks/*.bin` bits (96 seeds × {G16,G4,G1} at p=0.25; format reverse-engineered: bit-packed from byte 0, bitorder `big`, 1 bit per tile/pixel). This directly validates the **spatial** structure the variance model assumes:

| grid | units | inclusion (obs) | adjacent-cov | per-seed std (obs) | binomial std √(p(1−p)/n) |
|---|---|---|---|---|---|
| G16 (34×51) | 1734 | 0.2491 | −0.00032 | 0.0112 | 0.0104 |
| G4 (135×204) | 27540 | 0.2501 | +0.00005 | 0.0022 | 0.0026 |
| G1 (540×813) | 439020 | 0.2500 | +0.00003 | 0.00064 | 0.00065 |

**Findings (OBSERVED, not derived):** (a) inclusion ≈ 0.25 on all grids; (b) **max |adjacent covariance| = 0.00032** → selection is spatially Bernoulli-**independent**; (c) per-seed inclusion std matches the binomial expectation on all three grids → each mask is an independent Bernoulli(p) draw and the 96 seeds are independent. This **directly confirms the variance model's iid-independence assumption and the CRN sampling design on real rendered mask bits** — the one piece of the sampling machinery that could be validated without the checkpoint/GPU. Artifacts: `stage_b_mask_decode.json`.

**CRN endpoint identities (validated in the runner self-test, Round 7):** the two-stage parent→child mask reduces *bitwise* to the single-level baselines at its endpoints — `G16 == HWCS(q=p, r=1)` and `G4 == HWCS(q=1, r=p)` under the shared `(U_parent, U_child)`. This confirms the hierarchical sampler is algebraically consistent with the flat granularities it interpolates between (the property the HT estimator's unbiasedness rests on). Same assertions added to the turnkey patch `_wv0_grad_stage_b.py`.

### 5.2 Quality-gate status under the blend (Round 8)

The goal's quality gate is "HWCS NMSE below the G16/G4 mixture". The blend identity `NMSE_HWCS = (1−α)·NMSE_G4 + α·NMSE_G16` means the blend predicts HWCS NMSE **equals** the mixture *exactly* (max deviation 2.22e-16 across all 9 states, both configs). Verified on `stage_a_hwcs_predictions.csv`:

| state | G16 | G4 | HWCS-1 pred | mixture | HWCS-1 − mixture |
|---|---|---|---|---|---|
| bicycle__iter15000 | 1.965 | 1.154 | 1.425 | 1.425 | 0.000 |
| room__iter5000 | 1.800 | 0.747 | 1.098 | 1.098 | 0.000 |

**Implication (important):** the quality gate "HWCS below mixture" is **NOT predicted** — the blend gives equality. Any quality *advantage* requires negative within-parent cross-terms (s2<s1), which is scene-dependent (correction C3) and not guaranteed. Combined with §4 (HWCS loses to N2 on cost at κ=1), HWCS is **not** predicted to dominate N2; it equals N2 on quality and costs more, unless real N2 overhead o_N2 > 1.14–1.29× (§4.2). **Stage B-min test:** measure real HWCS NMSE vs per-seed mixture with paired CI — HWCS < mixture (CI excl 0) ⇒ scene-dependent advantage realized; ≈ ⇒ blend confirmed, no quality edge; > ⇒ drop quality claim. Artifacts: `stage_b_quality_gate.csv` + `.json`.

## 6. Stage B-min runner (BUILT, CPU self-test PASS)

`stage_b_min_runner.py` implements the controlling protocol: CRN masks, paired bootstrap CI, N1/N2 counts, variance-identity check, raw per-sample CSV schema, SUPERVISION_ORACLE timing labels, MX-dispatch guard.

Self-test output (this host, CPU): shapes OK; per-pixel inclusion 0.2497–0.2506 (≈p); variance-identity <0.4%; N1/N2 audit reproduces (HWCS-1 loses to N2 at κ=1). `--gpu-run` aborts cleanly with a dispatch message (torch unavailable here).

**MX patch points** (per `README_DSH.md`): patch a *copy* of `_wv0_grad.py` to call `generate_common_uniforms`/`make_mask` (in the runner) and keep one common `(U_parent, U_child)` per camera/seed across all methods; assert same image/target/dense-grad/loss-denominator/geometry; assert `G16 == HWCS(q=p,r=1)` and `G4 == HWCS(q=1,r=p)` bitwise under CRN. Run on an idle A100 **not** reserved for TEMPORAL_COVERAGE-R1; snapshot `nvidia-smi` before/after; abort if no clearly idle device.

**Turnkey patch provided (Round 2):** `_wv0_grad_stage_b.py` — a faithful Python-only copy of the audited oracle that already implements CRN + HWCS + N1/N2 + paired metrics + raw-CSV writer, importing `_wv0_core.render_full` **unchanged** (3 call sites). Syntax-validated (`py_compile`); the CUDA core and no-CUDA/no-training constraints hold. The MX operator drops it next to `_wv0_core.py`, sets the three `*_SHA256` env vars, and calls `run_camera(gparams, cam, sh_degree, seed)` per (camera, seed). It is **not** executed here (no torch/MX).

## 7. Raw per-sample result schema (to be produced on MX)

`stage_b_raw_results.csv`, one row per (camera, seed, method):
`camera_id, seed, method, nmse_geometry_macro, cosine_geometry_macro, rel_l2_geometry_macro, dense_grad_norm, est_grad_norm, dot_dense_est, actual_inclusion_freq, active_parent_count, selected_child_count, selected_pixel_count, gaussian_parent_candidate_count, gaussian_child_candidate_count, child_mask_sha256, checkpoint_sha256, source_sha256, so_sha256, timing_class(=SUPERVISION_ORACLE)`.
Plus `stage_b_variance_identity.csv` (per-seed `d_1,d_2` + paired bootstrap CI per camera) and `stage_b_n1_n2_counts.csv` (real projected counts, labeled COUNTS/PROXY vs NOT_MEASURED emitted_intersections).

## 8. Next-stage verdict (allowed vocabulary only)

| verdict | claimable now? |
|---|---|
| `ORACLE_PASS` | **NO** — GPU gradient oracle not executed |
| `ORACLE_INCONCLUSIVE` | N/A — not executed |
| `MODEL_OR_IMPLEMENTATION_MISMATCH` | **NO** — identity verified in MC; no real-gradient evidence either way |
| `STRUCTURAL_POTENTIAL` | **YES, but WEAK + N2-conditional** — N1 audit shows modest edge; **N2 audit shows the edge largely vanishes** (0/18 at κ=1). Proxy only, not GPU time. |
| `TIMING_NOT_MEASURED` | **YES** — no sparse-CUDA or HWCS timing exists |

**Stage B verdict:** `STRUCTURAL_POTENTIAL_WEAK_N2_CONDITIONAL` + `TIMING_NOT_MEASURED`. The GPU gradient oracle is **PENDING MX dispatch**.

## 9. Explicit gate to any CUDA prototype (NOT authorized this round)

Any `CODEX_GO` for a CUDA prototype requires **independent design + performance thresholds**, none of which exist yet:
1. Measured GPU kernel/F+B time vs **both N1 and N2** Native-G4 on matched p (Stage C, not B-min).
2. Gradient-quality preservation vs G16 on real gradients.
3. A minimum E2E time-to-quality target.
4. HWCS must beat **N2** on measured time, not just N1 on counts.

**A count-based κ threshold (e.g. κ≤0.5) is explicitly NOT a valid CUDA GO gate.** `STRUCTURAL_POTENTIAL ≠ CODEX_GO`.
