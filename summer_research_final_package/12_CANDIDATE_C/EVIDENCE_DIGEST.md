# PLAN_C — Candidate C Evidence Digest (Certificate-Guided Backward Skip)

**Subject:** Candidate C = loss-aware pre-backward gradient certificate for 3D Gaussian Splatting (3DGS) rendering. Stages R3, R3.1, R4.
**Repo:** `C:\Users\36570\3dgs-renderer-benchmark` (git, ~298 commits; project period 2026-07 .. 2026-09)
**Compiled from reports and source under that repo. Every claim cites a file path below.**

> **Naming collision warning.** A *different* "Candidate C — Exact fan-out-separated backward aggregation" appears in `reports/codex-independent-candidate-discovery.md` §7 (LOW PRIORITY, decided DROP-backup). That is **not** the Candidate C of this digest. The certificate/certificate-bound Candidate C is the one tracked through R3 → R3.1 → R4 and is the subject here. The certificate idea is the one consistently labeled "Candidate C" in the R3/R3.1/R4 reports.

---

## 1. Candidate C idea

Candidate C is a **per-(tile, Gaussian) certificate bound** that, using only forward-pass-after-loss quantities, upper-bounds the magnitude of each of the four raster-backward gradient families (color, opacity, mean2d, conic) aggregated per Gaussian. Because the bound is computed from forward state (`o_i`, `c_i`, the conic precision matrix `P_i`, per-tile `Q_t = Σ_p ‖q_p‖`, `C_max_t = max_j ‖c_j‖`, and the exact continuous-rectangle `sigma_min(i,t)`), its cost scales as `O(I_tile)` rather than `O(I_pixel)`. At a chosen certified error budget `ε`, the bound identifies a **joint skip-set** of (tile, Gaussian) pairs whose removed gradient mass is provably ≤ `ε·total_bound` for *all four families simultaneously* — one interaction bit per pair suppresses every derivative family in the shared raster backward kernel. Candidate C's stated novelty signal is **loss-conditioned nonzero-support culling**: pairs that are inside ordinary alpha support (`α ≥ 1/255`) but skippable due to loss-conditioned derivative bounds, as distinct from the prior-art exact-zero (`α_max < 1/255`) culling of Speedy-Splat / AccuTile (`reports/r3_1/decision_protocol.md` §"Critical Distinction"; `reports/phase-r3-final-formal-audit.md` §6.4).

**Where first described:** The earliest concrete statement of the bound formulas and the falsifiable claim ("exists ε ∈ {0.1%,…,5%} such that O(I_tile) bounds permit certified skipping of ≥30% of tile-Gaussian work in ≥1 family with no bound violation") is `reports/phase-r3-certificate-tightness-gate.md` (dated 2026-09-14, semantic `REFERENCE_V1_ABSGRAD`, pinned commit `84f29bb`), §1 and §9. The full corrected bound table is `reports/phase-r3-final-formal-audit.md` §3. The decision protocol (CR1–CR4 → C_KEEP/C_MODIFY/C_DROP) is codified from `experiments/r3/r3_decision.py` in `reports/r3_1/decision_protocol.md`.

---

## 2. R3 audit — counting methodology, 7,494 violations, per-family breakdown, original verdict

**Methodology.** R3 replaced the per-tile, per-Gaussian scalar accumulation loop in `_accumulate_certificate_bounds` (`experiments/r3/r3_certificate_runner.py`) with a fully vectorized path (`[P,1]×[1,G]` broadcasting, `torch.cumprod` transparency accumulation, `scatter_add_` aggregation, one `.to(cpu)` per window instead of thousands of `.item()` syncs). A strict semantic-equivalence gate (**R3-VEC**) required the vectorized path to match a scalar reference on 200 sampled tile/particle pairs before publishing any measurement; it passed bit-identical for integer `W_count` weights and ≤1e-9 for float bounds (`reports/r3-full-optimization.md` §5; `reports/r3-vectorization-gate.md`). Full measurement ran **3 training windows × 30 iterations = 90 certificate iterations** on Mip-NeRF 360 `room` (`reports/r3-full-optimization.md` §2). Source of truth for counts: `results/reference_v1/r3/aggregated_summary.json` (copied into `audit_packages/candidate_c_r3_final/analysis/`).

**Correctness gate (CR1):** `B_i^f + tol ≥ ‖g_i^actual‖` per-Gaussian aggregate, with `B_i = Σ_t B_it` and `g_i = Σ_t g_it` from autograd; tolerance 1e-6 (`reports/phase-r3-certificate-tightness-gate.md` §4; `reports/phase-r3-final-formal-audit.md` §1.2).

**The 7,494 violations claim.** `aggregated_summary.json`: `total_violations = 7494`, `zero_violations = False`, `n_iterations = 90` (`reports/r3-full-optimization.md` §3, §3 code block). The 5000 and 15000 windows show many non-zero iterations; the 29970 window is mostly clean (8 of 30 non-zero), consistent with convergence (`reports/r3-full-optimization.md` §3.2). Note: the 29970 checkpoint is a **symlink to `iter_30000.pt`** (same 236,187,191 bytes), documented in `checkpoints/checkpoint_manifest.json` (`reports/r3-full-optimization.md` §4).

**Per-family breakdown (R3):**

| Family | Violations | Proxy? | Notes |
|---|---:|:---:|---|
| color_coarse | 0 | No | bound holds |
| color_tight | 0 | No | bound holds |
| opacity | 260 | Yes | coarse opacity bound exceeded |
| **opacity_tight** | **5,602** | Yes | dominant failure mode |
| mean2d | 8 | Yes | rare |
| conic | 210 | Yes | moderate |
| mean2d_sigmamin | 19 | Yes | rare |
| **conic_sigmamin** | **1,395** | Yes | second-largest failure mode |
| **TOTAL** | **7,494** | — | 6 of 8 families violate |

Source: `reports/r3-full-optimization.md` §3.1 table; cross-checked against `reports/r3_1/candidate-c-r3-1-final.md` §2.2 table and `reports/r3_1/common-factor-audit.md` §3 table. Concentration: `opacity_tight` (5,602) + `conic_sigmamin` (1,395) = 6,997 = 93% of all violations.

**Original verdict.** Applying the C1–C6/C R1–CR4 protocol by hand: C2 (zero violations) **FAIL** because total = 7,494 ≠ 0; C3 (tightness) **FAIL** (pooled medians large, e.g. opacity_tight median ≈ 1121). Provisional recommendation **CERTIFICATE_INVALID / C_DROP**, driven by CR1/C2 failure (`reports/r3-full-optimization.md` §6). The automated decider (`r3_decision.py`) reported `AWAITING_DATA / DATA_INCOMPLETE` due to a decider input-path issue (it expected per-family summary files as separate artifacts whereas the aggregator merged them into `aggregated_summary.json`) — a tooling issue, not a data issue; all six families were present (`reports/r3-full-optimization.md` §6). The R3.1 final report restates the original decision as **"R3: C_DROP (CR1 FAIL: 7,494 violations across 6 families)"** (`reports/r3_1/candidate-c-r3-1-final.md` §6). The red-team gate rule is explicit: *"CR1 failure (any violation) → automatic C_DROP, regardless of opportunity numbers. Correctness is a hard gate."* (`reports/r3_1/decision_protocol.md` §Final Decision Logic).

---

## 3. R3.1 repair — root-cause chain, corrected formula, re-audit (0 violations), 90 measurements, VEC gate

### 3.1 Root-cause chain: the "conic-as-color proxy" bug

R3.1-A (`reports/r3_1/common-factor-audit.md`, frozen commit `32ab80e773f74f4d8e40ff4e338c86b29c7957f5`, source SHA256 `98eec016…`) found a single shared defect: `_accumulate_tile_bounds` computed appearance-factor bounds using **`‖conic_i‖₂` (geometric conic / covariance norm) as a proxy for `‖c_i‖₂` (SH-evaluated per-Gaussian RGB color norm)** in **6 of 8** certificate families. The source explicitly documented this proxy with a false justification — `compute_C_max_t` (lines 221–228) comments: *"C_max_t = max_{j in G_t} ‖c_j‖_2 over Gaussian conic norms per tile. NOTE: This uses conic norms as proxy for Gaussian color norms. The proper C_max_t would use ‖color_j‖ from SH output, but the certificate derivation holds for any norm-bounded quantity."* The audit's central finding is that this justification is **false**: the conic norm and color norm are independent quantities (a sharp splat can be bright; a diffuse splat can be dark), so there is no inequality `‖c_i‖ ≤ f(‖conic_i‖)` valid for all Gaussians. Using the proxy produces a bound that is *sometimes smaller than the true gradient magnitude* — exactly what a "violation" means (`reports/r3_1/common-factor-audit.md` §5).

Mechanism in code (`reports/r3_1/common-factor-audit.md` §2): line 433 `conic_norm_batch = conic_batch.norm(dim=-1)`; line 445 `C_max_t_val = float(conic_norm_batch.max().item())`; line 451 alias `c_norm = conic_norm_batch`; line 462 `factor_op_v = (c_norm + C_max_t_val) * tile_Q_val`. Crucially, the true color **was already computed** at line 1016 (`colors_rgb = spherical_harmonics(model.active_sh_degree, dirs, shs.unsqueeze(0))`, shape `[1, G, 3]`) but **never passed** to `_accumulate_tile_bounds` (line 1076 omits `colors_rgb.detach()`).

**Evidence of the proxy = the violation pattern.** Perfect binary partition: every family using the proxy (6/6) had violations; every family not using it (color_coarse, color_tight — opacity-only, no conic proxy — 2/2) had zero (`reports/r3_1/common-factor-audit.md` §4; `reports/r3_1/candidate-c-r3-1-final.md` §2.2). This perfect correlation confirms the conic-as-color proxy as the dominant root cause.

### 3.2 Repaired formula (R3.1-B)

Replace conic norm with the true SH-evaluated color norm and pass `colors_rgb` into the bound accumulator (`reports/r3_1/corrected-certificate-derivation.md` §2.3; `reports/r3_1/common-factor-audit.md` §6):

```python
# BEFORE (buggy):  c_norm = conic_batch.norm(dim=-1);  C_max_t_val = conic_norm_batch.max()
# AFTER  (repaired):
color_norm_batch = colors_rgb[0, g_unique].norm(dim=-1)   # ‖c_i‖₂ from SH
c_norm = color_norm_batch
C_max_t_val = float(color_norm_batch.max().item())
# opacity bound becomes: B_opacity_it = E_it * (‖c_i‖ + C_max_t) * Q_t  (mathematically meaningful)
```

The repaired opacity bound derives from `∂C/∂α_i = T_i·c_i − [Σ_{j>i} T_j α_j c_j]/(1−α_i)`, bounded by `(‖c_i‖ + C_max_t)·Q_t` under canonical RGB / L1+SSIM semantics where `v_render_a = 0` and `‖buffer/(1−α_i)‖ ≤ C_max_t` (`reports/r3_1/corrected-certificate-derivation.md` §2.1–2.2; `reports/phase-r3-final-formal-audit.md` §1.3). The color certificate (`α·Q_t`) was already correct and unchanged. The geometry families (mean2d: `√(λ_max/e)`; conic: `√(3/2)/(e·λ_min)`) now multiply the true color factor `base_geo_s = α_i·(‖c_i‖ + C_max_t)·Q_t` (`reports/phase-r3-final-formal-audit.md` §1.4–1.5). A noted secondary caveat: the simplified tail factor `(‖c_i‖ + C_max_t)·Q_t` assumes `N_downstream = 1` and small `α_i`; for tiles with many downstream Gaussians it may be under-conservative, but this was *not* the dominant R3 error and was deferred to float64 forensics (`reports/r3_1/corrected-certificate-derivation.md` §2.3 "Assessment").

### 3.3 Re-audit result: 0 violations; 90 measurements; float64 forensics

Scene Mip-NeRF 360 `room`; windows 5K / 15K / 30K mature snapshot, 30 cameras each = **90 measurements**; pinned commit `32ab80e` (`reports/r3_1/candidate-c-r3-1-final.md` header). Re-audit across all 8 families and all 90 measurements: **0 violations** (per-window and total all zero) (`reports/r3_1/candidate-c-r3-1-final.md` §4.1 table). Float64 forensics with **pre-declared** constants (tolerance 1e-6, borderline threshold 1%): total violations 0 in both float32 and float64; worst-case ratio `max(|g|/B) = 0.9998` (margin 0.0002); 2 borderline cases (ratio > 0.99 but < 1.0) flagged for transparency, not violations. Identical float32/float64 results confirm the margin is genuine, not a rounding artifact (`reports/r3_1/candidate-c-r3-1-final.md` §4.2). **CR1: PASS ✅** (0 violations, all 8 families, all 90 measurements). **CR3: PASS ✅** (3 windows). **CR4: PASS ✅** (SPD-disabled fraction = 0.0%, threshold <10%).

### 3.4 VEC gate

**VEC** is the semantic-equivalence validation gate comparing the scalar reference implementation against the vectorized bound/weight computation. For R3.1 it ran on 200 tile-Gaussian pair samples / 5,096 Gaussians compared across 200 tiles (synthetic data), pass threshold `max_rel_diff < 1e-5` (abs < 1e-7). All 8 families passed; overall `max_abs_diff = 2.09e-06`, `max_rel_diff = 6.25e-06` (`reports/r3_1/vec_gate_report.json` `overall`; `reports/r3_1/candidate-c-r3-1-final.md` §5 table). The earlier R3-VEC gate (W-equivalence) verified bit-identical integer `W_count` work weights on 256 random + 800 edge-case tiles and `UNSAFE_SIGMA_MIN_COUNT = 0` for the `scatter_reduce_("amin")` sigma-min computation (`reports/r3-vectorization-gate.md`). **VEC Gate: PASS ✅** — confirms the repaired vectorized path faithfully reproduces the scalar reference, so the 0-violation re-audit is not a vectorization artifact.

---

## 4. R4 status — CUDA implementation plan and current state

**Plan.** R4 inserts the certificate skip at the **(tile, Gaussian) intersection** level inside the gsplat 1.5.3 backward kernel `rasterize_to_pixels_bwd_kernel` (`rasterize_to_pixels_bwd.cu`). The insertion point is the inner loop over Gaussians in a tile batch: a precomputed `skip_mask[n_isects]` (bool, computed Python-side from forward state + budget ε=5%) gates the *gradient computation* (v_rgb, v_conic, v_xy, v_opacity) and the *atomicAdd* writes, while the alpha/T/buffer update **always** executes to keep subsequent gradients correct (`reports/r4/r4-cuda-insertion-audit.md` §2, §4, §5). The skip branch is **warp-uniform** (`isect_idx = batch_end - t` depends only on loop variable `t`, not thread index) → no warp divergence. Work avoided per skipped intersection: ~30 FMA + 5 warpSum reductions + 9 atomicAdds per pixel per skipped Gaussian; at the 62.5% / 5% figures and ~8M tile-Gaussian interactions/camera at 30K, ≈4.5 billion atomicAdd operations avoided per backward pass (`reports/r4/r4-cuda-insertion-audit.md` §3). Pinned commit `8c2d7b0` (R3.1 final) (`reports/r4/r4-cuda-insertion-audit.md` header).

**What was implemented (source present in repo):**
- `experiments/r4/rasterize_to_pixels_bwd_with_skip.cu` (394 lines) — modified backward kernel with `skip_mask` parameter; template `rasterize_to_pixels_bwd_with_skip_kernel<CDIM,S>` explicitly instantiated for CDIM=3 (confirmed via `grep`: kernel definition line 26, launcher lines 294–345, explicit instantiation line 374).
- `experiments/r4/ext_skip.cpp` — pybind11 binding exposing `rasterize_to_pixels_bwd_with_skip<3>` (confirmed via `grep`).
- `experiments/r4/build_cuda_extension.py` (258 lines) — JIT build script, targets A100 sm_80 (`TORCH_CUDA_ARCH_LIST="8.0"`), via `ssh mx`.
- `experiments/r4/r4_certificate_skip.py` (356 lines) — Python-level wrapper with `compute_skip_mask_python` using the **R3.1 corrected** formula (`c_norm = colors[0].norm(dim=-1)`, true color norm, not conic proxy) for families color_tight, opacity, mean2d_sigmamin, conic_sigmamin. Documents MODE0 (baseline) / MODE1 (R4 path, skip disabled) / MODE2 (R4 path, skip enabled); notes the Python path computes full gradients then zeros some (correctness verification only — no compute savings); actual savings require the CUDA extension.
- `experiments/r4/r4_room_sanity.py`, `experiments/r4/r4_sanity.py` — correctness sanity checks (MODE0 vs MODE1 gradient agreement).

**What remains / status:** **In progress, not validated end-to-end.** The sanity script `r4_room_sanity.py` (lines 274, 277) explicitly states: *"For now, this is the same as MODE0 since we haven't compiled the custom kernel yet"* and `# TODO: call custom compiled kernel with skip_mask=all False` — i.e., the CUDA extension **has not been compiled/invoked** in the sanity path; MODE1 currently re-runs the baseline. The 13-scene training launcher `experiments/r4/run_r4_13scene.sh` references `experiments/r4/r4_train_scene.py`, but **that file does not exist** in the repo (glob `experiments/r4/r4_train_scene*` → no files). No R4 training-result reports or result JSONs were found under `reports/` or `results/` for candidate C. Net: **R4 kernel + binding + build + Python skip-mask are written; CUDA compilation, MODE0/MODE1/MODE2 numerical agreement, and the 13-scene MODE2-vs-baseline training validation are not yet executed/complete.**

---

## 5. The 62.5% weighted-work opportunity at 5% budget

**Where it comes from.** CR2 = JOINT weighted-work removal fraction at 5% budget, the primary opportunity gate. `JOINT_SKIP_WEIGHTED_WORK_FRACTION = Σ(W_it for skipped) / Σ(W_it for all)`, where `W_it` = number of pixel lanes in tile `t` where Gaussian `i` is active (`α ≥ 1/255`) — measured as `counts[j]` from `torch.unique(g_indices, return_counts=True)` on depth-sorted `flatten_ids` per tile, the exact number of pixel lanes executing derivative work for that pair in the raster backward (`reports/phase-r3-final-formal-audit.md` §6.3; `reports/r4/r4-cuda-insertion-audit.md` §3). The JOINT skip rule (R3.11) sets **one bit per (tile, Gaussian) pair** that suppresses all four derivative families together (R2.1A had invalidated geometry-only gating because the shared alpha/T/buffer traversal is common to all families) (`reports/phase-r3-final-formal-audit.md` §6.2).

**How computed.** Aggregated across all 90 measurements (3 windows × 30 cameras), the budget table (`reports/r3_1/candidate-c-r3-1-final.md` §4.3) is:

| Budget | Mean pair fraction | Mean weighted-work fraction | Mean loss-cond fraction |
|---|---|---|---|
| 0.1% | 65.12% | 49.89% | 51.54% |
| 0.5% | 66.89% | 52.40% | 53.51% |
| 1% | 67.78% | 54.24% | 54.83% |
| 2% | 68.85% | 56.93% | 56.71% |
| **5%** | **70.75%** | **62.51%** | **60.52%** |

Per-window at 5% budget: 5K window weighted-work 68.6% (loss-cond 65.9%) → PASS; 15K window 59.1% (58.1%) → PASS; 30K window 59.9% (57.5%) → PASS; **cross-window mean weighted-work 62.5%** (loss-cond 60.5%) → **PASS (≥30% threshold)** (`reports/r3_1/candidate-c-r3-1-final.md` §4.3). **CR2: PASS ✅** (62.5% at 5% budget). The "~62.5% at 5% budget" headline is the cross-window mean of `JOINT_SKIP_WEIGHTED_WORK_FRACTION` at ε=5%; the per-iteration mean in the table is 62.51%, rounded to 62.5%.

**Caveats.** (a) This is a *certificate-predicted* opportunity (bound-based), not a measured wall-clock speedup — actual removed mass depends on the unknown per-pair `g_it`; the R4 CUDA kernel is the mechanism that would convert it to real savings, and R4 is unvalidated (§4). (b) The weighted-work fraction is the primary gate metric, but the **loss-conditioned** fraction (60.5%) — Candidate C's novelty signal excluding exact-zero prior-art culling — is reported separately and also clears the ≥30% threshold (`reports/r3_1/decision_protocol.md` §CR2; `reports/r3_1/candidate-c-r3-1-final.md` §4.3). (c) The simplified opacity tail (`N_downstream = 1`) is a known secondary looseness; float64 forensics confirmed no violations arose from it at the measured margin 0.0002, but it is flagged for future tiles with many downstream Gaussians (`reports/r3_1/corrected-certificate-derivation.md` §2.3). (d) Measurement is single-scene (`room`); cross-scene generalization is the R4 13-scene plan, not yet run.

---

## 6. Final accumulated verdict for Candidate C

**C_KEEP (KEEP)** — exact wording from `reports/r3_1/candidate-c-r3-1-final.md` §6:

```
CR1 (correctness):     PASS ✅ (0 violations, all 8 families, all 90 measurements)
CR2 (weighted work):   PASS ✅ (62.5% at 5% budget, ≥30% threshold)
CR3 (signal coverage): PASS ✅ (3 windows: 5K, 15K, 30K, 90 iterations)
CR4 (SPD):             PASS ✅ (SPD-disabled fraction = 0.0%)
Final Decision: C_KEEP (KEEP) ✅
```

Trajectory: **R3 → C_DROP** (CR1 FAIL: 7,494 violations across 6 families) → **R3.1 → C_KEEP** (CR1 PASS: 0 violations; CR2 PASS: 62.5% weighted work). Root cause was a single shared defect — conic norm as proxy for color norm in appearance-factor certificate bounds; replacing it with the true SH-evaluated RGB color norm eliminated all 7,494 violations while preserving the opportunity signal (62.5% JOINT weighted-work removal at 5% budget) (`reports/r3_1/candidate-c-r3-1-final.md` §6, §8). Decision protocol: CR1 pass AND CR2 pass (≥30%) AND CR3 pass AND CR4 pass → C_KEEP (`reports/r3_1/decision_protocol.md` §Final Decision Logic). Constraints honored: correctness first, no magic constants / tolerance relaxation, no custom loss to force PASS, float64 forensics with pre-declared constants, R3 frozen at `32ab80e` (not modified) (`reports/r3_1/candidate-c-r3-1-final.md` §8).

---

## 7. Contradictions / gaps

1. **Date inconsistency.** The R3.1 final report and the R4 CUDA audit are both dated **2025-09-18** in their headers (`reports/r3_1/candidate-c-r3-1-final.md` line 3; `reports/r4/r4-cuda-insertion-audit.md` line 3), but the git commit `8c2d7b0` ("R3.1 Certificate Repair … → C_KEEP") is dated **2026-09-18 13:42:41 +0800**, and the common-factor audit / corrected-derivation are dated **2026-09-17** (`reports/r3_1/common-factor-audit.md`; `reports/r3_1/corrected-certificate-derivation.md`). The 2025 vs 2026 year is an authoring typo in two report headers; the project period (2026-07..2026-09) and git dates confirm 2026.

2. **Automated decider never issued a machine-verified R3 verdict.** `r3_decision.py` reported `AWAITING_DATA / DATA_INCOMPLETE` for R3 due to a decider input-path issue (expected separate per-family summary files; aggregator merged them into `aggregated_summary.json`). The R3 C_DROP is therefore a **hand-applied** protocol verdict, not machine-verified (`reports/r3-full-optimization.md` §6). The R3.1 final report's CR1–CR4 verdicts are likewise reported, not accompanied by a `final_decision.json` artifact path. *Gap: no `final_decision.json` path is cited for either R3 or R3.1.*

3. **R4 not executed/validated.** Kernel/binding/build/Python-skip source exist, but `r4_room_sanity.py` states the custom kernel "haven't compiled … yet" (line 274) with a `TODO` (line 277); `r4_train_scene.py` (referenced by `run_r4_13scene.sh`) is **absent** from the repo. No R4 result reports/JSONs found. The 62.5% figure remains a *certificate-predicted* opportunity, not a measured speedup. The R4 audit's "≈4.5 billion atomicAdd avoided per backward pass" is a projection from the 62.5% figure, not a measurement (`reports/r4/r4-cuda-insertion-audit.md` §3).

4. **Single-scene evidence.** R3/R3.1 measured only Mip-NeRF 360 `room`. Cross-scene generalization (the 13-scene R4 plan: bicycle, bonsai, counter, flowers, garden, kitchen, room, stump, treehill, train, truck, drjohnson, playroom) is **planned, not run** (`experiments/r4/run_r4_13scene.sh`).

5. **Two "Candidate C" concepts.** The certificate-bound Candidate C (this digest) vs. "Candidate C — Exact fan-out-separated backward aggregation" (`reports/codex-independent-candidate-discovery.md` §7, LOW PRIORITY, distinct mechanism: high-fan-out two-pass reduction). The discovery doc's ranking lists the fan-out C as #3 / LOW PRIORITY. The R3/R3.1/R4 reports use "Candidate C" exclusively for the certificate idea. A reader must disambiguate by context (R3/R3.1/R4 = certificate C).

6. **Tight margin.** Re-audit worst-case ratio 0.9998 (margin 0.0002) with 2 borderline cases — extremely tight. Float64 confirmed it is genuine, but there is little headroom; robustness under different scenes/checkpoints is unverified (links to gap #4).

7. **Checkpoint symlink semantics.** The 29970 window is `iter_30000.pt` via symlink (documented, SHA256-identical), so "3 windows" are effectively 5K / 15K / 30K-mature, not a distinct 29970-step model (`reports/r3-full-optimization.md` §4). Not a contradiction, but a precision note for any "90 measurements across 3 distinct checkpoints" claim.

---

## Source index (files cited)

Reports: `reports/r3-full-optimization.md`; `reports/r3-full-optimization-template.md`; `reports/r3-vectorization-gate.md`; `reports/phase-r3-certificate-tightness-gate.md`; `reports/phase-r3-final-formal-audit.md`; `reports/r3_1/candidate-c-r3-1-final.md`; `reports/r3_1/common-factor-audit.md`; `reports/r3_1/corrected-certificate-derivation.md`; `reports/r3_1/decision_protocol.md`; `reports/r3_1/vec_gate_report.json`; `reports/r4/r4-cuda-insertion-audit.md`; `reports/README.md`; `reports/codex-independent-candidate-discovery.md`.
Scripts/source: `experiments/r3/r3_certificate_runner.py`; `experiments/r3/r3_decision.py`; `experiments/r4/rasterize_to_pixels_bwd_with_skip.cu`; `experiments/r4/ext_skip.cpp`; `experiments/r4/build_cuda_extension.py`; `experiments/r4/r4_certificate_skip.py`; `experiments/r4/r4_room_sanity.py`; `experiments/r4/r4_sanity.py`; `experiments/r4/run_r4_13scene.sh`.
Audit packages: `candidate_c_source_audit_20260914_manifest.txt`; `candidate_c_source_audit/PACKAGE_INFO.md`; `candidate_c_source_audit/reports/phase-r2-attribute-decoupled-backward-gate.md`; `@_build_candidate_c_audit.py`; `audit_packages/candidate_c_r3_final/` (archive SHA256 `6c22102c…`); `audit_packages/candidate_c_r3_1_final/`.

**Key commits (hash → date → subject):**
- `84f29bb` — 2026-09-14 — MD-V2 pre-freeze; R3 certificate-tightness-gate pinned commit (`REFERENCE_V1_ABSGRAD`).
- `e494458` — 2026-09-15 — R3: Fix n_lanes → faithful W_color_it/W_unclamped_it per-pixel replay.
- `fd08eb8` — (R3) Vectorize `_accumulate_tile_bounds` — pass W-equivalence gate.
- `9bc3a19` — (R3-VEC) batch CPU transfer to single stack call (16x fewer syncs).
- `32ab80e` — R3.1 frozen source commit (`r3_certificate_runner.py`, common-factor audit).
- `8c2d7b0` — 2026-09-18 — **R3.1 Certificate Repair: conic-as-color proxy fix → C_KEEP** (R4 pinned commit).
