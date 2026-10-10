# TFC-P0-MX — REPORT

- Protocol: `TFC_P0_NOVELTY_GATE_MX/DSH_EXECUTION_PROMPT.md` (Stage P0 only: read-only, no GPU)
- Verdict: **`TFC_FOOTPRINT_ONLY_NO_NOVELTY`** (N0 fail; footprint-only default per `DECISION_PROTOCOL.md`)
- GPU seconds this task: **0**
- Fail-closed: **true**
- Outputs archived independently in `tfc_p0_novelty_gate_20261008/P0_OUTPUT_MX_20261008/`; no input modified.

---

## 0. One-paragraph disposition

TFC (Tight / ellipse-exact tile footprint) is **not** a distinct CVPR-2027 renderer-mechanism contribution. Its mode-1 kernel emits exactly the ellipse-exact Gaussian→tile set that gsplat upstream **AccuTile** (PR #927, Apr 2026; attributed in-source to Speedy-Splat) already emits, is a **rows-only special case that omits AccuTile's `isY` shorter-axis heuristic**, never produces a tile AccuTile does not (`tfc_only_total = 0` on 10 500 real projected-Gaussian cases), and has asymptotic work `O(rows)` that is **never smaller** than AccuTile's `O(min(rows,cols))`. Every TFC number that looks like a speedup (1.3564×, 1.1988×, 1.0317×, 38.2% pairs) is either a TIER-C proxy or a footprint-count statistic measured against the **weak gsplat-1.5.3 AABB baseline**, not a production training speedup vs the strong AccuTile baseline. The only measured real compute that survives AccuTile is **orthogonal to footprint** (backward block-level atomic aggregation, forward staging floor / per-tile reject). **Stop all new TFC CUDA/training.**

---

## 1. Gate N0 — novelty (FAIL → footprint-only)

### 1.1 What TFC mode-1 actually is
TFC mode-1 replaces the gsplat-1.5.3 per-axis AABB rectangle with an **ellipse-exact per-row tile range**: for each screen row `y` inside the ellipse's vertical span, solve the conic `A·dx² + 2B·dx·dy + C·dy² = κ` analytically for the horizontal tile span `[tmin_x(y), tmax_x(y)]` and emit the contiguous tile range. (`_tfc1_oracle/src/tfc1_rowwise.cu`, frozen SHA-256 `a44e4ee63c7ba9d713a4489693d189cb3fd6e7ba95764e491027c0639e34e68e` — re-hashed live this task, **intact**.)

### 1.2 What upstream AccuTile actually is
gsplat upstream main `IntersectTile.cu` (local copy `public_baselines/gsplat/gsplat/cuda/csrc/IntersectTile.cu`, SHA-256 `1f331b6dc73cf16f1e496f4a525f3abec3ef1bc132b126408f04a46f2599e712`, == current-main commit `512d366` snapshot) provides `accutile_ellipse_intersection()` + `accutile_process_tiles()` with `bool isY = y_span < x_span;`. It computes the opacity-threshold ellipse, its **SnugBox**, iterates strips along the **shorter** dimension, does one analytic ellipse cross-range per strip boundary, and emits the contiguous tile range. In-source comments attribute this to **Speedy-Splat** (`reports/final_sprint/accutile_source_audit.md`; upstream commit `28e794ca44a4c25ffc39175370c5ee7b38bfcc36`, PR #927 merge `3d4f9027`).

### 1.3 Direct source-level equivalence (`_tfc_prior_art_audit/TFC_VS_ACCUTILE_MEMBERSHIP.json`)
Device-side CUDA enumeration, both kernels embedded verbatim, nvcc 11.8.89, sm_80, cohort MX, tile 16:
- **Synthetic** 330 cases (8 categories): exact-set-equal 330/330, `tfc_only_total = 0`, `acc_only_total = 0`.
- **Real** Mip-NeRF360 projected Gaussians, N=3500 each × {room, bicycle, garden} = 10 500 cases: exact-set-equal 8 243, disagreements 2 257, **`tfc_only_tiles_total = 0`**, `acc_only_tiles_total = 2257`, `max_acc_only = 1`, overflow 0.

**Interpretation (verbatim from the artifact):** TFC mode-1's exact tile set is a **strict subset** of AccuTile's conservative set. Every disagreement is AccuTile emitting **exactly one extra outward-rounded boundary tile** per Gaussian in some rows; there are **zero TFC-only tiles**. This is floating-boundary conservatism, not an algorithmic difference. TFC1 mode-1 is the **rows-only special case of AccuTile that omits the `isY` shorter-axis heuristic** and uses exact per-row min/max instead of conservative SnugBox rounding.

### 1.4 Asymptotic work is never better (`_tfc_prior_art_audit/TFC_COMPLEXITY_OPERATION_AUDIT.json`)
- TFC mode-1: `O(rows) = O(2·ry/ts)`, always rows, `~5 sqrt + 2 div` per strip, contiguous emit.
- AccuTile: `O(min(rows,cols))`, `isY = y_span<x_span`, `~2 sqrt + 1 div` per strip, contiguous emit.
- `rows / min(rows,cols) >= 1.0` ⇒ **TFC does MORE strip iterations whenever the ellipse is tall (rows>cols)**, never fewer. The only structural lever TFC could have (fewer strips) is exactly the shorter-axis selection TFC omits.

### 1.5 Why the footprint-count win does not survive the strong baseline (`_tfc_prior_art_audit/TFC_CURRENT_GSPLAT_BASELINE_AUDIT.json`)
gsplat-1.5.3 AABB == TFC mode-0 (0% delta). Current-main AccuTile == TFC mode-1 tile set (both ellipse-exact, same roots, up to boundary rounding). Therefore the **42.91%** intersection-count advantage of TFC over gsplat-1.5.3 is **inherent to ellipse-exact enumeration**, which AccuTile also achieves. Against current main, **TFC's intersection-count advantage is ~0**.

**N0 verdict:** the ellipse-exact footprint, "exact tile overlap", and "tighten the bounding box before binning" are all explicitly covered by Speedy-Splat SnugBox/AccuTile, FlashGS fast ellipse-tile test, and gsplat upstream AccuTile. TFC adds **no previously missing execution-level mechanism**. Automatic-fail clause (Stage 0C) is triggered.

---

## 2. Gate E0 — economics / measurable headroom

### 2.1 Evidence-tier reconciliation (details in `EVIDENCE_TIER_RECONCILIATION.md`)
| Number | What it is | Tier | Status |
|---|---|---|---|
| 38.2% pairs (TFC2 pooled, 11 scenes) | frozen-state Gaussian–tile **pair count**, not GPU time | TIER-C_FROZEN_PLY | OBSERVED |
| 1.3564× (TFC2) | legacy forward-rendering-pipeline timing | TIER-C_FWD_PIPELINE | HISTORICAL, not production |
| 1.1988× (PBR0 native) | decomposed stage-sum proxy | TIER_C_DECOMPOSED | NOT production |
| 1.0317× (PUB0 current-cohort) | decomposed proxy | TIER_C_DECOMPOSED | NOT production |
| TFC_FC0 30K wall geomean **1.0594×** | production 30K wall-clock, **vs gsplat-1.5.3 AABB** | TIER-A | real, but vs weak baseline; **quality gate FAIL** |
| AccuTile30K 30K wall geomean **1.1039×** | production 30K, **B1A vs B1 (AABB)**, anysplats torch 2.4.1 | TIER-A | real, but prior art; different cohort |

Per `PUBLICATION_CONVERGENCE_PACK_V2_1/PAPER1_OPEN_GAPS.json` (inspected **before** citing older PAPER0/PBR0/PUB0 scalars, as Stage 0A.5 requires): "TFC 1.3564× is forward-rendering-pipeline timing only; PBR0 1.1988× is a decomposed proxy, NOT production." None of 1.3564× / 1.1988× / 1.0317× is promoted to a production training speedup.

### 2.2 No-TFC-vs-B2 production measurement (Stage 0D)
- **B1** = clean frozen gsplat v1.5.3 per-axis AABB (weak for 2026 novelty).
- **B2** = gsplat upstream main AccuTile (PR #927), strip-based SnugBox ellipse intersection (prior art).
- There is **no same-cohort production run of TFC against B2**. Per protocol this is recorded `NOT_MEASURED` and **no unsupported speedup is computed**. Analytically (§1.3–1.5) TFC removes **~0 additional intersection work** beyond AccuTile.
- The AccuTile 30K run (`reports/accutile30k/`, MX A100-PCIE, anysplats torch 2.4.1, gsplat-true-accutile-v153) is the **strong-baseline** reference: geomean full-training speedup **1.1039×** vs B1 AABB, fwd 1.248× / bwd 1.118× / tot 1.178× (3-scene SXM4, `reports/accutile/accutile-a100-final.md`). This is **prior art**, not a TFC contribution.

### 2.3 The real remaining compute after AccuTile (orthogonal to TFC) — `reports/accutile/accutile-b1a-r6a-rescreen.md`
Direct atomic instrumentation (CUDA kernel, self-contained `bin_final`):
- AccuTile reduces intersection count **46–53%** but **atomic count reduction = 0.0** — **B1 and B1A have IDENTICAL warp-atomic counts** (max diff 217/292M = 0.00007%).
- Backward time split: **~20%** scales with intersections (iteration overhead), **~65% is atomics (unchanged)**, ~15% other.
- Block-level gradient aggregation (replace 8 warp-leader atomics per (block,Gaussian) with 1 block-leader) has **7.1–7.9× duplication factor** on B1A (identical to B1) ⇒ **orthogonal and additive** with AccuTile. Conservative E2E opportunity on B1A: **5–12%** at 20–30% atomic-reduction efficiency.
- These are the only **measured** real-work levers, and none is a footprint mechanism.

Cross-check vs `CLOSED_DIRECTIONS_20260930.md`: forward staging-floor / per-tile reject levers are already bounded — `LAW-AABB-TILE` (TS16 1.32–1.45×; TS32 1.63–1.85×), `LAW-ALPHAWALK` (~1.2×), per-pixel transmittance early-stop = gsplat built-in (1–2%). All are forward-side and orthogonal to TFC.

---

## 3. Gate C0 — correctness (NOT RUN)
Requires future explicit GPU authorization. Not started in P0. (See §4 for what the existing TFC_FC0 gradient evidence already shows.)

---

## 4. TFC gradient difference — is it real, does it need repair?

**Real, and it accumulates — but it is not a TFC-specific bug to "fix."**
- PBR0 (`_pbr0_oracle/PBR0_H1_HARNESS_AUDIT.json`, `_pbr0_oracle/PBR0_BENCHMARK.json`): train-cam151 A-vs-A cosine 0.9999522864818573 vs A-vs-B cosine 0.9993930160999298 ⇒ `ab_vs_aa = REAL_DIFF` above the near-deterministic A-A self-noise floor. Mechanism: `footprint_intersect` runs on **detached** `means2d/conics/opacity` ⇒ the discrete tile-membership choice is **not** differentiated ⇒ the changed tile set changes gradient routing/aggregation, not a direct derivative of tile selection.
- TFC_FC0 (`_tfc_fc0_pack/deliverables/`): over full 30K the difference **accumulates** — garden dPSNR **−0.0717 dB** (exceeds the 0.05 dB prereg gate) while room/bicycle pass ⇒ **quality gate FAIL, fail-closed**. Densification counts also diverge (garden B retains ~7.2K fewer Gaussians).
- The TFC2 self-imposed `gradient_cosine ≥ 0.99999` gate "fails" on train, but the determinism analysis (`TFC2_VERDICT.json`) shows train non-determinism stdev 1.8e-5 > gate gap 7.3e-6 — i.e. that specific proxy sits at the CUDA atomic-add noise floor. **This does NOT license a blanket "only atomic noise" claim** for the footprint perturbation itself: PBR0 + TFC_FC0 demonstrate a real, above-self-noise A-vs-B difference that accumulates into a measurable quality delta on garden.
- **This perturbation is inherent to any footprint culling.** AccuTile B1A's own 30K run shows garden ΔPSNR **−0.714 dB** (`reports/accutile30k/final-verdict.md`) — larger than TFC's −0.0717 dB — i.e. ellipse-tile culling changes the trajectory regardless of which implementation. Therefore: **do not invest in a TFC-specific gradient correction**; treat the trajectory/quality perturbation as a known, must-report property of any adopted footprint method.

---

## 5. Answers to the research lead's five questions

1. **What is novel beyond Speedy-Splat, FlashGS, upstream gsplat AccuTile? (quote functions/equations)** — **Nothing.** TFC mode-1 tile set == AccuTile `accutile_process_tiles`/`accutile_ellipse_intersection` (`isY = y_span < x_span`) ellipse-exact set; `tfc_only_total = 0` on 10 500 real cases; `O(rows) ≥ O(min(rows,cols))`; TFC omits the `isY` shorter-axis heuristic. No distinct execution-level mechanism, no new backward/sorting-aware interface, no independently-measured new work removed beyond AccuTile.

2. **Which expensive production work does the novel delta eliminate beyond the strong baseline?** — **None.** TFC removes ~0 additional intersection work beyond AccuTile. The 1.3564×/1.1988×/1.0317×/38.2% figures are TIER-C proxies/counts vs the weak AABB baseline. The only measured real compute that survives AccuTile is orthogonal (backward block-level atomic aggregation ~65% of backward, 7.5× duplication, conservative E2E 5–12%, additive with AccuTile) — and that is **not TFC**.

3. **Does the TFC gradient difference require new correction / further research?** — **Characterize, do not repair (as a TFC bug).** The A-vs-B difference is real (above A-A self-noise) and accumulates over 30K (garden −0.0717 dB). It is inherent to any footprint culling (AccuTile B1A garden −0.714 dB), not a TFC-specific defect. A blanket "only atomic noise" explanation is disallowed.

4. **Is TFC_FC0 actually complete, and what Tier-A metric supports it?** — **Yes, complete**, but vs the weak baseline. Real 30K: room/garden/bicycle, seed 42, gsplat v1.5.3 EPIC-05 A100-SXM4, torch 2.10. Tier-A metric: `TFC_FC0_TIMING` `speedup_30k` geomean **1.0594×** (room 1.0467, garden 1.0679, bicycle 1.0639) = AABB-baseline wall-clock ratio. **Quality gate FAIL** (garden dPSNR −0.0717 dB > 0.05 dB) ⇒ fail-closed. It is **not** a claim vs AccuTile and is on a different cohort than the AccuTile 30K run (1.1039×, MX, torch 2.4.1).

5. **Most economical next falsification experiment, if one exists? If no delta, STOP.** — **STOP for TFC** (footprint-only; DECISION_PROTOCOL default). No non-overlap credible delta exists, so `STAGE1_OPTIONAL_PREREG.md` is intentionally **not** produced. The only orthogonal real-work direction with measured headroom is **backward block-level atomic aggregation** (R6-A-class) — a separate mechanism, not TFC — which would need its own prereg and explicit GPU authorization; it is **not** started here.

---

## 6. Stop / permission policy

- **STOP** all new CUDA and training investment in the TFC / ellipse-footprint direction (Stage 0C automatic-fail; N0 fail).
- Historical TFC / TFC2 / PBR0 / TFC_FC0 results are **retained** as implementation baselines and negative-method evidence; nothing is deleted.
- Do **not** preempt `TEMPORAL_COVERAGE-R1`, `WV-HIER`, or C51 archived results; no frozen gate/checkpoint/camera/loss/LR/seed altered.
- All 1.3564× / 1.1988× / 1.0317× / 38.2% claims remain **non-headline** per `DECISION_PROTOCOL.md` forbidden-claims list.
