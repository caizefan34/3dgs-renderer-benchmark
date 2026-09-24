# Authoritative Experiment Registry — Trainable HiGS Renderer

**Status:** `AUTHORITATIVE_EXPERIMENT_REGISTRY_FROZEN` (R1 refresh)
**Date:** 2026-09-22 (initial) / **R1 refresh:** new P2 evidence registered (P2-1A 1799facc INVALID + R2 PREFLIGHT; P2-1C DROP; P2-0 reuse projections superseded; P2-1A-R0 static evidence completed; C0 timing-attribution language corrected; decision tree updated to R2 branches)
**Role:** This is the **single source of truth** for all future reports/paper writing. Every authoritative numeric claim carries a source-artifact reference. Superseded values are marked and NOT usable as publication numbers. The registry uses ONLY the frozen status vocabulary.

Machine-readable: `artifacts/research-registry/{authoritative_results.json, supersession_map.json, candidate_status.json, prohibited_claims.json}`.

**Status vocabulary (only):** `INVALID`, `SUPERSEDED`, `DROP`, `DEFER`, `KEEP_CANDIDATE`, `SUCCESSFUL_EXACT_MODULE`, `SUCCESSFUL_EXACT_COMPOSITION`, `AUTHORIZED`, `FROZEN`, `PREFLIGHT`. Ambiguous phrases ("looks promising", "probably successful", "basically done") are **prohibited**.

---

## 1. C0 authoritative state (V3)

| Field | Value |
|---|---|
| Candidate | **C0 V3** = F9 + SCALAR_ADJOINT + H8-MR (variant V3) |
| Status | **SUCCESSFUL_EXACT_COMPOSITION** |
| Correctness | **PASS** (9/9 from C0 exact composition, unmodified) |
| Publication timing | **C0-T2** (order-balanced nested F+B confirmation) |
| Timing grade | **C0_TIMING_PUBLICATION_GRADE** |
| 30K | **30K_AUTHORIZED** (decision only; the 30K benchmark is NOT yet run) |
| Deployment candidate | **V3 only** (do NOT ship F9 without H8-MR) |

**Authoritative C0-T2 nested F+B reductions (publication-grade, median ms, n=500/cell, 0 invariant failures):**

| Scene | V0 | V1 | V2 | V3 | **V3 vs V0** |
|---|---|---|---|---|---|
| room | 5.727 | 5.753 | 4.918 | **4.780** | **−16.54%** |
| bicycle | 8.178 | 8.175 | 7.250 | **6.972** | **−14.74%** |
| garden | 3.650 | 3.642 | 3.587 | **2.897** | **−20.62%** |

**Paired medians (window-matched):** room 16.88%, bicycle 14.70%, garden 20.59%.
**Fraction V3 faster:** room 98.2%, bicycle 96.4%, garden 99.4%.

Source: `reports/higs/c0-t2-order-balanced-confirmation.md` §5-6; `artifacts/higs-c0-t2/final_gate.json`, `paired_analysis.json`.

---

## 2. Superseded C0 numbers (SUPERSEDED_FOR_PUBLICATION)

The following are **NOT** publication-authoritative. They are marked, NOT deleted.

| Superseded value | Status | Remaining valid role | Superseded by |
|---|---|---|---|
| C0-T1 nested F+B 15.12/15.87/21.16% used as **publication** timing | SUPERSEDED_FOR_PUBLICATION | **ENGINEERING_TIMING_CLOSURE** (C0-T1 = engineering timing closure; NOT final publication timing) | C0-T2 balanced 16.54/14.74/20.62% |
| Original `artifacts/higs-c0/timing.csv` + `composition_accounting.json` + `final_gate.json` | SUPERSEDED_FOR_PUBLICATION | HISTORICAL only | C0-T1 then C0-T2 artifacts |
| **E_compose = 1.0** | SUPERSEDED / INVALID (tautological) | INVALID | **E_compose = NOT_COMPUTABLE** (isolated-protocol deltas are protocol-state-dependent; the additive prediction fails its validity precondition) |

C0-T1 remains the authoritative record for per-module decomposition, kernel attribution, anomaly resolutions, and E_compose adjudication. Source: `reports/higs/c0-t1-timing-closure.md` §13, §16.

---

## 3. Invalid / prohibited C0 claims

These must NOT be reused as final quantitative conclusions. Full "why invalid" in `artifacts/research-registry/prohibited_claims.json`.

| Prohibited claim | Status | Why invalid |
|---|---|---|
| **H8 ≈ 45-53% whole-backward gain** | INVALID | Instrumentation/protocol-state artifact. The room direct-protocol reading is internally inconsistent with its own kernel content (>2× gap). The discrepancy is described ONLY as protocol/GPU-state-dependent timing variability (isolated protocols with idle gaps vs the sustained nested protocol sample different timing states). **No hardware-level causal attribution** — a GPU clock/power-state oscillation mechanism is NOT established (no telemetry was collected). Decomposition: 0.462 ms kernel-attributable (true gain) + ~1.8 ms protocol-state time. The in-context nested whole-backward H8 gain is 5.10% (room) / 3.59% (bicycle) — matching H8-T0 standalone 4.31%/4.11%. |
| **E_compose = 1.0** | INVALID | Tautological; the additive prediction is NOT_COMPUTABLE on this GPU. |
| **Old impossible forward > F+B** (e.g. V0 room forward 8.716 ms > V0 room F+B 6.817 ms) | INVALID | Physically impossible; produced by measuring forward/backward/F+B in SEPARATE SUBPROCESSES with independently constructed fixture states. The nested in-iteration protocol is authoritative. |
| **Old zero-variance timing summaries** (std/p10/p90 = 0 despite a 100-sample protocol) | INVALID | Single merged values, not distributions. C0-T1/C0-T2 preserve full distributions + 6000 raw nested observations. |

Source: `reports/higs/c0-t1-timing-closure.md` §2, §8, §10, §12, §13.

---

## 4. F9 registry entry

| Field | Value |
|---|---|
| Status | **SUCCESSFUL_EXACT_FORWARD_MODULE** |
| Variant | ALGEBRAIC_EXACT_FP_REASSOCIATED |
| Patch | `1faa05c10bd18c24f2fa1daecca6b11b6a69751692d9c777d520d66c98279c13` (`patches/higs-f9-gatherless-final.patch`) |

**Authoritative F9 standalone findings:**

| | room | bicycle | garden |
|---|---|---|---|
| **forward gain** | +18.9% | +14.8% | +23.2% |
| **F+B gain** | +7.63% | +5.40% | +10.56% |

**Composition caveat (recorded separately):** Later **composed-context (C0) behavior DIFFERS** from standalone. In C0-T1 nested context, F9 forward gain was confirmed (0.916/1.986/0.761 ms, −27 to −35% forward) but F9 WITHOUT H8-MR (V2) **regresses garden** (+33.6% F+B); V3 is immune. **Do NOT substitute standalone F9 numbers for C0 composition timing.**

Source: `artifacts/higs-f9-final/final_gate.json`, `timing_breakdown.csv`; `reports/higs/c0-t1-timing-closure.md` §7, §11.

---

## 5. H8 registry entry

| Field | Value |
|---|---|
| Status | **SUCCESSFUL_EXACT_BACKWARD_MODULE** / **SAFE_COMPOSITION_MODULE** |
| Patch | `8fb23acca9841ae4d24035de9b2d0ba7725742186b7db149c44cc656415fe312` (`patches/higs-h8-mr.patch`) |

**Authoritative H8 standalone direct-backward gain (H8-T0):**

| | room | bicycle | garden | geomean |
|---|---|---|---|---|
| **direct backward** | ~4.31% | ~4.11% | ~3.54% | 3.97% |
| **F+B** | ~2.37% | ~2.14% | ~1.80% | 2.09% |

**C0 nested-context whole-backward contribution (recorded separately where supported):** room 5.10%, bicycle 3.59% (matching standalone); garden nested V2→V3 is interaction-dominated (C0-T1 item 11). The **~45-53% "H8 whole-backward gain" interpretation is PROHIBITED** (instrument artifact — see §3). The previous "30.81% room" was a TIMING_SUBTRACTION_ARTIFACT (baseline forward std=72-78 ms amplified into the backward estimate).

Source: `artifacts/higs-h8-t0/final_classification.json`; `reports/higs/c0-t1-timing-closure.md` §8, §10.

---

## 6. SCALAR_ADJOINT registry

| Field | Value |
|---|---|
| Status | **SUCCESSFUL_EXACT_BACKWARD_MODULE** |
| Historical standalone | Direct-blend redundant-multiply elimination benefit **exists** |
| C0 nested-context contribution | **Small**: room 0.050 ms (1.2%), bicycle 0.007 ms (0.1%), garden 0.004 ms (0.3%) |

**Record both contexts separately.** Do NOT imply the isolated gain transfers directly to C0. Source: `reports/higs/c0-t1-timing-closure.md` §7.

---

## 7. H4 registry

| Field | Value |
|---|---|
| Original H4-0 | **INVALID_ORACLE / SUPERSEDED** |
| Authoritative corrected oracle | **H4-0R** |
| Family status | **DROP_H4_FAMILY** |

**H4-0R corrected B16 saved visited work:**

| room | bicycle | garden | pooled |
|---|---|---|---|
| 36.34% | 53.79% | 30.74% | 43.91% |

H4-0's "97% savings" was 2-3× inflated by three accounting errors (savings formula, support condition σ≥0 instead of α-gate, V count ignoring termination).

**Integrated H4-P0 (measured — made F5 SLOWER on all 3 scenes):**

| room | bicycle | garden |
|---|---|---|
| 0.7076 → 1.3875 ms | 0.9144 → 1.8237 ms | 0.4137 → 0.8919 ms |

**Status: DROP_H4_FAMILY.** Reason: **transformation overhead + register-pressure/resource cliff overwhelms the theoretical visited-work elimination.** The corrected 31-54% arithmetic saving is real but is dominated by mask generation, transpose/publication, and register pressure at integration time.

Source: `reports/higs/h4-0r-oracle-repair.md` §1; `reports/higs/h4-p0-b16-prefilter.md`; `reports/hig-h4/h4-0-pixel-sparsity.md` (superseded banner).

---

## 8. Other closed candidates (closed; do NOT reopen without genuinely new evidence)

| Candidate | Mechanism | Status | Result |
|---|---|---|---|
| **R6-A** | Block-level gradient aggregation | **DROP** | Minimal CUDA prototype: net overhead exceeded savings (block-local reduction for every flattened intersection). Closed. |
| **WARP_ALL** | Warp-cooperative emit (1 warp/projected Gaussian) | **DROP** | Isolated pre-raster optimization positive but not promoted; 13-scene/training gates unavailable. Closed. |
| **H5 frontier load gating** | Frontier load gating | **DROP/WEAK** | WEAK. |
| **H5 batch-factored backward** | Batch-factored backward | **DROP** | DROP. |
| **H6** | One-touch coverage | **WEAK** | WEAK. |
| **H7** | Gaussian-major | **DROP** | DROP. |
| **Z/Morton locality E1** | Z/Morton locality | **DROP** | Gather-order gid stride already ~99% within stride-32 window (0.994-0.999); adds <1% full-iter. PRIOR ART / engineering, not a contribution. |
| **F9→partition fusion D** | Producer-partition fusion | **DEFER/DROP** | <5% forward gate fails; occupancy-cliff risk (43→~72 regs) mirrors closed H4. |

Negative results remain in the evidence record (skill §28). Source: `reports/r6a/r6a-minimal-cuda-prototype.md`, `reports/final_sprint/warp_emit_final.md`, `reports/higs/p2-0-hierarchy-opportunity-map.md` §15, §19.

---

## 9. P2-0 evidence levels (MEASURED / DERIVED / HYPOTHETICAL — mandatory distinction)

**MEASURED** (torch.profiler, real composed forward, 30 reps):

- Macro entries / fine pairs:
  - room: 124150 / 953144
  - bicycle: 311675 / 1412193
  - garden: 66724 / 533928
- Compression: room 7.68×, bicycle 4.53×, garden 8.00×
- Measured dead (mini-batch, fine-tile) groups the forward already resolves inactive but backward still loads: room **20.9%**, bicycle **30.1%**, garden **19.8%** (mean 23.7%)
- C0 stage breakdown (F4 flat partition 28-36% of forward kernel; F5 raster 44-50%)

**DERIVED / projected:**
- ~~13-20% backward reuse estimate~~ — **SUPERSEDED_FOR_PUBLICATION** (see supersession_map: group-count opportunity overestimated runtime-removable work; weighted entries much smaller; processing work already removed by `last_id`; only load/control remains)
- ~~~9.2% F+B reuse estimate~~ — **SUPERSEDED_FOR_PUBLICATION** (same reason)
- F4 flat partition removable (28-36% forward) — still a valid structural observation
- Metadata cost for reuse ≈ 2-5% of saved traffic (under the 25% cap)
- VJP→Adam (E2) ≈ 4% full-iter, low risk

**SUPERSEDED vs NOT superseded (critical distinction):** the **MEASURED dead-group counts themselves (20.9/30.1/19.8%) are NOT superseded** — they remain MEASURED P2-0 evidence. What IS superseded is the **reuse projection** derived from them (13-20% backward, ~9.2% F+B), because entry-weighted removable work is much smaller (10.8/4.7/8.2%) and 100% overlaps the existing `last_id` processing skip.

**HYPOTHETICAL:**
- **16-21% native hierarchical forward potential** (mean 19.1%) — UNTIL P2-1A R2 runtime executes. No equivalent native trainable forward has been executed, so this is a derived ceiling, not a measurement.

**Mandatory:** these three levels are NOT interchangeable. P2-0's observed **gross** opportunity (dead groups 20.9/30.1/19.8%) is **NOT equivalent to final incremental removable work** — the P2-1C empirical analysis established this (see §11).

Source: `reports/higs/p2-0-hierarchy-opportunity-map.md`.

---

## 10. P2-1A registry

**Overall P2-1A performance hypothesis remains UNRESOLVED. Do NOT classify P2-1A STRONG/MARGINAL/WEAK until R2 executes.**

| Variant | Status | Detail |
|---|---|---|
| **P2_1A_1799FACC** | **INVALID** | Deterministic FP16/FP32 entry-contract failure. `.so` = `1799facc0fa3c30375aad4edeff28b091ce35891cbb794f467c0984e76e14089`. A correctness/contract failure, NOT a performance result. |
| **P2_1A_R2** | **PREFLIGHT** | Contract-true projected-state adapter repair. NOT yet executed; no performance/correctness claim until R2 runs. |

Original patch: `ab860d771b4adb1afd72615ecaa5342d020de43d421ff653990f083cd16b496c`.

### P2-1A-R0 — completed static resource evidence (**RESOURCE_DELTA_MEDIUM**)

| Static fact | Value |
|---|---|
| FP16 raster | **3 CTA/SM** |
| FP32 raster | **3 CTA/SM** |
| Occupancy | **46.875% both** |
| FP32 `<8>` spill | **AMPLIFIED** |
| FP32 `<16>` spill | **NEW** |

The static residency audit is **COMPLETE** (no longer "pending corrected audit"): the resource delta is **MEDIUM** — the FP16→FP32 port does not cliff to 1 CTA/SM (registers and shared memory tie; 47760/163840 B = 29.15% permits three CTAs by smem). `47760 B → 1 CTA/SM` is NOT fact. Spills are present but the static delta is medium, not catastrophic. **This is static evidence only; runtime residency/performance still requires P2-1A R2 execution.**

---

## 11. P2-1C registry — **DROP**

| Field | Value |
|---|---|
| Status | **DROP** (empirical implementation decision) |
| Formal oracle/preflight gate | may be noted **MARGINAL** — but the empirical implementation decision is **DROP** |

**Empirical evidence (room / bicycle / garden):**

| Quantity | room | bicycle | garden |
|---|---|---|---|
| Gross dead groups | 20.9% | 30.1% | 19.8% |
| Entry-weighted | **10.8%** | **4.7%** | **8.2%** |
| Overlap with existing `last_id` processing skip | **100%** | **100%** | **100%** |
| Frozen 128-G fully skippable work | **0.40%** | **2.00%** | **1.08%** |
| Equivalent measured H5-0R timing | **+1.54%** | **-0.01%** | **0.00%** |

**Why DROP:** the group-count opportunity overstated runtime-removable work — weighted entries are much smaller than gross (10.8/4.7/8.2% vs 20.9/30.1/19.8%); the processing work was **already removed by the existing `last_id` loop** (100% overlap), so only load/control remains; the frozen 128-G fully skippable work is tiny (0.40/2.00/1.08%); and the equivalent measured H5-0R timing is +1.54/-0.01/0.00%.

**NOT superseded:** the MEASURED dead-group counts (20.9/30.1/19.8%) remain valid P2-0 MEASURED evidence. What is superseded is the reuse projection derived from them (§9).

**P2-1C is removed from the planned final stack** in all decision-tree branches (§13).

---

## 12. Prior-art / novelty language (claim control)

**Allowed current high-level research framing:**
- **Phase 1:** `Exact Work Elimination + Sufficient-Statistic Differentiation`
- **Phase 2 (if P2 succeeds):** `Hierarchy-Shared Forward/Backward Execution`

**Do NOT register** these unsupported claims **without a separate literature audit:**
- ❌ "first differentiable HiGS"
- ❌ "first hierarchical Gaussian renderer"
- ❌ "first optimizer fusion"
- ❌ "first tile-local culling"

Novelty claims require prior-art evidence (skill §26). Source: `artifacts/research-registry/prohibited_claims.json` → `prior_art_novelty_claim_control`.

---

## 13. Final renderer decision tree (ACTIVE)

```
P2-1A R2
│
├─ correctness fail
│    → DROP native hierarchy → C0 V3
│
├─ STRONG
│    → keep native forward
│
├─ RESOURCE_LIMITED
│    → exactly one targeted raster rescue → retest
│
└─ WEAK
     → DROP native hierarchy → C0 V3

P2-1C = DROP in all branches

After final forward architecture:
→ E2 gate
→ final candidate
→ FINAL 30K
```

Notes on the tree:
- **P2-1A is NOT classified STRONG/MARGINAL/WEAK until R2 executes** — the branches are the pre-registered outcomes of R2, not current claims.
- **P2-1C = DROP in all branches** — it is removed from the planned final stack regardless of the R2 outcome.
- **RESOURCE_LIMITED** permits exactly ONE targeted raster rescue followed by a retest — not an open-ended sequence of rescues.
- **E2** is gated only after the final forward architecture is decided.

The FINAL 30K harness is frozen to accept **C0_V3** (ready now) and **P2_FINAL_V2** (accept-later, no protocol change) under either branch. P2_FINAL_V2's composition is now: **C0 V3 + P2-1A native hierarchical forward (if R2 passes) + optional E2** (P2-1C removed).

---

## 14. Validation checklist (before completion)

- ✅ **All authoritative numeric claims have source-artifact references** — every number in `authoritative_results.json` carries a `source` field pointing to a report/artifact.
- ✅ **All superseded values are marked** — `supersession_map.json` marks the superseded entries SUPERSEDED_FOR_PUBLICATION/INVALID (none deleted), now including the P2-0 reuse projections (13-20% backward, ~9.2% F+B) and the 1799facc prototype .so.
- ✅ **All candidate states are mutually consistent** — `candidate_status.json` uses ONLY the frozen vocabulary; C0_V3=SUCCESSFUL_EXACT_COMPOSITION+FROZEN+AUTHORIZED, F9/SCALAR/H8=SUCCESSFUL_EXACT_MODULE, H4=DROP, P2_1A_1799FACC=INVALID, P2_1A_R2=PREFLIGHT, P2_1A_R0=RESOURCE_DELTA_MEDIUM (static evidence), P2-1C=DROP, P2-1A overall = UNRESOLVED (no STRONG/MARGINAL/WEAK until R2).
- ✅ **No invalid C0 timing remains publication-authoritative** — C0-T2 balanced values (16.54/14.74/20.62%) are the ONLY C0 publication timing; C0-T1 = ENGINEERING_TIMING_CLOSURE; original C0 timing + E_compose=1.0 = SUPERSEDED/INVALID.
- ✅ **P2-0 measured/derived/hypothetical distinctions are explicit** — §9 separates MEASURED (macro/fine counts, compression, dead groups — NOT superseded) / DERIVED (reuse projections — SUPERSEDED) / HYPOTHETICAL (16-21% forward, until R2).
- ✅ **No hardware-level causal attribution for the H8/C0 timing anomaly** — only "protocol/GPU-state-dependent timing variability" / "instrumentation/protocol-state artifact"; the GPU clock/power-state oscillation mechanism claim is PROHIBITED (no telemetry established it).
- ✅ **Final benchmark has identical protocol across candidates** — `protocol.json` frozen; `validate_candidates.py` Gate 3 ABORTS on any experiment-critical mismatch.
- ✅ **TTQ censoring semantics frozen** — ttq=null + TTQ_NOT_REACHED + censor_time (never treated as achieved TTQ); geomean over comparable reached/reached pairs only, with explicit reach counts.
- ✅ **Garden quality policy frozen** — garden included in the primary 13-scene aggregate; optional pre-labeled sensitivity analysis excluding garden; never excluded only from unfavorable conclusions.

**Result: `AUTHORITATIVE_EXPERIMENT_REGISTRY_FROZEN` = PASS** (refreshed; see final response item 20).
