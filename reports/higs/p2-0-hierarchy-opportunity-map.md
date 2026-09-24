# P2-0 — HiGS Hierarchy Opportunity Map

> Oracle only. No kernel implemented, no final-stack change, no 30K run.
> Environment: A100-PCIE-40GB / SM80, torch 2.9.1+cu128, CUDA 12.8.
> Frozen stack: **F9 (gatherless projected producer) + SCALAR_ADJOINT + H8-MR**, backward_mode `higs_native`, enable_culling.
> Provenance, raw tables: `artifacts/higs-p2-0/*.json`.

---

## 1. Current C0 stage breakdown (MEASURED, torch.profiler, real composed forward; summed over 30 reps → per-rep)

Resolution 2048×1365/1361/1327; `F4` = flat fine-tile partition (`intersect_tile` + sort + offset), `F5` = fine raster.

| Stage (ms/rep) | room | bicycle | garden |
|---|---|---|---|
| F1 gather (visible) | 0.077 | 0.283 | 0.043 |
| F2 projection (F9 producer) | 0.033 | 0.110 | 0.021 |
| F3 SH eval | 0.003 | 0.004 | 0.003 |
| F4 intersect_struct | 0.272 | 0.344 | 0.182 |
| F4 sort + offset | 0.188 | 0.230 | 0.134 |
| F5 raster | 0.683 | 0.899 | 0.399 |
| (unclassified / bookkeeping) | 0.113 | 0.159 | 0.105 |
| **kernel total** | **1.368** | **2.029** | **0.887** |

Full-forward (event median): room 1.549 ms, bicycle 1.942 ms, garden 1.178 ms.

**Readout:** `F5` raster dominates (44–50% of forward kernel) and is **not** reduced by the hierarchy. The flat partition `F4` (intersect + global sort + offset) is **28–36%** of forward kernel; the global sort + offset alone is **11–15%**.

## 2. Native-HiGS structural forward (DERIVED lower bound; no equivalent native trainable forward was executed)

Macro tile = 128×64 px, 32 fine tiles/macro, 1024-G macro batches, 32-G mini-batches. Structural counters (measured): macro entries room 124150 / bicycle 311675 / garden 66724 vs fine pairs 953144 / 1412193 / 533928 → compression 7.68× / 4.53× / 8.00×. The native forward replaces flat fine-tile global sort with macro partition + active-fine-tile scheduling, operating on ~5–8× fewer partition entries. F1/F2/F3+F5 stay.

## 3. Forward hierarchy ceiling (HIPOTHETICAL / derived)

`realistic ≈ 16–21% forward (mean 19.1%) → ≈ 7.7% F+B` (aggressive upper bound if all F4 removed: 10–16% F+B). Passes the ≥10% forward signal on all scenes; misses the ≥20% *strong* signal on bicycle (16.5%). Since forward is only ~38% of F+B, the E2E ceiling stays ~4–6% full-iteration.

## 4. Macro/mini-batch distribution (MEASURED)

Active fine tiles owned per macro entry: room p50 5 / p90 18 / p99 32; bicycle p50 3 / p90 9; garden p50 5 / p90 20. Active-tile popcount per 32-G batch: room mean 25.3, bicycle 22.4, garden 25.8 (max 32). Backward active mini-batches per fine tile: mean 4–8, but **p99 = max = 32** (dense-tile long tail).

## 5. Backward tail imbalance (MEASURED)

Backward search length p50/p90/p99/max: room 80/150/250/424; bicycle 86/272/831/1563; garden 43/89/175/299. Tail skip already removed by current `last_id` loop: room 13.8%, bicycle 5.3%, garden 10.1%. Bicycle shows the strongest long tail.

## 6. Hierarchical backward realistic ceiling (derived)

B1 (macro): 5–7% backward; B2 (32-G): 8–11% backward. Non-free prefix/suffix + partial-adjoint + sync + reduction consume the modest exposed parallelism (typical tiles have only 3–5 active mini-batches). **Below the ≥15% promotion signal → DEFER (B).**

## 7. Forward-discovered inactive-work fraction (MEASURED)

Dead (mini-batch, fine-tile) groups that the forward already resolves as inactive, and the current backward still **loads data for**: room **20.9%**, bicycle **30.1%**, garden **19.8%** (mean 23.7%). At 1024-G macro granularity this is only 0.3–2.3% → the 32-G mini-batch level is where the hierarchy wins. This is algorithmic forward output (fine-tile active masks), **not** a visibility re-prediction → distinct from closed H4.

## 8. Metadata/state cost for reuse (derived)

Per-frame forward+backward metadata ≈ 1.5 MB (last_active_batch int16 + active bitset), ≈ 2–5% of the saved backward load traffic — **well under the 25% cap**.

## 9. Work-reuse realistic net ceiling

Realistic backward time saving ≈ 13–20% (mean 15.4%) → **≈ 9.2% F+B** (room 8.2%, bicycle 12.3%, garden 7.2%). Passes the ≥20% exact-backward-removable signal on 2/3 scenes (bicycle 30.1%, room 20.9%; garden 19.8% marginal). **Strongest candidate.**

## 10. F9→partition temporary traffic (derived)

F9 writes 48 B/visible compact projected state + F4 rereads; partition-only temp footprint room 7.2 MB, bicycle 18.3 MB, garden 3.9 MB. F2 producer itself is only <6% of forward.

## 11. Producer-partition fusion realistic ceiling

Realistic ≈ **3–3.5% forward** (upper bound ~6%); **fails the ≥5% full-forward gate** on realistic. → DEFER / conditional DROP.

## 12. Projected fusion resource risk (measured + derived)

F9 producer = 43 regs / 75% occupancy / 0 spills; projection VJP = 96 regs / 31.25%. Fusing macro-coverage logic projects regs to ~55–72 → occupancy 75%→50–62%, spill risk MEDIUM-HIGH — the same occupancy-cliff failure mode as closed H4. This is the binding constraint that rejects D.

## 13. Z-order locality ceiling (measured)

Gather-order gid stride is already **~99% within a stride-32 window** (0.994–0.999) across all scenes. Z/Morton reorder adds <1% full-iter → **DROP (E1)**. (PRIOR ART / engineering, not a contribution.)

## 14. VJP→Adam fusion ceiling (derived)

Removable grad write + reread ≈ 31–274 MB/frame (plus launch overhead); m/v/param traffic is non-removable. Realistic **≈ 4% full-iter** with low risk → **PROMOTE_ENGINEERING (E2)**.

---

## 15. Ranked Phase-2 candidate table

| Id | Mechanism | HiGS-spec. | Exactness | Measured opportunity | Realistic net | Complexity | Resource risk | E2E | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **C** | Forward-discovered hierarchical work reuse | HIGH | exact | dead batch-tile 21–30% | ~9.2% F+B | MED | LOW (meta 2–5%) | ~5–7% | **PROMOTE_CORE #1** |
| **A** | Native hierarchical forward gap | HIGH | exact | flat F4 = 28–36% fwd | ~19% fwd / ~7.7% F+B | HIGH | MED (F5 dominates) | ~4–6% | **PROMOTE_CORE #2** |
| **E2** | VJP→Adam fusion | LOW (eng) | exact | 31–274 MB grad rt | ~4% iter | LOW | LOW | ~4% | **PROMOTE_ENGINEERING** |
| **B** | Hierarchical backward parallelism | HIGH | exact | active mb/tile 4-8, tail 32 | 5–11% bwd (<15%) | HIGH | MED-HIGH | ~2–4% | DEFER |
| **D** | F9→partition fusion | MED | exact | F2 <6% fwd | 3–6% fwd (<5%) | HIGH | HIGH (regs 72) | ~1–2% | DEFER/DROP |
| **E1** | Z/Morton locality | LOW (eng) | exact | gid stride<32 ~99% | <1% iter | MED | LOW | <1% | DROP |

## 16. PROMOTE_CORE candidate #1 — C, Forward-discovered hierarchical work reuse

Reuse the forward's already-resolved active (mini-batch, fine-tile) masks / last-active-batch frontier so the exact backward **skips data loading + processing of the 20–30% dead groups** (currently loads unconditionally). Exact, no re-prediction, HiGS-native mask granularity; metadata ≈ 2–5% of savings. Best measured evidence (bicycle 30%).

## 17. PROMOTE_CORE candidate #2 — A, Native hierarchical forward gap

Adopt the native macro→active-fine-tile scheduler in the trainable forward to remove the flat global sort + offset (11–15% forward) and restructure flat intersect (28–36% total). Exact; HiGS-specific. **Flagged:** F5 raster (44–50% of forward) is not reduced and dominates; ceiling is DERIVED/HYPOTHETICAL (no equivalent native trainable forward was executed); porting complexity is HIGH. Promoted as the second, riskier core.

## 18. PROMOTE_ENGINEERING — E2, VJP→Adam fusion

In-place grad consume (write grad, fused Adam step) removes grad write+reread+launch ≈ 4% full-iter, low risk, standard engineering.

## 19. DEFER / DROP

- **DEFER — B:** backward parallelism 5–11% < 15% gate; non-free prefix/suffix/reduction consume the tail-only parallelism.
- **DEFER — D (conditional):** producer-partition fusion <5% forward and occupancy-cliff risk (43→~72 regs) mirrors closed H4; defer unless a lean coverage kernel design can hold ≤64 regs, else DROP.
- **DROP — E1:** Z-order locality already ~99% coherent within 32-gid window; <1% full-iter.

## 20. Exact next experiment per promoted candidate

- **C (work reuse):** Instrument the native forward to emit per (mini-batch, fine-tile) active masks + last-active-batch (already generated by scheduling); replay exact backward in a **non-production oracle harness** that feeds the masks and skips dead-group loads; measure backward wall-time delta on room/bicycle/garden cam0 2048. Gate: confirm ≥15–20% backward on ≥2/3 scenes with metadata ≤10% of savings. If it holds, then measure on a **frozen 30K checkpoint** for E2E.
- **A (native forward gap):** Build a **non-production DERIVED model** (or a throwaway forward-only kernel for the macro scheduler) that reproduces the exact flat raster output but replaces F4 with macro partition + active-fine-tile scheduling; measure the forward-floor delta and confirm no occupancy cliff and byte-identical frame. Gate: ≥15% forward on ≥2/3 scenes and F5 dominance confirmed, before any trainable-path port.
- **E2 (VJP→Adam fusion):** Add a fused Adam-stable wrapper over the existing scalar-adjoint/H8-MR VJP output (write grad into optimizer-ready layout, consume in a single fused step); verify FP32-grad equivalence (reuse the h2-bwd determinism harness) and measure optimizer-kernel + grad round-trip removal on the frozen checkpoint. Gate: ≥3% full-iter with no correctness drift.

*No kernels implemented; all numbers above are oracle-level (MEASURED / DERIVED / HYPOTHETICAL as labeled).*