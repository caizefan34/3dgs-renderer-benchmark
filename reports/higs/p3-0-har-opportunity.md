# P3-0 — Hierarchical Adjoint Reduction (HAR) Opportunity Oracle

> **ORACLE / MEASUREMENT ONLY.** No CUDA kernels implemented. C0 not modified. P2-1A not reopened. P2-1C not reopened. No native HiGS mask used as rendering truth. No hypothetical implementation benchmarked.
> **Classification: `P3_HAR_WEAK` — DROP. P3-1 prototype NOT authorized.**
> Environment of record: A100-PCIE-40GB (SM80, 108 SMs, 40 MB L2), CUDA 12.8, torch 2.9.1+cu128. All MEASURED values come from that cohort's frozen experiments; this session derived the oracle from them (local host has only an RTX 5070 = different cohort, excluded per protocol; mx fixtures not mounted locally).
> Artifacts: `artifacts/higs-p3-0/` (23 JSONs). Structural recomputation script: `scripts/p3/p3_0_har_measure.py`.

---

## 0. Frozen baseline (untouched)

C0 V3 = F9 + SCALAR_ADJOINT + H8-MR, status `SUCCESSFUL_EXACT_COMPOSITION`, timing `C0_TIMING_PUBLICATION_GRADE` (C0-T2 balanced nested F+B: V3 vs V0 −16.54% / −14.74% / −20.62%; V3 F+B medians **4.780 / 6.972 / 2.897 ms**). C0 publication timing was **not rerun or reinterpreted**. P2-1A and P2-1C remain **DROP** per the task directive; nothing here reopens either. The deployed backward is `higs_blend_bwd_px_kernel<CDIM=3, PX=2, SCALAR_ADJOINT=true, H8_MR=true>` (block 128 = 4 warps, one 16×16 fine tile per block) → `higs_projection_bwd_kernel<true>` → `higs_sh_vjp_grid_kernel`.

## 1. The hypothesis, stated correctly

HAR is **not** "use hierarchy to decide forward contributors" (closed). It is: pixel-local exact differentiation produces associative adjoint contributions `A_{g,t}`, and those may be reduced hierarchically (macro = 8×4 fine tiles) before global per-Gaussian accumulation. Only the reduction order changes; the hierarchy **never decides whether a splat contributes** — a (macro, gid) packet exists *iff* a pixel-local contribution already exists (backward-active support = exact fine support × `last_id` frontier, H7-B0). Target exactness class: **`ALGEBRAIC_EXACT_FP_REASSOCIATED`** — achieved (§3).

## 2. The minimal exact adjoint packet (§3 of the task)

Audited from the frozen C0 V3 backward (`adjoint_packet_spec.json`):

| Packet | Scalars | Bytes/partial | Fields | Status |
|---|---:|---:|---|---|
| **P_MIN** | 4 | 16 | v_colors[3] + v_opacity(=R0) | incomplete (no geometry) |
| **P_GEOM** | **9** | **36** | v_colors[3], R0, Sx, Sy, Sxx, Sxy, Syy | **minimal complete** |
| P_FULL | 11 | 44 | P_GEOM + v_means2d_abs[2] | densification absgrad — **not in the frozen path** (`compute_abs=false`) |

**P_GEOM is the answer**: 9 FP32 scalars = **36 bytes** per (producer-unit, Gaussian) partial. v_means2d/v_conics *real* gradients are excluded — under H8-MR they are reconstructed from the moments + forward conic inside the projection VJP (`vx=A·Sx+B·Sy`, `vy=B·Sx+C·Sy`, `vA=0.5·Sxx`, `vB=Sxy`, `vC=0.5·Syy`, `v_opacity=R0`), so they are "reconstructible later" and must not be carried. No depth-channel fields exist in the frozen RGB path (depth_channel = −1). Downstream consumers: projection VJP (moment reconstruction → 10 master atomics per visible G), SH VJP (48+3 atomics per visible G), reduce/optimizer.

**Key structural fact:** the naive non-H8 geometry packet is *also* 9 scalars / 36 bytes. H8 reparameterizes the same 6-dimensional adjoint; it does **not** compress it (§9).

## 3. Exactness proof (§4)

Per-packet-component classification (`exactness_proof.json`):

- **SAFE_TO_REDUCE** (associative/commutative up to FP32 reassociation): v_colors[3], v_opacity/R0, Sx, Sy, Sxx, Sxy, Syy — the entire P_GEOM packet. Each is a per-contribution scalar summed by unordered `atomicAdd` today; any regrouping computes the same multiset sum. H8 moments are sufficient statistics — the reconstruction is a *linear map applied after the full sum*, using only the forward conic, so `linear(Σ) = Σ(linear)` exactly. (v_means2d_abs[2] would be equally reducible if enabled; v_backgrounds is reducable but HAR-irrelevant.)
- **REQUIRES_PIXEL_RECURRENCE**: T (transmittance product), SCALAR_ADJOINT's `buffer_dot` (running per-pixel state fed back into the chain), `last_id`/`bin_final` (read-only forward state). These stay pixel-local by construction.
- **ORDER_DEPENDENT**: the alpha/transmittance backward recurrence itself — HAR does not touch it.

HAR provably does **not** change: the T recurrence, the alpha derivative (eval_gaussian_weight / MAX_ALPHA branch), `last_id` semantics (bin_final per pixel, warp max, loop bound), H8 moment semantics (same 5-FMUL bodies, same reconstruction site), SCALAR_ADJOINT semantics (same 2-scalar pixel state), densification semantics (compute_abs=false; no absgrad exists to change). Harness corroboration: H2-BWD-0 batch-VJP validation 10/10 (max error 6.78×10⁻²¹), H5-1 FP32 reassociation class (rel_L2 2.9×10⁻⁸, cosine ≈ 1.0), C0 9/9 gradient pairs.

## 4. Exact macro reuse — recomputed from C0 support (§5)

`D_macro = N_fine_pairs / N_unique_(macro,G)`, macro = 8×4 fine tiles, **recomputed from the exact C0 fine-tile workload** (exact R2 macro representation, validated 0/0/0/0 missing/extra/duplicate/wrong-ID vs B2 fine pairs on room+garden). The old native HiGS 4.5–8× numbers were **not** reused.

| Scene | N_fine_pairs | N_unique(macro,G) | **D_macro (support)** | backward-active basis | **D_backward** |
|---|---:|---:|---:|---|---:|
| room | 953,144 | 124,017 | **7.69** | 821,777 / 119,696 | **6.87** |
| bicycle | 1,412,193 | 311,610 | **4.53** | 1,337,828 / 306,512 | **4.37** |
| garden | 533,928 | 66,682 | **8.01** | 479,864 / 62,148 | **7.72** |

Per-(macro,G)-entry fine-tile coverage: room mean 7.69 / p50 5 / p75 10 / p90 18 / p95 26 / p99 32 / max 32; bicycle 4.53 / 3 / 5 / 9 / 14 / 31 / 32; garden 8.01 / 5 / 11 / 20 / 26 / 32 / 32 (all MEASURED, `exact_macro_reuse.json`). The backward-effective factor (inside the `last_id` frontier — the contributions that actually exist) is the one HAR can harvest: **6.87 / 4.37 / 7.72**.

## 5. Multiplicity distribution (§6)

Unweighted (records) vs weighted (contribution volume), `macro_multiplicity.json`:

| Bucket | room unweighted / weighted | bicycle u / w | garden u / w |
|---|---|---|---|
| 1 | 10.8% / 1.4% | 19.5% / 4.3% | 11.3% / 1.4% |
| 2 | 14.8% / 3.8% | 25.3% / 11.2% | 14.5% / 3.6% |
| 3–4 | 20.9% / 9.5% | 24.7% / 19.1% | 19.4% / 8.5% |
| 5–8 | 24.6% / 20.8% | 19.8% / 28.4% | 23.2% / 18.9% |
| 9–16 | 17.5% / 28.5% | 7.1% / 19.7% | 17.8% / 27.8% |
| 17–24 | 5.7% / 15.1% | 1.7% / 7.9% | 8.2% / 20.9% |
| 25–32 | 5.7% / 21.2% | 1.8% / 11.4% | 5.6% / 20.0% |

45–69% of records touch ≤4 fine tiles (reuse ≈ nil for them); the 17–32-tile heavy tail is 11.3% / 3.5% / 13.8% of records but carries **36.3% / 19.3% / 40.9%** of the volume. Reuse is real but concentrated.

## 6. Current accumulation topology (§7)

Per (warp, t) flush, the warp leader issues **9 scalar atomicAdds** after 3 warpSums: v_colors[3] (12 B), v_means2d→(Sx,Sy)[2] (8 B), v_conics→(Sxx,Sxy,Syy)[3] (12 B), v_opacities→R0[1] (4 B) — 36 RMW bytes per flush, all in the flat per-visible-Gaussian rows. Master-side: projection VJP 10 atomics/visible G (40 B), SH VJP 51 atomics/visible G (204 B) — low contention, downstream of the packet. Measured flush volumes: **3,690,242 (room) / 5,023,034 (bicycle)**; garden ≈ 2,066,300 (DERIVED, room warp-coverage 3.8717 assumed). ncu hardware counters are unavailable on the host (ERR_NVGPUCTRPERM), so counts are structural simulations from exact support — but the atomic *time* is independently bounded by the measured oracle (§7), which is what the gate uses.

## 7. Atomic cost upper-bound oracle (§8) — **UPPER_BOUND_ONLY**

Mechanism: ATOMIC_FREE_ORACLE — two same-structure microbench kernels (identical eval/VJP/warpSum/early-exit), variant A scatters with contended global atomicAdd, variant B writes uncontended scratch. **Room MEASURED: 2.091 → 1.965 ms, O_atomic = 0.126 ms = 6.0% of blend, 2.8% of F+B.**

| Scene | T_backward (V3 direct) | T_pixel_recurrence | T_accum_ub | T_proj_vjp | T_sh_vjp | **accumulation_fraction** |
|---|---:|---:|---:|---:|---:|---:|
| room | 1.986 ms | 1.858 | 0.119 | 0.028 | 0.036 | **6.0%** |
| bicycle | 4.770 ms | 4.34 | 0.162 | 0.06 | 0.183 | **3.4%** (5.7% uniform UB) |
| garden | 1.228 ms | 1.03 | 0.067 | 0.02 | 0.021 | **5.4%** (6.0% uniform UB) |

**Cross-check that seals the mechanism**: NO_EXP (remove *all* expf) saves 0.1%; NO_VJP (remove *all* adjoint/VJP arithmetic) saves 0.2% (room) and is 1.6% *slower* (bicycle). The blend backward is **memory-latency-bound**: ~93.7% of it is batch smem loads, global attribute loads, loop control, barriers — none of which HAR touches. **If global gradient accumulation were entirely free, backward could drop by at most ~6.0% / 3.4–5.7% / 5.4–6.0%.**

## 8. Contention (§9)

Per visible Gaussian (means): warp-flush contributions **82.2 / 27.7 / 84.4**; backward-active fine tiles **18.30 / 7.37 / 19.60**; macro tiles **2.67 / 1.69 / 2.54**. Macro entries per G (MEASURED, h6-0): room mean 3.13 / median 2 / p90 6 / p95 8 / p99 16 / max 352; bicycle 1.81 / 1 / 3 / 4 / 8 / 330; garden 3.16 / 2 / 6 / 9 / 16 / 132. Concentration is **highly skewed** (DERIVED from measured buckets): the 17+-tile entry class carries 19–41% of the volume; estimated top-10% Gaussian share of atomics ≈ 20–45%, top-1% ≈ 8–15%. Skew would indeed favor hierarchical aggregation — **but the entire accumulation is ≤6% of the kernel, so absorbing the skew perfectly still cannot exceed that ceiling**, and the gradient working set (1.6/6.5/0.9 MB) is 2–8% of L2 (atomics are already L2-served).

## 9. H8-specific opportunity (§15) — the central scientific question

**"Does sufficient-statistic differentiation make hierarchy practical?" → NO.**

| | naive geometry packet | H8 moment packet |
|---|---|---|
| scalars / bytes per contribution | 9 / 36 B | 9 / 36 B |
| producer arithmetic per valid contribution | 14 ops (12 FMUL + 2 FFMA) | **5 FMUL** (−64.29%) |
| reconstruction | none | 11 ops per G, once, at projection VJP (forward conic only) |
| reconstruction amortization | — | 28.7–41.5× (MEASURED flush multiplicity) |

**Bytes saved: 0. Atomic scalar count saved: 0.** The sufficient statistic is a reparameterization of the same 6-dimensional adjoint — sufficiency lives in the reconstruction inputs, not the dimension. What H8 gives HAR: cleanest-possible reduce (pure elementwise 9-scalar adds, zero forward-geometry knowledge needed) and 62% cheaper producers. What it doesn't give: time — NO_VJP proves the producers' arithmetic is worth ~0.2% of a memory-latency-bound kernel, and H8-MR already banked that arithmetic advantage inside C0 V3.

## 10. SCALAR_ADJOINT compatibility (§16)

**`COMPATIBLE_AFTER_LOCAL_RECURRENCE`.** The SCALAR_ADJOINT state (T, buffer_dot) is per-pixel running recurrence state — order-dependent, non-Gaussian, never hierarchically accumulable. Its *outputs* are exactly the reducible P_GEOM scalars. SCALAR_ADJOINT adds no packet fields and changes no associativity class: it neither enables nor obstructs HAR. (Its 67→56 register benefit cannot stack with HAR, whose packet registers push the other way.)

## 11. SH/color boundary (§17)

**Best boundary: reduce v_color first, then SH VJP — the current structure.** Packet: 12 B vs 192 B (16×) for local SH VJP. SH VJP work: once per visible G vs 7–20× (per-contribution). SH atomics: 2.16M/8.73M/1.18M (current) vs 5.75M/14.7M/2.98M (option B — *more*, because a Gaussian appears in ~2.5 macros > its visibility). SH degree-3 coefficient gradients never enter the packet at the correct boundary — **SH does not dominate HAR packet traffic**; only the 12 B color share of the 36 B packet is SH-adjacent.

## 12. Geometry boundary (§18)

**Prefer H8 moments** (what C0 V3 already carries): equal bytes (36 B), equal atomics (6 geometry per final unit), producers 5 vs 14 ops, reconstruction amortized 28–42×. Both representations are equally associative; both fail the gate for the same reasons. O_net is equal-or-better for moments, so keep moments — but this is a preference, not an enabler.

## 13. HAR designs and traffic (§11–12)

| Design | Structure | HAR total bytes (room/bicycle/garden) | Logical ratio vs current | Physical |
|---|---|---|---|---|
| HAR-A (fine partial → macro reduce) | packet stream + reduce kernel | 78.1 / 135.7 / 45.0 MB | **0.54 / 0.60 / 0.56** | ~20× MORE DRAM (stream vs L2-resident) |
| HAR-B (macro-local packet cache) | macro CTA owns 32 tiles, smem table | 4.3 / 11.0 / 2.2 MB (floor) | 0.03 / 0.06 / 0.03 (**unreachable**) | coordination cost replaces traffic (R6-A anchor) |
| HAR-C (compact + sort + segmented) | packet stream + sort | 162.1 / 267.9 / 94.4 MB (3-pass) | 1.13 / 1.19 / 1.17 (0.58–0.62 in-place) | ~25–60× MORE DRAM |

HAR-B's floor is unreachable: a macro is 8,192 pixels = 4,096 threads at PX=2 (>1024 CTA limit → cooperative groups/persistent CTAs), and bicycle's packet table (p99 180 KB, max >244 KB) exceeds the 163 KB opt-in smem ceiling → mandatory global spill. Logical ratios of 0.54–0.62 do not convert to time because the removed atomics are **L2-resident** (h7-b0 §5: physical DRAM reduction ceiling ≈ 0%) while the added packet/sort traffic is DRAM-streaming.

## 14. Transformation overhead — mandatory accounting (§13)

`O_net = O_removable − C_transformation` (H4's mistake not repeated):

| Component | room | bicycle | garden | Label |
|---|---:|---:|---:|---|
| O_removable (accumulation ceiling) | 0.175 | 0.168 | 0.063 ms | DERIVED from MEASURED oracle |
| macro metadata construction (C_macro) | 0.259 | 0.152 | 0.040 ms | **MEASURED** (h6-0/h7-b0; mandatory since P2-1A is DROP) |
| packet materialization + indexing | 0.063 | 0.108 | 0.036 ms | DERIVED (1555 GB/s) |
| sort/segmented passes (HAR-C) | 0.299 | 0.427 | 0.248 ms | **MEASURED** anchor (h6-0 sort, same record class) |
| scan/zero-init | 0.024 | 0.027 | 0.014 ms | MEASURED anchor |
| coordination/sync (HAR-B) | R6-A anchor: −55.9…−91.5% backward at *smaller* scope | | | **MEASURED** |

**O_net: HAR-A = −0.18 / −0.13 / −0.04 ms (negative 3/3). HAR-C = −0.51 / −0.58 / −0.30 ms (worst — the sort alone exceeds the entire removable budget on every scene). HAR-B = negative at central/conservative.** Even if C_macro were free (it is not), HAR-A's best case is +0.078/+0.023/+0.003 ms = 2.7/0.5/0.3% of backward.

## 15. Resource feasibility (§14)

Fine-tile-scope packet tables (P_MIN/P_GEOM/P_FULL × 32/64/128 G): 0.6–6.1 KB smem, +~9–11 registers (56→~66), 9→7 blocks/SM (−22% occupancy), no spills → **LOW/MEDIUM risk**. R6-A measured 68 regs / +384 B / no spills — resources were never its failure mode. Macro-scope ownership (HAR-B): room mean 14.1 KB / p99 41 KB / max 53 KB (fits, opt-in); **bicycle mean 35 KB / p99 180 KB / max >244 KB → HIGH risk (spill mandatory)**. Temporary buffers: 19–54 MB packet streams (+38–107 MB sort workspace) — fit VRAM, add allocator churn. **Resources are not the rejection reason; O_net is.**

## 16. Macro ownership, parallelism, tail risk (§19–21)

- Ownership: unique G/macro mean 352/885/198, p99 1027/4509/—, max 1319/>6100/~700. Shared-memory ownership fails on the bicycle tail; a 32-G smem mini-batch hybrid always fits (1.3 KB) but reintroduces H7-B0's pixel-work explosion or packet traffic.
- Parallelism: 352/352/336 macro units = **3.3/SM** vs 11,008 fine-tile blocks today (31× collapse). A one-CTA-per-macro design drops to single-CTA residency per SM — the occupancy cliff H7-B0 flagged. Splitting heavy macros restores parallelism but breaks (macro,G) ownership → second combine level → packet traffic returns.
- Tail: macro work p50/p90/p99/max (room) 2,521/4,482/6,395/9,426 → p99/median 2.54, max/median 3.74; bicycle worst macro >6× mean batches; owner-lane p90 max/mean 4.97/6.55/4.38 — **fails the ≤4 hard gate on 3/3** (MEASURED). Any HAR needs dynamic scheduling AND macro splitting — both erode the ownership advantage. (Diagnosis only; no scheduler implemented.)

## 17. Prior art and R6-A (§22–23)

| Prior mechanism | Classification |
|---|---|
| Faster-GS / per-Gaussian backward | PARTIAL_OVERLAP — changes recurrence ownership (per-Gaussian prefix recompute); HAR provably never touches the recurrence |
| H8 current moment reduction | **SAME** for packet semantics — HAR's P_GEOM *is* the H8 packet; HAR adds only an intermediate macro combine between warpSum and global atomicAdd |
| R6-A block aggregation | PARTIAL_OVERLAP — same mechanism at smaller scope |
| native HiGS hierarchy | DISTINCT_MECHANISM for this use (reduction index only; forward-scheduling use is closed and not reopened) |
| GS-TG / tile grouping | PARTIAL_OVERLAP (shared spatial structure, different purpose) |

**R6-A comparison (decisive).** R6-A compressed writer events 7.56–7.97× and still measured **−91.5/−76.9/−55.9% backward** because 2 barriers/Gaussian + uniform loops cost more than the atomics saved. HAR offers the same-order compression (4.4–7.7×) at 32× scope with strictly heavier transformation: cross-tile coordination R6-A never needed, packet/sort traffic R6-A never paid, and macro metadata R6-A never built. **HAR recreates R6-A at larger scope with similar-or-worse transformation overhead → the task's DROP rule applies immediately.** The candidate distinction ("order-dependent pixel recurrence stays pixel-local while order-independent sufficient-statistic adjoints reduce through an exact spatial hierarchy") is *stated, not claimed* — it is real as an exactness statement but targets a ≤6% cost.

## 18. Net timing oracle (§24–25)

Basis: C0-T2 publication F+B (4.780/6.972/2.897 ms), nested backward share 0.61/0.71/0.40 (DERIVED from C0-T1), F+B ≈ 50% of iteration.

| Scene | optimistic (unreachable) bwd / F+B / iter | central bwd / F+B / iter | conservative (measured anchors) |
|---|---|---|---|
| room | 6.0% / 3.7% / 1.9% | 2.2% / 1.4% / 0.7% | ≤0; H7-B0 realistic F+B **−3.16%** |
| bicycle | 3.4% / 2.4% / 1.2% | 1.3% / 0.9% / 0.45% | H7-B0 **+0.49%** |
| garden | 5.4% / 2.2% / 1.1% | 2.0% / 0.8% / 0.4% | H7-B0 **+2.02%** |

Labels: removable ceiling DERIVED from the MEASURED atomic-free oracle; C_macro, sort costs, R6-A outcome, H7-B0 net = MEASURED; optimistic rows = HYPOTHETICAL bounds. **Even the optimistic ceiling misses ≥10% backward and ≥5% F+B on 3/3 scenes.**

## 19. Gate (§26–27)

- **P3_HAR_STRONG**: FAIL — optimistic backward max 6.0% (<10%) on 0/3; optimistic F+B max 3.7% (<5%) on 0/3; HIGH risk on every design.
- **P3_HAR_MARGINAL**: FAIL — the only 5–10% readings are UPPER_BOUND_ONLY no-op-oracle values before any transformation cost; with mandatory accounting all scenes fall below 5% backward at central (2.2/1.3/2.0%), and implementation risk is HIGH, not "clearly isolated low-risk".
- **P3_HAR_WEAK**: SATISFIED on all three criteria — central <5% backward on 3/3; transformation consumes most/all atomic savings (O_net negative on 3/3 for HAR-A/C); mechanism duplicates R6-A at larger scope (and overlaps the already-frozen H8 reduction).

**Classification: `P3_HAR_WEAK` → DROP. `P3-1 HAR prototype` is NOT authorized.**

## 20. Deliverables

```
reports/higs/p3-0-har-opportunity.md          (this file)
scripts/p3/p3_0_har_measure.py                (structural recomputation script, mx-runnable)
artifacts/higs-p3-0/
    provenance.json            adjoint_packet_spec.json     exactness_proof.json
    exact_macro_reuse.json     macro_multiplicity.json      accumulation_topology.json
    atomic_cost_oracle.json    contention_distribution.json current_traffic.json
    har_designs.json           har_traffic_model.json       transformation_cost.json
    resource_feasibility.json  h8_har_interaction.json      scalar_har_interaction.json
    sh_boundary_analysis.json  geometry_boundary_analysis.json
    macro_ownership.json       reduction_parallelism.json   tail_risk.json
    prior_art_overlap.json     r6a_comparison.json          net_timing_oracle.json
    final_gate.json
```

No production source changed. C0, P2-1A, P2-1C untouched.

## 21. Next action

With P2-1A DROP, P2-1C DROP, and P3-0 HAR DROP, the renderer-architecture question is closed: **C0 V3 is the final forward/backward architecture**. The decision tree's remaining branches are the optimizer side and the final benchmark: **run the E2 gate (VJP→Adam fused optimizer wrapper on the frozen C0 V3 checkpoint; gate ≥3% full-iteration, FP32-grad equivalence via the h2-bwd determinism harness, no correctness drift), then the FINAL 30K benchmark with C0 V3 (+E2 if passed).**
