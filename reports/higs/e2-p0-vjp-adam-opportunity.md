# E2-P0 — VJP→Adam Fusion Opportunity Remeasure (Frozen C0 V3)

**Status:** `COMPLETE`
**Date:** 2026-10-08
**Decision:** `E2_DROP`
**Implementation authorization:** `NO`
**Timing grade:** `DIAGNOSTIC_SHARED_GPU` (all absolute ms)
**Scope:** measurement/oracle only — no source modifications, no production CUDA changes, no C0 publication timing rerun, no wait on P2-1A-R2.

---

## 0. Executive summary

The historical "~4% full iteration" prior for VJP→Adam fusion is **refuted by current measurement**. On the frozen C0 V3 stack:

| | room | bicycle | garden |
|---|---:|---:|---:|
| Realistic **O_net** (full iteration) | **0.79%** | **2.25%** | **0.77%** |
| Perfect-fusion upper bound (C=0) | 1.32% | 3.31% | 1.22% |

Three findings killed the prior:

1. **The current optimizer is already fused Adam.** `torch.optim.Adam(fused=True)`, 5 param groups → **10 optimizer-side launches/iteration** (5 `FusedAdam` kernels + 5 state-step `foreach_add_`), not dozens of per-tensor launches. The launch-prize is ~10 launches ≈ 25µs GPU-side.
2. **Only 1/7 of Adam's traffic is removable.** Fused Adam reads grad+param+m+v and writes param+m+v. The grad read is the only removable pass; the mandatory 6-pass m/v/param traffic (163/822/94 MB per scene) **must survive any fusion** and gets *relocated into* producer kernels at worse (visible-scattered) access efficiency.
3. **The removable bytes are small against the V3 iteration.** Gradient round-trip (write + reread, strict) is 37.8/179.8/21.4 MB → at measured clean bandwidth (1.0–1.44 TB/s) that is ~33/160/19µs against 5.41/8.13/4.02ms iterations.

Gate: O_net < 1% on 2/3 scenes (room, garden) → **DROP**. KEEP (≥3% on ≥2/3) is unreachable in every modeled scenario — even the zero-cost upper bound reaches 3% only on bicycle. No blocking semantic issue exists, but the restructure stack (SH-VJP one-writer grid rewrite, full-N momentum-decay extension, densify-signal re-emit, topology state sync) dominates a ≤2.25% best-case prize, and the geometry producer sits 2 registers from the 128-reg occupancy cliff.

---

## 1. Baseline and environment (recorded actuals)

```text
host:            bms-39468022-001 (ssh alias mx)
GPU:             NVIDIA A100-PCIE-40GB, SM80, 108 SMs
measurement GPU: GPU-6d75016d-fc44-9443-9190-396375f20f6a (CUDA_VISIBLE_DEVICES=4,
                 the same physical GPU as C0-T1/T2)
co-tenant:       pid 2092275, 22264 MiB resident, ~40% util during the window
env:             /mnt/storage_pool/liaoyuanjun/higs-13scene-env
torch:           2.9.1+cu128    CUDA runtime: 12.8
commit:          77ab983ffe43420b2131669cb35776b883ca4c3c (higs_c0_worktree)
composed .so:    experimental_gaussian_render_inference_scene_cuda.so
                 sha256 7ca1c6bf6c8e4307ecb8fcdbcaf2953bf95305d2c9814f84f3fb5130859301f6
gsplat_cuda.so:  sha256 361b216bcc11609a0ebb8fb44ad2e0c6170948112b6294e85123df45541c8c98
gsplat_scene.so: sha256 (recorded in provenance.json)
V3 env:          HIGS_PX_RUNTIME=2, HIGS_BWD_SCALAR_ADJOINT=scalar_adjoint,
                 HIGS_BWD_H8_MR=1, HIGS_DISABLE_F9 unset
fixture:         c0t1_timing.py load_fixture — 2048px cam0, 30K speedy-splat PLYs
```

| Scene | N_GS | N_visible | N_visible/N | n_isects | resolution |
|---|---:|---:|---:|---:|---|
| room | 115,278 | 44,908 | 39.0% | 953,144 | 2048×1365 |
| bicycle | 580,416 | 181,525 | 31.3% | 1,412,193 | 2048×1361 |
| garden | 66,282 | 24,483 | 36.9% | 533,928 | 2048×1327 |

Protocol: nested CUDA events in one iteration (fwd | loss | bwd | grad-consumer | opt | zero_grad), 20 warmup + 5 reps × 100 samples = 500 raw observations/scene, persistent leaves+handle+optimizer (training-loop shape), fixture byte-identical to C0-T1. Cross-check: room fwd+loss+bwd = 4.677ms vs C0-T2 V3 nested F+B = 4.780ms — consistent.

---

## 2. Producer → optimizer map (per family)

Backward chain (measured chrome traces + `_native_backward` source):

```text
B0  5× zeros_like(master N) + I×N_v accumulator zeros     [FillFunctor ×13/iter]
B1  higs_blend_bwd_px_kernel<3,2,true,true>   2369µs (room)  ← I×N_v atomic adjoint
B2  higs_projection_bwd_kernel<true>            19.4µs      ← FINAL PRODUCER means/quats/scales
B3a higs_camera_positions_kernel + higs_sh_vjp_grid_kernel 2.3+34.1µs ← FINAL PRODUCER shs
B3b higs_reduce_master_kernel                    3.5µs      ← FINAL PRODUCER opacities
B5  grad_means2d zeros + index_copy_            2.2+6.8µs   ← densification-facing means2d grad
C   means.grad.norm reduce + accum add          7.3+2.8µs   ← densification grad-norm consumer
OPT 5× (state-step foreach_add_ + FusedAdam)                ← Adam consumer
```

| Family | Final VJP producer | Gradient buffer | Intermediate consumer(s) | Adam kernel |
|---|---|---|---|---|
| means | `higs_projection_bwd_kernel<true>` | means.grad [N,3] fp32 | **grad-norm accumulation (every iter) + densify/prune (every densify_every)** — *after* opt.step in `run_higs_train_benchmark.py` L1218-1219, L1231 | FusedAdam group[0] (lr 1.6e-4) |
| quats | `higs_projection_bwd_kernel<true>` | quats.grad [N,4] fp32 | none | FusedAdam group[1] (lr 1e-3) |
| scales | `higs_projection_bwd_kernel<true>` | scales.grad [N,3] fp32 | none | FusedAdam group[2] (lr 5e-3) |
| opacities | `higs_reduce_master_kernel` | opacities.grad [N] fp32 | none | FusedAdam group[3] (lr 5e-2) |
| shs | `higs_sh_vjp_grid_kernel` | shs.grad [N,16,3] fp32 | none | FusedAdam group[4] (lr 2.5e-3) |
| means2d (densif.) | blend bwd `v_means2d` + `index_copy_` | grad_means2d [1,N,2] | grad-norm accumulation, densify | **none** (not a parameter) |

Visibility bookkeeping: sorted `visible_ids` (int64 [N_v]) from forward culling; VJPs write master rows through sorted scatter; F9 producer reads master parameters directly (gatherless).

### F0/F1/F2 classification

| Family | Class | Reason |
|---|---|---|
| means | **F2** | post-step consumers of means.grad (grad-norm accumulation every iteration; densify/prune). Direct fusion starves densification; becomes F1 only if the fused kernel re-emits per-Gaussian grad norms and materializes gradients on densify steps. |
| quats | **F1** | Adam-only consumer, but producer grid is I×N_v while Adam must decay momentum over all N → full-N extension required; producer at 96 regs (cliff risk). |
| scales | **F1** | same producer as quats. |
| opacities | **F1** | LOW resource risk (32 regs), but 1 float/Gaussian → negligible prize. |
| shs | **F1** | Adam-only consumer; (N_visible × D)-decomposed atomicAdd grid → per-row fused Adam needs a one-writer **REQUIRES_RESTRUCTURE**; dominant prize (81% of bytes). |
| means2d | **F2** | not an Adam input; different consumer (densification); out of scope. |
| backgrounds | F0 | trivial 3-float gradient; negligible. |

**Key structural fact: NO family is F0 on these checkpoints.** Every producer writes only N_visible rows (39/31/37% of N) while fused Adam must cover all N rows — momentum decay is nonzero for zero-gradient Gaussians (m_t = 0.9·m_{t−1}, update ≠ 0).

---

## 3. Measured iteration decomposition (DIAGNOSTIC_SHARED_GPU)

Medians, n=500/scene (full distributions in `raw_e2_timing.csv`, rep medians in `timing_decomposition.json`):

| Stage (ms) | room | % | bicycle | % | garden | % |
|---|---:|---:|---:|---:|---:|---:|
| forward | 2.716 | 50.2 | 3.245 | 39.9 | 1.490 | 37.0 |
| loss | 0.212 | 3.9 | 0.213 | 2.6 | 0.207 | 5.1 |
| blend backward (inside bwd) | ~2.37 | 43.8 | ~3.18 | 39.1 | ~0.86 | 21.5 |
| **bwd total** | 1.749 | 32.3 | 3.442 | 42.3 | 0.855 | 21.3 |
| grad-consumer (densif. norms) | 0.014 | 0.3 | 0.028 | 0.3 | 0.013 | 0.3 |
| **optimizer (fused Adam)** | **0.691** | 12.8 | **1.175** | 14.5 | **1.459** | 36.3 |
| zero_grad | 0.003 | 0.1 | 0.003 | 0.0 | 0.003 | 0.1 |
| **total iteration** | **5.412** | 100 | **8.133** | 100 | **4.023** | 100 |

Note on the bwd column: the blend backward kernel (2369µs room) exceeds the bwd event window (1749µs) because backward launches overlap forward's tail on the stream in the sustained loop; the C0-T1 nested protocol showed the same sustained-state behavior. The decomposition that matters for E2 — final-VJP producers + optimizer — is unaffected.

Shared-GPU caveat: small-kernel durations are inflated to a ~125µs co-tenant time-slice floor (garden's opt window is the most contaminated: clean 5-group Adam is ~85µs vs 1459µs window). Large contiguous kernels are bandwidth-consistent (bicycle B0 fills 119µs measured ≈ analytic 137MB @ 1.15TB/s; room shs-Adam 22.1MB×7/139µs ≈ 1.1TB/s). Clean optimizer estimates: **room ~0.16–0.19ms, bicycle ~0.60–0.64ms, garden ~0.09ms** (3.0% / 7.9% / 2.1% of iteration).

---

## 4. Optimizer implementation (measured, not assumed)

```text
Implementation:  FUSED Adam — torch.optim.Adam(fused=True)
Source:          benchmark/run_higs_train_benchmark.py make_optimizer (default fused=True),
                 used by the F9-1R 5K gates and the current training harness
Groups:          5 (xyz 1.6e-4, rotation 1e-3, scaling 5e-3, opacity 5e-2, shs 2.5e-3),
                 eps=1e-8 default, betas (0.9, 0.999), no weight decay, amsgrad=False
Kernels/iter:    5 × multi_tensor_apply_kernel<FusedOptimizerTensorListMetadata<4>,
                   FusedAdamMathFunctor<float,4,ADAM_MODE::0,false>>   (one per group)
                + 5 × multi_tensor_apply_kernel<TensorListMetadata<1>,
                   BinaryOpScalarFunctor> (state_steps foreach_add_, ~3.2-3.7µs each)
                = 10 optimizer-side launches per iteration
Launch gaps:     intra-Adam kernel gap median 1.09µs (bicycle trace); step-increment
                 kernels interleave between the 5 FusedAdam launches
Median durations (shared-GPU, last-iteration trace): room 124-141µs/group;
                 bicycle ~128/~124/~465(shs)/~163/~128µs; garden 122-134µs/group
Total kernels per iteration (whole loop): 89-90
```

The `reference_v1` trainer (B2/gsplat comparisons) uses default foreach Adam with eps=1e-15 — a **different** harness; E2 evaluates the current HiGS stack, which is fused.

---

## 5. Gradient materialization traffic (per family, per iteration)

| Bytes | room | bicycle | garden |
|---|---:|---:|---:|
| grad buffer, full N (all families) | 27.2 MB | 137.0 MB | 15.6 MB |
| — of which shs | 22.1 MB | 111.4 MB | 12.7 MB |
| VJP grad write (visible rows only) | 10.6 MB | 42.8 MB | 5.8 MB |
| B0 zero-fill of grad buffers | 27.2 MB | 137.0 MB | 15.6 MB |
| **removable round-trip (strict = VJP write + Adam reread)** | **37.8 MB** | **179.8 MB** | **21.4 MB** |
| removable incl. B0 zero-fill | 65.0 MB | 316.8 MB | 37.0 MB |

Only the strict rows are "bytes fusion can truly remove" under the task's formula; the B0 row is removable only in a full-N fused design (gradients never materialize). `grad_means2d` (2.1/10.5/1.2 MB) is **not** removable — it feeds densification, not Adam.

## 6. Mandatory Adam traffic (remains after perfect fusion)

| Bytes | room | bicycle | garden |
|---|---:|---:|---:|
| param read + m read + v read | 3×27.2 = 81.7 MB | 3×137.0 = 411.1 MB | 3×15.6 = 46.9 MB |
| param write + m write + v write | 3×27.2 = 81.7 MB | 411.1 MB | 46.9 MB |
| **mandatory total (6 passes)** | **163.3 MB** | **822.1 MB** | **93.7 MB** |
| Adam arithmetic | 59 FLOP-pairs/Gaussian — compute-trivial, bandwidth-bound |
| full Adam traffic (7 passes incl. grad read) | 190.4 MB | 959.1 MB | 109.3 MB |

Mandatory traffic is **6× the removable grad round-trip**. Any fusion design that makes this traffic even 15% less efficient (visible-scattered access) loses more than it gains — this is the decisive structural argument.

---

## 7. Producer → optimizer gap (measured sequence)

Between the final VJP producer and the first Adam kernel (room trace):

```text
higs_reduce_master_kernel (final VJP producer)      3.5µs
FillFunctor (grad_means2d zeros)                    2.2µs
index_copy_ (B5: visible→full-N means2d scatter)    6.8µs   ← densification-facing
NormTwoOps reduce (means.grad.norm → grad_norm_acc) 7.3µs   ← consumer of grad_means
CUDAFunctor_add (accum +=)                          2.8µs
state-step foreach_add_                             3.7µs
FusedAdam group[0] (xyz) — first Adam kernel      134.0µs
```

- GPU kernels between producer and first Adam: **22.3µs**; **no synchronizations** (fully async, same stream).
- Python dispatch: `opt.step()` loops 5 groups → 5 `aten::_fused_adam_` + 5 `_foreach_add_` calls (~100–250µs CPU, mostly hidden; partially visible on short iterations).
- Densification events do **not** appear on normal iterations (frozen-topology fixture); in real training every `densify_every` iterations `_densify_gaussians`/`_prune_gaussians` + `sync_optimizer_state_for_topology_change` run after `opt.step()`.
- Required intermediate consumers: **grad_means** (grad-norm accumulation every iteration; densify on densify steps) and **grad_means2d** (densification signal). quats/scales/opacities/shs have **no** intervening consumer → direct fusion is semantically possible for those four families only.

## 8. Semantic constraints audit

| Constraint | Status | Detail |
|---|---|---|
| densification | **REQUIRES_RESTRUCTURE** | means.grad consumed after opt.step (grad-norm accumulation every iter; densify/prune periodically); fused kernel must re-emit per-Gaussian norms (exact, in-register) or means stays unfused |
| gradient accumulation | SAFE | single view per step; no multi-view .grad accumulation |
| zero_grad semantics | SAFE | set_to_none=True; next backward allocates fresh zeros_like (B0) — fusion removes B0 wholesale |
| gradient clipping | SAFE | none in harness |
| mixed precision | SAFE | all-FP32 pipeline |
| loss scaling | SAFE | none |
| visibility mapping | **REQUIRES_RESTRUCTURE** | sorted visible_ids scatter; fused producers must extend to full N |
| sparse/master-visible mapping | **REQUIRES_RESTRUCTURE** | B0 zeros master rows, VJP writes visible rows, Adam reads all rows → zero-grad path must fold into the full-N kernel |
| multi-view accumulation | SAFE | I=1 everywhere today; I×N_v design already generalizes |
| topology-change state sync | **REQUIRES_RESTRUCTURE** | densify/prune rebuilds optimizer state; fused path needs equivalent migration |
| **BLOCKING issues** | **NONE** | 4 REQUIRES_RESTRUCTURE items, all with known exact paths |

## 9. Resource risk (static cuobjdump, composed .so 7ca1c6bf…)

| Kernel | regs/thread | smem | spills | occupancy (256 t/b) | Adam-insertion risk |
|---|---:|---:|---:|---:|---|
| `higs_projection_bwd_kernel<true>` (means/quats/scales) | **96** | 0 | 0 | 25% (2 CTA/SM) | **HIGH** — +15..30 regs → 111–126; the 128-reg boundary cliffs to 12.5% (≈2× slowdown) |
| `higs_sh_vjp_grid_kernel` (shs) | 48 | 0 | 0 | 62.5% (5 CTA/SM) | **MEDIUM** — → ~68–80 regs, ~46–50% occupancy, no cliff; but needs one-writer grid restructure |
| `higs_reduce_master_kernel` (opacities) | 32 | 0 | 0 | 75% | **LOW** — negligible prize |
| `higs_blend_bwd_px_kernel<3,2,true,true>` | 64 | 0 | 0 | — | not a fusion target (produces I×N_v intermediates) |
| `f9_projected_producer_kernel` | 43 | 0 | 0 | — | forward-only |

Zero spills everywhere today. The geometry producer's 2-register margin to the cliff is the binding resource constraint — the same failure mode that closed H4 and D.

---

## 10. Traffic oracle (TRAFFIC_BOUND_ORACLE)

Measured effective bandwidth (shared-GPU microbench, large contiguous kernels — reliable): fill 1.15–1.44 TB/s; contiguous copy (r+w) 0.99–1.30 TB/s; random-row scatter (39%, worst case) 0.09–0.39 TB/s. Oracle assumptions (conservative for the sorted-visible VJP scatter): fill 1.30, read 1.20, sorted-scatter write 1.00, Adam 1.25 TB/s. **A100 peak (1.55 TB/s) was NOT used as achieved bandwidth.**

| TRAFFIC_BOUND_ORACLE | room | bicycle | garden |
|---|---:|---:|---:|
| B0 zero-fill removable | 21.0µs | 105.4µs | 12.0µs |
| VJP write removable (visible) | 10.1µs | 40.6µs | 5.5µs |
| Adam grad-read removable | 22.7µs | 114.2µs | 13.0µs |
| **traffic-bound removable (best subset)** | **53.8µs** | **260.2µs** | **30.5µs** |

## 11. Launch oracle (separate accounting — no double counting)

| | room | bicycle | garden |
|---|---:|---:|---:|
| optimizer launches removable | 10/iter (5 FusedAdam + 5 state-step) | same | same |
| GPU-side removable (launch latency + gaps) | ~25µs | ~25µs | ~25µs |
| CPU dispatch removed (5 `_fused_adam_` + 5 `_foreach_add_`) | ~100–250µs, mostly hidden | mostly hidden | partially visible (short iterations) |

Launch savings are added to the traffic oracle once (in `O_removable`), never also inside kernel-time savings.

## 12. Net opportunity (O_net = O_removable − C_transformation)

Best subset = **all-except-means {quats, scales, opacities, shs}** (means excluded: post-step densification consumers).

| | room | bicycle | garden |
|---|---:|---:|---:|
| O_removable (traffic + B0 + launches) | 71.4µs | 269.0µs | 49.3µs |
| C: mandatory-traffic relocation penalty (visible-scatter efficiency) | −18.8µs | −75.9µs | −10.2µs |
| C: register/control overhead | −10.0µs | −10.0µs | −8.0µs |
| **O_net** | **42.7µs → 0.79%** | **183.1µs → 2.25%** | **31.0µs → 0.77%** |
| O_net upper bound (C=0, perfect fusion) | 1.32% | 3.31% | 1.22% |

Per-family net (room/bicycle/garden): shs **0.61/1.89/0.53%**; quats 0.14/0.21/0.16%; scales 0.13/0.18/0.15%; opacities 0.10/0.10/0.13%; means 0.13/0.18/0.15% (excluded — restructure required).

## 13. Best fusion subset

Evaluated: geometry-only, shs-only, opacity-only, all-eligible.

| Subset | room | bicycle | garden |
|---|---:|---:|---:|
| geometry only (means+quats+scales) | ~0.4% | ~0.6% | ~0.45% |
| shs only | 0.61% | 1.89% | 0.53% |
| opacity only | 0.10% | 0.10% | 0.13% |
| **all-except-means (best)** | **0.79%** | **2.25%** | **0.77%** |
| all families (incl. means, +restructure) | ~0.9% | ~2.4% | ~0.9% |

**Best realistic subset: all-except-means**, driven by shs (81% of gradient bytes). Even this subset does not assume fusing all parameters is optimal — geometry families add <0.6pp combined while importing the 96-register cliff.

## 14. Decision gate

```text
KEEP  requires O_net >= 3% on >=2/3 scenes  -> 0/3 scenes (max: bicycle 2.25%).
      Even the zero-cost upper bound reaches 3% only on 1/3 scenes (bicycle 3.31%).
DROP  triggers on O_net < 1% on >=2/3 scenes -> room 0.79%, garden 0.77% (2/3)  -> MET.
      Corroborated by: semantic restructuring cost dominates (4 REQUIRES_RESTRUCTURE
      items vs a <=2.25% best-case prize) and the 96-reg cliff on the geometry producer.
DEFER band (1-3%) holds only for bicycle (1/3 scenes).

DECISION: E2_DROP
Implementation authorization: NO
```

The historical "~4% full iteration" estimate (P2-0 §14/§18, `artifacts/higs-p2-0/optimizer_fusion_ceiling.json` — left untouched) is superseded-in-conclusion by this measurement: it was DERIVED analytic, assumed a pre-fused-Adam optimizer, a pre-V3 (slower) denominator, and implicitly treated more of the optimizer as removable than the 1/7 grad-read pass that is actually removable.

**Resolution sensitivity (documented, not acted on):** at the benchmark-harness default 960px the fwd/bwd denominator shrinks ~2× while optimizer bytes are unchanged → estimates move to roughly 1.2–1.4% (room), 3.3–3.8% (bicycle), 1.6–1.7% (garden). Bicycle would cross 3% but still only on 1/3 scenes (DEFER band, not KEEP). The DROP is taken at the frozen C0 V3 authoritative context (2048px, C0-T1/T2 fixture). If the program benchmark context moves to low-res/large-N, recompute (below) before reopening.

## 15. Interaction with P2-1A

Not waited on. If P2-1A-R2 lands and changes **only forward latency**, E2 recomputes without new measurements:

```text
new_O_net% = same removable net ms (room 42.7µs, bicycle 183.1µs, garden 31.0µs)
             / new total iteration ms
```

Example: a 20% forward reduction moves bicycle 2.25% → ~2.4% (still < KEEP). A ~60% forward reduction would be required before bicycle-alone reaches 3%. P2-1A cannot flip the 2/3-scene gate.

## 16. Deliverables

```text
reports/higs/e2-p0-vjp-adam-opportunity.md            (this report)

artifacts/higs-e2-p0/
    provenance.json                  env, hashes, co-tenant, protocol, per-scene N/N_visible
    producer_consumer_map.json       B0..OPT chain + per-family flow (§2)
    parameter_fusibility.json        F0/F1/F2 per family (§2)
    grad_materialization_bytes.json  per-family grad/write/read bytes (§5)
    adam_traffic.json                mandatory vs removable Adam passes (§6)
    launch_accounting.json           10 optimizer launches, durations, gaps (§4)
    timing_decomposition.json        stage medians + rep stability + shared-GPU caveat (§3)
    producer_optimizer_gap.json      22.3µs measured gap sequence (§7)
    traffic_oracle.json              TRAFFIC_BOUND_ORACLE per family/subset (§10)
    launch_oracle.json               launch-count oracle (§11)
    resource_risk.json               regs/smem/spills/occupancy + insertion risk (§9)
    semantic_constraints.json        SAFE / REQUIRES_RESTRUCTURE audit (§8)
    per_family_opportunity.json      per-family removable/penalty/net (§12)
    full_iteration_opportunity.json  O_net per scene (§12)
    final_gate.json                  decision + rationale + recompute rule
    raw_e2_timing.csv                1500 raw nested observations (500/scene)
    supplementary: bandwidth_oracle.json, stage_timing_summary.json,
                   kernel_attribution_{room,bicycle,garden}.json, resource_audit.json,
                   e2p0_harness.py / e2p0_oracle.py / e2p0_artifacts.py / e2p0_gate.py
    remote copies: mx:/mnt/storage_pool/liaoyuanjun/higs_e2_p0_results/ (incl. chrome traces)
```

No C0 or P2 artifacts were overwritten.

## 17. Final answer

1. **Optimizer implementation:** fused Adam (`torch.optim.Adam(fused=True)`, 5 groups, eps=1e-8) — `multi_tensor_apply<FusedAdamMathFunctor>` per group.
2. **Producer kernels:** means/quats/scales → `higs_projection_bwd_kernel<true>` (96 regs); shs → `higs_sh_vjp_grid_kernel` (48 regs, (N_v×D)-decomposed); opacities → `higs_reduce_master_kernel` (32 regs); means2d → blend bwd `v_means2d` + `index_copy_`.
3. **Adam consumer kernels:** 5× `multi_tensor_apply_kernel<FusedOptimizerTensorListMetadata<4>, FusedAdamMathFunctor<float,4,ADAM_MODE::0,false>>` + 5× state-step `foreach_add_` per iteration.
4. **Classification:** no F0 on these checkpoints; means F2 (post-step densification consumers); quats/scales F1 (full-N extension + 96-reg cliff); shs F1 (grid restructure); opacities F1 (negligible); means2d F2 (not an Adam input).
5. **Gradient buffer bytes/family (room/bicycle/garden):** means 1.38/6.97/0.80 MB; quats 1.84/9.29/1.06; scales 1.38/6.97/0.80; opacity 0.46/2.32/0.27; shs 22.13/111.44/12.73 (totals 27.2/137.0/15.6 MB).
6. **Removable round-trip (strict):** 37.8 / 179.8 / 21.4 MB (65.0 / 316.8 / 37.0 MB incl. B0 zero-fill).
7. **Mandatory Adam traffic:** 163.3 / 822.1 / 93.7 MB per iteration (6 passes: param+m+v read, param+m+v write) — 6× the removable bytes; Adam arithmetic is bandwidth-trivial.
8. **room optimizer ms:** 0.691 (window, DIAGNOSTIC_SHARED_GPU); clean ~0.16–0.19.
9. **bicycle optimizer ms:** 1.175 (window); clean ~0.60–0.64.
10. **garden optimizer ms:** 1.459 (window, most contaminated); clean ~0.09.
11. **Final-VJP ms per scene (producer kernels, room/bicycle/garden):** ~0.059 / ~0.217 / ~0.040 (sum of projection VJP + camera positions + SH VJP + reduce_master).
12. **Optimizer launch count:** 10 per iteration (5 FusedAdam + 5 state-step); whole iteration 89–90 kernels.
13. **Producer→optimizer intervening work:** 22.3µs GPU (means2d zeros+scatter 9.0µs, grad-norm reduce+add 10.1µs, first state-step 3.7µs), no synchronizations, 5-group Python dispatch; required intermediate consumers: grad_means (densification) and grad_means2d.
14. **Traffic oracle (TRAFFIC_BOUND_ORACLE):** removable 53.8/260.2/30.5µs at measured 1.0–1.3 TB/s (fill/copy/scatter measured; A100 peak NOT assumed).
15. **Launch oracle:** 10 launches removable ≈ 25µs GPU-side + 100–250µs CPU dispatch (mostly hidden).
16. **Resource risk:** HIGH on geometry producer (96→111-126 regs, 2 regs from the 128-reg cliff, 25%→12.5% occupancy), MEDIUM on shs (48→~70-80, grid restructure), LOW on opacity; zero spills today.
17. **Semantic blockers:** NONE BLOCKING; four REQUIRES_RESTRUCTURE (densification signal, visibility/full-N mapping, sparse-master mapping, topology state sync).
18. **Best fusion subset:** all-except-means {quats, scales, opacities, shs}; shs alone carries the prize.
19. **room realistic full-iteration opportunity:** **0.79%** (upper bound 1.32%).
20. **bicycle realistic full-iteration opportunity:** **2.25%** (upper bound 3.31%).
21. **garden realistic full-iteration opportunity:** **0.77%** (upper bound 1.22%).
22. **Decision:** **E2_DROP** — O_net < 1% on 2/3 scenes at the frozen C0 V3 context; KEEP unreachable in any modeled scenario; restructure cost dominates.
23. **Implementation authorization:** **NO**.
24. **Exact next action:** Close E2 as measured-negative on the frozen C0 V3 stack (2048px context). Record the recompute trigger in the registry: if the benchmark context moves to 960px-class denominators or large-N-only scene sets, recompute `O_net% = 42.7/183.1/31.0µs ÷ new total iteration ms` per §15 before reopening; do not implement fusion. Direct effort to the surviving candidates (P2-1A-R2 execution) instead.
