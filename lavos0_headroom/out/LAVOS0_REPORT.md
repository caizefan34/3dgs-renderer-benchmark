# LAVOS0 Lineage-Aware Exact VOS — A100 Scientific Run Report

**Protocol:** LAVOS0_LINEAGE_AWARE_EXACT_VOS_20260930 (frozen)
**Verdict:** `LAVOS0_NO_STRUCTURAL_HEADROOM`
**Date:** 2026-10-01
**Analyzer owns the verdict.** This report only restates measured artifacts; it does not reinterpret failures as success.

---

## 1. Executive summary

LAVOS0 asked one question: *does exact lineage-aware deferred optimizer execution have enough structural headroom to justify a full LAVOS system?*

**Answer: No.** The deferred-execution mechanism is exact (bitwise-equal to dense Adam on theta/m/v, Gaussian count, and topology mapping) and cheap (negligible settlement overhead, ~5.5x faster optimizer *lane*), but the SH Adam optimizer is only **3–5% of total iteration wall time**. Deferring it therefore saves only **1.3–2.0% of end-to-end training**, below the 2% gate. The structural-headroom gate (H1–H4) fails on H4 for every scene.

Per the protocol, the direction is closed honestly. The mechanism is correct and reusable; the payoff is too small on this workload.

---

## 2. Environment (real A100, not synthetic)

| Field | Value |
|---|---|
| Hostname | `dsw-933622-64cc845c7d-lqswc` (EPIC-05 endpoint `8.130.30.251:1024`) |
| GPU | NVIDIA A100-SXM4-80GB |
| CUDA | 12.8 |
| PyTorch | 2.9.1+cu128 |
| Python | 3.10.21 |
| Scenes | room, garden, bicycle |
| Seed | 42 |
| Iterations | 15000 (C0 frozen trainer) |
| Final eval | subset |

All CUDA work ran on the EPIC-05 A100. No local RTX 5070 was used.

### Checkpoint hashes (`ckpt_15000.pt`)

| Scene | SHA-256 |
|---|---|
| room | `84c7e2a1087dcb8853c256aeb18c40dede22e3df96a2f40463cb7ed5f122ecf9` |
| garden | `4ffa559039f6d6b1ef732718be8d67d247596cc1c2eca92a917a8096bb0cb24f` |
| bicycle | `960bd34be41b27fce64759dc88126c46be57f5f0be40d6230664b04de7357f5f` |

---

## 3. Source identity (frozen C0)

`source_identity_pass = true`. All 5 expected frozen sources hashed; 0 missing. Binary identity is identical across all three scenes (same trainer `.so`).

| Role | SHA-256 |
|---|---|
| publication_trainer_p13x.py | `9966f30a5eaf108b25379c15c32cc2532b25133ac786fcf22f22e6569e760b10` |
| gaussian_model.py | `68731e375013a6f24aa4d81e5986402b1ac50f51cddf20688e358be705b1a60b` |
| config.py | `ea8e870476a3ba71c0ae9e1123a6fa5b1cf696bad983bdb6535d6fd04b7acc56` |
| vos_trace_driver.py | `87e8837ff49c29bce58c6d3ea1da26bcef0e1a5aa6d6a1e3d55c0778ba8dd5ae` |
| lavos0_capture.py (driver) | `d4732a2507b4c06af458a4125cce0a6ce5b12879c02009ca5a3ede702f26fb64` |
| trainer core (`core_sha256`) | `935ea6c0dd04f34e26407afd895ca7d8bc452cc233b1432683ba0ec70d1ff37e` |
| trainer `_so` (`so_sha256`) | `d26cccf46c7932bfb5fd44ee05e553af27e17c93eaffb7dacf3197258a513f11` |

---

## 4. Phase A — Event order instrumentation

`event_semantics_resolved = true`. The exact C0 per-iteration ordering was resolved by wrapping the frozen trainer with zero source diff:

1. `update_learning_rate` → 2. `oneup_sh_degree` → 3. `render_and_loss` (reads SH) → 4. `backward` (reads SH) → 5. `update_max_radii2D` → 6. `add_densification_stats` → 7. `densify_and_prune` (clone → split → prune) → 8. `reset_opacity` → 9. `optimizer_step` (reads SH) → 10. `zero_grad`.

Across all scenes: **144 topology events, 0 opacity resets, invariants OK.**

---

## 5. Phase B — Real headroom measurement

Four hypotheses gate structural headroom. **H1–H3 pass; H4 fails on every scene.**

| Gate | Threshold | room | garden | bicycle | Result |
|---|---|---|---|---|---|
| H1 avoidable fraction | >= 0.40 | 0.889 | 0.886 | 0.907 | PASS |
| H2 topology-forced settle | <= 0.25 | 0.00114 | 0.00081 | 0.00142 | PASS |
| H3 p90 debt | >= 3 | 10 | 10 | 17 | PASS |
| H4 SH-Adam wall fraction | >= 0.10 | 0.0309 | 0.0495 | 0.0363 | **FAIL** |

- **Avoidable transitions are abundant (~89%)** and **topology forcing is negligible (~0.1%)**, and **debt accumulates** (p90 10–17, max ~13.8k–14.3k). So deferral is *possible*.
- **But SH Adam is only 3–5% of iteration wall** (H4 needs >=10%). Consumer block fraction is 9–11%. This is the binding constraint: the deferrable work is real but tiny relative to forward+backward+densification.

Aggregate: `scenes_ge_0.55 = 3/3` (avoidable), yet `structural_headroom_pass = false` because H4 fails universally.

---

## 6. Phase C — Debt replay / topology exactness

`topology_exactness_pass = true`. Streaming debt replay (numpy-vectorized, proven bitwise-equal to the frozen pure-Python `debt_replay.replay_debt`) was cross-checked against a topology oracle over the real densification fences.

| Scene | Fences | Mismatches | Exact |
|---|---|---|---|
| room | 2 | 0 | true |
| garden | 3 | 0 | true |
| bicycle | 2 | 0 | true |
| **Total** | **6** | **0** | **true** |

Bitwise equality holds for theta, m, v, Gaussian count, and topology mapping. No `LAVOS0_TOPOLOGY_EXACTNESS_FAIL`.

---

## 7. Phase D — Performance oracle (CUDA event timing)

| Metric | room | garden | bicycle |
|---|---|---|---|
| mean iter (ms) | 41.62 | 139.66 | 126.31 |
| SH-Adam ms/iter | 1.287 | 6.910 | 4.585 |
| SH-Adam wall fraction | 0.0309 | 0.0495 | 0.0363 |
| optimizer-lane speedup | 5.94x | 4.21x | 6.72x |
| gross saving (ms/iter) | 1.070 | 5.270 | 3.903 |
| net saving (ms/iter) | 1.070 | 5.269 | 3.902 |
| overhead fraction of gross | 0.015% | 0.012% | 0.015% |
| incremental full-30k reduction | 1.33% | 2.00% | 1.65% |

- **Optimizer lane:** geomean **5.52x** speedup, every scene >= 1.35x, **0 regressions > 0.5%**, overhead well within gate.
- **End-to-end:** `scenes_full30k_reduction_ge_2pct = 0`. The lane is fast, but it is too small a slice of the loop to move total training time by >= 2%.

`microbench_speedup = null`: the isolated dense-vs-lazy settlement microbench was not the binding measurement; H4 is decided by the real per-iteration SH-Adam wall fraction, which is measured and fails.

---

## 8. Phase E — Frozen analyzer verdict

```
verdict = LAVOS0_NO_STRUCTURAL_HEADROOM
source_identity_pass      = true
event_semantics_resolved  = true
topology_exactness_pass   = true
structural_headroom_pass  = false   (H4 fails on all scenes)
performance_headroom_pass = false   (0 scenes reach 2% end-to-end reduction)
strong_headroom_pass      = false
```

This is one of the protocol's valid outcomes. It is an honest negative, not a failure of execution: the run is real, exact, and reproducible.

---

## 9. Scientific interpretation

The purpose of LAVOS0 was only to test whether exact lineage-aware deferred optimizer execution has enough structural headroom to justify a full LAVOS system. It does not, on this workload:

- **What works:** exactness (bitwise), settlement overhead (~0.01%), and optimizer-lane speedup (~5.5x) are all strong. The lineage-aware deferred-execution *primitive* is sound and reusable.
- **What blocks:** the deferrable SH-Adam optimizer is a small fraction (3–5%) of 3DGS iteration wall. Forward, backward, and densification dominate. Deferring the optimizer cannot move end-to-end training by the required >= 2%.

**Recommendation:** close LAVOS0 honestly. Do not claim acceleration or novelty. If revisited, target a regime where the optimizer is a larger share of wall time (e.g., much higher SH degree, or workloads where forward/backward are cheaper), not the current fused-L1 3DGS loop.

---

## 10. Reproducibility & deviations

- Frozen sources were not modified; thresholds, analyzer, VOS1, ReproTopo0, checkpoints, and the Adam implementation were untouched. Bitwise requirements were not relaxed.
- Streaming debt replay was validated bitwise-equal to the frozen pure-Python replay before the full run.
- **Deviation (documented):** the protocol's camera split `holdout = i % 8 == 0` is not implemented by the frozen C0 trainer; scenes were run as-is with seed 42. LAVOS0 dynamics (deferrable fraction, SH-Adam wall fraction) are robust to this split.
- **Co-tenant note:** garden and bicycle shared the A100 with other tenant jobs during the run, inflating absolute wall-clock. All gate-relevant metrics (SH-Adam wall fraction, optimizer-lane speedup, end-to-end reduction fraction) are ratios measured via CUDA events on the same device, so co-tenant contention affects numerator and denominator together and does not change the verdict. room ran on an otherwise idle GPU.
- Verify locally: `sha256sum -c MANIFEST.sha256` in the mirror directory.