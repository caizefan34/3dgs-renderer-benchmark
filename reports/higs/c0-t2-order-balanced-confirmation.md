# C0-T2 — Order-Balanced Nested F+B Confirmation

**Date:** 2026-09-22
**Status:** COMPLETE — **C0_T2_PUBLICATION_PASS → C0_TIMING_PUBLICATION_GRADE** — C0 timing frozen permanently
**Role:** C0-T1 = engineering timing closure; **C0-T2 = publication-grade balanced confirmation**
**Does NOT reopen C0.** No correctness rerun, no direct backward, no forward-only, no profiler attribution, no rebuild, no module modification. Only the fixed execution-order confound in C0-T1's timing was addressed.

---

## 1. Task and scope

C0-T1 established `SUCCESSFUL_EXACT_COMPOSITION` / `30K_AUTHORIZED` with nested F+B reductions of 15.12% (room), 15.87% (bicycle), 21.16% (garden), measured with a fixed per-sample variant order V0→V1→V2→V3. Because C0-T1 also observed protocol/GPU-state-dependent timing variability (bimodal distributions), the fixed order was a remaining confound: any systematic advantage or penalty attached to execution position would have biased variant comparisons. C0-T2 repeats **only** the nested F+B protocol with a **deterministic Latin rotation** so that every variant occupies every execution position exactly equally, and confirms — or refutes — the C0-T1 result under this balanced design.

## 2. Frozen environment and binary identity

Exactly the C0-T1 authoritative environment: host `bms-39468022-001`, A100-PCIE-40GB (SM80), UUID `GPU-6d75016d-fc44-9443-9190-396375f20f6a`, `CUDA_VISIBLE_DEVICES=4`, torch 2.9.1+cu128, CUDA 12.8, commit `77ab983f…`.

Binary identity verified in-harness before any timing (abort on mismatch):

| Component | SHA256 (prefix) | Note |
|---|---|---|
| composed extension | `7ca1c6bf…` | single binary for V0–V3, runtime env toggles only |
| core extension | `361b216b…` | `/tmp/h1_b2_authoritative/gsplat_cuda/gsplat_cuda.so` |
| scene pack module | `5f86d0cb…` | prebuilt from the C0 composed build (higs_c0_worktree sources), pre-registered in `sys.modules`; the same binary the C0-T1 run loaded via its extensions cache (verified by cache-timestamp forensics: `build_params.json` rewritten 13:13, three minutes before T1's 13:16 results; `.so` unchanged since the 03:37 composed build). No JIT build, no ninja dependency, no rebuild. |
| F9 patch | `1faa05c1…` | frozen |
| H8-MR patch | `8fb23ac8…` | frozen |

Variant toggles identical to C0-T1 (recorded per variant in `provenance.json`): V0 = scalar off / F9 disabled / H8 off; V1 = scalar on / F9 disabled / H8 off; V2 = scalar on / F9 enabled / H8 off; V3 = scalar on / F9 enabled / H8 on. Deployment candidate remains **V3 only**.

Fixture identity asserted in-harness against the C0-T1 values (abort on mismatch): room 115,278 GS / 44,908 visible / 2048×1365; bicycle 580,416 / 181,525 / 2048×1361; garden 66,282 / 24,483 / 2048×1327 — all matched exactly (`fixture_matches_c0_t1: true`).

## 3. Balanced ordering design

Deterministic Latin rotation, no randomization:

```text
sample mod 4 == 0:  V0 V1 V2 V3
sample mod 4 == 1:  V1 V2 V3 V0
sample mod 4 == 2:  V2 V3 V0 V1
sample mod 4 == 3:  V3 V0 V1 V2
```

With 100 samples per rep (100 % 4 == 0), each variant occupies each execution position exactly 25 times per rep and 125 times over 5 reps. Verified from the raw CSV: **every (variant × position) cell contains exactly 375 observations** (3 scenes × 125) — the design is exactly balanced, not approximately.

## 4. Protocol

One process, one fixture, one binary, one CUDA stream. Nested CUDA events per iteration: `total_start → forward_start → forward() → forward_end → [loss construction] → backward_start → loss.backward() → backward_end → total_end`, all on the explicit current stream, synchronized before `total_start` and after `total_end`. Upstream gradient tensors (`vr`, `va`, seed 4200) pre-generated once per scene, outside every timed region. 20 warmup rotation samples (= 80 iterations, balanced) per scene, then 5 reps × 100 samples × 4 variants = **500 observations per variant per scene, 6,000 total** — nested F+B only.

Per-observation invariants asserted in-harness and re-verified locally from the raw CSV after download: `fb ≥ forward`, `fb ≥ backward` (0.01 ms tolerance), `gap = fb − forward − backward ≥ −0.5 ms`. **Result: 0 failures in 6,000 observations** (both checks agree). The gap (loss-construction bridge) is stable at 0.162–0.167 ms median across all scenes and variants — identical to C0-T1.

## 5. Authoritative balanced nested F+B (median ms, n=500 per cell)

| Scene | V0 | V1 | V2 | V3 | **V3 vs V0** |
|---|---|---|---|---|---|
| room | 5.727 | 5.753 | 4.918 | **4.780** | **−16.54%** |
| bicycle | 8.178 | 8.175 | 7.250 | **6.972** | **−14.74%** |
| garden | 3.650 | 3.642 | 3.587 | **2.897** | **−20.62%** |

## 6. Paired analysis (primary comparison)

Each (rep, sample) contains all four variants, so V3-vs-V0 pairs share the same time window. 500 pairs per scene:

| Scene | paired median | paired mean | p10 | p90 | fraction V3 faster |
|---|---|---|---|---|---|
| room | 16.88% | 14.64% | 10.05% | 18.39% | **98.2%** |
| bicycle | 14.70% | 14.28% | 11.28% | 16.44% | **96.4%** |
| garden | 20.59% | 20.17% | 14.89% | 22.14% | **99.4%** |

All paired medians are within 0.4 pp of the median-based reductions — the paired (window-matched) and marginal comparisons agree, i.e., the result is not an artifact of time-window allocation.

## 7. Position-bias analysis (the reason for C0-T2)

**Does execution position materially shift time? No.** The position effect independent of variant (median of each observation's fb normalized by its own variant's overall median) is tiny on every scene: room 0.9987/1.0016/1.0008/0.9992 (positions 1–4), bicycle 1.0001/1.0000/1.0002/0.9997, garden 1.0034/0.9989/0.9994/0.9994 — i.e., **±0.34% maximum**. C0-T1's fixed order therefore carried no material position privilege: V0 was not systematically favored or penalized by always running first.

Per-variant position spreads tell a complementary story:

| Scene | V0 | V1 | V2 | V3 (max) | per-position V3-vs-V0 reduction |
|---|---|---|---|---|---|
| room | 0.16% | 0.23% | 1.92% | **8.08%** | 17.6 / 10.7 / 15.2 / 17.3% |
| bicycle | 0.08% | 0.05% | 0.23% | 0.12% | 14.8 / 14.7 / 14.7 / 14.7% |
| garden | 1.01% | 0.11% | 0.23% | **6.93%** | 16.0 / 20.6 / 20.7 / 20.6% |

On room and garden the per-position spread is concentrated in **V3** (V0 is flat: e.g., room V0 position medians 5.722–5.731 ms), and the "slow" position differs by scene (room: position 2, V3 median 5.110 ms; garden: position 1, V3 median 3.092 ms) — while the same position is *not* slow for the other variants. This is therefore not a causal execution-position effect (which would shift all variants at that position) but **protocol/GPU-state-dependent timing variability of V3's moment-space backward across sample-window classes**, of the same nature C0-T1 documented. No hardware cause (clocks, power, P-state) is claimed — no such telemetry was measured. The balanced design does exactly what it was designed to do: V3 occupies every position/window class equally, the overall medians and paired comparisons average across all windows, and **V3 remains decisively faster than V0 at every position on every scene (minimum +10.7%)** — no order-specific reversal anywhere (12/12 scene×position cells positive).

## 8. Rep stability

V3-vs-V0 median-based reduction per rep (all five shown per scene):

| Scene | rep 0 | rep 1 | rep 2 | rep 3 | rep 4 | positive all 5 | ≥10% all 5 |
|---|---|---|---|---|---|---|---|
| room | 17.18 | 16.32 | 16.88 | 17.27 | 15.16 | ✓ | ✓ |
| bicycle | 14.65 | 14.85 | 14.83 | 14.67 | 14.64 | ✓ | ✓ |
| garden | 20.67 | 20.59 | 20.61 | 20.77 | 20.57 | ✓ | ✓ |

Required (positive in all 5 reps on ≥2/3 scenes): **3/3 scenes**. Preferred (≥10% in all five reps on ≥2/3 scenes): **3/3 scenes** — the preferred tier is met on all scenes.

## 9. Confirmation against C0-T1

| Scene | C0-T1 (fixed order) | C0-T2 (balanced) | Δ |
|---|---|---|---|
| room | 15.12% | 16.54% | +1.4 pp |
| bicycle | 15.87% | 14.74% | −1.1 pp |
| garden | 21.16% | 20.62% | −0.5 pp |

**The C0-T1 result is confirmed**: all three reductions reproduce within 1.4 pp under the balanced design. The absolute time level of this run sat in a faster GPU-state regime than C0-T1's run (e.g., bicycle V0 median 8.178 ms here vs 13.906 ms in C0-T1) — the same protocol/GPU-state-dependent variability documented in C0-T1, which is why T1's conclusions were drawn from within-run comparisons; the T2 confirmation likewise relies only on within-run, window-paired comparisons. The gain percentages are stable across both regimes.

## 10. Distribution notes

- **Bicycle V0 contains one extreme transient**: max 105.1 ms in 500 observations (p90 = 8.260 ms, tight). The outlier inflates V0's std to 4.39 ms but affects neither medians nor paired statistics; V3's bicycle distribution contains no such event (max 7.585 ms).
- **Garden V2 did not reproduce T1's regression in this run's regime** (V2 fb median 3.587 ms ≈ V0's 3.650 ms, backward 1.715 ms, tight p90 3.606 ms — vs T1's nested-context V2 backward degradation to 3.599 ms). This is consistent with T1's characterization of that effect as a **configuration-dependent execution interaction** — state-dependent in magnitude rather than a deterministic constant. The C0-T1 deployment rule (V3 only; do not ship F9 without H8-MR) was set from T1's worst-case evidence and remains unchanged; C0-T2 did not re-qualify V2, which is not the deployment candidate.

## 11. Gate evaluation — C0_T2_PUBLICATION_PASS

| Condition | Result | Evidence |
|---|---|---|
| V3 nested F+B reduction ≥10% on ≥2/3 scenes | **PASS (3/3)** | 16.54 / 14.74 / 20.62% |
| No scene regression >2% | **PASS** | all scenes strongly positive |
| 0 nested timing invariant failures | **PASS** | 0 / 6,000 (in-harness + local recheck) |
| V3 faster than V0 in >75% of pairs on ≥2/3 scenes | **PASS (3/3)** | 98.2 / 96.4 / 99.4% |
| No order-specific reversal | **PASS** | 12/12 scene×position cells positive (min +10.7%) |
| Rep stability (positive all 5 reps, ≥2/3 scenes — required; ≥10% all 5 — preferred) | **PASS (3/3 required, 3/3 preferred)** | see §8 |

Position dependence does not remain a concern: the variant-independent position effect is ±0.34% and the balanced design removes window allocation from the comparison (paired ≈ marginal reductions, §6).

**Status: C0_T2_PUBLICATION_PASS → C0_TIMING_PUBLICATION_GRADE.**

## 12. Freeze decision, artifacts, and language discipline

**C0 timing is now frozen permanently.** Final publication numbers for the frozen V3 stack on this A100 cohort are the C0-T2 balanced values: **room −16.54%, bicycle −14.74%, garden −20.62%** (nested F+B, medians, 500 balanced observations per cell), corroborated by the C0-T1 engineering values. No further timing runs unless the environment or code changes.

Artifacts in `artifacts/higs-c0-t2/` (7 files, C0-T1 not overwritten): `raw_balanced_nested_timing.csv` (6,000 rows with execution_position), `timing_summary.json` (median/mean/std/p10/p25/p75/p90/min/max per scene/variant/metric), `paired_analysis.json`, `position_bias.json` (per-variant and variant-normalized effects, per-position reductions), `rep_stability.json`, `provenance.json` (hashes incl. scene module, env snapshots, workloads, protocol), `final_gate.json` (six conditions, per-scene detail, freeze decision).

Language discipline per spec: all cross-protocol/cross-run absolute-time differences are described as **protocol/GPU-state-dependent timing variability** (no clock/power/P-state cause is claimed — no hardware telemetry was collected); the garden V2 effect is a **configuration-dependent execution interaction**; the position-class-correlated V3 variation (§7) is described behaviorally, without a microarchitectural mechanism claim.
