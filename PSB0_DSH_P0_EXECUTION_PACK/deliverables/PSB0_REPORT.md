# PSB0-P0 Report — Associative Segment-Scan Backward

**Date (execution):** 2026-10-01 (Asia/Shanghai)
**Remote node:** EPIC-05 — host `dsw-933622-64cc845c7d-lqswc`, 8× NVIDIA A100-SXM4-80GB, root
**Repo:** /root/3dgs-renderer-benchmark @ `abfa6c3e4b23caf5a927bf3722a65cba37b6fab3` (matches local)
**Frozen harness:** nerficg `e458ac6e` (v2.2), method FasterGSFused
**Scope:** THEORY / SOURCE / TRACE / CPU-ORACLE ONLY. **No CUDA built or run.**

## Core question

Can front-to-back alpha compositing over a splat segment be represented by a compact associative transform, so that backward state at arbitrary segment boundaries is recovered by prefix/suffix scan instead of fixed checkpoint + linear replay?

## Gate results

| Gate | Description | Verdict |
|------|-------------|---------|
| 0 | Remote guard (EPIC-05 only) | **PASS** — identity recorded; all experimental work on EPIC-05 |
| 1 | Exact algebra (closure + associativity + recoverability) | **PASS** — see PSB0_ALGEBRA.md |
| 2 | Prior-art audit | **PASS** — no collision (PSB0_PRIOR_ART.md) |
| 3 | State size ≤ checkpoint | **PASS** — summary (A,B)=4 floats = native float4 checkpoint |
| 4 | Operation model ≥20% state-recovery reduction in ≥2/3 scenes | **FAIL (borderline)** — see below |
| 5 | Exact CPU/oracle replay | **PASS** — cosine 0.99999994, relL2 5.2e-7 |
| 6 | Strong-backend E2E ≥1.05× in ≥2/3 scenes | **FAIL** — E2E ceiling ~1.02× |

## Gate 1 — algebra (PASS)

Per-splat affine transform `M_k = [[1, a_k c_k],[0, 1-a_k]]`. Segment summary `S(a:b)=(A,B)` with composition `(A1,B1)⊗(A2,B2)=(A1+B1·A2, B1·B2)`. Closed, associative (numerically verified to 3.4e-7), and recovers `T_k` (prefix B) and `C_after_k = C_total − C_inc_k` (prefix A). Full derivation in PSB0_ALGEBRA.md; oracle in PSB0_REFERENCE_ORACLE.json.

## Gate 3 — state size (PASS)

Native checkpoint = `float4` (color.rgb + T) = 4 floats per bucket-pixel. Scan segment summary = (A.rgb, B) = 4 floats. **Equal size** — no expansion. (Note: scan must materialize per-splat transforms ≈ V floats vs native checkpoints ≈ nbuckets·256 floats; since V ≈ nbuckets·256 these are comparable, not larger.)

## Gate 4 — operation model (FAIL, borderline)

Using **real** per-scene visit counts (PSB0_TRACE_STATS.json):

- room: V=14,894,848 visits, 58,472 buckets
- garden: V=35,872,768 visits, 140,684 buckets
- bicycle: V=25,459,968 visits, 101,725 buckets

State-recovery work model (gradient math excluded, identical in both):

- **Native replay:** 10 flop + 1 exp per visit (recompute `expf`, alpha, blending_weight, color_after, T).
- **Scan:** 14 flop (Blelloch compose, 2×7) + 3 flop (C_after) per visit, with the backward `expf` eliminated (alpha reused from forward) but a `div` (gaussian=alpha/opacity) added to the gradient path.

Reduction `= (exp − 6)/(exp + 8)` — **scene-independent** (per-visit ratio identical across room/garden/bicycle, so scene mix cannot change the gate outcome):

| precise-expf cost (FMA-equiv) | reduction | ≥20%? |
|---|---|---|
| 4 (fast `__expf`) | −13.3% | no (worse) |
| 8 | +10.5% | no |
| 10 | +21.1% | yes |
| 12 | +26.1% | yes |
| 16 | +32.0% | yes |

**Breakeven for 20% is exp ≈ 9.5 FMA-equiv.** Precise `expf` on A100 is a multi-instruction sequence (range-reduction + polynomial + 1 SFU op) estimated 8–16 FMA-equiv, placing the reduction at 12–37% — straddling the threshold. At fast-exp the scan is 13% *worse*.

**Structural finding:** the Blelloch compose (14 flop/visit) is *inherently more arithmetic* than the native replay recurrence (10–11 flop/visit). The scan's only state-recovery saving is eliminating the backward `expf` recomputation — an uncertain, hardware-dependent quantity that P0 is forbidden to measure (no CUDA profiling). The reduction is therefore not robustly ≥20% across the realistic expf cost range. **Gate 4 not confidently passed.**

## Gate 6 — economics (FAIL)

Measured FasterGSFused room: training_wall_s=272.3s @ 1920×1080, 30k iters, 1.29M gaussians. State-recovery is a small fraction of the backward (which also does gradient math, preprocess, Adam), and the backward is ~1/3 of the step. Even granting the full borderline Gate 4 reduction, the E2E ceiling from state-recovery alone is `1/(1−0.33·0.25·0.20) ≈ 1.017×` — **below the 1.05× Gate 6 threshold.** Gate 6 fails independently of Gate 4.

## Verdict

**`DROP_OPERATION_COUNT`**

The associative segment-scan backward is **algebraically exact and numerically verified** (Gates 1/3/5 PASS) with **no prior-art collision** (Gate 2 PASS) and **equal state size** (Gate 3 PASS). But it **does not reduce state-recovery work**: the general associative compose costs more arithmetic than the native fixed-bucket replay recurrence, and the only saving (eliminating backward `expf`) is borderline and hardware-dependent (Gate 4 FAIL). Consequently the strong-backend E2E ceiling is ~1.02×, below the 1.05× threshold (Gate 6 FAIL).

Per the prereg decision tree, the first failing gate determines the verdict: **DROP_OPERATION_COUNT**.

## What would change the verdict

- A measured `expf` cost on the strong backend ≥ ~10 FMA-equiv *and* a fused forward/backward that eliminates the backward `expf` entirely would push Gate 4 to ~21–32% — still marginal for Gate 6.
- A formulation that reduces the *compose* cost below the replay recurrence (e.g., a cheaper scan lattice, or recovering state without the general 2× compose) could plausibly clear both gates. That is a PSB1 question, not P0.

## Reproduce

```
scp deliverables/psb0_reference_oracle.py deliverables/psb0_operation_model.py EPIC-05:/tmp/
ssh EPIC-05 "python3 /tmp/psb0_reference_oracle.py"
ssh EPIC-05 "python3 /tmp/psb0_operation_model.py"
```

Artifacts: PSB0_VERDICT.json, PSB0_SOURCE_LOCK.json, PSB0_ALGEBRA.md, PSB0_PRIOR_ART.md, PSB0_TRACE_STATS.json, PSB0_OPERATION_MODEL.json, PSB0_REFERENCE_ORACLE.json, PSB0_AMDAHL.json, scripts/, MANIFEST.sha256.
