# N1 Report — ReBudgetGS Mechanism Measurement

Campaign: **N1_ReBudgetGS_mechanism_measurement** (frozen preregistration `N1_PREREG.md`, operationalization `N1_SPEC_OPERATIONALIZATION.json`, both fixed before any probe gradient was computed).

Host: EPIC-05, 8× A100-SXM4-80GB. Frozen P12/C0 baseline (trainer sha256 `675b0267…`, c0 ext `.so` `d26cccf4…`, fused-ssim `.so` P11 lock `dd6a4c79…`). Scenes: **room** (311 cams) and **bicycle** (194 cams), MipNeRF360, 1080p, seed 42, `--arm c0 --eps2d 0.3 --loss fused`.

## Verdict

```
N1_SMALL_BUT_ACTIONABLE
```

Room shows **+11.5%** matched-cost improvement of the offline {A0, A1, A2-random} assignment over the *strictest* best-0/1-skipping oracle at ρ=0.75 (≥5% rule met); bicycle shows +4.0% (below 5%). No catastrophic counterevidence. H2 (paired-view rescue) and H3 fail in both scenes, and the strong reallocation gate fails on bicycle, so the strong verdicts do not apply. Per the prereg: *may still support a smaller publishable optimization; do not kill solely for missing 3×-scale headroom.* Stage 6 (cross-scene policy) was **not run** — its gate (strong opportunity in both scenes) failed.

## Stage 0 — source & checkpoint audit (PASS, both scenes)

- Snapshot runs `n1snap_c0f_s42_{room,bicycle}` (save-only edit of the frozen P12 trainer, sha256 `49a79538…`) reproduced the canonical seed-42 camera stream bit-exactly (25,000/30,000 entries prefix-identical to the canonical `p12a` runs) and produced checkpoints at 10K/15K/20K/25K.
- Bit-exact counterfactual restore verified at every checkpoint: after random perturbation **and** after a real A1 action, `StateMaster.restore()` returns all parameters, Adam moments, and step counters to the saved state bit-for-bit; held-out renders are bitwise deterministic; re-rendering after restore reproduces the held-out metrics bit-exactly (restore–render invariance), and switching checkpoints back and forth preserves bit-exactness.
- **Adam-step semantics (expected baseline behavior, not a defect):** `state['step']` equals the iteration minus one lost increment per densification event (events at 500–14900 replace the parameter objects, whose `.grad` is `None` at the event iteration, so torch Adam skips the increment). At 15K all five parameter groups read step 14856 and increment on every subsequent step. The probe's `update_learning_rate(C+1)` + `optimizer.step()` therefore reproduces training iteration C+1 exactly.
- Execution non-determinism (CUDA atomics) is real but tiny: three A1 replicas from the same restored state differ by ~1e-8 absolute on one-step held-out deltas (~1e-4 relative). σ_replica (robust MAD scale): room 6.9–8.6e-10, bicycle 1.0–1.4e-9.
- HIGS dynamic-scene tracker discipline (P13-Y/P17): `reset()` only when parameter *tensor objects* change (checkpoint switch); in-place `copy_` restore is read live by the renderer handle — verified by the restore–render invariance test.

## Stage 1 — deterministic probe list

Written to `N1_PROBE_LIST.json` before any gradient was computed. 64 anchors per (scene, checkpoint ∈ {15K, 20K, 25K}) = 384 anchors total; anchor k at checkpoint C is training opportunity C+k with view `seq[C+k-1]` from the canonical 30,000-entry seed-42 stream (25K anchors read the true canonical continuation 25001–25064). Held-out sets (8 cams/scene, even stride over the complement of all anchor views): room {0,36,74,110,160,207,242,275}, bicycle {1,25,55,86,103,125,157,174}. Partner pools: 7 per anchor by the frozen geometric-overlap score (high×3, medium×2, low×2). A2-random partners drawn from a frozen per-scene RNG stream. All selection purely geometric.

## Stage 2 — counterfactual measurements

**Gradient conflict (ground truth, 1,344 pairs/scene).** True per-view fused-loss gradients at the checkpoint state are **near-orthogonal**: median global cosine room 0.0071, bicycle 0.0034. Conflict concentrates exactly where geometry predicts: high-overlap pairs (median visibility Jaccard 0.81 room / 0.65 bicycle) have median cosine 0.354 / 0.061, while medium and low pairs are ≈0 with 29–42% *negative* cosine. Pairwise gradient conflict at the 15–25K stage is the exception (similar views), not the rule.

**Sketches (frozen dense Rademacher, 256/group × 4 groups).** 1,014 room + 712 bicycle unit-normalized 1024-dim sketches at 10K/15K/20K/25K, plus per-view fused losses for the EMA feature chain.

**Counterfactual actions (4,608 rows).** Every action from the bit-exact restored state; held-out (8 cams) re-evaluated after each action.

Per-checkpoint heterogeneity of the one-step A1 held-out delta (fused loss, positive = progress):

| scene | ckpt | median Δ(A1) | sign rate | σ_onestep |
|---|---|---|---|---|
| room | 15000 | −9.10e-5 | 1.6% | 4.17e-5 |
| room | 20000 | −1.07e-4 | 4.7% | 4.73e-5 |
| room | 25000 | −2.12e-5 | 31.3% | 4.55e-5 |
| bicycle | 15000 | −1.26e-4 | **0.0%** | 2.35e-5 |
| bicycle | 20000 | **+1.91e-4** | **100%** | 1.52e-5 |
| bicycle | 25000 | −3.77e-5 | 15.6% | 4.17e-5 |

The sign of one-step held-out progress is strongly checkpoint-dependent (bicycle flips from 0% to 100% positive between 15K and 20K) — a phase structure in the loss landscape, not smooth noise. Most single steps at most states make the held-out fused loss *worse* (room pooled sign rate 12.5%).

## Stage 3 — action timing (10 warm-ups + 30 repeats, restore untimed, no profiler)

| scene | ckpt | A0 (gate) | A1 | A2 |
|---|---|---|---|---|
| room | 15000 | 4.04 ms | 12.43 ms | 26.20 ms |
| room | 25000 | 4.38 ms | 15.44 ms | 24.72 ms |
| bicycle | 15000 | 8.10 ms | 30.20 ms | 52.04 ms |
| bicycle | 25000 | 7.70 ms | 28.14 ms | 50.23 ms |

Gating (A0) costs 26–33% of a training step; A2 costs 1.6–1.8× A1. Backward dominates A1 (6.3–8.3 ms room, 10.1–13.4 ms bicycle); the optimizer step is bicycle-heavy (10.2 ms at N=3.65M).

## Stage 4 — sketch predictivity

Spearman between sketch-predicted disagreement (1 − cosine of concatenated unit-normalized group sketches) and true-gradient disagreement (1 − global-normalized cosine), 1,344 pairs/scene:

| scene | stale (C−5000) | fresh (C) |
|---|---|---|
| room | **0.758** | 0.798 |
| bicycle | **0.463** | 0.429 |

Five-thousand-iteration-stale sketches lose almost nothing (room −5% relative; bicycle stale is actually *better* than fresh). Predictivity is concentrated in the high-overlap tier (room 0.72/0.82 stale/fresh) and vanishes in the low tier (≈0). Sketches rank *similar-view* conflict well but carry almost no signal for dissimilar pairs — precisely where the A2 value turns out to live (below).

## Stage 5 — budget frontier and gates

Exact MCKP (0.01 ms grid) over 192 anchors/scene; T(a) = Stage-3 medians; E(A0):=0.

| ρ=0.75, primary set | room | bicycle |
|---|---|---|
| assignment progress | **+1.481e-3** | **+1.350e-2** |
| all-A1 progress | −1.153e-2 | +3.059e-3 |
| best fixed 0/1 mixture | 0.0 | +2.294e-3 |
| oracle-01 control (best 0/1 skipping) | +1.328e-3 | +1.298e-2 |
| assignment / max control | **1.115** | 1.040 |
| assignment / all-A1 | −0.13 (all-A1 negative) | 4.41 |
| shares (A0/A1/A2r) | 84.4/10.9/4.7% | 58.9/17.2/24.0% |

The room assignment turns an *overall harmful* all-A1 policy (held-out progress −0.0115) into positive progress at 75% of the cost. Both scenes' optima are essentially saturated by ρ=0.6.

**Gates (frozen):**
- **H1 (skip opportunity): PASS.** Skip shares 87.5% (room) and 61.5% (bicycle) vs the 20% gate. Single-view steps at the 15–25K stage mostly do not move held-out quality forward — descriptive, and the campaign's most robust effect.
- **H2 (paired-view rescue): FAIL.** 0.52% (room), 0.00% (bicycle) vs the 15% gate.
- **H3 (argmax-E preference): FAIL.** Room prefers A0 84.9%/A1 12.0%/A2-best 3.1% (A1 < 30% gate); bicycle A0 54.7%/A1 38.5%/A2-best 6.8% (A2-best < 10% gate).
- **Strong ρ=0.75 gate: room PASS, bicycle FAIL** (1.040 < 1.10 vs the 0/1 oracle control) → Stage 6 not run; `N1_CROSS_SCENE_POLICY.json` records the gate failure.

**Mechanistic highlight:** the best A2 partner is a **low-overlap** partner for **all 384 anchors** (192+192; high/medium tiers never win). Where a second view helps, it helps by adding *new* information, not by reinforcing the same view — while the sketches (which only predict high-tier conflict) are blind to exactly those pairs. "Rescue" as preregistered (similar-view pairing) is not the operative mechanism; the value that exists comes from skip decisions plus occasional genuinely-diverse second views.

## Verdict chain (prereg §11, in order)

1. `N1_STRONG_REALLOCATION_SIGNAL` — ✗ (H2/H3 fail both scenes; strong gate fails on bicycle; policy diagnostic not reached).
2. `N1_MECHANISM_EXISTS_POLICY_WEAK` — ✗ (requires H2/H3 + reallocation opportunity in both scenes).
3. `N1_SMALL_BUT_ACTIONABLE` — ✓ **room +11.5% matched-cost over best 0/1 skipping (≥5%); no catastrophic counterevidence** (restore bit-exact at all checkpoints, renders deterministic, replica noise 5 orders below the effect, no GPU contamination).
4. `N1_NO_REALLOCATION_VALUE` — ✗ (room ≥ 5%).

## STOP compliance

No final online controller. No full 15K→30K ReBudgetGS training (the only trainings were plain frozen P12/C0 runs to 25K for probe-state generation). No threshold tuning after inspecting bicycle (spec + probe list precede all measurements). No learned network (gated Stage 6 not run). No renderer change (binaries hash-locked and re-verified). No densification change (probes take no densification actions; last event 14,900 < 15,000). No garden.

## Artifacts

All 14 preregistered artifacts are in `artifacts/` with sha256 in `MANIFEST_N1.json`: `N1_SOURCE_IDENTITY.json`, `N1_CHECKPOINT_IDENTITY.json`, `N1_PROBE_LIST.json`, `N1_ACTION_RESULTS.parquet` (4,608 rows), `N1_ACTION_TIMING.json`, `N1_GRADIENT_CONFLICT.parquet` (2,688 rows), `N1_SKETCH_PREDICTIVITY.json`, `N1_ACTION_HETEROGENEITY.json`, `N1_BUDGET_FRONTIER.json`, `N1_CROSS_SCENE_POLICY.json` (gate-failed record), `N1_CONTAMINATION.json`, `N1_VERDICT.json`, `N1_REPORT.md` (this file), `MANIFEST_N1.json`. Supporting evidence (sketches `.npz`, per-view losses, per-scene raw JSONs, audit raws, contamination CSVs, logs, scripts) is listed in the manifest.
