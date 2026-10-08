# D0 — 3DGS Training Waste Census: Report

Campaign: D0 (discovery-first, non-optimization census over the frozen P12 C0 baseline)
Host: EPIC-05, physical GPU 2 (UUID GPU-bcca92b9-0234-7ec9-a2f0-a63925d52146)
Date: 2026-09-28
Canonical verdict: **D0_NO_3X_HEADROOM** (via bundle tool `d0_verdict.py`, sha256 393af7e0…)
No D1 was started. No production optimizer, no densification modification, no git commit.

## 0. Research question and answer

Question: *Does the frozen P12 C0 30K trajectory contain enough retrospectively
identifiable and pre-birth predictable cumulative work to make a >3×
end-to-end acceleration physically plausible without lowering the final C0
quality target?*

Answer (pre-registered measurement, both scenes): **No.** The 95% bootstrap
lower bound on the admissible removed-work fraction R is ≈1.4% (room) and
≈0.5% (bicycle), giving an optimistic speedup ceiling S_upper ≈ 1.016 (room)
and ≈ 1.005 (bicycle) — two orders of magnitude below the 3× gate. In
addition, no checkpoint before 30K lies inside the frozen quality envelope
(t_star = 30000 for both scenes), so *no* tail truncation is admissible at
all. Even the most generous alternative partition (init-root sensitivity
variant, reported only, never substituted) yields S_upper ≈ 1.19 (room) and
≈ 1.04 (bicycle) — still nowhere near 3×.

Interpretation: on this baseline, gaussians/prims that are
"retrospectively dead" (no final descendant) die *young and cheap* — their
cumulative render work is a tiny fraction of the trajectory. All removals
happen inside the frozen densify/prune window (iteration ≤ 14,900); after
15,000 the population is frozen, so there is no late, fat, removable
population. With ~30% of founding lineages dead (room) translating to only
~1.9% of tile-intersection work, removing "waste" cannot plausibly accelerate
end-to-end training by 3×.

## 1. Scope, protocol, and constraints (frozen)

- Frozen P12 C0 recipe: seed 42, 30000 iterations, densify 500..15000 @100,
  opacity reset every 3000 (<15000), SH oneup every 1000, tile16,
  `--eps2d 0.3 --loss fused`, eval SepSSIM + LPIPS(vgg), `--final-eval all`.
- Two scenes: room (311 eval cameras) and bicycle (194 eval cameras).
- Constraints honored verbatim: no birth suppression; no densification
  change; no prune-rule change; no selector; no threshold tuning; no camera
  sampling change; no loss/optimizer change; no D1; no invented P20/P21; no
  speedup claim from instrumented runtime (traced runs are FUNCTIONAL_ONLY;
  instrumented wall time is not primary timing evidence).
- `RETROSPECTIVELY_DEAD` does **not** mean "safe to remove". Low-contribution
  lineages are **not** called safe to remove. No causal or removal claim is
  made anywhere in this census.
- Garden was not touched. Raw tensors stay on EPIC-05 (hashes + paths in
  `D0_FINAL_MANIFEST.json`).
- GPU discipline: physical GPU 2 only; clean-GPU gate (10 checks × 30 s with
  abort) before every launch; GPUs 0/1 never touched.

## 2. Frozen identity and instrumentation (Stage 0)

- `D0_SOURCE_IDENTITY.json`: all_match = true (frozen trainer
  `/root/epic05_p12/publication_trainer_p12.py`, frozen renderer, fused-SSIM
  binary, patch lineage as pinned by the D0 prereg pair).
- Instrumentation only *adds* logging; it does not alter semantic tensors or
  training RNG: metadata-only UID/lineage tracking; birth-feature logging
  strictly before materialization; N_t/I_t per-iteration logging; per-UID
  intersection accounting every 20th iteration (sample offset 11 for room,
  15 for bicycle); 5K/10K/15K/20K/25K/30K checkpoints + eval outputs; source
  diff in `D0_SOURCE_DIFF.diff` / `D0_RENDERER_DIFF.diff`.
- Forward determinism check: iteration-1 loss is **bit-identical** across
  every run including the traced run and the historical P12 prefix — room
  0.2896811068058014 (C3room1, C3room2, T3room, ref30k_prefix), bicycle
  0.20873187482357025. Backward atomicAdd reduction order injects ~1e-7
  noise from iteration 2; all run pairs (clean–clean included) first separate
  at iteration 600–800; divergence amplifies at densification thresholds.
- Camera-sequence check passed (frozen camera order preserved).

## 3. Trace-validity calibration (Stages 0c–2) — full honest record

The trace-validity screen went through a documented calibration sequence;
every step below is preserved as an artifact.

1. First smoke run was judged INVALID by the initial *max-envelope* gate
   (`D0_TRACE_VALIDITY_smoke_run1_INVALID.json`, preserved). Root-cause
   analysis showed the 1× max-envelope comparison has a coin-flip false-
   failure probability ≈ n_ct/n_total because it compares a max over ct pairs
   against a max over a smaller number of within-group pairs. No frozen
   recipe element was touched to "fix" this.
2. Pre-registered replacement screen (`D0_TRACE_VALIDITY_SCREEN_PREREG.json`,
   written before any new data): **INVALID iff clean_traced_max > 2.0 ×
   within-group envelope max** (env = max(cc_max, tt_max)) for loss and dN_rel
   (and final_N at 30K); literal 1× values reported verbatim. Acceptance was
   pre-registered at 2.0× for exactly this small-sample reason.
3. Expanded smoke ensemble (Stage 0c rep3; 7 clean + 6 traced samples per
   scene) → `D0_TRACE_VALID_smoke.json`: D0_TRACE_VALID_SMOKE, no reasons.
   Spread diagnostic (`D0_TRACE_SPREAD_DIAGNOSTIC.json`, 20000 permutations,
   seed 20260928): room mean-diff p = 0.911 (dN) / 0.963 (loss), max-ct
   p = 0.535 / 0.542 — traced runs are not distinguishable from clean runs.
   Smoke screen values: room loss ct 0.158449 vs env 0.158303 (ratio 1.001),
   dN ct 0.022187 vs env 0.021122 (ratio 1.050); bicycle loss ct 0.007168 vs
   env 0.007584 (ratio 0.945), dN ct 0.012433 vs env 0.010563 (ratio 1.177).
4. Stage 1 (room 30K trace) → D0_TRACE_VALID_30K room gates:
   loss ct 0.137009 vs cc 0.137168 (ratio 1.001) pass; n_gs ct 20 867 vs
   cc 14 906 pass; final_N ct 2.382% vs cc 1.553% (ratio 1.534) pass;
   quality trace inside the reference envelope: true.
5. Stage 2 (bicycle 30K trace) initially fired the pre-registered final_N
   screen: ct 0.6658% vs cc 0.2826% (ratio 2.356 > 2.0) →
   D0_TRACE_INVALID (`D0_TRACE_VALIDITY_30k.json`, pre-escalation state).
   Forensics (recorded in the escalation prereg): eval cadence identical
   (the frozen recipe evaluates at 5K/10K/15K/20K/25K/30K regardless;
   the traced run only *adds* `d0_ckpt` rows), iter-1 loss bit-identical,
   divergence onset 600–800 for all pairs including clean–clean, and the
   30K final_N of the three clean replicas had converged unusually tight
   (0.283% max) while the trace sat at its trajectory-frozen distance
   (0.666%). Room's clean–clean spread at 30K was 1.55% — larger than the
   bicycle ct distance that tripped the screen.
6. Pre-registered one-round escalation
   (`D0_BICYCLE_FINALN_SCREEN_ESCALATION_PREREG.json`, written before any
   new data existed; one round only, fixed at +3 frozen-recipe clean bicycle
   30K runs; envelope re-estimated over ALL clean replicas + P12 reference;
   thresholds unchanged; verdict final): CLEAN3/4/5_bicycle ran rc=0
   (20:24–21:12, gated and clean).
7. **Escalated outcome: D0_TRACE_VALID_30K, reasons empty for both scenes.**
   Extended clean final_N set {CLEAN 3 667 776; CLEAN2 3 657 439;
   CLEAN3 3 664 111; CLEAN4 3 645 472; CLEAN5 3 715 720; ref 3 663 539} vs
   trace 3 681 789: ct_max = 0.9962% (vs CLEAN4), cc_max = 1.8906%
   (CLEAN4 vs CLEAN5, two frozen-recipe clean runs) → ratio 0.527 ≤ 2.0,
   screen PASS. loss screen ct 0.009064 vs cc 0.008648 (ratio 1.048) pass;
   n_gs screen ct 44 567 vs cc 70 248 pass; quality envelope true.
   The earlier fire is attributed to small-sample envelope estimation, not
   to instrumentation; the escalation protocol was pre-registered before the
   new data and its outcome is final.

## 4. Stage 3 — per-lineage accounting, tail quality, headroom

All accounting validity gates passed before the headroom numbers were
accepted (`D0_COST_ACCOUNTING.json`):

| gate | room | bicycle |
|---|---|---|
| sampled reconstruction error (≤ 0.03) | 0.006491 | 0.005715 |
| per-sample exact alignment (rel max) | 0.0 | 0.0 |
| sample-shard crosscheck | true | true |
| isect sumcheck mismatches | 25 (first at iter 1) | 0 |
| isect sumcheck max relative | 5.93e-08 | 0 |
| grad_missing_iters | 0 | 0 |

(The ±1-unit sumcheck deltas are frozen-op-internal rounding between the
pass-1 AccuTile count and the pass-2 emit count; relative magnitude ≤ 6e-8.
Raw mismatch count reported verbatim.)

Lineage partitions (post-init birth roots, frozen spec 4.5):

| quantity | room | bicycle |
|---|---|---|
| uids / post-init births | 1 616 133 / 1 503 506 | 6 566 757 / 6 512 482 |
| roots | 46 095 | 41 027 |
| dead roots | 13 853 (30.05%) | 8 972 (21.87%) |
| exact I_t total (tile isects) | 92 314 238 919 | 125 673 275 013 |
| weighted estimate (×20) | 92 913 457 440 | 126 391 509 280 |

Headroom (`D0_HEADROOM.json`; F_FB = 13.915/19.369, F_OPT = 4.140/19.369,
T = 30000; bootstrap 2000, seed 20260928, r_N_dead held exact):

| quantity | room | bicycle |
|---|---|---|
| t_star | 30000 | 30000 |
| r_I_dead | 0.018663 | 0.006602 |
| r_N_dead | 0.004898 | 0.000616 |
| R_upper | 0.015506 | 0.005230 |
| R_upper 95% CI | [0.014121, 0.016921] | [0.004636, 0.005868] |
| S_upper | 1.0158 | 1.0053 |
| S_upper at CI-low | 1.0143 | 1.0047 |
| gate_headroom_ge_070 | false | false |
| gate_no_headroom_lt_060 | **true** | **true** |

t_star rationale (`D0_TAIL_QUALITY.json`): the frozen envelope is
PSNR ≥ final − 0.10, SSIM ≥ final − 0.003, LPIPS ≤ final + 0.005.
- room, final (32.2485, 0.9283395, 0.3010942): ckpt@25000 = (32.0127,
  0.9256307, 0.3049983) — PSNR misses the 32.1485 floor by 0.136 dB.
- bicycle, final (26.3694, 0.8360808, 0.2465939): ckpt@25000 = (26.0709,
  0.8284808, 0.2544608) — misses PSNR floor by 0.199 dB, SSIM floor, and
  LPIPS ceiling.
No checkpoint k ∈ {5000, …, 25000} is inside the envelope on either scene →
t_star = 30000 → zero admissible tail truncation.

Sensitivity (init-root partition; reported only, never substituted, spec
4.5): room r_I_dead 0.1980, r_N_dead 0.0163, R_upper 0.1563 (S 1.185);
bicycle r_I_dead 0.0529, r_N_dead 0.0015, R_upper 0.0411 (S 1.043).
`D0_SENSITIVITY_INIT_ROOTS.json` carries the same numbers.

Stage-3 verdict: **D0_NO_3X_HEADROOM.**

## 5. Stage 4 — predictability: not run (spec gate)

The spec runs Stage 4 only when both scenes show headroom. The Stage-3
verdict is D0_NO_3X_HEADROOM, so Stage 4 was not started and
`D0_PREDICTABILITY.json` / `D0_PREDICTABILITY_ROWS.json` are intentionally
absent (recorded as missing artifacts). Substantively the question is moot:
with an optimistic ceiling of ≈1.01–1.02× there is no cumulative work left
for a pre-birth predictor to remove, and the canonical verdict precedence
checked the headroom gate before predictability in any case.

## 6. Stage 5 — final-state descriptive audit

`D0_FINAL_CONTRIBUTION_AUDIT.json` (+ per-scene parquets). All structural
gates passed on both scenes: rows match final N; audit row uids = alive uid
set; audit covers all surviving lineages; visibility mask popcount = count;
per-camera tiles vs isects max |delta| = 0.
- room: 32 242 surviving lineages (46 095 − 13 853); 2 328 with zero
  recorded footprint; top 1% carry 45.4% of the alpha-footprint proxy
  (top 5%: 74.6%, top 10%: 86.2%).
- bicycle: 32 055 surviving lineages; 1 632 zero-footprint; top 1% 50.2%
  (top 5%: 77.5%, top 10%: 87.1%).
- RGB contribution proxy: unavailable (not cheaply exposed by the C0
  renderer) — recorded as unavailable rather than approximated.
- Interpretation guard (verbatim, embedded in the artifact): *descriptive
  only: low recorded contribution does NOT mean "safe to remove"; no causal
  or selection claim is made by this stage.*

## 7. Verdict and what it does/does not say

`D0_VERDICT.json` = `{"verdict": "D0_NO_3X_HEADROOM", "reason":
"headroom_lower_bound_below_0.60"}` produced by the bundle-provided
canonical tool run verbatim over `D0_SCENE_SUMMARY.json` (trace_valid true
both scenes; reconstruction gate ok; CI-low < 0.60 both scenes).

What it says: on the frozen P12 C0 30K trajectory (seed 42), the cumulative
render work attributable to retrospectively-dead lineages — measured
exactly and reconstruction-gated — is ≈1.9% (room) and ≈0.7% (bicycle) of
tile intersections; no admissible tail truncation exists under the frozen
quality envelope. A >3× end-to-end speedup from removing "waste" of this
kind is not physically plausible without lowering the final quality target.

What it does not say: nothing about other acceleration mechanisms, other
seeds, other scenes, or other recipes; nothing about "safe to remove"
lineages; no speedup claim is made from instrumented runtime; this is a
discovery census, not an optimization result.

## 8. Limitations and caveats

- Single seed (42) per scene; two scenes only. The negative result is
  established with pre-registered measurement on these two trajectories.
- Run-to-run nondeterminism is real in the frozen system (within-group
  spread up to ~1.9% in final N over 30K at fixed seed). Trace validity was
  therefore judged by the pre-registered 2.0× spread screen over an
  extended clean-replica envelope (5 clean replicas + P12 reference), with
  literal 1× values reported verbatim in the artifacts.
- Wall-times from traced runs are FUNCTIONAL_ONLY and are excluded from
  timing evidence (`D0_CONTAMINATION.json` records environment provenance:
  GPU UUID ownership, watchdog, per-run rc, snapshots; 30 runs indexed).
- The dead-waste fractions are trajectory-specific measurements, not
  general claims about 3DGS training.

## 9. Artifact map (canonical, in /root/epic05_d0/artifacts/)

D0_SOURCE_IDENTITY.json; D0_TRACE_VALIDITY.json (= 30k); D0_TRACE_VALIDITY_30k.json;
D0_TRACE_VALIDITY_30k_room.json; D0_TRACE_VALIDITY_smoke.json;
D0_TRACE_VALIDITY_smoke_run1_INVALID.json (preserved calibration evidence);
D0_TRACE_VALIDITY_SCREEN_PREREG.json; D0_BICYCLE_FINALN_SCREEN_ESCALATION_PREREG.json;
D0_TRACE_SPREAD_DIAGNOSTIC.json; D0_LINEAGE_SCHEMA.json; D0_BIRTH_TABLE.parquet;
D0_LINEAGE_OUTCOMES.parquet; D0_POPULATION_TRAJECTORY.parquet;
D0_RENDER_COST_SAMPLES.parquet; D0_COST_ACCOUNTING.json; D0_TAIL_QUALITY.json;
D0_HEADROOM.json; D0_SENSITIVITY_INIT_ROOTS.json; D0_FINAL_CONTRIBUTION_AUDIT.json;
D0_SCENE_SUMMARY.json; D0_CONTAMINATION.json; D0_VERDICT.json; D0_REPORT.md;
D0_SOURCE_DIFF.diff; D0_RENDERER_DIFF.diff; d0_source_hashes.txt;
frozen_renderer_sha_before/after.txt; D0_FINAL_MANIFEST.json (+ return_package/).

Raw tensors (stay on EPIC-05; sha256 + paths in D0_FINAL_MANIFEST.json):
`/root/epic05_d0/runs/TRACE_{room,bicycle}/` — d0_uid_table.npz,
d0_birth_records.npz, d0_population_trajectory.npz, d0_sample_summary.npz,
d0_final_audit.npz, d0_trace_meta.json, d0_ckpt_*.pt, d0_cost_samples/
(1500 shards per scene).

D0_PREDICTABILITY.json / D0_PREDICTABILITY_ROWS.json: intentionally absent
(Stage 4 gated off; see section 5).
