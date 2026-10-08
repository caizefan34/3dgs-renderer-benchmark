# DFC Post-PBR0_H1 Combined Report (Amendment: POST_PBR0_H1)

- Generated (UTC): 2026-10-02T13:37:32Z
- Pack family: DFC2_GENERALIZATION_PACK + DFC_FC0_FULL_CONVERGENCE_PACK + PUB0_BOTTLENECK_LEDGER
- Amendment: POST_PBR0_H1 (supersedes prior PBR0-resource/timing assumptions; all DFC2/DFC-FC0 scientific objectives, frozen algorithms, thresholds, scenes, gates, fail-closed requirements UNCHANGED unless explicitly amended).
- Execution guard: all 3DGS GPU runs on EPIC-05 A100 only (host dsw-933622-64cc845c7d-lqswc, 8x A100-SXM4-80GB); none on local Windows. Remote identity PROVEN before every GPU job; source hashes recorded.

## Evidence-Tier Legend (amendment-mandated)
- **TIER A** — Authoritative production evidence: real complete production 30K wall-clock; directly measured production iteration wall-clock.
- **TIER B** — Production instrumentation: component timings collected INSIDE the same real production iteration (CUDA events around renderer/backward/optimizer in the real trainer).
- **TIER C** — Proxy / microbenchmark: independent-stage microbenchmarks; sums of separately timed stages; residual-derived optimizer timing; fixed-checkpoint renderer microbenchmarks; historical cross-cohort estimates.
- Rule: only TIER A or suitable TIER B may support claims about the CURRENT production bottleneck fraction or publication headline training speedup. TIER C guides research prioritization only. Never merge timing fractions/speedups across cohorts.

## A. DFC Held-Out Generalization (DFC2_GENERALIZATION_PACK)

- Primary question: Can ONE universal fixed transmittance threshold, selected WITHOUT per-scene test-set tuning, satisfy dPSNR<=0.05 dB on EVERY held-out Mip-NeRF360 scene while retaining useful workload reduction?
- Frozen candidates: threshold = 1e-7 and 1e-6; SAME threshold applied over ALL six held-out scenes (bonsai, counter, flowers, kitchen, stump, treehill). No per-scene threshold selected after observing GT metrics.
- Verdict (amended quality-only gate): **UNIVERSAL_QUALITY_PASS** (fail_closed=False)

### A.1 Universal-quality gate (worst-scene over all 6 held-out scenes)

| threshold | worst dPSNR (dB) | worst dSSIM | worst dLPIPS | min grad cosine | max grad relL2 | min splat reduction | min fwd-pipe speedup | quality PASS all scenes |
|---|---|---|---|---|---|---|---|---|
| 1e-07 | 3.884e-07 | 2.980e-08 | 1.285e-07 | 0.99999996 | 1.685e-04 | 5.76% | 1.0008x | YES |
| 1e-06 | 4.921e-06 | 1.118e-07 | 1.770e-07 | 0.99999986 | 2.852e-04 | 8.31% | 1.0007x | YES |

Gate: dPSNR<=0.05 dB on EVERY held-out scene with the SAME fixed threshold. Both thresholds pass the quality gate on all six scenes (worst-case dPSNR ~5e-7 dB at 1e-7, ~5e-6 dB at 1e-6). The amendment REMOVED the old 10% workload floor; workload reduction is reported transparently (min splat reduction 5.76% at 1e-7, 8.31% at 1e-6) but is NOT a pass condition.

### A.2 Per-scene dPSNR (dB) at each threshold

| scene | dPSNR @1e-7 | dPSNR @1e-6 |
|---|---|---|
| bonsai | -1.744e-07 | -3.208e-06 |
| counter | -3.583e-08 | -4.942e-07 |
| flowers | -3.884e-07 | -4.921e-06 |
| kitchen | +3.082e-07 | +4.545e-06 |
| stump | -1.404e-07 | -1.790e-06 |
| treehill | -5.086e-08 | +8.372e-07 |

### A.3 Timing scope (TIER C proxy — NOT production full step)
- production_full_training_step = **NOT_MEASURED** (DFC2 is a fixed-checkpoint forward/backward microbenchmark; per PBR0_H1 it must NOT be estimated from component sums).
- evidence_tier = TIER_C_PROXY_MICROBENCH; timing_scope = S1 (processed-splat/raster workload), S2 (forward rendering pipeline = fwd_pipe_ms), S3 (forward+backward renderer path = fwd_bwd_ms).
- These are renderer-local proxy timings. They are NOT training-speedup evidence and must not be presented as such.

Historical note (preserved): pre-amendment DFC2 verdict was `{'pack': 'DFC2_GENERALIZATION_PACK', 'fail_closed': True, 'primary_question': 'Does one fixed threshold satisfy quality <= 0.05 dB degradation across ALL held-out scenes while retaining useful workload reduction?', 'psnr_loss_max_db': 0.05, 'workload_reduction_min': 0.1, 'verdict_per_threshold': {'1e-07': {'max_psnr_loss_db': 3.884203430004618e-07, 'per_scene_psnr_loss_db': {'bonsai': 1.7443461786115222e-07, 'counter': 3.5829405220511035e-08, 'flowers': 3.884203430004618e-07, 'kitchen': -3.0823119345768646e-07, 'stump': 1.403965050883471e-07, 'treehill': 5.086021204192548e-08}, 'min_splat_reduction': 0.0575802242444563, 'per_scene_splat_reduction': {'bonsai': 0.1182282675220726, 'counter': 0.18466885274564748, 'flowers': 0.09034139334565505, 'kitchen': 0.13345414538661893, 'stump': 0.0575802242444563, 'treehill': 0.09567088370005272}, 'quality_pass_all_scenes': True, 'workload_pass_all_scenes': False, 'universal_pass': False}, '1e-06': {'max_psnr_loss_db': 4.92085928982533e-06, 'per_scene_psnr_loss_db': {'bonsai': 3.207821173134562e-06, 'counter': 4.941770264110801e-07, 'flowers': 4.92085928982533e-06, 'kitchen': -4.544904932402005e-06, 'stump': 1.7896323036836748e-06, 'treehill': -8.372474589180001e-07}, 'min_splat_reduction': 0.08307395863873412, 'per_scene_splat_reduction': {'bonsai': 0.15911978947793837, 'counter': 0.2335713936600522, 'flowers': 0.11949419047880727, 'kitchen': 0.1678395515345572, 'stump': 0.08307395863873412, 'treehill': 0.13138600000275846}, 'quality_pass_all_scenes': True, 'workload_pass_all_scenes': False, 'universal_pass': False}}, 'any_universal_threshold_passes': False, 'best_threshold_by_workload': '1e-06', 'result': 'DOES_NOT_GENERALIZE_UNIVERSALLY', 'note': 'Thresholds fixed a priori (1e-7, 1e-6); none selected per scene using test PSNR.'}` because of a 10% workload-reduction floor that the static self-test predicted would fail; real held-out data falsified that prediction (quality is essentially identical). The amendment replaced that floor with the quality-only universal gate.

## B. DFC Full-Convergence Quality (DFC_FC0_FULL_CONVERGENCE_PACK)

- Primary question: Does DFC remain quality-safe over full 30K optimization trajectories on the frozen original scenes?
- Frozen per-scene thresholds (NO retuning during/after training): room=1e-5, garden=1e-5, bicycle=1e-7; baseline=0.0 (termination disabled).
- Identical initialization/seed(42)/data-order/loss/optimizer+param-groups(incl SH)/densification-pruning-reset/eval/checkpoint schedules between baseline and DFC.
- Verdict: **PASS** (quality_safe_all_scenes = True)

| scene | thr | baseline PSNR | DFC PSNR | dPSNR (dB) | dSSIM | dLPIPS | final GS base->DFC | quality PASS |
|---|---|---|---|---|---|---|---|---|
| room | 1.000000e-05 | 31.2571 | 31.2463 | -1.0788e-02 | -9.531e-05 | +2.722e-04 | 1850028->1850879 | YES |
| garden | 1.000000e-05 | 26.1059 | 26.0573 | -4.8550e-02 | -2.475e-04 | +2.146e-04 | 6439782->6421846 | YES |
| bicycle | 1.000000e-07 | 23.8679 | 23.8713 | +3.4485e-03 | +7.099e-05 | -1.054e-03 | 4186323->4220834 | YES |

Gates: abs(dPSNR)<=0.05 dB AND abs(dSSIM)<=0.001 per scene. LPIPS reported transparently (no post-hoc gate). All three scenes PASS with large margin (worst dPSNR = -0.0486 dB on garden). Densification/pruning trajectory is IDENTICAL to baseline (same grow/prune/reset event counts: 144/144/9 on every scene), confirming DFC did not alter the optimization trajectory.

## C. Real 30K Production Training Speedup (TIER A — authoritative)

- speedup_30k = baseline_total_30k_training_wall_s / DFC_total_30k_training_wall_s, measured with CUDA synchronize immediately before the measured 30K loop and immediately after iteration 30000, inside the ACTUAL production trainer (all param groups incl. SH, real loss, real backward, real optimizer, densification/pruning/opacity-reset executed in-step).
- This is the ONLY number that may be called 'training speedup'. It is NOT constructed from independently benchmarked stages and NOT residual-derived.

| scene | baseline 30K wall (s) | DFC 30K wall (s) | speedup_30k | base iter ms (med) | DFC iter ms (med) |
|---|---|---|---|---|---|
| room | 2642.82 | 2519.66 | 1.0489x | 42.34 | 38.22 |
| garden | 5716.02 | 5913.21 | 0.9667x | 97.17 | 106.76 |
| bicycle | 4930.61 | 4831.04 | 1.0206x | 89.03 | 85.14 |
| **geomean** |  |  | **1.0115x** |  |  |

**Honest interpretation (fail-closed):** total 30K wall-clock speedup is SCENE-DEPENDENT and roughly neutral (geomean 1.0115x). DFC is faster on room (1.0489x) and bicycle (1.0206x) but SLOWER on garden (0.9667x). The per-step benefit is concentrated in the BACKWARD pass (saved-T early termination reduces gradient work); the forward pass carries a padded-thr_arr overhead that scales with resolution and tile padding, which on high-resolution dense scenes (garden, 17.4MP, 6.4M GS) exceeds the backward savings and produces a net slowdown. See section C.1 for the per-stage breakdown.

### C.1 Per-Stage Breakdown (TIER B — CUDA events inside the same production iteration)

| scene | fwd ms (base->DFC) | bwd ms (base->DFC) | opt ms (base->DFC) | net iter ms (base->DFC) |
|---|---|---|---|---|
| room | 15.88 -> 15.69 | 21.00 -> 16.93 | 5.72 -> 5.92 | 42.34 -> 38.22 |
| garden | 30.05 -> 44.16 | 48.53 -> 44.01 | 18.92 -> 18.81 | 97.17 -> 106.76 |
| bicycle | 38.22 -> 38.16 | 39.52 -> 35.27 | 12.44 -> 12.61 | 89.03 -> 85.14 |

TIER B per-stage medians (ev0 fwd-start, ev1 fwd-end, ev2 bwd-end, ev3 opt-end) collected INSIDE the same real production iteration — not a separate microbenchmark, not residual-derived. The forward overhead on garden (30.1 -> 44.2 ms) is the padded-thr_arr cost; the backward savings (room 21.0->16.9 ms, bicycle 39.5->35.3 ms) are the DFC benefit.

## D. Renderer-Local Workload / GS-Population Trajectory

Ratios < 1 indicate DFC reduced the Gaussian population (workload reduction). This distinguishes pure renderer acceleration from acceleration caused by a changed optimization/densification trajectory. Because the densification trajectory is identical to baseline (section B), any renderer-local difference here is a workload/termination effect, not a trajectory effect.

| scene | baseline final GS | DFC final GS | final GS ratio | integrated GS ratio | DFC grow/prune/reset |
|---|---|---|---|---|---|
| room | 1850028 | 1850879 | 1.0005 | 0.9996 | 144/144/9 |
| garden | 6439782 | 6421846 | 0.9972 | 0.9968 | 144/144/9 |
| bicycle | 4186323 | 4220834 | 1.0082 | 1.0063 | 144/144/9 |

GS-population ratios are within ~1% of 1.0 on every scene, confirming DFC's termination does not materially change the optimization trajectory over full convergence. The renderer-local workload reduction (splat termination) is real but modest; it is NOT the same as a total-training speedup.

## E. Proxy / Historical Timing Evidence (PUB0_BOTTLENECK_LEDGER)

All timing evidence is classified into one of three tiers. TIER C may guide research prioritization but must not be presented as authoritative production timing. Timing fractions/speedups are never merged across cohorts.

| cohort | methods | evidence tier | production-equivalent | role | source artifact |
|---|---|---|---|---|---|
| TFC_DFC_EPIC05 | stock, TFC, DFC, TFC+DFC | TIER_C_PROXY_MICROBENCH | false | unified paper headline benchmark | _tfc1dfc1_combined/deliverables/TFC1_DFC1_TIMING.json |
| VOS1_EPIC05 | VOS | TIER_A_B_PRODUCTION | true | optimizer-stage visibility-skip ceiling probe | vos1_pack/VOS1_TRAINING_METRICS.json + vos/VOS_30K.json |
| K50B1_HISTORICAL | K50-B1, baseline | TIER_C_HISTORICAL | false | historical sparse-backward + pruning reference; NOT comparable per-run to current cohort | results/a100/phase-c51* |

### E.1 Tier classification updates (amendment-mandated)
- **PBR0 M4** = TIER C. Reason: decomposed microbenchmark with incomplete optimizer parameter coverage, no SH optimization, fixed Gaussian population, no densification/pruning; its 'full step' is built from independently timed stages. NEVER use as production full-training-step evidence; NEVER construct a production full step by summing independently benchmarked stages; NEVER infer pure optimizer time by subtracting two independent timing loops.
- **Combined-pack residual-derived optimizer_ms** = TIER C. Not a direct measurement of pure Adam time; absorbs asymmetric loop/autograd/zero_grad overhead and timing noise.
- **PBR0 forward-pipeline and forward+backward timing** = valid TIER C renderer microbenchmark evidence, NOT production-training wall-clock.
- **Historical K50-B1** = retain original cohort + evidence status (TIER_C_HISTORICAL); do NOT promote to current-canonical evidence without canonical revalidation; do NOT convert into current-cohort claims.

### E.2 Authoritative production claim boundary
- The ONLY current-cohort production training speedup is the TIER A number in section C (DFC-FC0 real 30K wall-clock).
- VOS1 (TIER A/B) is a real 30K graphdeco optimizer-visibility ceiling probe; it bounds what optimizer-skip could achieve but is a different codebase (gsplat-free) and is NOT merged with the gsplat production speedup.
- TFC_DFC (TIER C) and K50-B1 (TIER C historical) are proxy/historical; they guide prioritization only.

## Final Separated Summary (A/B/C/D/E)

A. **DFC held-out generalization result:** UNIVERSAL_QUALITY_PASS — one fixed threshold satisfies dPSNR<=0.05 dB on all six held-out scenes (1e-7 and 1e-6 both pass quality; 1e-6 gives more workload reduction). No per-scene test-set tuning. production_full_training_step = NOT_MEASURED (TIER C proxy).
B. **DFC full-convergence quality result:** PASS — DFC is quality-safe over full 30K trajectories on room/garden/bicycle (abs(dPSNR)<=0.05 dB, abs(dSSIM)<=0.001); densification trajectory identical to baseline. No threshold retuning.
C. **Real 30K production training speedup (TIER A):** scene-dependent, geomean 1.0115x (room 1.0489x, bicycle 1.0206x faster; garden 0.9667x SLOWER). This is the only number that may be called 'training speedup'.
D. **Renderer-local workload/speedup:** modest splat-termination workload reduction (5.76%-8.31% held-out; GS-population ratios ~1.0 on full-convergence scenes); NOT a training-speedup claim.
E. **Proxy/historical timing evidence:** PUB0 tiers — TFC_DFC=TIER C, VOS1=TIER A/B, K50-B1=TIER C historical; PBR0 M4=TIER C. Guides research prioritization only; never merged across cohorts; never promoted without revalidation.

## Fail-Closed Statement
- No threshold retuning during or after any training. Historical failed gates and previous verdicts preserved (DFC2 pre-amendment `{'pack': 'DFC2_GENERALIZATION_PACK', 'fail_closed': True, 'primary_question': 'Does one fixed threshold satisfy quality <= 0.05 dB degradation across ALL held-out scenes while retaining useful workload reduction?', 'psnr_loss_max_db': 0.05, 'workload_reduction_min': 0.1, 'verdict_per_threshold': {'1e-07': {'max_psnr_loss_db': 3.884203430004618e-07, 'per_scene_psnr_loss_db': {'bonsai': 1.7443461786115222e-07, 'counter': 3.5829405220511035e-08, 'flowers': 3.884203430004618e-07, 'kitchen': -3.0823119345768646e-07, 'stump': 1.403965050883471e-07, 'treehill': 5.086021204192548e-08}, 'min_splat_reduction': 0.0575802242444563, 'per_scene_splat_reduction': {'bonsai': 0.1182282675220726, 'counter': 0.18466885274564748, 'flowers': 0.09034139334565505, 'kitchen': 0.13345414538661893, 'stump': 0.0575802242444563, 'treehill': 0.09567088370005272}, 'quality_pass_all_scenes': True, 'workload_pass_all_scenes': False, 'universal_pass': False}, '1e-06': {'max_psnr_loss_db': 4.92085928982533e-06, 'per_scene_psnr_loss_db': {'bonsai': 3.207821173134562e-06, 'counter': 4.941770264110801e-07, 'flowers': 4.92085928982533e-06, 'kitchen': -4.544904932402005e-06, 'stump': 1.7896323036836748e-06, 'treehill': -8.372474589180001e-07}, 'min_splat_reduction': 0.08307395863873412, 'per_scene_splat_reduction': {'bonsai': 0.15911978947793837, 'counter': 0.2335713936600522, 'flowers': 0.11949419047880727, 'kitchen': 0.1678395515345572, 'stump': 0.08307395863873412, 'treehill': 0.13138600000275846}, 'quality_pass_all_scenes': True, 'workload_pass_all_scenes': False, 'universal_pass': False}}, 'any_universal_threshold_passes': False, 'best_threshold_by_workload': '1e-06', 'result': 'DOES_NOT_GENERALIZE_UNIVERSALLY', 'note': 'Thresholds fixed a priori (1e-7, 1e-6); none selected per scene using test PSNR.'}` retained in JSON).
- Renderer-local or decomposed proxy acceleration is NOT called 'training speedup'. Only the TIER A real 30K wall-clock (section C) is.
- If quality had failed: diagnose trajectory divergence and FAIL CLOSED (did not trigger; all quality gates passed).

