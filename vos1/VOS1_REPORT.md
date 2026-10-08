# VOS1 -- Publication Hardening Report

**Verdict: QUALITY_FAIL**  (strong=False)

VOS1 hardens the surviving VOS0 SH-only virtual-optimizer-state mechanism for 3DGS. It does NOT invent a new mechanism; it hardens the evidence.

## 1. What VOS1 changes vs VOS0

- Eliminates the per-step `torch.nonzero` host synchronization via device-side compaction (zero ordinary-step host syncs).
- Uses corrected unseen true-holdout (camera index `i % 8 == 0` excluded from ALL training) for the primary PSNR/SSIM/LPIPS.
- Uses 3 fixed seeds {42, 123, 3407} (9 primary cells).
- Forks A_f / A_s / VOS1 from the SAME 15K prefix (no arm retrains the prefix).
- Optional composition with deterministic DPC (dependency-gated).

## 2. Frozen mechanism scope

- SH DC + SH rest only; active at iterations 15001..30000.
- Topology fixed after the 15K densification window.
- Exact deferred logical-time settlement relative to single-tensor Adam A_s (`foreach=False`), as frozen by the VOS0 exactness oracle.
- No change to forward renderer, loss, camera semantics, densification policy, SH-degree schedule, LR schedule, or other optimizer groups.

## 3. Exactness oracle (VOS1 CUDA compaction path)

- all_pass: **True**
  - vos1_oracle_bicycle_ckpt30k.json: bitwise_vs_A_s=True fp32_equiv=True pass=True
  - vos1_oracle_edge_random.json: bitwise_vs_A_s=True fp32_equiv=True pass=True
  - vos1_oracle_edge_zeromv.json: bitwise_vs_A_s=True fp32_equiv=True pass=True
  - vos1_oracle_garden_ckpt30k.json: bitwise_vs_A_s=True fp32_equiv=True pass=True
  - vos1_oracle_room_ckpt30k.json: bitwise_vs_A_s=True fp32_equiv=True pass=True

## 4. Device-compaction sync audit

- ordinary_step_host_sync_count: **0**
- VOS1 per-step syncs: {'nonzero': 0, 'item': 0, 'synchronize': 0, 'total': 0}
- VOS0 per-step syncs: {'nonzero': 1, 'item': 1, 'synchronize': 0, 'total': 2}
- geomean VOS-stage speedup vs VOS0 nonzero path: **0.890602457360541**

## 5. Corrected true-holdout

- rule: camera index i % 8 == 0; excluded from ALL training incl. prefix
  - room: n_cameras=311 holdout_n=39 N@15k=869412
  - garden: n_cameras=185 holdout_n=24 N@15k=2549604
  - bicycle: n_cameras=194 holdout_n=25 N@15k=3722583

## 6. Primary quality (V - S, semantic target)

- all_9_cells_complete: **None**
- max_scene_abs_median_VS_psnr: None (gate <= 0.10)
- max_abs_VS_psnr: None (gate <= 0.25)
- pooled_abs_median_VS_psnr: None (gate <= 0.05)
- max_scene_abs_median_VS_ssim: None (gate <= 0.002)
- max_scene_abs_median_VS_lpips: None (gate <= 0.004)

Per-scene median V-S PSNR:

## 7. Deployment comparison (V - F)

- min_scene_median_VF_psnr: None (gate >= -0.20)
- min_VF_psnr: None (gate >= -0.35)
- min_scene_median_VF_ssim: None (gate >= -0.004)
- max_scene_median_VF_lpips: None (gate <= 0.008)
- pooled_median_VF_psnr: None (gate >= -0.10)

Per-scene median V-F PSNR:

## 8. Attribution (V-F = (S-F) + residual(V-S))


## 9. Primary speed gates

- min_scene_median_optimizer_speedup_V_vs_S: 1.6069177115195323 (gate >= 1.50)
- n_scenes_full_wall_reduction_ge_5pct_V_vs_F: 2 (gate >= 2)
- worst_scene_median_full_wall_regression_V_vs_F: -0.03557114631509579 (gate <= 0.01)
- geomean_full_wall_speedup_V_vs_F: 1.079047844356716 (gate >= 1.05)
- max_overhead_fraction_of_gross_saving: 0.029819486083687428 (gate <= 0.25)
- nan_inf_count: 0 (gate == 0)

Per-scene optimizer speedup V/S:
  - room: 1.6069177115195323
  - garden: 2.294153221196302
  - bicycle: 2.3641555684771647
Per-scene full-wall reduction V/F:
  - room: 0.03557114631509579
  - garden: 0.07317890169826703
  - bicycle: 0.1095453853594964
Per-scene overhead fraction:
  - room: compaction=0.0355ms gross_saving=1.1899ms frac=0.029819486083687428
  - garden: compaction=0.0931ms gross_saving=4.2505ms frac=0.02191482158303461
  - bicycle: compaction=0.1477ms gross_saving=5.9916ms frac=0.024653216488968416

## 10. Deterministic-composition dependency

- available: False (DPC0_TOO_SLOW)
- composition state: VOS1_DPC_COMPOSITION_BLOCKED (NOT a VOS1 failure)

## 11. Per-cell raw results

| cell | seed | V-PSNR | S-PSNR | F-PSNR | V-S | V-F | opt_spd_V/S | wall_red_V/F |
|---|---|---|---|---|---|---|---|---|
| room_s42 | 42 | 30.402312603954076 | 30.51811820000994 | 30.42178517931495 | -0.11580559605586416 | -0.019472575360872213 | 1.6630006217050215 | 0.03557114631509579 |
| room_s123 | 123 | 30.19859753188545 | 30.299548749872518 | 30.323795360981883 | -0.10095121798706685 | -0.1251978290964324 | 1.5443364386777152 | -0.06511426263761506 |
| room_s3407 | 3407 | 30.252842453234628 | 30.297196165458253 | 30.337352215732928 | -0.04435371222362505 | -0.08450976249829978 | 1.6069177115195323 | 0.0396312971086289 |
| garden_s42 | 42 | 25.912557611838412 | 25.942250563355742 | 25.921254177959252 | -0.029692951517329647 | -0.008696566120839577 | 2.309829937862662 | 0.07727102432871982 |
| garden_s123 | 123 | 26.024225434640062 | 26.043726793959795 | 26.0538879101459 | -0.01950135931973307 | -0.02966247550583745 | 2.294153221196302 | 0.07317890169826703 |
| garden_s3407 | 3407 | 26.08300833045673 | 26.100845951855742 | 26.10873420911949 | -0.017837621399010573 | -0.025725878662758106 | 2.264880137757928 | 0.0724063996351731 |
| bicycle_s42 | 42 | 21.676398589460177 | 21.709304011989747 | 21.705878693915423 | -0.03290542252956996 | -0.029480104455245737 | 2.5941739596473643 | 0.1095453853594964 |
| bicycle_s123 | 123 | 21.631731043975876 | 21.643278482353313 | 21.638196833765356 | -0.011547438377437658 | -0.006465789789480425 | 2.2592242625987184 | 0.045323787905407226 |
| bicycle_s3407 | 3407 | 21.714406865941736 | 21.718312730632803 | 21.72772008593506 | -0.003905864691066796 | -0.01331321999332502 | 2.3641555684771647 | 0.1385909504957355 |

## 12. Provenance

- torch 2.9.1+cu128, CUDA 12.8, NVIDIA A100-SXM4-80GB
- execution: EPIC-05 A100 only; no local GPU
- VOS1 source identity: 7 components hashed
- VOS0 frozen result: KEEP (stage speedups None)

## 13. Allowed attribution language

VOS1 is exact relative to A_s by the in-process replay oracle. Full-training quality variation between independently executed A_s and VOS1 continuations is interpreted against native raster-backward run-to-run nondeterminism. V-F = (S-F) + residual(V-S) with replay-oracle exactness evidence.
