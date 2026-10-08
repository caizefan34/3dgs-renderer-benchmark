# PAPER0 REPORT - Publication-Readiness Evidence Matrix

**Generated:** 2026-10-02  
**Mode:** read-only synthesis. NO GPU work. NO new runs. NO cohort merging.  
**Source of truth:** per-method *_VERDICT.json / *_REPORT.md / *_SOURCE_LOCK.json deliverables.  
**Scope:** TFC, DFC, K50-B1, VOS, IGF, CGC, KFC, AGS (+ predecessor oracles).

## 1. Role summary

| Role | Methods |
|---|---|
| MAIN | TFC, K50-B1 |
| SUPPORTING | DFC (DFC1-R1 saved-T backward fix sub-result) |
| DROP | DFC (DFC1-R2 adaptive), VOS (VOS1 hardened), IGF (IGF1), CGC (CGC1), KFC (KFC1), AGS (AGS1) |
| ANALYSIS (preserved negatives) | CGC0 oracle, IGF0 oracle, VOS0 SH-only, AGS0 oracle, KFC0 |

## 2. Method ledger (one row per method; cohorts NOT merged)

| Method | Role | Real CUDA | Scenes | Cams | Full-conv | Workload red. | Fwd | Fwd+Bwd | Full-train-step | Total-train-time | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TFC | MAIN | yes | room,garden,bicycle,bonsai,counter,kitchen,stump,flowers,treehill,truck,train | 88 | NO | 38.2% pooled tile-intersection reduction (per-scene 28.8-41.9%) | intersect kernel 1.38x faster; per-scene fwd render 1.15-1.42x | E2E (fwd+bwd) 1.17-1.51x per scene; measured e2e 1.3564x | NOT_MEASURED | NOT_MEASURED | TFC0_TILE_FOOTPRINT_TIGHTENING_CERTIFIED -> TFC2_CERTIFIED |
| K50-B1 | MAIN | yes | room,garden,bicycle | {"room": "311 (MipNeRF360 room 1920x1080)", "garden": "MipNeRF360 garden", "bicycle": "MipNeRF360 bicycle"} | YES_30K | Gaussian count reduced ~30% (gs_ratio room 0.698, bicycle 0.694, garden 0.643); clone/split ratios <1 (fewer densification events). | NOT_APPLICABLE | MICROBENCH_ONLY | SUPPORTED_30K | IMPLIED_POSITIVE | C51-R simulation oracle -> Stage 4A/4B real CUDA |
| DFC | DROP | yes | room,garden,bicycle | 8 cameras/scene (24 total); resolution room 1557x1037, garden 2593x1680, bicycle 2473x1643 (down=2). | PARTIAL_RENDERER_STEP | splat reduction room 0.2877, garden 0.2072, bicycle 0.1706 (best quality-preserving policy). | SUPPORTED | SUPPORTED | ROOM_ONLY | NOT_MEASURED | DFC0_PASS_EARLY_TERMINATION_HEADROOM -> DFC1_R1 backward fix ADOPT -> DFC1_R2_ADAPTIVE FAIL_CLOSED |
| VOS | DROP | no | room,garden,bicycle | room/garden/bicycle @1920x1080 (MipNeRF360). | YES_30K_MULTISEED | SUPPORTED_LOGICAL_ONLY | NOT_APPLICABLE | NOT_APPLICABLE | NONE | NONE | QUALITY_FAIL (device-side compaction, corrected unseen true-holdout eval, 3 seeds, same-15K-prefix fork) |
| IGF | DROP | no | room,garden,bicycle | room/garden/bicycle (MipNeRF360). | YES_30K | SUPPORTED_BUT_INSUFFICIENT | NOT_APPLICABLE | NOT_APPLICABLE | NEGATIVE | NEGATIVE | DROP_NO_IMPORTANCE_FREEZE_HEADROOM (real training) |
| CGC | DROP | yes | room,garden,bicycle | 8 eval cameras/scene; room 1,593,376 G (58.15M candidates), garden 5,834,784 G (89.34M), bicycle 6,131,954 G (63.85M). | NO | SUPPORTED_BUT_INSUFFICIENT | PARTIAL | PARTIAL_BELOW_GATE | NOT_MEASURED | NOT_MEASURED | DROP_PRUNING_FIDELITY (real CUDA kernel) |
| KFC | DROP | yes | room | 1 workload (room_iter10000 single camera microbenchmark). | NO | ANALYTICAL_ONLY | FAIL | FAIL | NOT_RUN | NOT_MEASURED | DROP_NO_KERNEL_GAIN (real CUDA) |
| AGS | DROP | yes | room | 12 consecutive COLMAP room cameras (sort I/O captured); garden/bicycle frozen step30000 checkpoints MISSING on EPIC-05 (only room m3_sh3 exists). | NO | FAIL | FAIL | NOT_REACHED | NOT_REACHED | NOT_MEASURED | DROP_SORT_KERNEL_GAIN (real CUDA) |

## 3. Key findings

- K50-B1 is the ONLY method with a real full-training-step speedup at 30K: room 1.082x (+8.2%), bicycle 1.244x (+24.4%), garden 1.289x (+28.9%), PSNR neutral-to-positive (Stage 4B clean cohort). Gate A (cosine>=0.999, skipped=0) passes; forward bit-exact.
- TFC is the strongest renderer-level result: ellipse-exact footprint removes 38.2% of tile intersections across 11 scenes / 88 cameras, pixel-identical (max dPSNR 2.59e-07 dB), e2e renderer speedup 1.356x. Renderer-local only (frozen PLY) - NOT a training speedup.
- DFC1-R2 adaptive termination is FAIL_CLOSED: >=1.2x E2E gate met only on sparsest room (1.206x, 4/8 cams); garden 1.140x, bicycle 1.061x fail (O(N) projection dominates dense-scene budget). DFC1-R1 saved-T backward fix is a real SUPPORTING correctness contribution (548x grad-error reduction at t=0) but a no-op at early-termination thresholds. A later DFC2_GENERALIZATION amendment (gate changed to quality-only, 10% workload floor removed) reaches UNIVERSAL_QUALITY_PASS (dPSNR<=0.05dB on all six held-out scenes at t=1e-07/1e-06) but yields no usable speedup (min fwd-pipe ~1.0007x, splat reduction ~6-8%, production_full_training_step=NOT_MEASURED); the DROP role stands. Historical pre-amendment DFC2 verdict (FAIL_CLOSED) preserved in source.
- VOS (VOS1 hardened) is QUALITY_FAIL with no wall-clock benefit: ~40% logical update reduction but Amdahl-bound (optimizer ~15% of wall, SparseGaussianAdam O(N)); room seed-0 -0.271 dB instability. VOS0 SH-only KEEP is a superseded-generation result.
- IGF / CGC / KFC / AGS are all DROP with real CUDA but failing economic or fidelity gates: IGF (no kernel, slower end-to-end, quality fails), CGC (fidelity gates fail at all thresholds, rasterize overhead-bound), KFC (fused kernel 0.98x + cannot launch on real workload, shared-mem overflow), AGS (0% tile reuse, 2.18x slower than radix).
- Historical failed/superseded gates are preserved (KFC0, AGS0, CGC0, IGF0, DFC0, VOS0, TFC0) and must not be silently dropped.

## 4. Cohort integrity

- K50-B1 stage4a (room-only, A100-PCIE-40GB, CUDA 11.8, config-buggy 5K) and stage4b (3-scene, A100-80GB, clean 30K) are SEPARATE cohorts - never merged.
- Renderer-frozen-PLY cohorts (TFC/DFC/CGC/KFC/AGS) are NOT full-training cohorts; their speedups MUST NOT be presented as total-training-time speedups.
- EPIC-05 A100-SXM4-80GB (remote) != A100-PCIE-40GB (stage4a).

## 5. Top open gaps

- [BLOCKER] G1] Only K50-B1 has a real full-training-step speedup; TFC/DFC/CGC/KFC/AGS are renderer-local (frozen PLY) and MUST NOT be presented as training speedups.
- [HIGH] G2] No method has a measured TOTAL-training-time speedup end-to-end (only per-iter / renderer-step). K50-B1 implies it but no separate total-time field.
- [HIGH] G3] K50-B1: single seed, no multi-seed CI; LPIPS not measured; stage4a config-buggy 5K cohort.
- [HIGH] G4] DFC: >=1.2x E2E gate fails on dense scenes (garden 1.140x, bicycle 1.061x); O(N) projection dominates 39-50% of budget; no transmittance headroom. Room pass marginal (4/8 cams).
- [HIGH] G5] VOS: Amdahl-bound (optimizer ~15% of wall, SparseGaussianAdam O(N)); no wall-clock speedup; room seed-0 instability (-0.271 dB).
- [HIGH] G6] CGC: fidelity gates (SSIM/grad-cos/LPIPS) fail at every threshold on every scene; rasterize kernel overhead-bound; no trained ckpt for garden/bicycle.
- [HIGH] G7] KFC: fused kernel cannot launch on real workload (804KB shared >> 164KB A100 max); bwd compute-bound; autograd boundary blocks E2E; KFC0 DRAM claim disproven.

See PAPER0_OPEN_GAPS.json for the full gap register (13 gaps).

## 6. Artifacts

- PAPER0_EVIDENCE_MATRIX.json - authoritative per-method ledger (17 fields each).
- PAPER0_COHORT_MAP.json - cohort definitions + cross-cohort prohibitions.
- PAPER0_OPEN_GAPS.json - 13-item gap register.
- PAPER0_MAIN_TABLE_DRAFT.json - draft main comparison table (cohort-tagged).
- PAPER0_REPORT.md - this report.
- MANIFEST.sha256 - SHA256 of all artifacts above.
