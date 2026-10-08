# SB0-FGS Report — Frozen C0 x Faster-GS 2026 Strong-Baseline Uplift

Date: 2026-09-30 (corrected)  |  Host: EPIC-05 (A100-SXM4-80GB)  |  Status: COMPLETE — PROJECT SUCCESS via path B (STRONGER_EXTERNAL_BASELINE)

## 0. Environment & identity (verified on EPIC-05)
- C0 extension sha256 `d26cccf4...` -> `/root/epic05_build_cache/c0_ext/.../experimental_gaussian_render_inference_scene_cuda.so` (MATCHES task pin).
- C0 core sha256 `935ea6c0...` -> `/root/epic05_build_cache/core/gsplat_cuda/gsplat_cuda.so` (MATCHES task pin).
- C0 env: `/root/miniforge3/envs/gsplat/bin/python` (torch 2.9.1+cu128). Faster-GS env separate (torch 2.10.0+cu128).
- Authoritative C0 = **P12_C0**: `publication_trainer_p12.py` (sha256=675b0267) + `--loss fused` (0.8*L1+0.2*(1-SSIM), full-resolution fused-ssim), eps2d 0.3, densify 500..15000 grad0.0008 absgrad, H8_MR, sh3, tile16, seed42, 30K, black bg, 1920x1080 non-uniform stretch.
- Native res: room 3114x2075, garden 5187x3361, bicycle 4946x3286 (aspect ~1.50).

## 1. COMPONENT AUDIT (C0 vs Faster-GS)
See `SB0_COMPONENT_MATRIX.json`. Counts: ALREADY_IN_C0={C2,C4,C5,C6}; DISTINCT_AND_EXACT={C1,C7,C8,C9}; DISTINCT_BUT_APPROXIMATE={C3}; INCOMPATIBLE={}; UNKNOWN={}.
- C7 = backward->optimizer fusion. Port constraints (prohibition-compliant): STRIP `n_touched_tiles==0` skip; STRIP bias-correction, keep C0 no-bias Adam + `valid[]` mask; preserve H8_MR SerialBatch absgrad path. => ALGEBRAIC_EXACT_FP_REASSOCIATED, not bit-exact.
- C8 = morton/Z-order densification (Faster-GS `morton_ordering` @ interval 5000, end 15000; C0 has NONE).

## 2. OPPORTUNITY ORACLE
See `SB0_OPPORTUNITY_ORACLES.json`. Authoritative C0 stage timings (mean over seeds):
- room: 29.441ms (fwd 3.671 / loss 7.323 / bwd 15.088 / opt 2.513); opt=8.5% E2E; bwd+opt=59.8%.
- bicycle: 41.058ms (bwd 18.970 / opt 8.422); opt=20.5% E2E; bwd+opt=66.7%.
- garden: 37.867ms (bwd 18.055 / opt 6.539); opt=17.3% E2E; bwd+opt=64.9%.

## 3. REFERENCE REPRODUCTION (criterion B) — COMPLETE + CORRECTED
See `SB0_FASTERGS_REFERENCE.json`. Faster-GS 30K seed42 reproduced on EPIC-05 (aspect-preserve + 1920x1080 stretch, matched 1080p):
- room:   32.05 / 0.936 / 0.243 | N=1,148,197 | 228.7s | 1621x1080 (aspect-preserve)
- garden: 32.45 / 0.947 / 0.176 | N=1,054,944 | 150.4s | 1667x1080
- bicycle: 32.44 / 0.947 / 0.178 | N=1,097,330 | 155.2s | 1626x1080
These are OUR EPIC-05 measurements, NOT upstream-paper numbers.

### 3a. Prior "C0 anomaly" — RESOLVED (wrong trainer)
The profile's C0 garden 24.04 / bicycle 26.40 came from `/root/3dgs-publication-epic05/full30k/c0_s4{2,3,4}_*`, which used `publication_trainer.py` (sha256=1b67171d, SepSSIM loss, **no** `--loss fused`) — NOT the task-pinned `publication_trainer_p12.py`. The authoritative C0 (P12_C0, `--loss fused`) trains fine: train-view room 32.16 / garden 28.61 / bicycle 26.41. So 24/26 was a wrong-trainer artifact, not a C0 renderer weakness.

### 3b. TRUE unseen-view comparison — DECISIVE (this corrects the prior invalid "MIXED" verdict)
The prior same-protocol comparison used C0 numbers computed on the **all-camera-trained** P12_C0 checkpoint evaluated on every-8th cameras (`metrics_common`, `c0_unseen_eval.py`). Those every-8th cameras were IN C0 training, so they are **train-view subset numbers, NOT unseen-view**. Comparing them to Faster-GS's true held-out (`-s test`) was apples-to-oranges. The "train-test gap ~ 0" observation compared two subsets that were BOTH in training — trivial, and says nothing about generalization.

**Fix:** built `publication_trainer_p12_holdout8.py` (derived from the pinned trainer; 3-line change: `HOLDOUT_STEP=8`, `viewpoint_stack` excludes `i%8==0`, `final_eval` on `heldout_indices`). Renderer/loss/densification/optimizer UNCHANGED; only the training camera pool differs. Verified zero leakage via `camera_sequence.npy` (unique_trained == train_pool: room 272=311-39, garden 161=185-24, bicycle 169=194-25; `heldout_leaked_into_training=[]`).

| Scene | C0 true-holdout PSNR | Faster-GS stretch held-out PSNR | Δ (FGS−C0) | winner |
|---|---|---|---|---|
| room | 30.80 | 31.63 | +0.83 | Faster-GS |
| garden | 26.01 | 26.89 | +0.88 | Faster-GS |
| bicycle | 21.70 | 24.74 | **+3.04** | Faster-GS |

Protocol: 1920x1080 non-uniform stretch, 30K, seed42, black bg, 0.8*L1+0.2*(1-SSIM) fused, TRUE holdout (every-8th excluded from training on BOTH sides), same physical held-out cameras (C0 `cameras.json` `img_name` order == FGS sorted filename order; FGS `i%8==0`→test). Evidence: `SB0_TRUE_HOLDOUT_EVAL.json`, `SB0_FASTERGS_REFERENCE.json.corrected_unseen_view_comparison`.

**C0 generalization gap** (all-camera train-view minus true-holdout): room 1.36 / garden 2.60 / bicycle 4.72 dB — grows with scene complexity. C0's H8_MR renderer overfits more than Faster-GS (bicycle over-densifies to 3.70M gaussians). Faster-GS also trains faster (stretch iso-protocol wall 396/651/579s vs C0 516/816/869s).

**Conclusion:** Under a valid true-unseen-view iso-protocol, **Faster-GS is stronger than frozen C0 on all three scenes** (decisively on bicycle). Criterion B (STRONGER_EXTERNAL_BASELINE) is SUPPORTED.

## 4. COMPONENT EVALUATIONS (no-CUDA, pinned C0 .so read-only)
### 4a. C7 backward->optimizer fusion — DROP
See `SB0_C7_EVALUATION.json`. Oracle upper-bound (optimizer_ms 8.5/20.5/17.3% E2E) assumed fusion eliminates optimizer_ms. EMPIRICALLY FALSIFIED on C0: C0 H8_MR backward (HigsNativeBackward.cu) is MULTI-KERNEL tile-based (blend->projection->sh_vjp) with atomicAdd scatter + data deps; Adam (AdamCUDA.cu, separate core module) must run AFTER all stages => sequential, cannot overlap. Fusion only removes kernel LAUNCH overhead. Measured CUDA launch overhead = 4.80us x5 = 24us/iter = <0.15% E2E. Full optimizer_ms elimination requires per-primitive backward restructure = out-of-scope research, not a port.
### 4b. C8 morton/Z-order densification — DROP
See `SB0_C8_EVALUATION.json`. No-CUDA microbench (c8_microbench3.py, interleaved A/B/A/B inverse-perm restore, gsplat env torch2.9.1, A100). fwd+bwd +1.35%/+0.52%/+3.31% (room/garden/bicycle, N=852K/2.57M/3.70M); per-iter E2E 0.54/0.34/2.08%; full-train E2E 0.27/0.17/1.04%. Idealized ceiling (zero reorder overhead + all-iter application): 0.54/0.34/2.08% < 3% gate on ALL scenes. Benefit confined to backward atomicAdd scatter; C0 backward is TILE-SORTED (not global-order) => weak morton coherence vs Faster-GS fused per-primitive backward. DROP.
### 4c. C1 two-stage sorting — DROP (perfect UB<3% E2E). C9 frontend alloc — DROP (not an E2E lever). C3 opacity filter — PROHIBITED (changes rendering/quality).

## 5. VERDICT — PROJECT SUCCESS (path B)
- **Criterion A (C0-PLUS): NOT_ACHIEVED.** Every DISTINCT_AND_EXACT component DROP on frozen C0 (C1 UB<3%; C7 empirically falsified <0.15% E2E; C8 proven <3% ceiling; C9 not a lever). C2/C4/C5/C6 ALREADY_IN_C0; C3 prohibited. No exact component clears the >=3% E2E implementation gate.
- **Criterion B (STRONGER_EXTERNAL_BASELINE): SUPPORTED.** Under a rigorous true-unseen-view iso-protocol, Faster-GS is stronger than frozen C0 on all three Mip-NeRF360 scenes (room +0.83, garden +0.88, bicycle +3.04 dB) and trains faster. C0's H8_MR renderer overfits more (generalization gap 1.36/2.60/4.72 dB). Faster-GS should be formally adopted into the external baseline matrix for A0/Q0/PSC.
- **This is a PROJECT SUCCESS via path B.** Neither an original research contribution, but a legitimate, rigorous strong-baseline result.
- Caveats: single seed (42) for both; room/garden margins modest (<1 dB) though consistent 3/3; bicycle decisive. A multi-seed true-holdout sweep would tighten the modest margins but is not needed to establish direction (FGS never weaker, decisive on bicycle).
