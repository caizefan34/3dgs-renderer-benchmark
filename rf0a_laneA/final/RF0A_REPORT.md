# ReproFast0 Lane A - Official FastGS Reproduction Report

**Lane:** RF0A (ReproFast0 Lane A)  
**Generated (UTC):** 2026-09-30T15:42:40Z  
**Host:** dsw-933622-64cc845c7d-lqswc  
**GPU:** NVIDIA A100-SXM4-80GB, 81920 MiB, 570.133.20  
**Verdict:** REPRODUCED  

## 1. Primary Scientific Question

> Does the published fast-training regime reproduce on EPIC-05 under corrected held-out cameras?

**Answer: REPRODUCED.** 
Official FastGS (train_base.sh per-scene Mip-NeRF360 settings) at 30k iters, seed 42, on A100-SXM4-80GB, under the corrected i%%8==0 true-holdout split, trains in 170-236s wall and reaches holdout PSNR 25.2-32.1 / SSIM 0.74-0.92 / LPIPS 0.11-0.27, consistent with published FastGS/3DGS-quality on Mip-NeRF360. It is 2.6-4.6x faster (wall) and uses ~7-8x fewer Gaussians and less VRAM than the faithful same-backend vanilla 3DGS comparator, at comparable true-holdout quality.

## 2. Source Identity (frozen)

| Field | Value |
|---|---|
| Remote URL | https://github.com/fastgs/FastGS.git |
| Branch | main |
| Commit SHA | `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f` |
| git describe | 44e02a5 |
| Working tree (tracked source) clean | yes (only untracked build artifacts: __pycache__, input.ply, cameras.json) |
| Git submodules | NONE - rasterizer/simple-knn/fused-ssim vendored into main tree |
| Vendored tree SHAs | rasterizer `774904ee22ae`, simple-knn `21a25b056d91`, fused-ssim `4bc85a6d4005` |

Built CUDA extension (rasterizer) `_C.so` sha256: `92ea2c4644e8eb8c9252225a841b533ea94f60358facfd21630bfa323e7696c7`

## 3. Environment

| Component | Value |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 (CUDA runtime 12.8) |
| nvcc | Build cuda_12.8.r12.8/compiler.35583870_0 |
| gcc | gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0 |
| GPU | NVIDIA A100-SXM4-80GB, 81920 MiB, 570.133.20 |

Note: the FastGS `environment.yml` pins CUDA 11.6 / py3.7 / torch1.12, which is unavailable on EPIC-05. The vendored rasterizer was rebuilt from the frozen source against torch 2.10.0+cu128 / nvcc 12.8. Source identity is pinned by commit + vendored tree SHAs (section 2), so the algorithm is unchanged; only the build toolchain differs.

## 4. Data & Corrected True-Holdout Split

Rule: original camera index i (COLMAP image order) with i % 8 == 0 => HOLDOUT (test); all others => train. Holdout cameras are NEVER used for training (train.py samples only getTrainCameras()).

| Scene | Total cams | Train | Holdout (test) | cameras_extent radius |
|---|---|---|---|---|
| room | 311 | 272 | 39 | 6.4228 |
| garden | 185 | 161 | 24 | 4.9229 |
| bicycle | 194 | 169 | 25 | 4.9705 |

Holdout cameras (i %% 8 == 0) are rendered for PSNR/SSIM/LPIPS only; training samples exclusively from `getTrainCameras()`. No holdout camera is ever used for optimization.

## 5. Arms & Comparator Choice

- **Arm A (fastgs):** official FastGS, `train_base.sh` per-scene Mip-NeRF360 settings, verbatim `train.py` loop, `mult=0.5`, `optimizer_type=default`, seed 42, 30k iters.
- **Arm B (vanilla):** faithful original INRIA 3DGS on the **same** `diff_gaussian_rasterization_fastgs` backend (same `Scene`/`GaussianModel`), differing only in: original mean-gradient clone/split densification, per-iteration dense Adam, `mult=1.0` (full compact box), no `final_prune_fastgs`. Canonical 3DGS Mip-NeRF360 hyperparameters.

**Comparator justification:** the official `fastgs/FastGS` repo exposes **no** in-repo vanilla baseline (grep for vanilla `render`/`densify_and_prune` is empty; `optimizer_type` only selects `default` vs `sparse_adam`, both FastGS densification). Per the task spec, the faithful vanilla/comparator is the original INRIA 3DGS the repo is forked from (`LICENSE_ORIGINAL.md`), run on the identical rasterizer backend so that wall-time differences reflect the algorithm, not the backend.

## 6. Per-Scene Results

### 6.1 Timing & model size

| Scene | Arm | Wall (s) | Internal (s) | Peak VRAM (GB) | Final Gaussians |
|---|---|---|---|---|---|
| room | fastgs | 170.1 | 164.4 | 5.65 | 196863 |
| room | vanilla | 436.3 | 421.5 | 7.56 | 1527522 |
| garden | fastgs | 235.6 | 225.8 | 4.39 | 693503 |
| garden | vanilla | 1042.8 | 996.7 | 11.23 | 5508770 |
| bicycle | fastgs | 217.4 | 207.7 | 4.88 | 808878 |
| bicycle | vanilla | 997.0 | 949.6 | 11.96 | 5928890 |

### 6.2 Speedup (vanilla / fastgs, same backend)

| Scene | Speedup (wall) | Speedup (internal) | Gaussian reduction | VRAM ratio (van/fast) |
|---|---|---|---|---|
| room | 2.56x | 2.56x | 7.8x fewer | 1.34x |
| garden | 4.43x | 4.41x | 7.9x fewer | 2.56x |
| bicycle | 4.59x | 4.57x | 7.3x fewer | 2.45x |

### 6.3 True-holdout quality (i %% 8 == 0)

| Scene | n | Arm | PSNR | SSIM | LPIPS | Train PSNR (diag) |
|---|---|---|---|---|---|---|
| room | 39 | fastgs | 32.076 | 0.9233 | 0.2121 | 33.035 |
| room | 39 | vanilla | 31.203 | 0.9260 | 0.1931 | 33.435 |
| garden | 24 | fastgs | 27.400 | 0.8445 | 0.1671 | 28.634 |
| garden | 24 | vanilla | 27.563 | 0.8653 | 0.1147 | 29.754 |
| bicycle | 25 | fastgs | 25.226 | 0.7439 | 0.2713 | 25.085 |
| bicycle | 25 | vanilla | 25.385 | 0.7622 | 0.2239 | 26.469 |

### 6.4 Quality delta (FastGS - vanilla, holdout)

| Scene | dPSNR | dSSIM | dLPIPS | dTrainPSNR |
|---|---|---|---|---|
| room | +0.873 | -0.0027 | +0.0190 | -0.400 |
| garden | -0.163 | -0.0208 | +0.0524 | -1.121 |
| bicycle | -0.159 | -0.0182 | +0.0475 | -1.383 |

## 7. Interpretation

- **Speed regime reproduces:** FastGS trains in 170-236 s wall (30k iters) vs vanilla 436-1043 s, a 2.6x-4.6x speedup on the same A100 backend.
- **Model compression:** FastGS uses ~7.3x-7.9x fewer Gaussians than vanilla at convergence.
- **Quality is comparable:** holdout PSNR is within ~0.9 dB of vanilla (above on room, marginally below on garden/bicycle); SSIM/LPIPS slightly lower for FastGS on garden/bicycle. With only 24-39 holdout views per scene, per-scene deltas have limited sample size.
- **Faster == fewer Gaussians == less VRAM**, consistent with the FastGS paper's claims.

## 8. Reproduction Mismatches / Caveats

- Environment differs from the paper's stated spec (paper: CUDA 11.6 / py3.7 / torch1.12; here: torch2.10.0+cu128 / py3.12 / nvcc12.8). The vendored rasterizer was rebuilt from frozen source (commit 44e02a5) against the available toolchain; source identity pinned by commit + vendored tree SHAs.
- True-holdout PSNR: FastGS is +0.87 dB above vanilla on room but -0.16/-0.16 dB below on garden/bicycle; SSIM slightly lower and LPIPS slightly higher for FastGS on garden/bicycle. With only 24-39 holdout views per scene, per-scene metrics have limited sample size; deltas are small and quality is comparable overall.
- Vanilla comparator uses mult=1.0 (full compact box) vs FastGS mult=0.5; this box difference is part of the method, not a confound. Both arms use the identical diff_gaussian_rasterization_fastgs backend.
- Timing is single-GPU wall clock on one A100 per run; no cross-backend wall is compared as algorithmic speedup.

## 9. Exact Commands

All runs used `rf0a_runner.py` (verbatim official FastGS loop for arm A; faithful vanilla 3DGS for arm B) on EPIC-05, seed 42, 30k iters, `--eval` (corrected i%%8==0 holdout). Example (room, fastgs):

```
cd /root/rf0a_laneA/FastGS && export PYTHONPATH=/root/rf0a_laneA/FastGS
CUDA_VISIBLE_DEVICES=0 python /root/rf0a_laneA/rf0a_runner.py --method fastgs \
  -s /root/epic05_p16rb/common_data/mipnerf360/room -i images --eval \
  --mult 0.5 --eval_mult 0.5 --optimizer_type default --iterations 30000 \
  --densification_interval 500 --highfeature_lr 0.02 --grad_abs_thresh 0.0008 \
  --out /root/rf0a_laneA/final/RF0A_raw_room_fastgs.json
```

Vanilla (room): same command with `--method vanilla --mult 1.0 --eval_mult 1.0` and the canonical 3DGS densification hyperparams (`--densification_interval 100 --densify_grad_threshold 0.0002 --percent_dense 0.01 --opacity_lr 0.05 ...`).

## 10. Deliverables

`RF0A_ENVIRONMENT.json`, `RF0A_SOURCE_LOCK.json`, `RF0A_CAMERA_SPLIT.json`, `RF0A_CONFIG.json`, `RF0A_RAW.json`, `RF0A_QUALITY.json`, `RF0A_TIMING.json`, `RF0A_SUMMARY.json`, `RF0A_REPORT.md`, `MANIFEST.sha256`.

Verify: `cd /root/rf0a_laneA/final && sha256sum -c MANIFEST.sha256`

## 11. Scientific Status

External baseline reproduction (NOT author novelty). Lane A only; Lane B/C not started.
