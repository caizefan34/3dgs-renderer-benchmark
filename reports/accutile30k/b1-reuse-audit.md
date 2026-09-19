# B1 Reuse Audit — 13-Scene 30K Full-Training Validation

## Decision Summary

`REUSE_B1 = NO` for all 13 scenes. B1 must be re-run in the B1A execution
environment so that wall-clock training-speed comparisons are measured on a
single, homogeneous hardware/software cohort.

## 1. Candidate artifacts audited

Three prior B1-equivalent artifact families were inspected for reuse.

### 1.1 C42 13-scene `reference` runs

Location: `/mnt/storage_pool/liaoyuanjun/c42_13scene_results/{dataset}/{scene}/reference/`

Each leaf contains `training_results.json`, `provenance.json`,
`camera_sequence.npy`, `train.log`.

Provenance (sampled from `mipnerf360/counter/reference/provenance.json`):

| Field | Value |
|-------|-------|
| git_head | `32ab80e773f74f4d8e40ff4e338c86b29c7957f5` |
| git_describe | `baseline/reference-v1-absgrad-3-g32ab80e` |
| git_dirty | true |
| pytorch_version | `2.7.1+cu118` |
| cuda_version | `11.8` |
| gsplat_version | `1.5.3` |
| gpu_name | `NVIDIA A100-PCIE-40GB` |
| config_base | `REFERENCE_V1_ABSGRAD` |
| seed | 42 |
| iterations | 30000 |
| dataset_resolution | `1080p` |
| densify_grad_threshold | 0.0008 |
| absgrad | true |
| grow_grad2d | 0.0008 |
| loss_definition | `(1-lambda)*L1_fullres + lambda*d_ssim_downsampled(scale=1.0), lambda=0.2` |
| ssim_implementation | `SepSSIM window=11 sigma=1.5` |

Coverage: 10/13 scenes
(`counter`, `kitchen`, `bonsai`, `flowers`, `stump`, `treehill`,
`drjohnson`, `playroom`, `truck`, `train`). The three scenes used for the
3-scene AccuTile renderer validation (`room`, `bicycle`, `garden`) are NOT
present in this family.

### 1.2 3-scene AccuTile renderer-validation checkpoints

Location: `/mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts/{bicycle,garden,room}/checkpoints/iter_{5000,15000,30000}.pt`

These are frozen 30K checkpoints used only for renderer-forward/backward
timing (CUDA Events). They were produced to validate AccuTile renderer
correctness/speed, not full-training convergence/timing. No
`training_results.json` (per-checkpoint PSNR/SSIM/LPIPS/timing/N_GS
trajectory) is present.

### 1.3 R4 / R5 candidate-C runs

Explicitly excluded by protocol: "Do not reuse R4/R5 results merely because
the scene name and seed match."

## 2. Reuse criteria evaluation

| Criterion | C42 reference | Reuse? |
|-----------|---------------|--------|
| exact gsplat baseline (1.5.3) | YES | — |
| same training code (REFERENCE_V1_ABSGRAD) | YES | — |
| same dataset preprocessing | YES (1080p, SepSSIM) | — |
| same resolution/downsampling | YES | — |
| same seed (42) | YES | — |
| same camera ordering | YES (same `random.Random(seed)` sequence) | — |
| same optimizer / LR schedule | YES | — |
| same densification config | YES (absgrad, 0.0008) | — |
| same loss | YES (L1 + d_ssim, lambda=0.2, scale=1.0) | — |
| same SH schedule | YES | — |
| same training length (30000) | YES | — |
| same A100 hardware cohort | YES (A100-PCIE-40GB) | — |
| **compatible CUDA/PyTorch environment** | **NO** (`2.7.1+cu118` vs B1A `2.4.1+cu124`) | **FAIL** |
| **no relevant code/env changes affecting timing** | **NO** (different CUDA toolkit + PyTorch build changes kernel timing) | **FAIL** |

### Blocking condition

The B1A execution environment (see §3) is `torch 2.4.1+cu124` (the `anysplat`
conda env), because that is the environment in which the true strip-based
AccuTile gsplat tree is built and validated (`gsplat/csrc.so` present;
`accutile=True` parameter confirmed in `rasterization`). The C42 reference
runs used `torch 2.7.1+cu118`. PyTorch/CUDA-toolkit build differences affect
kernel launch overhead, cuBLAS/cuDNN selection, and CUDA-graph-free timing,
so a wall-clock training-speedup measured across these two environments would
not isolate the AccuTile variable. The protocol requires
`training_speedup = T_B1 / T_B1A` on a single cohort and forbids substituting
renderer-only speedup for full-training speedup.

Therefore B1 cannot be reused from the C42 reference family for the timing
comparison. Quality (PSNR/SSIM/LPIPS) is not environment-sensitive at this
scale, but the protocol demands a paired B1/B1A on the same cohort, and the
primary gate is on full-training speedup — so reuse is rejected wholesale.

## 3. Chosen execution environment (for both B1 and B1A)

| Item | Value |
|------|-------|
| Host | `mx` (bms-39468022-001), 8 × NVIDIA A100-PCIE-40GB |
| Free GPUs | 0,1,2,3,5,6,7 (GPU 4 in use by another user) |
| Driver / CUDA runtime | 595.71.05 / CUDA 13.2 |
| Conda env | `anysplat` |
| PyTorch | `2.4.1+cu124` |
| gsplat (B1) | pip-installed 1.5.3 (AABB tile enumeration, `accutile` param absent) |
| gsplat (B1A) | source tree `/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153` (strip-based AccuTile, `accutile=True` in `rasterization`, `gsplat/csrc.so` built) |
| Scratch output | `/dev/shm/accutile30k` (320 GB free; home & storage_pool ~full) |
| Seed | 42 (matches existing authoritative B1 artifact / config.py) |
| Iterations | 30000 |
| Checkpoints logged | 5K,10K,15K,20K,25K,30K (metrics only; only 30K .pt kept) |

## 4. Per-scene decision

| Scene | Dataset | REUSE_B1 | Reuse source | Reason |
|-------|---------|:--------:|--------------|--------|
| bicycle | Mip-NeRF360 | NO | — | no full-training B1 artifact in target env |
| bonsai | Mip-NeRF360 | NO | — | env mismatch (cu118 vs cu124) |
| counter | Mip-NeRF360 | NO | — | env mismatch (cu118 vs cu124) |
| flowers | Mip-NeRF360 | NO | — | env mismatch (cu118 vs cu124) |
| garden | Mip-NeRF360 | NO | — | no full-training B1 artifact in target env |
| kitchen | Mip-NeRF360 | NO | — | env mismatch (cu118 vs cu124) |
| room | Mip-NeRF360 | NO | — | no full-training B1 artifact in target env |
| stump | Mip-NeRF360 | NO | — | env mismatch (cu118 vs cu124) |
| treehill | Mip-NeRF360 | NO | — | env mismatch (cu118 vs cu124) |
| train | Tanks & Temples | NO | — | env mismatch (cu118 vs cu124) |
| truck | Tanks & Temples | NO | — | env mismatch (cu118 vs cu124) |
| drjohnson | Deep Blending | NO | — | env mismatch (cu118 vs cu124) |
| playroom | Deep Blending | NO | — | env mismatch (cu118 vs cu124) |

`REUSE_B1 = NO` × 13.

## 5. Action

Re-run B1 (clean gsplat 1.5.3, AABB, `accutile=False`) and B1A (true strip-based
AccuTile, `accutile=True`) for all 13 scenes, 30000 iterations, seed 42, in the
`anysplat` (torch 2.4.1+cu124) environment on the same A100 cohort. Pair the
runs per scene on identical seed/camera-sequence/optimizer/densification/SH/loss.
