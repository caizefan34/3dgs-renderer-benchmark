# R5-A0: R4 Seed/Config Audit

## Question

> Are the R4 train/truck paired results (seed=0) reusable as one of the 3 required paired seeds for R5-A?

## Audit Method

Inspected `provenance.json` and `config.json` for all 4 R4 runs:
- train/baseline
- train/candidate_c
- truck/baseline
- truck/candidate_c

Compared: seed, camera_seed, git_commit, git_dirty, git_diff_sha256, training_script_sha256, config_sha256, gsplat_version, torch_version, gpu_name, initial_gaussian_count, camera_sequence.npy hash.

## Findings

### Seed
All 4 runs: **seed=0**, **camera_seed=0**.

### Git state
All 4 runs: git_commit=`32ab80e773f74f4d8e40ff4e338c86b29c7957f5`, git_dirty=True, git_diff_sha256=`701aa817d6147bd87bb63b3715f22a16799a00419b127baf0b6e4cb87fca9ace`.

Identical dirty state — same uncommitted changes were present for all runs.

### Script/config hashes
- training_script_sha256: `ca4f9744daa0edb14d94d43012f6d509b4db3b87d960a081bade3e49f1f452f3` (all 4 runs)
- config_sha256: `b440df61f49b00af5a29cc266147b7fdeb706c0386bb6ede798c1fb10998dce4` (all 4 runs)

### Software environment
- gsplat: 1.5.3+pt24cu124
- torch: 2.4.1+cu124
- CUDA runtime: 12.4
- GPU: NVIDIA A100-PCIE-40GB

### Camera sequence
- train/baseline and train/candidate_c: identical hash (-175044028655383977)
- truck/baseline and truck/candidate_c: identical hash (-6119879248611936296)

### Initial Gaussian count
- train: 182686 (both baseline and candidate)
- truck: 136029 (both baseline and candidate)

### Config parameters (all 4 runs identical)
- iterations: 30000
- sh_degree: 3
- resolution: 1080p
- densify_grad_threshold: 0.0008
- densify_from_iter: 500
- densify_until_iter: 15000
- densification_interval: 100
- opacity_reset_interval: 3000
- lambda_dssim: 0.2
- tile_size: 16
- packed: false
- checkpoint_iterations: [30000]
- eval_iterations: [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000]

## Verdict

**ORIGINAL_R4_PAIR_REUSABLE = YES**

### Reason
1. Seed is explicitly 0 for all runs ✅
2. Baseline and candidate used the same seed (0) ✅
3. Same git commit, same dirty diff, same script/config hashes ✅
4. Same gsplat/torch/CUDA versions ✅
5. Same GPU model (A100-PCIE-40GB) ✅
6. Camera sequences are byte-identical between baseline and candidate ✅
7. Initial Gaussian counts match ✅

The R4 seed=0 paired results can serve as one of the 3 required paired seeds.

### R4 seed=0 results (for reference)
| Scene | B_PSNR | C_PSNR | ΔPSNR | B_SSIM | C_SSIM | ΔSSIM | B_N | C_N |
|-------|--------|--------|-------|--------|--------|-------|-----|-----|
| train | 21.86 | 22.34 | +0.48 | 0.8313 | 0.8166 | -0.0146 | 419478 | 472289 |
| truck | 22.33 | 24.46 | +2.13 | 0.8520 | 0.8506 | -0.0014 | 1074296 | 1185928 |

## R5-A seed plan

- **Seed 0**: Reuse R4 paired results (verified above)
- **Seed 1**: New paired runs (B + C for train, B + C for truck)
- **Seed 2**: New paired runs (B + C for train, B + C for truck)

Total new runs: 8 (2 scenes × 2 methods × 2 new seeds)
