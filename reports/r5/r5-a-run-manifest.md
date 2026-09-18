# R5-A Run Manifest

## Experiment Design

- **Scenes**: train, truck (Tanks & Temples)
- **Methods**: B (baseline), C (frozen R4 Candidate C)
- **Seeds**: 0 (reused from R4), 1 (new), 2 (new)
- **Total paired experiments**: 2 scenes × 3 seeds × 2 methods = 12 runs
- **R4 reuse**: 4 runs (seed=0: train B/C, truck B/C) — verified in r5-a0 audit
- **New runs**: 8 (seeds 1 and 2)

## Hardware/Software Cohort

- **Machine**: mx (36.140.146.31:26372)
- **GPU**: 8× NVIDIA A100-PCIE-40GB, driver 595.71.05
- **CUDA runtime**: 12.4
- **Python**: 3.10.19
- **PyTorch**: 2.4.1+cu124
- **gsplat**: 1.5.3+pt24cu124
- **Git commit**: 32ab80e773f74f4d8e40ff4e338c86b29c7957f5 (dirty, diff sha256: 701aa817d6147bd87bb63b3715f22a16799a00419b127baf0b6e4cb87fca9ace)
- **Training script**: baseline/reference_v1/trainer.py (sha256: ca4f9744daa0edb14d94d43012f6d509b4db3b87d960a081bade3e49f1f452f3)
- **Config**: baseline/reference_v1/config.py (sha256: b440df61f49b00af5a29cc266147b7fdeb706c0386bb6ede798c1fb10998dce4)
- **CUDA extension**: experiments/r4/build/ext_skip.so (R4 compiled, frozen)

## Run Schedule

### Wave 1 (4 GPUs, parallel)

| GPU | Scene | Seed | Method sequence |
|-----|-------|------|-----------------|
| 0 | train | 1 | B → C |
| 1 | truck | 1 | B → C |
| 2 | train | 2 | B → C |
| 3 | truck | 2 | B → C |

Each GPU runs baseline first, then candidate_c sequentially.
Total wall time estimate: ~2× single-scene time (B + C sequential per GPU).

### Seed 0 (reused from R4)

| Scene | Method | Path |
|-------|--------|------|
| train | B | /mnt/storage_pool/liaoyuanjun/r4_13scene_v2/train/baseline/ |
| train | C | /mnt/storage_pool/liaoyuanjun/r4_13scene_v2/train/candidate_c/ |
| truck | B | /mnt/storage_pool/liaoyuanjun/r4_13scene_v2/truck/baseline/ |
| truck | C | /mnt/storage_pool/liaoyuanjun/r4_13scene_v2/truck/candidate_c/ |

## Frozen Candidate C Configuration

- **Budget**: 0.05 (5%)
- **Skip mask**: certificate-guided, per (tile, Gaussian) pair
- **CUDA kernel**: rasterize_to_pixels_bwd_with_skip (R4, frozen)
- **Monkey-patch**: _RasterizeToPixels.backward → patched_rasterize_backward
- **No modifications** from R4: same budget, same kernel, same Python wrapper

## Training Configuration (identical for B and C except monkey-patch)

- Iterations: 30000
- SH degree: 3
- Resolution: 1080p
- Loss: 0.2×L1 + 0.8×DSSIM (SepSSIM)
- Densify from: 500, until: 15000, interval: 100
- Densify grad threshold: 0.0008
- Opacity reset: 3000
- Tile size: 16, packed: false, absgrad: true
- Optimizer: Adam, eps=1e-15
- Eval iterations: [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000]
- Checkpoint: [30000] only (to avoid disk issues)

## Output Paths

- New runs: `/mnt/storage_pool/liaoyuanjun/r5_a/{scene}_seed{seed}/{method}/`
- Logs: `/mnt/storage_pool/liaoyuanjun/r5_a_logs/{scene}_seed{seed}_{method}.log`
- Aggregated results: `reports/r5/r5-a-results.json`
