# HiGS method paper workspace

## Thesis

Hierarchy-aware differentiable rendering can reduce time-to-quality for 3DGS
training while preserving the final reconstruction target.

The full-training result gate has been executed end-to-end. A frozen 132-job
3-seed confirmatory matrix (4 methods x 11 scenes x 3 seeds, 30k steps, one
A100) completed with zero failed or missing jobs and passes all pre-registered
gates for the frozen candidate `gsplat_30k_fused_prune10_rclip05` (fused
renders + opacity-based pruning every 10th densification + radius clamp 0.05):
PSNR paired-delta 95% CI lower bound -0.022 (>= -0.10 dB), SSIM -0.0011
(>= -0.003), LPIPS upper bound +0.0025 (<= +0.005), wall-clock speedup ratio
mean 1.164x (>= 1.111x) with CI lower bound 1.034 (> 1.0), and time-to-quality
faster (CI upper bound -19.0 s). The speedup is not ordinary early-stop:
`gsplat_25k` fails the quality gates, and the visibility-masked
`higs_visible_only` variant fails quality and speed gates. The earlier frozen
210-job from-scratch A100 matrix (original 3DGS 33, gsplat 48, higs_full 48,
higs_proposed 48, Speedy-Splat 33) supports trainability and a memory
reduction (21.6% lower mean peak GPU memory) and remains the source of those
findings.

## Contribution and evidence map

| Contribution | Main evidence | Paper item | Current state |
| --- | --- | --- | --- |
| Native HiGS backward | Finite differences, gradcheck, native/recompute parity, topology lifecycle | Figure 2 system diagram; Table 1 gradient and kernel validation | Implementation exists; freeze a tracked aggregate artifact |
| Hierarchy-aware training algorithm | Frozen pseudocode and the 210-job from-scratch execution (masked Adam + progressive resolution vs same-backend control) | Algorithm 1; Figure 3 component diagram; Table 2 ablation | Implemented and executed from scratch; ablation table available in `tables/summary.md` |
| Faster converged training | Full wall-clock and quality curves; 132-job 3-seed confirmatory matrix | Figure 4 time-to-quality; Table 3 complete dataset results | Supported: accel15 candidate `gsplat_30k_fused_prune10_rclip05` passes all pre-registered gates (PSNR CI lo -0.022, SSIM CI lo -0.0011, LPIPS CI hi +0.0025, speed ratio mean 1.164x / CI lo 1.034, TTQ faster); `gsplat_25k` early-stop control fails quality gates, so the gain is not ordinary early-stop |
| Generalization and scaling | Gaussian-count, resolution, and hardware cohorts | Figure 5 scaling; Table 4 hardware results | Consumer and second data-center GPU blocked |
| Failure analysis | Low-N and high-N behavior, perceptual-quality limits | Figure 6 failure cases and quality-speed frontier | Short-horizon evidence exists; repeat under full training |

## Required headline tables

1. Complete per-dataset final quality and total training time. Report every
   scene, seed uncertainty, and failure count.
2. Time-to-matched-quality relative to Original 3DGS, official gsplat, full
   HiGS, and official acceleration baselines.
3. Component ablation under the same initialization, seed, and timing boundary.
4. Cross-hardware results as separate cohorts, never pooled into one speedup.

## Commands

```bash
python src/scripts/prepare_higs_paper_source.py --variant official
python src/scripts/prepare_higs_paper_source.py --variant higs

python src/scripts/validate_higs_paper_protocol.py \
  --output-plan artifacts/higs-paper/experiment-plan.json

python src/scripts/build_higs_training_command.py \
  --method gsplat --scene mipnerf360/garden --seed 0 \
  --data-dir /datasets/360_v2/garden \
  --result-dir results/paper/higs/gsplat-garden-s0

python src/scripts/validate_higs_paper_results.py \
  artifacts/higs-paper/results/*.json --require-complete
```

## Writing boundary

The abstract may claim a native differentiable implementation, a released
evaluation protocol, a frozen 210-job from-scratch execution in which the
proposed method reduces mean peak GPU memory relative to gsplat, and a
pre-registered 132-job 3-seed confirmatory matrix in which the frozen accel15
candidate passes all quality-preservation and >=10% wall-clock speedup gates
(1.164x mean, CI lower bound 1.034, TTQ faster). It may not claim
cross-hardware generalization or superiority to official training baselines;
and neither the visibility-masked HiGS mechanism alone (`higs_visible_only`)
nor the 25k early-stop control passes the gates, so any speedup claim must
reference the frozen system-level candidate and its exact gate table.
