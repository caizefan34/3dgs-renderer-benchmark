# HiGS method paper workspace

## Thesis

Hierarchy-aware differentiable rendering can reduce time-to-quality for 3DGS
training while preserving the final reconstruction target.

The full-training result gate has been executed end-to-end: a frozen 210-job
A100 matrix (original 3DGS 33, gsplat 48, higs_full 48, higs_proposed 48,
Speedy-Splat 33) with zero failed or missing jobs. The executed evidence supports trainability and a
memory reduction, but not a quality-preserving training speedup: the proposed
method converges from SfM initialization and uses less peak GPU memory, while
final PSNR is lower than the same-backend control on 10 of 11 scenes (tied on
stump). The manuscript is therefore submission-ready for the trainability and
memory contributions; the speed claim remains blocked by the measured quality
gap.

## Contribution and evidence map

| Contribution | Main evidence | Paper item | Current state |
| --- | --- | --- | --- |
| Native HiGS backward | Finite differences, gradcheck, native/recompute parity, topology lifecycle | Figure 2 system diagram; Table 1 gradient and kernel validation | Implementation exists; freeze a tracked aggregate artifact |
| Hierarchy-aware training algorithm | Frozen pseudocode and the 210-job from-scratch execution (masked Adam + progressive resolution vs same-backend control) | Algorithm 1; Figure 3 component diagram; Table 2 ablation | Implemented and executed from scratch; ablation table available in `tables/summary.md` |
| Faster converged training | Full wall-clock and quality curves | Figure 4 time-to-quality; Table 3 complete dataset results | 210-job executed (official Speedy-Splat baseline included); mean wall time lower on 7/11 scenes but aggregate time-to-quality higher (+4.3%) and final PSNR lower on 10/11, so no quality-preserving speedup is claimed |
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
evaluation protocol, and a frozen 210-job from-scratch execution in which the
proposed method reduces mean peak GPU memory relative to gsplat. It may not
claim full-training acceleration, cross-hardware generalization, or
superiority to official training baselines: the executed data shows final
PSNR below the same-backend control on 10 of 11 scenes and aggregate
time-to-quality above the control. Those claims unlock only after the
corresponding machine-readable gates pass.
