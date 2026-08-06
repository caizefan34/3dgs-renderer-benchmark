# HiGS method paper workspace

## Thesis

Hierarchy-aware differentiable rendering can reduce time-to-quality for 3DGS
training while preserving the final reconstruction target.

This thesis is not yet supported by the existing short-horizon experiments.
The manuscript becomes submission-ready only when the full-training result gate
passes without missing or failed jobs.

## Contribution and evidence map

| Contribution | Main evidence | Paper item | Current state |
| --- | --- | --- | --- |
| Native HiGS backward | Finite differences, gradcheck, native/recompute parity, topology lifecycle | Figure 2 system diagram; Table 1 gradient and kernel validation | Implementation exists; freeze a tracked aggregate artifact |
| Hierarchy-aware training algorithm | Final frozen pseudocode and component ablation | Algorithm 1; Figure 3 component diagram; Table 2 ablation | Blocked until one final from-scratch trainer replaces the R60-R65 recipe history |
| Faster converged training | Full wall-clock and quality curves | Figure 4 time-to-quality; Table 3 complete dataset results | Blocked by 333-job paper protocol |
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

The command builder is also a runner-readiness gate. At present it emits the
official gsplat control and rejects both HiGS methods because native backward
does not yet expose `DefaultStrategy`-compatible densification information.
This is an explicit implementation blocker, not a missing documentation item.

## Writing boundary

The abstract may currently claim a native differentiable implementation and a
released evaluation protocol. It may not claim full-training acceleration,
cross-hardware generalization, or superiority to official training baselines.
Those claims unlock only after the corresponding machine-readable gates pass.
