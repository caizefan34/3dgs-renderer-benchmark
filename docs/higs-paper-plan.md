# Differentiable HiGS paper plan

## Central question

Can a hierarchical 3DGS renderer provide correct native gradients and reduce
end-to-end training wall time while preserving full-convergence quality?

This is a separate method paper. Renderer benchmarking and storage compression
are supporting infrastructure, not co-equal contributions.

## Proposed contributions

1. A native CUDA backward for HiGS with explicit blend, projection, and SH VJPs,
   versioned scene state, and safe dynamic-topology updates.
2. A hierarchy-aware training method combining sparse tile work, progressive
   resolution, visibility-aware optimization, and full-resolution quality
   anchors under one ablated design.
3. A multi-scene analysis of time-to-quality, final quality, memory, and scaling
   across Gaussian count, resolution, and GPU architecture.

The second contribution needs a concise algorithm. A collection of 65 tuning
rounds is not itself a method. The paper should present one final algorithm,
pseudocode, complexity, and a component ablation; the chronological research
report remains supplementary material.

## Frozen proposed method (protocol v5, 2026-08-07)

The proposed method is frozen in
[`benchmark/higs-paper-protocol.json`](../benchmark/higs-paper-protocol.json)
under `methods.higs_proposed.algorithm` and routed through the same from-SfM
trainer as the `higs_full` control. One configuration, no tuning after freeze:

- **Renderer**: HiGS dynamic scene with the native CUDA backward
  (`backward_mode="higs_native"`), packed and sparse-grad paths disabled, same
  as the `higs_full` control.
- **Optimizer**: visibility-masked Adam (`visible_adam=True`; gsplat
  `SelectiveAdam` over the per-Gaussian union-visibility mask derived from the
  rendered radii). Culled Gaussians keep both parameters and moments untouched,
  which removes optimizer memory traffic for out-of-view rows.
- **Progressive resolution**: training renders at 0.5x resolution (intrinsics
  and images downsampled jointly) for steps `[0, 24000)`, then switches to full
  resolution for the final `[24000, 30000)` anchor stage. Densification uses
  the half-resolution `width`/`height` normalization that `DefaultStrategy`
  already applies, so gradient thresholds stay resolution-consistent.
  Evaluation and trajectory rendering always use the full dataset resolution.

Everything else (DefaultStrategy, learning rates, SH schedule, loss weights,
seed, 30k budget) is identical to `higs_full`. The two methods therefore differ
only by the visibility-masked optimizer and the progressive-resolution
schedule, which is exactly the ablated design.

Component ablations required by the submission contract:

- `higs_full` (no masked Adam, no progressive resolution) is the same-backend
  control;
- a `higs_full + visible_adam` variant isolates the optimizer contribution;
- a `higs_full + progressive resolution` variant isolates the schedule;
- the switch step (`higs_full_res_step`) and initial scale are sensitivity axes
  after the primary matrix completes.

The tile-sampling API (`tile_sampling_ratio`, `sampling_mode`, `tile_mask`) and
the fused cull-masked Adam kernel in
[`benchmark/higs_masked_adam.py`](../benchmark/higs_masked_adam.py) remain
documented levers from the R60-R65 research log; they are intentionally not part
of the frozen primary method.

## Current evidence boundary

- Native backward correctness is covered by finite differences, `gradcheck`,
  native-versus-recompute comparisons, camera modes, SH degrees, mixed
  precision, and topology lifecycle tests.
- The tracked R60 artifact contains three-seed short-horizon groups for Train,
  Garden, and Bicycle on one A100.
- The R65 research log reports quality-prioritized 3000-step configurations that
  reduce per-step time by 8.6% to 21.7% against the same-backend full-resolution
  control on four high/mid-N scenes, while Train is a stated exception.
- `benchmark/run_higs_train_benchmark.py` loads `point_cloud.ply`; R60-R65 are
  short-horizon optimization of an existing Gaussian scene, not from-scratch
  reconstruction from SfM initialization.
- These are not yet proof of full from-scratch convergence. They must not be
  advertised as a universal 1.8x to 2.5x quality-preserving training speedup.

The executable submission contract is frozen in
[`benchmark/higs-paper-protocol.json`](../benchmark/higs-paper-protocol.json).
It expands to 333 auditable jobs: a complete 11-scene primary matrix and a
five-scene cross-hardware matrix. The protocol validator reports 177
executable A100 jobs: original_3dgs (33), gsplat (48), higs_full (48), and
higs_proposed (48). Speedy-Splat, Turbo-GS, consumer, and second-data-center
cohorts remain explicit blockers.

All four executable methods have fail-closed runners bound to pinned source
trees with SHA-256 locks:

- `gsplat` runs the clean upstream gsplat checkout (77ab983f) through
  `benchmark/run_higs_full_training.py` with the protocol seed and SfM
  initialization.
- `higs_full` and `higs_proposed` run the same trainer against the patched
  HiGS tree whose native CUDA backward exports the screen-space gradients,
  radii, and Gaussian ids consumed by gsplat `DefaultStrategy` densification.
  The two methods differ only by the visibility-masked optimizer and the
  progressive-resolution schedule, exactly the ablated design.
- `original_3dgs` runs the official graphdeco-inria/gaussian-splatting
  trainer (54c035f) plus the audited seed patch
  (`patches/original-3dgs-seed.patch`) through
  `benchmark/run_original_3dgs_training.py`. It trains on the same factor-4
  images and official every-8th-image test split as the gsplat/HiGS cohort,
  and each checkpoint is scored with the official PSNR/SSIM and VGG LPIPS
  implementations (`src/scripts/eval_original_3dgs_checkpoint.py`).

## Running the full-training matrices

Both schedulers are 8-GPU, session-resume, power-sampled drivers that write
protocol-valid result JSONs:

```bash
# 144 jobs: gsplat + higs_full + higs_proposed x 11 scenes x 3 seeds
bash scripts/linux/run_higs_paper_a100_matrix.sh

# 33 jobs: original_3dgs x 11 scenes x 3 seeds (train_original env)
bash scripts/linux/run_original_3dgs_a100_matrix.sh
```

A job that trains successfully but fails assembly is recorded as
`needs_assembly` in the session JSON; re-running the wrapper reassembles it
from the run directory without retraining. Completed runs are never
retrained, and a result JSON is only written after it passes the
`validate_higs_paper_results.py` contract.

## Submission-critical experiment

Use official from-scratch training recipes on complete benchmark datasets, not
only initialized checkpoints or short optimization horizons.

| Axis | Minimum credible design |
| --- | --- |
| Datasets | Full Mip-NeRF 360, Tanks and Temples, and Deep Blending scene sets |
| Repeats | At least three seeds per method and scene |
| Baselines | Original 3DGS, gsplat, full HiGS, official Speedy-Splat and Turbo-GS where applicable, plus random-tile ablation |
| Hardware | A100 plus one consumer GPU and one additional architecture |
| Time | End-to-end wall clock including data, densification, optimizer, evaluation, and synchronization |
| Quality | Final and time-indexed PSNR, SSIM, LPIPS; render consistency where ordering changes |
| Resources | Peak VRAM, energy or power-integrated cost, final Gaussian count, and artifact size |
| Statistics | Per-scene paired differences, confidence intervals, seed variance, and fixed-suite aggregate ratios |

Primary plots should be quality versus wall-clock curves and a quality-constrained
speed table. Per-step kernel latency is a mechanism result, not the headline.

Validate the protocol and export its job plan with:

```bash
python src/scripts/prepare_higs_paper_source.py --variant official
python src/scripts/prepare_higs_paper_source.py --variant higs

python src/scripts/validate_higs_paper_protocol.py \
  --output-plan artifacts/higs-paper/experiment-plan.json

python src/scripts/build_higs_training_command.py \
  --method gsplat --scene mipnerf360/garden --seed 0 \
  --data-dir /datasets/360_v2/garden \
  --result-dir results/paper/higs/gsplat-garden-s0
```

Completed job JSON files must pass `validate_higs_paper_results.py`. The final
paper gate uses `--require-complete`; failed or missing jobs cannot silently
disappear from an aggregate table.

## Required ablations

- native backward versus standard gsplat backward and explicit recomputation;
- tile sampling ratio and sampling policy;
- progressive-resolution schedule;
- masked Adam and union-decay components;
- full-resolution LPIPS and topology anchors;
- Gaussian-count and resolution scaling;
- failure cases, especially low-N Train and high-N perceptual-quality limits.

## Suitable claim

"Under the frozen full-training protocol, the proposed hierarchy-aware method
reaches the matched quality target in X% less wall time on Y of Z scenes."

Avoid "HiGS trains 2x faster" until the same operation point meets the final
quality criterion across the declared full-training cohort.
