# Benchmark taxonomy

The framework keeps four result classes separate. A row may be compared only
with rows from the same scene, camera trajectory, resolution, quality
reference, and measurement protocol.

## Synthetic Stress Benchmark

Measures scalability, overlap handling, tile-list growth, memory behavior,
and scheduling efficiency on generated workloads. It has no ground-truth
photographs. Synthetic speed is never evidence of quality preservation and is
never admitted to a GT-quality leaderboard, Pareto frontier, or quality-based
recommendation.

## Real Scene Quality Benchmark

Compares each renderer directly with the same held-out ground-truth images and
reports PSNR, SSIM, and LPIPS. The current official Train paired-reference
audit measures renderer fidelity, but the pretrained archive does not prove
that those images were held out from training. It is therefore presented as
quality verification rather than a held-out reconstruction leaderboard.

## Real Scene Speed Benchmark

Measures trained scenes with identical camera trajectories and output
resolution across renderers. Scene tensors, renderer metadata, commits,
timing protocol, and reproducibility metadata remain part of every result.

## Pareto Analysis

Consumes a compatible real-scene cohort. A renderer is dominated when another
is at least as fast, has no worse PSNR or SSIM, has no worse LPIPS, and is
strictly better in at least one dimension. Missing GT metrics and synthetic
rows are explicitly excluded.

## Scientific inclusion rules

- Every quality value is relative to the same original GT image, not another
  renderer.
- Missing quality remains `null`; it is never interpreted as equivalent.
- Synthetic and real-scene rows are not merged to manufacture a speed-quality
  point.
- Camera, scene, resolution, renderer commit, environment, and GT hashes stay
  attached to published evidence.
- Existing finite-output, camera-change, and quality gates remain mandatory.

See [Evaluation methodology](evaluation_methodology.md) and the existing
[renderer survey inclusion rules](renderer_survey.md#inclusion-rules).
