# Tier A Comparison Analysis

This is the human-readable interpretation of the generated
[Tier A leaderboard](leaderboard/ranking.md). Numeric tables remain generated
from committed `metrics.json` files; this document explains the trade-offs and
does not replace the machine-readable ranking.

## Scope and comparability

The published matrix contains 25 measurements: five renderers across Garden,
Truck, Train, Bicycle, and Bonsai at 1920x1080. Every row uses the same canonical
PLY checkpoint, ordered camera manifest, 100 GT images, protocol hash, A100 GPU
UUID, driver, PyTorch build, and benchmark commit.

The comparison is therefore valid for the repository's `common_representation`
track on the EPIC-05 A100 cohort. It is not a claim about training speed,
retrained/pruned models, mobile/WebGPU deployment, or other GPU generations.

## Overall result

| Renderer | Speed index vs original | Aggregate FPS | PSNR delta | SSIM delta | LPIPS delta | Max VRAM | Practical reading |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gsplat HiGS | 5.671x | 696.91 | -0.285 dB | -0.00312 | +0.00117 | 6,616 MiB | Maximum throughput and best efficiency; small measurable quality loss |
| Speedy-Splat | 2.385x | 293.03 | +0.001 dB | +0.00002 | -0.00000 | 4,276 MiB | Best balanced default; reference-level quality with much lower memory |
| TC-GS | 2.048x | 251.62 | +0.010 dB | -0.00020 | -0.00046 | 4,322 MiB | Strong quality-preserving option; speed is less stable across cases |
| gsplat packed/dense | 1.966x | 241.60 | -0.286 dB | -0.00310 | +0.00157 | 4,206 MiB | Lowest full-suite peak memory and solid speed, with the same quality trade-off as HiGS |
| Original 3DGS | 1.000x | 122.88 | baseline | baseline | baseline | 8,234 MiB | Scientific reference and compatibility baseline |

Positive PSNR/SSIM deltas and negative LPIPS deltas are improvements relative
to Original 3DGS. The tiny differences between Original, Speedy-Splat, and
TC-GS are close enough that application-specific image inspection remains more
important than the aggregate ordering.

## What each renderer is best at

### gsplat HiGS: throughput-first

HiGS is the fastest renderer in every canonical case and reaches 5.671x the
reference speed index. It is also the generated efficiency winner. Its cost is
the largest candidate VRAM footprint and an average quality shift of about
-0.285 dB PSNR and -0.0031 SSIM. Choose it for server rendering, batch
generation, or latency-sensitive workloads where that measured quality delta is
acceptable.

### Speedy-Splat: balanced default

Speedy-Splat is 2.385x faster than Original 3DGS while matching the reference
quality to rounding precision. It cuts the full-suite peak from 8,234 MiB to
4,276 MiB. The generated recommendation selects it for both highest combined
quality utility and best balance. It is the safest general recommendation when
the deployment can use its CUDA path.

### TC-GS: quality-preserving Tensor Core path

TC-GS has the best aggregate PSNR and LPIPS, while SSIM is effectively tied
with the reference group. Its aggregate FPS confidence interval is wider than
the other candidates because performance varies more by scene. It is attractive
when Tensor Core execution and quality preservation matter more than having the
most predictable speedup.

### gsplat packed/dense: memory-first

gsplat has the lowest maximum process VRAM at 4,206 MiB and almost doubles the
reference speed. Its aggregate quality delta is nearly identical to HiGS, so it
is not the quality-preserving choice, but it is the best measured option when
memory is the binding constraint.

### Original 3DGS: reference path

Original 3DGS remains the control implementation. It is the slowest and uses the
most peak VRAM in this cohort, but its role is reproducibility and compatibility,
not winning the optimized ranking.

## Case-level winners

| Case | Fastest | Lowest VRAM | PSNR leader | SSIM leader | LPIPS leader |
| --- | --- | --- | --- | --- | --- |
| Garden | gsplat HiGS, 493.60 FPS | gsplat, 4,206 MiB | TC-GS | Speedy-Splat | TC-GS |
| Truck | gsplat HiGS, 693.14 FPS | gsplat, 2,170 MiB | TC-GS | gsplat | TC-GS |
| Train | gsplat HiGS, 807.90 FPS | Speedy-Splat, 1,482 MiB | TC-GS | Speedy-Splat | TC-GS |
| Bicycle | gsplat HiGS, 556.28 FPS | gsplat, 4,154 MiB | TC-GS | Original/Speedy-Splat | Effectively tied |
| Bonsai | gsplat HiGS, 1,069.15 FPS | gsplat, 1,448 MiB | Original/Speedy-Splat/TC-GS | Original/Speedy-Splat | TC-GS |

The case table should not be read as five independent benchmark cohorts. The
overall recommendation requires complete five-case coverage and uses the
generated aggregate metrics.

## Decision guide

| Requirement | Recommended renderer | Why |
| --- | --- | --- |
| Maximum FPS | gsplat HiGS | Fastest in all five cases; 5.671x reference speed index |
| General CUDA deployment | Speedy-Splat | Best balance of speed, reference-level quality, and memory |
| Lowest peak VRAM | gsplat | Lowest full-suite maximum at 4,206 MiB |
| Best aggregate PSNR/LPIPS | TC-GS | Leads both aggregate metrics, with roughly 2.05x speed index |
| Reference reproduction | Original 3DGS | Official control path and ranking baseline |
| Browser/WebGPU | None yet | No renderer has a verified Tier A web adapter |

## Limitations and next measurements

- Results describe one A100 SM80 cohort; Ada, Blackwell, consumer GPUs, and
  other driver/toolkit combinations need separate cohorts.
- The common-representation track isolates rendering. It does not compare
  training, pruning, compression, or renderer-specific retraining.
- Quality values are averages over the fixed 100-view manifests. Temporal
  consistency, popping, and application-specific perceptual thresholds are not
  yet ranked.
- No WebGPU/WebGL path is verified, so the report makes no browser recommendation.
- A future self-hosted GPU workflow should reproduce the matrix periodically
  without mixing new hardware/software cohorts into this baseline.

For exact confidence intervals, startup timings, per-case values, and source
commits, use [ranking.json](leaderboard/ranking.json) or the adjacent published
result evidence under `results/measured/`.
