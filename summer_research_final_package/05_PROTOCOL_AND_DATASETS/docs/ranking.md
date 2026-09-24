# Ranking Design

The ranking page selects the best available evidence tier in this order: Measured, Reproduced, Paper Reported.
It never fills missing Tier A cells with Tier B or C values.

## Overall

Full suite coverage only.

| Renderer config | Speed index | FPS | Frame ms | PSNR | SSIM | LPIPS | Max VRAM | Startup | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## Real-time

Sort by normalized speed index for the full suite, or FPS inside one exact case.
Also publish “fastest satisfying the quality gate” because unconstrained FPS can reward degraded output.

## Quality

Publish three independent tables:

- PSNR descending;
- SSIM descending;
- LPIPS ascending.

Do not hide metric disagreement behind an undocumented lexicographic order.

## Efficiency

Sort by the versioned quality/resource score in `docs/methodology.md` and show its components.
The page must label the score as a decision aid.

## Pareto

List non-dominated renderer configurations for FPS–PSNR, FPS–LPIPS, and combined objectives separately.
Show dominated-by evidence and excluded rows.

## Recommendation matrix

| Use case | Selection rule |
| --- | --- |
| Highest FPS | First Tier A real-time row passing the quality gate |
| Highest quality | Winner of the relevant PSNR/SSIM/LPIPS table |
| Low VRAM | Lowest peak NVML process VRAM within a declared quality loss |
| Web viewer | Fastest Tier A config whose verified platform includes WebGPU/WebGL |
| Research | Reference renderer plus adapters with deterministic raw outputs |
| Production | Pareto config meeting platform, license, stability, startup, and memory constraints |

If no row satisfies a rule, the recommendation is “not yet measured,” not a paper-derived guess.
