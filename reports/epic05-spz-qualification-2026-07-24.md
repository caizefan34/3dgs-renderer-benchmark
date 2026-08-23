# EPIC-05 SPZ 8/8 qualification

Date: 2026-07-24  
Hardware: NVIDIA A100-SXM4-80GB on EPIC-05  
Renderer: gsplat packed, source commit `77ab983ffe43420b2131669cb35776b883ca4c3c`  
Codec: NianticLabs SPZ 1.1.0 / format v4, SH precision 8/8 bits

## Decision

SPZ 8/8 passes the repository near-lossless gate on all five canonical cases.
It is the highest-compression common-checkpoint codec measured by this
repository: 5.732x aggregate versus 3.840x for tile-codebook and 2.170x for
block-float. No pruning, training images, or fine-tuning are required.

| Case | PLY bytes | SPZ bytes | Ratio | PSNR delta | SSIM delta | LPIPS delta | Numeric | Visual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Garden | 1,447,027,964 | 259,687,442 | 5.572x | -0.001544 dB | -0.000138 | +0.000364 | pass | pass |
| Truck | 630,225,580 | 108,027,715 | 5.834x | -0.000926 dB | -0.000085 | +0.000443 | pass | pass |
| Train | 254,575,516 | 44,329,417 | 5.743x | -0.007268 dB | -0.000146 | +0.000467 | pass | pass |
| Bicycle | 1,520,726,124 | 263,049,614 | 5.781x | +0.014885 dB | -0.000054 | +0.000533 | pass | pass |
| Bonsai | 308,716,644 | 50,839,636 | 6.072x | -0.014548 dB | -0.000217 | +0.000666 | pass | pass |
| **Aggregate / worst** | **4,161,271,828** | **725,933,824** | **5.732x** | **-0.014548 dB worst** | **-0.000217 worst** | **+0.000666 worst** | **pass** | **pass** |

Thresholds are PSNR drop `< 0.2 dB`, SSIM drop `< 0.002`, and LPIPS increase
`< 0.005`. The measured worst case has more than an order of magnitude margin
on every threshold.

## Visual audit

Each scene audit compared all 100 decoded-render frames against its reference,
ranked frames by mean absolute pixel error, and reviewed a six-frame contact
sheet containing reference, candidate, and 12x absolute difference panels.

| Scene | Mean frame MAE | Worst frame MAE | Decision |
| --- | ---: | ---: | --- |
| Garden | 0.001713 | 0.001919 | pass |
| Truck | 0.001994 | 0.002452 | pass |
| Train | 0.002391 | 0.004299 | pass |
| Bicycle | 0.002122 | 0.002865 | pass |
| Bonsai | 0.001389 | 0.004612 | pass |

Differences are concentrated around high-frequency edges, thin geometry,
specular transitions, and low-amplitude color changes. The worst-frame review
found no missing structure, block artifacts, or visually material color shift
at normal viewing scale.

## Interpretation

- **Recommended same-checkpoint storage format:** SPZ v4 with SH 8/8.
- **Safer lower-compression fallback:** tile-codebook, whose measured aggregate
  ratio is lower but attribute deltas are also smaller.
- **Do not compare directly:** HAC++, HEMGS, and other learned/retrained
  representations may reach stronger rate-distortion points, but they change
  the model or require a decoder/training pipeline.
- **Research candidate:** GSICO is post-training and does not need training
  images, but its JPEG XL parameter-image decoder has not yet been integrated
  or measured in this repository.

The SPZ defaults are 5/4-bit SH. This qualification deliberately uses 8/8 bits;
the default setting must remain a separate, more lossy benchmark row.
