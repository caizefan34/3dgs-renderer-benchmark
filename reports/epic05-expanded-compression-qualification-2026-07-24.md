# EPIC-05 expanded near-lossless compression qualification

This report compares nine storage rows on the five canonical 1080p scenes:
native PLY, XZ lossless, block-float, tile-codebook, PlayCanvas Compressed PLY,
PlayCanvas SOG, and SPZ with 8/8, 6/6, and 5/4-bit SH precision. Every row was
decoded to a standard PLY and rendered with the same gsplat renderer, 100-camera
trajectory, ground truth, A100 cohort, and benchmark commit
`690697a8fed5d466d45780c46a0fcf12fc958798`.

## Final result

| Codec | Aggregate size | Ratio | Worst PSNR delta | Worst SSIM delta | Worst LPIPS delta | Overall passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XZ lossless | 3.569 GB | 1.166x | +0.000000 dB | +0.000000 | +0.000000 | 5/5 |
| block-float | 1.918 GB | 2.170x | -0.026202 dB | -0.002102 | +0.002783 | 4/5 |
| tile-codebook | 1.084 GB | 3.840x | -0.001600 dB | -0.000121 | +0.000138 | 5/5 |
| Compressed PLY | 1.028 GB | 4.047x | -0.232002 dB | -0.006788 | +0.006733 | 4/5 |
| **SPZ 8/8** | **725.9 MB** | **5.732x** | **-0.014548 dB** | **-0.000217** | **+0.000666** | **5/5** |
| SPZ 6/6 | 546.5 MB | 7.615x | -0.275095 dB | -0.002616 | +0.004858 | 4/5 |
| SPZ 5/4 | 413.4 MB | 10.067x | -1.806481 dB | -0.017052 | +0.027364 | 0/5 |
| PlayCanvas SOG | 223.1 MB | 18.655x | -2.451067 dB | -0.025942 | +0.033478 | 0/5 |

The strict gate is PSNR drop below 0.2 dB, SSIM drop below 0.002, LPIPS
increase below 0.005, plus a normal-view visual audit. The visual review covered
all 40 non-reference rows. SOG showed visible color and structure errors; the
other codecs passed visual inspection, while their independent numerical gates
still determined the overall result.

## Recommendation

- Use **SPZ v4 with SH 8/8** for the smallest same-checkpoint artifact that
  passed all five scenes. It reduces 4,161,271,828 bytes to 725,933,824 bytes.
- Use **tile-codebook** when a lower-risk, lower-compression fallback is more
  important than size. Its quality deltas are the smallest lossy deltas here.
- Use **XZ** only when bit-exact recovery is mandatory; its storage saving is
  limited to 1.166x.
- Do not label SOG, SPZ 5/4, or the complete SPZ 6/6 curve as five-scene
  near-lossless under this protocol. Compressed PLY also fails on Bonsai.
- FCGS remains a separate pretrained-codec track because its decoder contract
  differs from a direct same-checkpoint format conversion. Its completed
  five-scene safest point reaches 12.843x but passes the strict numerical gate
  on only 2/5 scenes. GSICO, HAC, and HAC++ remain pending research/native
  compression tracks.

Machine-readable evidence is in
[`compression-results.json`](generated/compression-expanded-final/compression-results.json),
the compact audit index is in
[`visual-audit-summary.json`](generated/compression-expanded-final/visual-audit-summary.json),
and the worst-frame review is
[`global-worst-visual-review.jpg`](generated/compression-expanded-final/global-worst-visual-review.jpg).
