# EPIC-05 expanded compression comparison

Benchmark commit: `690697a8fed5d466d45780c46a0fcf12fc958798`

| Codec | Cases | Aggregate ratio | Worst PSNR delta | Worst SSIM delta | Worst LPIPS delta | Numeric passes | Overall passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| block-float | 5 | 2.170x | -0.026202 dB | -0.002102 | +0.002783 | 4/5 | 4/5 |
| playcanvas-compressed-ply | 5 | 4.047x | -0.232002 dB | -0.006788 | +0.006733 | 4/5 | 4/5 |
| playcanvas-sog | 5 | 18.655x | -2.451067 dB | -0.025942 | +0.033478 | 0/5 | 0/5 |
| reference-ply | 5 | 1.000x | +0.000000 dB | +0.000000 | +0.000000 | 5/5 | 5/5 |
| spz | 5 | 5.732x | -0.014548 dB | -0.000217 | +0.000666 | 5/5 | 5/5 |
| spz-5-4 | 5 | 10.067x | -1.806481 dB | -0.017052 | +0.027364 | 0/5 | 0/5 |
| spz-6-6 | 5 | 7.615x | -0.275095 dB | -0.002616 | +0.004858 | 4/5 | 4/5 |
| tile-codebook | 5 | 3.840x | -0.001600 dB | -0.000121 | +0.000138 | 5/5 | 5/5 |
| xz-lossless | 5 | 1.166x | +0.000000 dB | +0.000000 | +0.000000 | 5/5 | 5/5 |
