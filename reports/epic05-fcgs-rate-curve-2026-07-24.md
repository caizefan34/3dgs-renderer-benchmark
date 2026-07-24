# EPIC-05 FCGS pretrained-codec rate curve

FCGS commit: `31e59a46f7e51505b46a7bbe6e53268ee3155fbf`  
MPEG G-PCC / TMC13 commit: `a3d15c5e73bae20fbe2ec79be60994038a66dc8d`  
Scene: Tanks and Temples Train, 254,575,516-byte canonical PLY, GPU 2 A100.

| Lambda | Compressed bytes | Ratio | PSNR delta | SSIM delta | LPIPS delta | Strict near-lossless |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.0001 | 21,097,542 | 12.067x | -0.046687 dB | -0.003521 | +0.005349 | fail |
| 0.0002 | 18,368,780 | 13.859x | -0.067837 dB | -0.003898 | +0.006350 | fail |
| 0.0004 | 15,849,448 | 16.062x | -0.090510 dB | -0.004841 | +0.008388 | fail |
| 0.0008 | 13,467,961 | 18.902x | -0.154043 dB | -0.006163 | +0.011690 | fail |
| 0.0016 | 11,381,408 | 22.368x | -0.212813 dB | -0.007898 | +0.017173 | fail |

FCGS achieves substantially higher compression than SPZ 8/8, but every tested
point fails this repository's strict same-checkpoint near-lossless limits. Even
the safest point exceeds both the SSIM-drop and LPIPS-increase limits. FCGS is
therefore reported as a separate pretrained-codec rate-distortion option, not
as the strict near-lossless storage winner.

The five-scene `lambda=0.0001` qualification is running separately so the
single-scene curve is not generalized without evidence.

Machine-readable values are in
[`fcgs-train-rate-curve.json`](generated/compression-expanded-final/fcgs-train-rate-curve.json).
