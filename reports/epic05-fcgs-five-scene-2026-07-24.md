# EPIC-05 FCGS five-scene qualification

FCGS commit: `31e59a46f7e51505b46a7bbe6e53268ee3155fbf`  
MPEG G-PCC / TMC13 commit: `a3d15c5e73bae20fbe2ec79be60994038a66dc8d`  
Operating point: `lambda=0.0001`, the safest point from the Train rate curve.

## Result

| Scene | Ratio | PSNR delta | SSIM delta | LPIPS delta | Numeric gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Garden | 12.187x | -0.019739 dB | -0.002052 | +0.002807 | fail |
| Truck | 14.270x | -0.011914 dB | -0.001764 | +0.003394 | pass |
| Train | 12.067x | -0.046687 dB | -0.003521 | +0.005349 | fail |
| Bicycle | 13.281x | -0.011329 dB | -0.004770 | +0.004507 | fail |
| Bonsai | 12.108x | -0.073730 dB | -0.000967 | +0.002062 | pass |

The aggregate artifact size is 323,999,906 bytes from 4,161,271,828 source
bytes, or **12.843x compression**. Only 2/5 scenes pass the numerical gate.
Garden narrowly exceeds the SSIM limit, while Train and Bicycle fail by wider
margins. A visual audit cannot turn those numerical failures into passes, so it
is not required to reject FCGS as the strict five-scene near-lossless winner.

## Decision

FCGS is useful as a pretrained high-compression, light-loss rate-distortion
codec. It is substantially smaller than SPZ 8/8 (12.843x versus 5.732x), but it
does not preserve the canonical checkpoint closely enough under this
repository's strict limits. **SPZ 8/8 remains the storage recommendation when
all five scenes must remain near-lossless.**

Machine-readable evidence is in
[`fcgs-five-scene.json`](generated/compression-expanded-final/fcgs-five-scene.json).
