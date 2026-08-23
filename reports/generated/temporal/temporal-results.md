# EPIC-05 ordered-camera temporal fidelity

This CPU analysis uses retained Tier A PNG evidence. It measures adjacent-frame
RGB-delta residual against GT and does not claim motion-compensated video quality.

| Config | Cases | Mean residual | Mean luma residual | Temporal delta PSNR |
| --- | ---: | ---: | ---: | ---: |
| gsplat | 5 | 0.133351 | 0.132231 | 14.094 dB |
| gsplat_dense | 5 | 0.133351 | 0.132231 | 14.094 dB |
| gsplat_higs | 5 | 0.133340 | 0.132219 | 14.095 dB |
| gsplat_higs_auto | 5 | 0.133360 | 0.132223 | 14.095 dB |
| gsplat_higs_sh16 | 5 | 0.134002 | 0.132379 | 14.075 dB |
| gsplat_higs_sh32 | 5 | 0.133360 | 0.132223 | 14.095 dB |
| gsplat_higs_tile16 | 5 | 0.133335 | 0.132213 | 14.096 dB |
| gsplat_higs_tile16_sh16 | 5 | 0.133996 | 0.132373 | 14.076 dB |
| gsplat_higs_tile16_sh32 | 5 | 0.133355 | 0.132217 | 14.095 dB |
| original_3dgs | 5 | 0.130292 | 0.129093 | 14.229 dB |
| speedy_splat | 5 | 0.130284 | 0.129084 | 14.230 dB |
| tcgs | 5 | 0.130248 | 0.129044 | 14.233 dB |
