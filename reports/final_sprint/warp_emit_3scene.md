# Warp-cooperative emit: 3-scene benchmark

Reference V1: commit `02375033388d4348376b6b607ab85f551e498a77`, A100 PCIe
40 GB, torch 2.4.1+cu124, CUDA runtime 12.4, tile size 16. All measurements
use 20 warmups, 40 CUDA-event repetitions, explicit synchronization, and
median as the primary statistic. Candidate timing was rerun on an idle GPU;
an intervening saturated-GPU run was discarded.

| Scene | visible | N_isects | intersections/visible | baseline emit ms | warp emit ms | emit speedup | baseline forward ms | warp forward ms | forward speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| room | 16,076 | 25,475,247 | 1,584.68 | 10.632 | 0.345 | 30.81× | 14.780 | 4.864 | 3.04× |
| bicycle | 13,654 | 18,550,853 | 1,358.64 | 6.584 | 0.315 | 20.88× | 10.880 | 4.581 | 2.38× |
| garden | 100,582 | 11,257,252 | 111.92 | 1.961 | 0.250 | 7.85× | 5.220 | 3.522 | 1.48× |

| Scene | baseline pre-raster ms | warp pre-raster ms | pre-raster speedup | baseline sort ms | warp sort ms | baseline raster ms | warp raster ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| room | 14.029 | 4.061 | 3.45× | 3.107 | 3.352 | 0.328 | 0.416 |
| bicycle | 9.474 | 3.194 | 2.97× | 2.651 | 2.668 | 0.897 | 0.908 |
| garden | 3.795 | 2.078 | 1.83× | 1.648 | 1.661 | 1.085 | 1.105 |

Emit geomean speedup is **17.16×** (corrected aggregate). Full-forward geomean
speedup is **2.20×**. Sort and raster remain effectively
unchanged; the saving is isolated to the serial emit work. `N_isects` is
identical to baseline in every measurement.

## Workload shape

Room has p50/p90/p95/p99/max intersections per visible Gaussian of
39/3,076.5/13,260/25,350/25,350; 15.3% of its Gaussians (`>1024` bucket)
perform 94.8% of intersection work. Bicycle is similarly heavy-tailed:
56/1,646.4/4,918.9/38,895.7/63,860, with 13.4% of Gaussians doing 92.1% of
work. Garden has many small splats (p50 2), but its `>1024` bucket still does
66.0% of work. This matches the speedup order room/bicycle > garden.

Raw data: `experiments/final_sprint/warp_emit/raw/w1_isolated_emit.json` and
`experiments/final_sprint/warp_emit/raw/w1_full_pipeline_128_clean.json`.
