# EPIC-05 HiGS calibrated tile selector

Benchmark commit: `e5cf510abceaf0e2667801a9a0de2f1cd5a1cef3`  
Host: EPIC-05, one A100-SXM4-80GB cohort, five canonical 1080p scenes.

`gsplat_higs_calibrated` times tile 8 and tile 16 with CUDA Events on the first
view, then fixes the faster choice for the renderer lifetime. This tests whether
measurement is a better selector than the existing Gaussian-count heuristic.

## Result

| Scene | Selected tile | Calibrated FPS | Fixed tile-8 FPS | FPS delta | Calibrated P99 | Fixed P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Garden | 8 | 489.01 | 493.60 | -0.93% | 2.45 ms | 2.55 ms |
| Truck | 8 | 708.10 | 693.14 | +2.16% | 1.77 ms | 1.78 ms |
| Train | 8 | 809.26 | 807.90 | +0.17% | 1.54 ms | 1.64 ms |
| Bicycle | 8 | 568.76 | 556.28 | +2.24% | 2.48 ms | 2.48 ms |
| Bonsai | 8 | 1009.94 | 1069.15 | -5.54% | 1.28 ms | 1.20 ms |

All five scenes selected tile 8. Mean FPS was 717.01 versus 724.01 for the
canonical fixed tile-8 runs, an aggregate -0.97%. Quality matches the fixed
path to numerical noise because the selected kernel and rendering contract are
identical. Peak VRAM also remains effectively unchanged.

## Decision

The prototype is reproducible and protects against choosing the slower tile-16
path, but it does not improve this five-scene cohort. The current canonical
scene densities all favor tile 8, so first-view calibration adds policy
complexity without a measured speed benefit. Ideas 11 and 29 are therefore
recorded as **measured, no gain on the canonical cohort**, not as successful
accelerations.

The result narrows the next renderer work: tile-size selection is not the
current bottleneck. The higher-value implementation target remains exact
conic-vs-fine-tile rejection, followed by a sparse-warp/dense-Tensor-Core
hybrid selected from actual per-tile occupancy.

Machine-readable evidence is in
[`five-scene-results.json`](generated/higs-calibrated/five-scene-results.json).
