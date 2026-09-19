# 13-Scene Training Speed — B1A vs B1 (30K)

Full-training wall-clock timing on the A100-PCIE-40GB cohort (anysplat env, torch 2.4.1+cu124). `training_speedup = T_B1 / T_B1A`.

| Scene | Dataset | B1 total (min) | B1A total (min) | Speedup | B1 mean iter (ms) | B1A mean iter (ms) | B1 steady iter (ms) | B1A steady iter (ms) | iter speedup |
|-------|---------|----------------:|----------------:|--------:|-------------------:|--------------------:|---------------------:|----------------------:|-------------:|
| bicycle | Mip-NeRF360 | 42.5 | 38.3 | 1.109× | 81.5 | 72.9 | 82.3 | 77.6 | 1.061× |
| bonsai | Mip-NeRF360 | 30.9 | 27.4 | 1.127× | 59.1 | 52.6 | 52.3 | 50.8 | 1.031× |
| counter | Mip-NeRF360 | 31.6 | 28.3 | 1.117× | 60.7 | 54.0 | 53.4 | 51.9 | 1.030× |
| flowers | Mip-NeRF360 | 37.9 | 33.9 | 1.118× | 73.0 | 65.4 | 70.9 | 68.0 | 1.043× |
| garden | Mip-NeRF360 | 38.7 | 34.8 | 1.113× | 74.5 | 67.1 | 71.8 | 67.6 | 1.063× |
| kitchen | Mip-NeRF360 | 34.3 | 30.9 | 1.110× | 65.7 | 59.0 | 58.8 | 56.6 | 1.039× |
| room | Mip-NeRF360 | 31.2 | 28.3 | 1.103× | 59.7 | 53.8 | 54.6 | 52.7 | 1.037× |
| stump | Mip-NeRF360 | 36.1 | 33.1 | 1.090× | 69.3 | 63.3 | 66.6 | 64.4 | 1.035× |
| treehill | Mip-NeRF360 | 38.7 | 35.5 | 1.091× | 74.5 | 67.9 | 74.5 | 71.6 | 1.041× |
| train | Tanks & Temples | 32.3 | 29.4 | 1.099× | 62.1 | 56.3 | 55.8 | 53.8 | 1.036× |
| truck | Tanks & Temples | 32.7 | 29.7 | 1.098× | 62.6 | 56.9 | 57.8 | 56.0 | 1.032× |
| drjohnson | Deep Blending | 29.3 | 27.1 | 1.084× | 56.1 | 51.6 | 50.0 | 49.7 | 1.005× |
| playroom | Deep Blending | 31.1 | 28.5 | 1.092× | 59.7 | 54.4 | 55.6 | 54.3 | 1.024× |

## Dataset / overall geomean speedup

| Dataset | n | geomean speedup | arith mean speedup |
|---------|---|----------------:|-------------------:|
| Mip-NeRF360 | 9 | 1.1086× | 1.1087× |
| Tanks & Temples | 2 | 1.0984× | 1.0984× |
| Deep Blending | 2 | 1.0881× | 1.0881× |
| **All 13** | 13 | **1.1039×** | **1.1040×** |

Note: `steady_state_mean_iter_ms` = mean of iterations 15000-30000 (after densification ends), a cleaner per-iteration comparison than the full-run mean (which includes early densification spikes).
