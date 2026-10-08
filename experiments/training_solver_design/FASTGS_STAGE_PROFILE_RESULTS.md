# FastGS native 30K stage profile results

**2026-09-28 data-provenance correction:** The garden trajectory used the EPIC-05 shared garden sparse model later shown to be a corrupt flat 1,839,236-point cloud with stripped image keypoints. Quarantine its garden stage timings and phase percentages as measurements of that corrupted-data workload; they are not valid official-garden baseline costs. The room trajectory is unaffected by this garden-only finding. See `P16_GARDEN_POINT_ORDER_AUDIT_20260928.md` and the official-data P16 garden reruns.

2026-09-27. Diagnostic measurements from the unchanged author FastGS trajectories for room and garden. Both runs completed all 30,000 iterations from initialization at the pinned author commit. These profiled timings are not an uninstrumented speed baseline and do not establish full-training savings.

The table reports summed CUDA kernel time per profiled step and mean synchronized iteration wall time while Kineto was active. Phase percentages use summed CUDA kernel time as the denominator. “Raster backward” is the native `_RasterizeGaussiansBackward` autograd operator; optimizer is `phase.optimizer`. Each cell shows phase CUDA milliseconds per step and its share of CUDA kernel time.

| Scene | First iteration | CUDA kernels ms/step | Profiled wall ms/step | Forward ms/step (%) | Raster backward ms/step (%) | Optimizer ms/step (%) | Adam calls (main + f_rest) |
|---|---:|---:|---:|---:|---:|---:|---:|
| room | 4096 | 7.561 | 10.235 | 1.830 (24.2%) | 2.928 (38.7%) | 1.194 (15.8%) | 128 + 8 |
| room | 10048 | 7.345 | 9.494 | 1.582 (21.5%) | 3.084 (42.0%) | 1.156 (15.7%) | 128 + 8 |
| room | 16000 | 5.915 | 7.137 | 1.362 (23.0%) | 3.119 (52.7%) | 0.077 (1.3%) | 4 + 4 |
| room | 24064 | 5.491 | 6.854 | 1.260 (22.9%) | 2.904 (52.9%) | 0.034 (0.6%) | 2 + 2 |
| garden | 4096 | 17.833 | 22.379 | 3.795 (21.3%) | 5.943 (33.3%) | 3.591 (20.1%) | 128 + 8 |
| garden | 10048 | 27.781 | 31.784 | 5.380 (19.4%) | 9.587 (34.5%) | 6.358 (22.9%) | 128 + 8 |
| garden | 16000 | 22.343 | 23.362 | 4.863 (21.8%) | 9.979 (44.7%) | 0.770 (3.4%) | 4 + 4 |
| garden | 24064 | 20.101 | 21.096 | 4.506 (22.4%) | 9.661 (48.1%) | 0.307 (1.5%) | 2 + 2 |

All eight windows contain exactly 128 steps. The optimizer-call counts match the expected accumulation schedule: 128 main-parameter and 8 `f_rest` updates in the early windows, then 4 + 4 at 16K and 2 + 2 at 24K. Both early windows include topology activity and extra forward calls (168 and 148 forward calls per 128-step window, respectively); their per-step forward figures therefore include topology-triggered work and are not ordinary-step costs. Do not treat these rows as complete-training speedup estimates. The late windows show the native raster backward occupying about 45–53% of summed CUDA kernel time, while optimizer time falls below 4%; neither share alone shows how much complete training can be removed.

The analyzer initially rejected same-name CUDA Kineto rows for `phase.*` annotations. The corrected analyzer excludes CUDA annotation rows whose names also occur as CPU user annotations, so annotation ranges and their child kernels are not counted twice. It uses `_RasterizeGaussiansBackward` for native raster backward because `phase.backward` spans work across autograd threads and its CUDA child total is incomplete. The stage-profile protocol’s explanatory note for this accounting was appended after data collection; the run parameters, thresholds, and experimental gate did not change. These windows remain profiler-affected samples and cannot be extrapolated into a complete 30K runtime or speed claim.

## Provenance

- Author source commit: `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f`.
- Author `train.py` SHA-256: `1fd700c341d109f13f95ff3a3cdf1328445f27909119e7eacadb31913e34c963`.
- Probe `fastgs_stage_profile.py` SHA-256: `ca1181cad38767a3f5a1f0870cd900cf4f1b5839d07807a3827ee1907ad3db82`.
- Analyzer `analyze_fastgs_stage_profile.py` SHA-256: `15e91d4f73b3e69ef6e7bb827e0814ee4a38ac3d769bf42ca280e78e13bbf91b`.
- Stage-profile protocol SHA-256 before its post-run accounting note: `9873c26b6a80487b36bfe10acedff76c03e2b6772be539d807206096093e9c21`; current protocol SHA-256: `da84d97e2e778fd36ab36b6b7431330549190a948d8b0d76bfc2f4f4f4ec0cd9`.
- Deferred-pullback protocol SHA-256: `842173411ec3b5d96fc4615572a467d1c417d9e4acd2fe542c5cce27e846456a`.

| Artifact | SHA-256 |
|---|---|
| `fastgs_stage_profile_20260927/room.json` | `38a73aad1b2e3ee138f115285d93ef28b2eeef0e802eb95cc0df00aca7862185` |
| `fastgs_stage_profile_20260927/garden.json` | `435dd456e2b1a300ac4f45acfdba6ba5031eb8bd4fd5ef021c8b782442f7583a` |
| `fastgs_stage_profile_20260927/room.log` | `2bdbc3b00af6c0e55a5800ad11d7d8fea1eb8f3c0acfbcfbdbc08e96e1a71e1f` |
| `fastgs_stage_profile_20260927/garden.log` | `de7305ff3794a3f0d2a31c93522a3fe1b4c19d855cbfcefae86072b34bce48c3` |
| `fastgs_stage_profile_20260927/summary.json` | `a57d9409429b34d14a47cf91a1159a79e2f79192ab7f75bb4fa625b94c6c65a9` |
| room final PLY | `b864f1a310272b541f14227eeef0bae8fd51f51cdf637d45ea977d44e3ee5d3a` |
| garden final PLY | `ba12264768db1e09ebdb32c8c57a6be506300d4a354c459404b520d6a0977eaa` |

The raw manifests, complete logs, and analyzer output are in [`fastgs_stage_profile_20260927`](fastgs_stage_profile_20260927/); the analyzer output is `summary.json`.
