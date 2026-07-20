# Benchmark report — partial, non-ranking evidence

The table below is the completed Train run from timestamp `20260717T022523Z`. It is **not an official Matrix result** because the required absolute NVML process peak is unavailable and only one of five required cases was executed before the hard blocker stop.

| Renderer | Wall FPS | Mean wall ms | PSNR dB | SSIM | LPIPS | Framework peak MiB (secondary only) | NVML process peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_3dgs | 57.6 | 17.3742 | 23.5410 | 0.809344 | 0.299728 | 1254.4 | unavailable (recorded 0) |
| gsplat | 151.8 | 6.5870 | 22.3809 | 0.798356 | 0.305869 | 728.1 | unavailable (recorded 0) |
| gsplat_higs | 516.6 | 1.9358 | 22.3817 | 0.798291 | 0.304925 | 553.4 | unavailable (recorded 0) |
| speedy_splat | 174.2 | 5.7413 | 23.5422 | 0.809371 | 0.299723 | 772.0 | unavailable (recorded 0) |
| tcgs | 227.7 | 4.3917 | 23.5434 | 0.809089 | 0.298932 | 780.2 | unavailable (recorded 0) |

Evidence completeness for every row:

- 500 synchronized wall samples and 500 CUDA-event samples.
- Five repeat arrays of 100 samples.
- 100 quality views.
- 100 hashed RGB PNG render outputs.
- Raw NVML samples at the configured 5 ms request interval. Sample counts varied with render duration: original 599, gsplat 287, HiGS 93, Speedy 115, TC-GS 101. Every process-memory value was unavailable/zero under WDDM.
- Cold-path fields: process startup, renderer init, scene load/parse, renderer preparation/upload, and time to first frame.

Ranking output: `generated/ranking/ranking.md` and `generated/ranking/ranking.json`. The official measured table is empty.
