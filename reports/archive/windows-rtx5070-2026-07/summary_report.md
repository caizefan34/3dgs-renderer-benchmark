# Matrix v3.1 benchmark summary

Status: **blocked — no official ranking generated**.

The repository was checked out at `b46e8f27fbc3beea89a12f25c35ce8b296f24cd9`. The exact protocol SHA-256 is `892e18890501c408dc6746af69f17e16973604f0c02a3caddf1954d3bf1fede2`.

Completed work:

- Verified all four official source archives.
- Prepared all five canonical cases and independently confirmed checkpoint, camera, and GT-manifest hashes.
- Prepared pinned environments for `original_3dgs`, `gsplat`, `gsplat_higs`, `speedy_splat`, and `tcgs`.
- Completed the full Train speed and quality phases for all five renderers: 30 warmup frames, 5 repeats × 100 measured frames, 100 GT views, PSNR, SSIM, LPIPS-vgg, raw frame timings, raw NVML observations, and 100 render PNGs per renderer.
- Ran the repository ranking generator. It correctly emitted no eligible renderer because complete five-case Tier A coverage does not exist.

Hard blocker:

The required primary memory metric is absolute NVML process peak MiB. This machine runs the GeForce GPU in Windows WDDM mode. NVIDIA reports `Used GPU Memory: Not available in WDDM driver model`, and every raw per-process NVML observation is `NVML_VALUE_NOT_AVAILABLE`. The collector therefore rejects `peak_vram_mb = 0`, as required by the schema. Framework memory cannot be substituted without changing the protocol.

Per the task's failure policy, the remaining four formal scene runs were not started after this hard blocker was proven. See `failure_report.md` for the exact code path and evidence.

Generated deliverables:

- `machine_report.md`
- `dataset_report.md`
- `renderer_report.md`
- `benchmark_report.md`
- `failure_report.md`
- `reproducibility.md`
- `generated/ranking/ranking.json`, `.csv`, and `.md`
