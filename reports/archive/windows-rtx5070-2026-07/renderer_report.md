# Renderer report

All tested backends used Python 3.10.20, PyTorch 2.12.1+cu130, CUDA runtime 13.0, CUDA toolkit 13.3, and NVIDIA driver 592.01.

| Config ID | Source commit | Backend / precision | Configuration | Train phase status |
| --- | --- | --- | --- | --- |
| `original_3dgs` | `9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0` | CUDA; float32 inputs/outputs | Official Graphdeco rasterizer, SH degree 3, black background | speed + quality complete; collector blocked by NVML |
| `gsplat` | `77ab983ffe43420b2131669cb35776b883ca4c3c` | CUDA; float32 inputs/outputs | packed rasterization, SH degree 3 | speed + quality complete; collector blocked by NVML |
| `gsplat_higs` | `77ab983ffe43420b2131669cb35776b883ca4c3c` | CUDA; backend frame float16, adapter output float32 | tile 8, no SH compression, SH degree 3 | speed + quality complete; collector blocked by NVML |
| `speedy_splat` | `34c45c6d9b8bd6110231864f2f358b6d3abbf73d` | CUDA; float32 inputs/outputs | all scores = 1, no pruning, SH degree 3 | speed + quality complete; collector blocked by NVML |
| `tcgs` | `0bb82f88fde211c34b42e1497f0fc7265461592b` | CUDA Tensor Cores; FP16 internal alpha path, float32 output | TC-GS Speedy integration, all scores = 1, SH degree 3 | speed + quality complete; collector blocked by NVML |

Not runnable in this native Windows environment:

- `fast_gauss`: repository status `environment_blocked`; EGL is unavailable on native Windows.
- `flashgs`, `local_gs`, `gemm_gs`: repository status `custom_adapter_required`.
- `stopthepop`: repository status `separate_track_required`.

Pinned clean source checkouts are under `../artifacts/renderer-sources/`. Build and environment details are under `../artifacts/environment-setup/`.
