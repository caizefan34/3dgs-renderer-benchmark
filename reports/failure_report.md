# Failure and blocker report

## Hard blocker: NVML process memory is unavailable under WDDM

Required protocol field:

`memory.primary = absolute NVML process peak MiB`

Observed platform state:

- GPU brand: GeForce.
- Driver model: WDDM, current and pending.
- `nvidia-smi -q` reports: `Used GPU Memory: Not available in WDDM driver model`.
- `nvidia-smi --query-compute-apps=...used_gpu_memory` returns `[N/A]` for every process.
- `pynvml.nvmlDeviceGetComputeRunningProcesses` / graphics process entries expose `NVML_VALUE_NOT_AVAILABLE` for `usedGpuMemory`.
- All retained Train raw NVML samples contain 0 after the sampler correctly discards the sentinel. Positive sample count: 0 for all five renderer processes.

Failure code path:

1. `src/benchmark_framework/nvml.py::_usage_mb` cannot obtain a numeric process value and returns 0.
2. Speed and quality phases still complete successfully.
3. `src/scripts/collect_matrix_result.py` uses the required NVML peak.
4. `src/benchmark_matrix.py::validate_result` rejects `$.metrics.performance.peak_vram_mb` because it must be positive.

Why no workaround was used:

- PyTorch allocated/reserved memory is a secondary framework metric, not absolute NVML process memory.
- Total device memory includes desktop applications and other processes.
- Subtracting a context baseline is explicitly forbidden by the protocol.
- Changing the metric or schema would modify benchmark rules.

Exact external requirement to unblock:

Run on a host/driver combination where NVML exposes numeric per-process used GPU memory, normally native Linux or a validated TCC-capable setup. This GeForce laptop remains WDDM and does not expose that field.

## Same-machine Linux-path audit

- `wsl --list --verbose` shows no Ubuntu or other general-purpose Linux distribution; only `docker-desktop` WSL2 exists.
- The `docker-desktop` distribution boots its WSL2 kernel, but its minimal userland cannot execute `/usr/lib/wsl/lib/nvidia-smi` because the required compatible dynamic loader/userland is absent.
- Docker CLI 29.2.1 is installed, but the Docker Desktop Linux engine is not running and no local image inventory can be queried.
- No new WSL distribution, container image, or driver mode was installed or activated because that would be a material external-state change and still would not prove numeric NVML process memory support.

Therefore there is no currently runnable same-machine Linux environment that can replace the WDDM measurement path.

## Resolved pre-run failures retained as evidence

1. Windows Git initially converted `benchmark/protocol.json` to CRLF, changing its byte hash. The file was restored byte-for-byte to the repository LF blob; the required hash now matches.
2. The official fixed cameras triggered a synthetic-camera scene-center heuristic. The launcher now skips that heuristic only when an external hashed camera manifest is provided.
3. LPIPS was initially visible only through user site-packages. It and SciPy were installed into the actual isolated run environment.
4. The pinned original rasterizer returns three values while another installed fork returned four. The adapter now accepts both without changing the rendered image.
5. HiGS returned a float16 backend frame; the adapter now casts its public output to the required float32 after rendering.

All failed attempts and GPU snapshots remain under `../results/measured/**/failures/` and `../artifacts/run-logs/`.
