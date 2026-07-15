# Benchmark Protocol v3

The authoritative, hashable protocol is [`benchmark/protocol.json`](../benchmark/protocol.json).
This page explains its boundaries; it does not redefine numeric defaults.

## Primary rules

- One renderer/case per fresh process.
- One physical GPU visible to the process.
- Identical checkpoint, ordered evaluation cameras, GT images, resolution, color space, background, and SH degree.
- Exactly 100 cameras ordered by image name and evenly selected including endpoints.
- Center-cropped 16:9 field of view with matched GT crop; GT is area-resized to 1920x1080 at metric time.
- End-to-end FPS from synchronized wall time; CUDA-event latency is diagnostic.
- Ordered raw frame samples retained.
- Parent-launch-to-child-ready startup, adapter initialization, shared scene parsing, renderer preparation/upload, and first-frame latency reported separately.
- NVML process peak memory is primary; framework allocation is secondary.
- PSNR, SSIM, and LPIPS use the same evaluation cameras as performance measurement.
- Every asset, protocol, renderer, benchmark revision, and raw sample artifact is hashed.

## Output contract

The ranking contract requires normalized float32 HWC RGB in `[0,1]`.
Depth and alpha may be collected as renderer capabilities, but they are not required for a renderer that only exposes RGB.
Changing antialiasing, precision, SH approximation, tile policy, pruning, or sorting semantics creates a distinct `config_id`.

## Statistical reporting

Retain samples by repeat and frame.
Report arithmetic mean, median, P95, P99, coefficient of variation, and confidence intervals over independent repeats when enough repeats exist.
Do not treat thousands of correlated frames as independent experimental replicates.

## Environment

Record GPU model/UUID/VRAM, CPU, RAM, OS, driver, CUDA, Python, PyTorch, renderer commit, build command, benchmark commit, power limit, and clock policy.
Any difference creates a separate cohort unless the suite explicitly declares it irrelevant.

Legacy Protocol v1/v2 artifacts remain historical evidence and are indexed in `results/quarantine/legacy-index.json`.
