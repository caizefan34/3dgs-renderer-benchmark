# Hardware Cohorts

Hardware identity is part of every comparison cohort.
Results from different GPUs never share a primary ranking.

Every run records:

- GPU model, UUID, VRAM, power limit, and clock policy;
- CPU model and logical core count;
- system RAM;
- operating system and kernel/build;
- NVIDIA driver;
- CUDA runtime/toolkit;
- Python, PyTorch, renderer package/commit, and benchmark commit;
- timestamp.

## Reference host template

```json
{
  "hardware_profile_id": "rtx4090-linux-cuda12",
  "gpu": "NVIDIA GeForce RTX 4090",
  "gpu_uuid": "GPU-...",
  "gpu_vram_mb": 24564,
  "cpu": "...",
  "ram_mb": 65536,
  "os": "Ubuntu 24.04 ...",
  "driver": "...",
  "cuda": "...",
  "clock_policy": "locked",
  "power_limit_w": 450
}
```

Thermal throttling, background GPU consumers, dynamic power limits, and MIG must be recorded as deviations.
The primary publication host should lock clocks and power policy where permitted; community hardware results remain separate cohorts.
