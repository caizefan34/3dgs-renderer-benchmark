# Reproducibility and resume instructions

## Published cohort

The Tier A evidence was collected from benchmark commit
`dc9bb4e9231ae2fdf90fa9c40bcd6e0dbd7d104f` with protocol SHA-256
`892e18890501c408dc6746af69f17e16973604f0c02a3caddf1954d3bf1fede2`.

All 25 renderer/case results share this immutable cohort:

- GPU UUID: `GPU-12b6b703-727b-0c6e-1433-4e6161c54938` (physical GPU 2)
- NVIDIA driver: 580.105.08
- PyTorch CUDA runtime: 12.8
- Python: 3.10.20
- PyTorch: 2.9.1+cu128
- OS fingerprint: `Linux-5.10.134-16.3.al8.x86_64-x86_64-with-glibc2.35`

The later report-serialization fix does not alter any collected metric or raw sample.

## Verify the checkout and protocol

```bash
git rev-parse HEAD
sha256sum benchmark/protocol.json
```

Use the pinned renderer commits in `benchmark/renderers.json`. The four isolated
environments used on EPIC-05 were:

```text
/root/miniforge3/envs/original3dgs
/root/miniforge3/envs/gsplat
/root/miniforge3/envs/speedy
/root/miniforge3/envs/tcgs
```

## Verify strict NVML process memory

Before collection, confirm that a live CUDA process exposes a numeric process
memory value:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
```

Do not substitute framework allocator memory for the Tier A peak.

## Run or resume the fixed matrix

```bash
cd /root/3dgs-renderer-benchmark
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
/root/miniforge3/envs/gsplat/bin/python \
  src/scripts/run_linux_tier_a_matrix.py --resume
```

The runner verifies canonical assets, enforces the canonical 25-step order,
adopts only one valid orphan metric, and refuses mixed cohorts. A successful run
must pass `validate_session_metrics` with 25 metrics, five renderers, five cases,
and one cohort.

Generate the public report with:

```bash
/root/miniforge3/envs/gsplat/bin/python benchmark.py report \
  --output-dir docs/leaderboard
```

Acceptance requires `rejected_files=[]` and five rows in Tier A `overall`.
The published run met both conditions. One unsuccessful Speedy-Splat/Train
attempt rounded a microsecond-scale initialization measurement to `0.00 ms` and
was rejected before `metrics.json` creation; it is not part of the session or
published evidence.
