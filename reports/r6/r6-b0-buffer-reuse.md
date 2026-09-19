# R6-B0 — Persistent Buffer / Full Clear

## Implemented scope

Only the unpacked 3DGS rasterizer backward buffers are reused:

- `v_means2d`
- `v_conics`
- `v_colors`
- `v_opacities`
- `v_means2d_abs`

`experiments/r6/r6_b/r6b_persistent_buffers.cuh` owns capacity-managed flat
storage and returns active-shaped views. A growth or logical-shape change
forces a full active-range clear. B0 (`r6b_set_mode(1)`) always clears the
entire active range before launching the unchanged raster backward kernel.

Projection, SH, parameter gradients, optimizer, rendering, loss, and training
schedule were not modified.

## Measurement

`benchmark_r6b.py` records backward and full training iteration time with CUDA
events after >=20 warmups and >=100 replays by default. The patched extension
also records a CUDA event pair around rasterizer buffer preparation; its value
is emitted as `T_raster_prepare_cuda_event_ms`.

Baseline allocation/init decomposition remains the existing R6 profiler's
CUDA-kernel/profiler measurement; B0 is intentionally a separate allocation
vs. clear ablation rather than a Python-wall-clock claim.

## Status

The required room/bicycle/garden 5K/15K/30K checkpoint set is not present in
this workspace, so no workload number is reported here.
