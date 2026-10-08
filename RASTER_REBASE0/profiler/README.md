# Profiler

Nsight Compute / Nsight Systems were NOT available in this environment.
Hotspot identity is INFERRED from forward-vs-backward CUDA-event fractions + source structure, not from kernel-level profiling.

To enable: install CUDANsight (ncu), then profile the forward and backward entry points
of each backend on a fixed scene and record kernel-level SASS/occupancy here.

