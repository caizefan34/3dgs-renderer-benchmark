# profiler/

Intentionally empty of captured profiles.

Runtime hardware profilers were UNAVAILABLE for this lane:
- Nsight Compute 2021.3.1: injecting the ncu wrapper into the torch 2.9.1+cu128
  / cudnn 9.10 process returns application error code 6 and profiles zero
  kernels (see ../logs/ncu_room.log).
- nvprof: v11.5, too old for CUDA 12.8.

Causal evidence is therefore intervention-based (paired interleaved R0-vs-variant
CUDA-event timing), which satisfies the spec two-independent-evidence-class
requirement without hardware counters. Per-scene fallback evidence is recorded
in BWD_DEPREG0_PROFILE_{ROOM,BICYCLE,GARDEN}.json.
