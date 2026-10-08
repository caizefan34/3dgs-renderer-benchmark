# profiler/

Intentionally empty of captured runtime profiles.

Runtime hardware profilers were UNAVAILABLE (ncu 2021.3.1 incompatible with the
torch 2.9.1+cu128 build, error 6; nvprof 11.5 too old for CUDA 12.8). F2 is
audited statically via cuobjdump PTX/SASS (decisive for instruction-mix) and is
contention-independent. F4 K50-B1 quality gates are contention-independent.
