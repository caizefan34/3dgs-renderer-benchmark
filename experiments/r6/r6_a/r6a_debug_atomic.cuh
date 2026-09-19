#pragma once
#include <cstdint>

// Host-side API for the R6-A debug atomic counter.
// The __device__ variable and these functions are defined IN THE KERNEL FILE
// (RasterizeToPixels3DGSBwd.cu) by the prepare_r6a_source.py patch, NOT in
// a separate .cu file.  This is required because CUDA __device__ variables
// do not have cross-TU linkage — extern __device__ is treated as static by
// nvcc, so a separate .cu file would get a different counter.

namespace gsplat::r6a {
    void reset_debug_counter();
    uint64_t get_debug_counter();
}
