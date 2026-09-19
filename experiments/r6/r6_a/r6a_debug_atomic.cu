// R6-A direct debug atomic instrumentation.
// Defines a __device__ counter incremented once per warp-leader atomicAdd
// execution set inside rasterize_to_pixels_3dgs_bwd_kernel.
//
// The kernel file (RasterizeToPixels3DGSBwd.cu) declares:
//   extern __device__ unsigned long long r6a_debug_count;
// and increments it inside the `if (warp.thread_rank() == 0)` block:
//   ::atomicAdd(&r6a_debug_count, 1ULL);
//
// This file provides the definition and host-side reset/read helpers.

#include "r6a_debug_atomic.cuh"
#include <cuda_runtime.h>
#include <cstdio>

namespace gsplat {

// Global device counter — one increment per warp-leader atomic execution.
// Defined here (not extern) so the linker resolves to this TU.
__device__ unsigned long long r6a_debug_count = 0ULL;

namespace r6a {

void reset_debug_counter() {
    unsigned long long zero = 0ULL;
    cudaError_t err = cudaMemcpyToSymbol(
        r6a_debug_count, &zero, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice
    );
    if (err != cudaSuccess) {
        fprintf(stderr,
            "r6a::reset_debug_counter: cudaMemcpyToSymbol failed: %s\n",
            cudaGetErrorString(err));
    }
}

uint64_t get_debug_counter() {
    unsigned long long val = 0ULL;
    cudaError_t err = cudaMemcpyFromSymbol(
        &val, r6a_debug_count, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost
    );
    if (err != cudaSuccess) {
        fprintf(stderr,
            "r6a::get_debug_counter: cudaMemcpyFromSymbol failed: %s\n",
            cudaGetErrorString(err));
    }
    return (uint64_t)val;
}

} // namespace r6a
} // namespace gsplat
