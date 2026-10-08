#pragma once
namespace faster_gs { namespace rasterization { namespace kernels {
    // BWD-ARITH1 R1: host-side globals shared between forward (allocates) and backward (reads).
    // Passed as kernel parameters to avoid cudaMemcpyToSymbol (invalid for dlopen'd extensions).
    extern float* bwd1_gauss_cache_host;
    extern unsigned int bwd1_cache_enable_host;
} } }
