#!/usr/bin/env python3
# BWD-ARITH1 API patch v2: allocate gaussian_cache, control via env var BWD1_CACHE_ENABLE.
import os
ROOT = "/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH1-RECOMPUTE-GAUSSIAN/rasterization"
API_CU = os.path.join(ROOT, "src", "rasterization_api.cu")
def rd(p): return open(p).read()
def wr(p,s): open(p,'w').write(s)
def rep(s, old, new, tag):
    if old not in s: raise SystemExit("ANCHOR MISSING: " + tag)
    return s.replace(old, new, 1)

a = rd(API_CU)
if "BWD1_R1" not in a:
    a = rep(a, '#include "rasterization_api.h"',
                '#include "rasterization_api.h"\n#include "bwd1_cache_symbol.h"\n#include <cuda_runtime.h>\n#include <cstdlib>\n#include "rasterization_config.h"\n\n// BWD1_R1: keep the cache tensor alive between forward() and backward() calls\nstatic torch::Tensor bwd1_cache_keepalive;',
        "api includes + keepalive")
    a = rep(a,
        '    int n_buckets;\n    cudaMemcpy(&n_buckets, tile_buffers.buckets_offset + n_tiles - 1, sizeof(uint), cudaMemcpyDeviceToHost);',
        '''    int n_buckets;
    cudaMemcpy(&n_buckets, tile_buffers.buckets_offset + n_tiles - 1, sizeof(uint), cudaMemcpyDeviceToHost);

    // BWD1_R1: exact FP32 Gaussian response cache control
    const char* bwd1_env = std::getenv("BWD1_CACHE_ENABLE");
    const unsigned int bwd1_enable = (bwd1_env != nullptr && std::atoi(bwd1_env) != 0) ? 1u : 0u;
    const float* bwd1_cache_ptr = nullptr;
    if (bwd1_enable) {
        const torch::Tensor gaussian_cache = torch::empty({(int64_t)n_buckets * 32 * (int64_t)config::block_size_blend}, float_options);
        bwd1_cache_ptr = gaussian_cache.data_ptr<float>();
        bwd1_cache_keepalive = gaussian_cache;  // keep alive for backward
    }
    cudaMemcpyToSymbol(faster_gs::rasterization::kernels::bwd1_gauss_cache, &bwd1_cache_ptr, sizeof(const float*));
    cudaMemcpyToSymbol(faster_gs::rasterization::kernels::bwd1_cache_enable, &bwd1_enable, sizeof(unsigned int));''',
        "forward_wrapper cache control")
    wr(API_CU, a); print("patched rasterization_api.cu")
else:
    print("rasterization_api.cu already patched")
print("API PATCH DONE")
