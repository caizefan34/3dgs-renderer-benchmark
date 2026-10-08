#!/usr/bin/env python3
# BWD-ARITH1 forward.cu patch: allocate gaussian_cache inside forward(), set device symbols.
import os
ROOT = "/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH1-RECOMPUTE-GAUSSIAN/rasterization"
FWD_CU = os.path.join(ROOT, "src", "forward.cu")
def rd(p): return open(p).read()
def wr(p,s): open(p,'w').write(s)
def rep(s, old, new, tag):
    if old not in s: raise SystemExit("ANCHOR MISSING: " + tag)
    return s.replace(old, new, 1)

fc = rd(FWD_CU)
if "BWD1_R1" not in fc:
    # add includes after last include
    fc = rep(fc, '#include "forward.h"',
                '#include "forward.h"\n#include "bwd1_cache_symbol.h"\n#include <cuda_runtime.h>\n#include <cstdlib>\n\n// BWD1_R1: keep the exact Gaussian response cache alive between forward() and backward() calls\nstatic torch::Tensor bwd1_cache_keepalive;',
        "forward.cu includes")
    # after n_buckets known, alloc cache + set symbols (before blend_cu launch)
    fc = rep(fc,
        '    int n_buckets;\n    cudaMemcpy(&n_buckets, tile_buffers.buckets_offset + n_tiles - 1, sizeof(uint), cudaMemcpyDeviceToHost);',
        '''    int n_buckets;
    cudaMemcpy(&n_buckets, tile_buffers.buckets_offset + n_tiles - 1, sizeof(uint), cudaMemcpyDeviceToHost);

    // BWD1_R1: exact FP32 Gaussian response cache control (env BWD1_CACHE_ENABLE=1 => R1)
    const char* bwd1_env = std::getenv("BWD1_CACHE_ENABLE");
    const unsigned int bwd1_enable = (bwd1_env != nullptr && std::atoi(bwd1_env) != 0) ? 1u : 0u;
    const float* bwd1_cache_ptr = nullptr;
    if (bwd1_enable) {
        const torch::Tensor gaussian_cache = torch::empty({(int64_t)n_buckets * 32 * (int64_t)config::block_size_blend}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
        bwd1_cache_ptr = gaussian_cache.data_ptr<float>();
        bwd1_cache_keepalive = gaussian_cache;
    }
    cudaMemcpyToSymbol(faster_gs::rasterization::kernels::bwd1_gauss_cache, &bwd1_cache_ptr, sizeof(const float*));
    cudaMemcpyToSymbol(faster_gs::rasterization::kernels::bwd1_cache_enable, &bwd1_enable, sizeof(unsigned int));''',
        "forward.cu cache control")
    wr(FWD_CU, fc); print("patched forward.cu")
else:
    print("forward.cu already patched")
print("FORWARD PATCH DONE")
