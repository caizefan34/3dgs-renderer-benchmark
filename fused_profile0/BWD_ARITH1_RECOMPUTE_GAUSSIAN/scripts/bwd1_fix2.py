#!/usr/bin/env python3
import os
ROOT = "/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH1-RECOMPUTE-GAUSSIAN/rasterization"
SYM_H  = os.path.join(ROOT, "include", "bwd1_cache_symbol.h")
BWD_CU = os.path.join(ROOT, "src", "backward.cu")
FWD_CU = os.path.join(ROOT, "src", "forward.cu")
def rd(p): return open(p).read()
def wr(p,s): open(p,'w').write(s)

# Fix 1: symbol header - use non-const float* so we can write the cache
wr(SYM_H, '#pragma once\n#include <cstdint>\nnamespace faster_gs { namespace rasterization { namespace kernels {\n    extern __device__ unsigned int bwd1_cache_enable;\n    extern __device__ float* bwd1_gauss_cache;\n} } }\n')
print("fixed symbol header (non-const)")

# Fix 2: backward.cu - remove early def, add after kernels_backward.cuh include
bc = rd(BWD_CU)
bc = bc.replace('\n// BWD1_R1_SYMDEF: cache control symbols (single definition)\nnamespace faster_gs::rasterization::kernels {\n    __device__ unsigned int bwd1_cache_enable = 0u;\n    __device__ const float* bwd1_gauss_cache = nullptr;\n}\n', '\n')
lines = bc.split('\n')
for i, l in enumerate(lines):
    if l.strip() == '#include "kernels_backward.cuh"':
        ins = '\n// BWD1_R1_SYMDEF: cache control symbols (single definition, after extern decl)\nnamespace faster_gs::rasterization::kernels {\n    __device__ unsigned int bwd1_cache_enable = 0u;\n    __device__ float* bwd1_gauss_cache = nullptr;\n}'
        lines.insert(i+1, ins)
        break
bc = '\n'.join(lines)
wr(BWD_CU, bc)
print("fixed backward.cu symbol def placement")

# Fix 3: forward.cu - non-const cache ptr
fc = rd(FWD_CU)
fc = fc.replace('const float* bwd1_cache_ptr = nullptr;', 'float* bwd1_cache_ptr = nullptr;')
fc = fc.replace('sizeof(const float*)', 'sizeof(float*)')
wr(FWD_CU, fc)
print("fixed forward.cu cache ptr type")
print("DONE")
