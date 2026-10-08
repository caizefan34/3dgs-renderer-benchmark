#!/usr/bin/env python3
# BWD-ARITH1 R1 patch v2: exact FP32 Gaussian response cache via extern device symbol.
import os
ROOT = "/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH1-RECOMPUTE-GAUSSIAN/rasterization"
FWD_H  = os.path.join(ROOT, "include", "kernels_forward.cuh")
BWD_H  = os.path.join(ROOT, "include", "kernels_backward.cuh")
BWD_CU = os.path.join(ROOT, "src", "backward.cu")
SYM_H  = os.path.join(ROOT, "include", "bwd1_cache_symbol.h")
def rd(p): return open(p).read()
def wr(p,s): open(p,'w').write(s)
def rep(s, old, new, tag):
    if old not in s: raise SystemExit("ANCHOR MISSING: " + tag)
    return s.replace(old, new, 1)

wr(SYM_H, '#pragma once\n#include <cstdint>\nnamespace faster_gs::rasterization::kernels {\n    extern __device__ unsigned int bwd1_cache_enable;\n    extern __device__ const float* bwd1_gauss_cache;\n}\n')
print("wrote bwd1_cache_symbol.h")

f = rd(FWD_H)
if "BWD1_R1" not in f:
    f = rep(f, '#include "rasterization_config.h"', '#include "rasterization_config.h"\n#include "bwd1_cache_symbol.h"', "fwd include")
    f = rep(f,
        '                const float gaussian = expf(fminf(exponent, 0.0f));\n                const float alpha = opacity * gaussian;',
        '                const float gaussian = expf(fminf(exponent, 0.0f));\n                // BWD1_R1: store exact FP32 Gaussian response for backward reuse\n                if (kernels::bwd1_cache_enable != 0u) {\n                    const uint cidx = bucket_offset * (config::block_size_blend * config::block_size_blend) + (j % config::block_size_blend) * config::block_size_blend + thread_rank;\n                    kernels::bwd1_gauss_cache[cidx] = gaussian;\n                }\n                const float alpha = opacity * gaussian;',
        "fwd cache write")
    wr(FWD_H, f); print("patched kernels_forward.cuh")
else: print("kernels_forward.cuh already patched")

b = rd(BWD_H)
if "BWD1_R1" not in b:
    b = rep(b, '#include "rasterization_config.h"', '#include "rasterization_config.h"\n#include "bwd1_cache_symbol.h"', "bwd include")
    b = rep(b,
        '        const float gaussian = expf(fminf(exponent, 0.0f));',
        '        float gaussian;\n        // BWD1_R1: load exact Gaussian response computed in forward, else recompute (production R0)\n        if (kernels::bwd1_cache_enable != 0u) {\n            const uint cidx = bucket_idx * (config::block_size_blend * config::block_size_blend) + lane_idx * config::block_size_blend + idx;\n            gaussian = kernels::bwd1_gauss_cache[cidx];\n        } else {\n            gaussian = expf(fminf(exponent, 0.0f));\n        }',
        "bwd cache read")
    wr(BWD_H, b); print("patched kernels_backward.cuh")
else: print("kernels_backward.cuh already patched")

bc = rd(BWD_CU)
if "BWD1_R1_SYMDEF" not in bc:
    anchor = '#include "backward.h"'
    assert anchor in bc
    bc = bc.replace(anchor, anchor + '\n\n// BWD1_R1_SYMDEF: cache control symbols (single definition)\nnamespace faster_gs::rasterization::kernels {\n    __device__ unsigned int bwd1_cache_enable = 0u;\n    __device__ const float* bwd1_gauss_cache = nullptr;\n}', 1)
    wr(BWD_CU, bc); print("patched backward.cu (symbol def)")
else: print("backward.cu already has symbol def")
print("SYMBOL PATCH DONE")
