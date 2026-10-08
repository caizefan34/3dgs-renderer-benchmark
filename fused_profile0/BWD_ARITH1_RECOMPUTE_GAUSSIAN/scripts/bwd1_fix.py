#!/usr/bin/env python3
import os
ROOT = "/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH1-RECOMPUTE-GAUSSIAN/rasterization"
FWD_CU = os.path.join(ROOT, "src", "forward.cu")
SYM_H  = os.path.join(ROOT, "include", "bwd1_cache_symbol.h")
def rd(p): return open(p).read()
def wr(p,s): open(p,'w').write(s)

# Fix symbol header: use extern "C" style? No - use plain extern __device__ with include guard.
# The redefinition error is because the header is included in backward.cu which ALSO defines them.
# Solution: header only declares (extern); backward.cu defines. Ensure header has include guard.
wr(SYM_H, '#pragma once\n#include <cstdint>\nnamespace faster_gs { namespace rasterization { namespace kernels {\n    extern __device__ unsigned int bwd1_cache_enable;\n    extern __device__ const float* bwd1_gauss_cache;\n} } }\n')
print("fixed symbol header")

# Fix forward.cu: ensure torch/extension.h is included before the keepalive static
fc = rd(FWD_CU)
# remove the misplaced keepalive + bwd1_cache_symbol include near top, re-add after helper_math
fc = fc.replace('#include "forward.h"\n#include "bwd1_cache_symbol.h"\n#include <cuda_runtime.h>\n#include <cstdlib>\n\n// BWD1_R1: keep the exact Gaussian response cache alive between forward() and backward() calls\nstatic torch::Tensor bwd1_cache_keepalive;\n', '#include "forward.h"\n#include <cuda_runtime.h>\n#include <cstdlib>\n')
# add after helper_math include
fc = fc.replace('#include "helper_math.h"\n', '#include "helper_math.h"\n#include <torch/extension.h>\n#include "bwd1_cache_symbol.h"\n\n// BWD1_R1: keep the exact Gaussian response cache alive between forward() and backward() calls\nstatic torch::Tensor bwd1_cache_keepalive;\n', 1)
wr(FWD_CU, fc)
print("fixed forward.cu includes")
print("DONE")
