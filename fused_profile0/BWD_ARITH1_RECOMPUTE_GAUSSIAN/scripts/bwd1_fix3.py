#!/usr/bin/env python3
import os
ROOT = "/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH1-RECOMPUTE-GAUSSIAN/rasterization"
BWD_CU = os.path.join(ROOT, "src", "backward.cu")
SYM_CU= os.path.join(ROOT, "src", "bwd1_cache_symbol.cu")
def rd(p): return open(p).read()
def wr(p,s): open(p,'w').write(s)

# Remove the definition from backward.cu (it's now in its own TU)
bc = rd(BWD_CU)
bc = bc.replace('\n// BWD1_R1_SYMDEF: cache control symbols (single definition, after extern decl)\nnamespace faster_gs::rasterization::kernels {\n    __device__ unsigned int bwd1_cache_enable = 0u;\n    __device__ float* bwd1_gauss_cache = nullptr;\n}\n', '\n')
wr(BWD_CU, bc)
print("removed symbol def from backward.cu")

# Create dedicated symbol TU
wr(SYM_CU, '#include "bwd1_cache_symbol.h"\n// BWD-ARITH1 R1: single definition of cache control symbols\nnamespace faster_gs::rasterization::kernels {\n    __device__ unsigned int bwd1_cache_enable = 0u;\n    __device__ float* bwd1_gauss_cache = nullptr;\n}\n')
print("created bwd1_cache_symbol.cu")
# Also add the symbol TU to setup_arith1.py
SETUP = "/mnt/storage_pool/liaoyuanjun/fused_profile0/BWD_ARITH1-RECOMPUTE-GAUSSIAN/setup_arith1.py"
s = rd(SETUP)
if "bwd1_cache_symbol.cu" not in s:
    s = s.replace(
        "        os.path.join(module_root, \"rasterization\", \"src\", \"backward.cu\"),",
        "        os.path.join(module_root, \"rasterization\", \"src\", \"backward.cu\"),\n        os.path.join(module_root, \"rasterization\", \"src\", \"bwd1_cache_symbol.cu\"),")
    wr(SETUP, s)
    print("added symbol TU to setup_arith1.py")
print("DONE")
