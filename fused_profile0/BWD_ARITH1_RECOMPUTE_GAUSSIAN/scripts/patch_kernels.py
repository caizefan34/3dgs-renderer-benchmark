import re
p = "rasterization/include/kernels_backward.cuh"
s = open(p).read()
# 1) add global counter declaration after namespace open
needle = "namespace faster_gs::rasterization::kernels::backward {"
assert needle in s, "namespace needle not found"
s = s.replace(needle, needle + "\n\n    // BWD-ARITH1: global counter for valid G evaluations (nullptr = production)\n    unsigned int* g_bwd1_counter = nullptr;", 1)
# 2) increment after grad_alpha_common load (the leader-thread block)
anchor = "                grad_alpha_common = grad_info_pixel.w;\n            }"
assert anchor in s, "anchor not found"
s = s.replace(anchor, "                grad_alpha_common = grad_info_pixel.w;\n                if (g_bwd1_counter != nullptr) atomicAdd(g_bwd1_counter, 1u);\n            }", 1)
open(p,"w").write(s)
print("kernels_backward.cuh patched")
