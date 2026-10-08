p = "rasterization/include/kernels_backward.cuh"
s = open(p).read()
s = s.replace("    unsigned int* g_bwd1_counter = nullptr;",
              "    __device__ unsigned int* g_bwd1_counter = nullptr;")
open(p,"w").write(s)
print("device qual added:", "__device__ unsigned int* g_bwd1_counter" in s)
