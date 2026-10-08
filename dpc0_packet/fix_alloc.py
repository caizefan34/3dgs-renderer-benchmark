path = "/root/epic05_dpc0/src_full/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackwardDpc.cu"
with open(path) as f:
    s = f.read()

# 1. remove the unconditional (out-of-scope D) allocation
alloc = "    at::Tensor part_shmeans = at::empty({I * N * D, 3}, opts);\n"
assert alloc in s, "alloc line not found"
s = s.replace(alloc, "", 1)

# 2. add allocation right after D is defined in the SH branch
anchor = "        const int64_t D = sh_coeffs.value().size(-1);\n"
assert anchor in s, "D anchor not found"
s = s.replace(anchor, anchor + "        at::Tensor part_shmeans = at::empty({I * N * D, 3}, opts);\n", 1)

with open(path, "w") as f:
    f.write(s)
print("OK allocation moved")