path = "/root/epic05_dpc0/src_full/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackwardDpc.cu"
with open(path) as f:
    s = f.read()
old = "        at::Tensor part_coeffs = at::empty({I * N * K * D}, opts);"
new = "        at::Tensor part_coeffs = at::zeros({I * N * K * D}, opts);"
assert old in s, "part_coeffs alloc not found"
s = s.replace(old, new, 1)
with open(path, "w") as f:
    f.write(s)
print("OK part_coeffs -> zeros")