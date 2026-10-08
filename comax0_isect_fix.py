#!/usr/bin/env python3
import os
SRC = "/mnt/storage_pool/liaoyuanjun/comax0_mx/COMAX0_CUDA_PROD/src"
p = os.path.join(SRC, "comax_envelope_kernel.cu")
s = open(p, encoding="utf-8").read()
old1 = """    const float* __restrict__ background, // [3]
    const int* __restrict__ tile_ids, const int* __restrict__ tile_offsets,
    float* __restrict__ out_rgb, float* __restrict__ out_alpha,"""
new1 = """    const float* __restrict__ background, // [3]
    float* __restrict__ out_rgb, float* __restrict__ out_alpha,"""
assert old1 in s, "bg block"
s = s.replace(old1, new1, 1)
old2 = """    const int*   __restrict__ node_pow2,  // [G] pow2 per group
    const int    n_tiles, const int tw, const int th,"""
new2 = """    const int*   __restrict__ node_pow2,  // [G] pow2 per group
    const int* __restrict__ tile_ids, const int* __restrict__ tile_offsets,
    const int    n_tiles, const int tw, const int th,"""
assert old2 in s, "pow2 block"
s = s.replace(old2, new2, 1)
open(p, "w", encoding="utf-8").write(s)
print("FIX_OK")
