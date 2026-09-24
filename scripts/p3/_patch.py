#!/usr/bin/env python3
"""Minimal fix for nvcc: '#' preprocessor directives embedded inside the
HIGS_LAUNCH_BLEND_BWD_PX macro body are mis-parsed by nvcc. Move the
conditional sink argument OUT of the macro by always declaring / passing the
diagnostic scratch tensor. For P3==0 (FULL), the scratch is an empty unused
tensor and the kernel's sink block is compiled out, so behavior is unchanged."""
import sys

F = "/mnt/storage_pool/liaoyuanjun/higs_p3h_worktree_gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu"
src = open(F, encoding="utf-8").read()

def rep(old, new, n=1):
    global src
    c = src.count(old)
    if c != n:
        raise SystemExit("pattern count %d != %d for:\n%r" % (c, n, old))
    src = src.replace(old, new)

# 1) kernel signature: always declare the sink param (unused for P3==0)
rep(
    "    float *__restrict__ v_backgrounds\n"
    "#if P3_SINK_MODE != 0\n"
    "    ,\n"
    "    float *__restrict__ v_p3h_scratch // diagnostic uncontended sink slot\n"
    "#endif\n"
    ")",
    "    float *__restrict__ v_backgrounds,\n"
    "    float *__restrict__ v_p3h_scratch // diagnostic sink slot (always declared; unused when P3==0)\n"
    ")",
)

# 2) scratch creation: keep allocating in sinks, empty unused for P3==0
rep(
    "#if P3_SINK_MODE != 0\n"
    "        // Diagnostic scratch: 16 floats per warp, capped at 8 warps/block.\n"
    "        const int64_t p3h_n_blk = (int64_t)grid.x * (int64_t)grid.y * (int64_t)grid.z;\n"
    "        at::Tensor v_p3h_scratch = at::zeros({p3h_n_blk * 128}, opts);\n"
    "#endif",
    "#if P3_SINK_MODE != 0\n"
    "        // Diagnostic scratch: 16 floats per warp, capped at 8 warps/block.\n"
    "        const int64_t p3h_n_blk = (int64_t)grid.x * (int64_t)grid.y * (int64_t)grid.z;\n"
    "        at::Tensor v_p3h_scratch = at::zeros({p3h_n_blk * 128}, opts);\n"
    "#else\n"
    "        // P3==0 (FULL C0 V3): no global sink; pass an unused empty destination.\n"
    "        at::Tensor v_p3h_scratch = at::zeros({0}, opts);\n"
    "#endif",
)

# 3) launcher macro: always emit the sink argument
rep(
    "            v_backgrounds.data_ptr<float>()\n"
    "#if P3_SINK_MODE != 0\n"
    "            , v_p3h_scratch.data_ptr<float>()\n"
    "#endif\n"
    "            );",
    "            v_backgrounds.data_ptr<float>(),\n"
    "            v_p3h_scratch.data_ptr<float>()\n"
    "            );",
)

open(F, "w", encoding="utf-8").write(src)
print("PATCHED OK")