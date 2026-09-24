#!/usr/bin/env python3
"""Revert the accidental extra argument on higs_background_bwd_kernel (line ~1535).

The launcher_macro_args replace-all matched 5 sites; only 4 are LAUNCH_BLEND_BWD
branches (px_safe 0/1/2/4). The 5th is the compacted background kernel, which
must keep its original 5-argument signature.
"""
import sys

P = "/mnt/storage_pool/liaoyuanjun/higs_c0_final30k_worktree/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu"

BAD = (
    "render_alphas.const_data_ptr<float>(), v_render_colors.const_data_ptr<float>(), "
    "v_backgrounds.data_ptr<float>(), v_means2d_abs_ptr);"
)
GOOD = (
    "render_alphas.const_data_ptr<float>(), v_render_colors.const_data_ptr<float>(), "
    "v_backgrounds.data_ptr<float>());"
)

with open(P, encoding="utf-8") as f:
    text = f.read()
n = text.count(BAD)
if n != 1:
    print(f"ANCHOR FAIL: found {n}, expected 1")
    sys.exit(2)
text = text.replace(BAD, GOOD)
with open(P, "w", encoding="utf-8") as f:
    f.write(text)

# verify: exactly 4 macro-branch injections remain, zero elsewhere
assert text.count("v_backgrounds.data_ptr<float>(), v_means2d_abs_ptr);") == 4
print("FIX OK: background kernel reverted; 4 macro branches retain the absgrad arg")
