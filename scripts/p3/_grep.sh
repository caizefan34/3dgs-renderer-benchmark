#!/usr/bin/env bash
G=/mnt/storage_pool/liaoyuanjun/higs_p3h_worktree_gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference
echo "===== v_p3h_scratch refs in .cu ====="
grep -n "v_p3h_scratch\|P3_SINK_MODE" $G/HigsNativeBackward.cu
echo "===== header ups ====="
grep -n "v_p3h_scratch\|higs_blend_bwd_px_kernel\|P3_SINK_MODE" $G/HigsNativeBackward.h
echo "===== cpp binding reference to backward params/logic ====="
grep -rn "higs_rasterize_backward\|skip_background_atomic\|compacted" $G/../../ext.cpp | head