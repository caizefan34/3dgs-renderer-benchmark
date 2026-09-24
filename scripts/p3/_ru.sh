#!/bin/bash
SO1=/mnt/storage_pool/liaoyuanjun/higs_c0_cache_composed/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so
SO2=/mnt/storage_pool/liaoyuanjun/higs_p3h_cache/V0/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so
echo "########## resource-usage context PROD ##########"
cuobjdump --dump-resource-usage "$SO1" 2>/dev/null | grep -i -A8 -B2 'higs_blend_bwd_px_kernel' | head -40
echo "########## resource-usage context V0 ##########"
cuobjdump --dump-resource-usage "$SO2" 2>/dev/null | grep -i -A8 -B2 'higs_blend_bwd_px_kernel' | head -40
echo "########## launch config / smem dispatch (diag .cu) ##########"
CU=/mnt/storage_pool/liaoyuanjun/higs_p3h_worktree_gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu
grep -n 'cudaFuncSetAttribute\|cudaFuncAttributeMaxDynamicSharedMemorySize\|shmem_size\|dim3 grid\|dim3 block\|higs_blend_bwd_px_kernel<<<\|static constexpr int PX\|kBlockSize\|block_size\|tile_size \* tile_size' "$CU" | head -40