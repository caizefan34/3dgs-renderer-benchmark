#!/bin/bash
set -e
export PATH=/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin:/home/liaoyuanjun/.local/bin:$PATH
export CUDA_HOME=/mnt/storage_pool/liaoyuanjun/higs-13scene-env

SRC=/tmp/higs_h3_fwd_1a_source/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu
BUILD=/tmp/higs_h3_fwd_1a_build/experimental_gaussian_render_inference_scene_cuda
PATCH=/tmp/higs-scalar-adjoint.patch

echo "=== H5-0R Kernel Build ==="

# Step 1: Backup original source
cp "$SRC" "$SRC.orig"
echo "Backed up original source"

# Step 2: Apply scalar-adjoint patch
cd /tmp/higs_h3_fwd_1a_source
if git rev-parse --git-dir 2>/dev/null; then
    git checkout -- gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu 2>/dev/null || true
fi
patch -p1 --forward < "$PATCH" 2>&1 || {
    echo "Patch may already be applied, checking..."
    grep -q "HIGS_BWD_SCALAR_ADJOINT" "$SRC" && echo "Scalar-adjoint already applied" || {
        echo "ERROR: Failed to apply scalar-adjoint patch"
        exit 1
    }
}
echo "Scalar-adjoint patch applied"

# Step 3: Add H5 frontier variants
python3 << 'PYEOF'
import re

src_path = "/tmp/higs_h3_fwd_1a_source/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu"
with open(src_path, "r") as f:
    code = f.read()

# 1. Add FRONTIER template parameter to higs_blend_bwd_px_kernel
# Change: template<uint32_t CDIM, uint32_t PX, bool SCALAR_ADJOINT = false>
# To:     template<uint32_t CDIM, uint32_t PX, bool SCALAR_ADJOINT = false, int FRONTIER = 0>
code = code.replace(
    "template<uint32_t CDIM, uint32_t PX, bool SCALAR_ADJOINT = false>",
    "template<uint32_t CDIM, uint32_t PX, bool SCALAR_ADJOINT = false, int FRONTIER = 0>"
)

# 2. Add block_bin_final computation after warp_bin_final
# Find: warp_bin_final = cg::reduce(warp, warp_bin_final, cg::greater<int>());
# Add block_bin_final computation after it
old_warp_reduce = """    warp_bin_final = cg::reduce(warp, warp_bin_final, cg::greater<int>());

    for(uint32_t b = 0; b < num_batches; ++b)"""

new_warp_reduce = """    warp_bin_final = cg::reduce(warp, warp_bin_final, cg::greater<int>());

    // H5-0R: Block-level frontier computation
    // block_bin_final = max(last_ids[pixel]) over ALL valid pixels in tile
    int32_t block_bin_final = warp_bin_final;
    if constexpr(FRONTIER != 0)
    {
        __shared__ int32_t s_warp_max[8];
        const uint32_t warp_id = warp.meta_group_rank();
        if(warp.thread_rank() == 0) s_warp_max[warp_id] = warp_bin_final;
        block.sync();
        block_bin_final = 0;
        const uint32_t n_warps = block.size() / 32;
        for(uint32_t i = 0; i < n_warps; ++i)
            block_bin_final = max(block_bin_final, s_warp_max[i]);
    }

    for(uint32_t b = 0; b < num_batches; ++b)"""

code = code.replace(old_warp_reduce, new_warp_reduce)

# 3. Add frontier load gating in the batch loop
# H5A (FRONTIER==1): skip entire batch if frontmost entry > block_bin_final
# H5B (FRONTIER==2): per-thread gate: only load if idx <= block_bin_final
# H5C (FRONTIER==3): same as H5B but with 32-G group awareness (same load count)
old_load = """        const int32_t idx        = batch_end - (int32_t)tr;
        if(idx >= range_start)
        {"""

new_load = """        const int32_t idx        = batch_end - (int32_t)tr;
        // H5-0R: Frontier-aware load gating
        if constexpr(FRONTIER == 1)
        {
            // H5A: skip entire batch if frontmost entry > block_bin_final
            const int32_t batch_front = batch_end - batch_size + 1;
            if(batch_front > block_bin_final)
            {
                block.sync();
                continue;
            }
        }
        if constexpr(FRONTIER == 2 || FRONTIER == 3)
        {
            // H5B/H5C: per-thread load gate
            if(idx > block_bin_final)
            {
                // This thread's entry is beyond the frontier — don't load
                // Still need to write dummy to shared mem to avoid UB
                id_batch[tr] = 0;
                xy_opacity_batch[tr] = {0.f, 0.f, 0.f};
                conic_batch[tr] = {0.f, 0.f, 0.f};
            }
            else if(idx >= range_start)
            {"""

code = code.replace(old_load, new_load)

# 4. Close the H5B/H5C else-if block — need to find the closing brace of the load block
# The original code has:
#         if(idx >= range_start)
#         {
#             ... loads ...
#         }
# We changed it to:
#         if constexpr(FRONTIER == 2 || FRONTIER == 3)
#         {
#             if(idx > block_bin_final) { ... }
#             else if(idx >= range_start)
#             {
#                 ... loads ...
#             }
#         }
#         (for FRONTIER == 0 or 1, the original if(idx >= range_start) still runs)
#
# Actually, this approach is getting complex. Let me use a simpler modification.
# For FRONTIER == 0 (baseline): original behavior
# For FRONTIER == 1 (H5A): check before load, skip batch
# For FRONTIER == 2 (H5B): check per-thread, skip load if beyond frontier
# For FRONTIER == 3 (H5C): same as H5B

# Let me rewrite the approach more carefully
# Revert and use a cleaner patch

PYEOF

echo "Python patch script complete (will use simpler approach)"
