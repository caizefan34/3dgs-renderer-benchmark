#!/usr/bin/env python3
"""H5-0R: Patch HigsNativeBackward.cu for frontier-aware load gating.

Prerequisites: scalar-adjoint patch must be applied first.
This script adds:
  - FRONTIER template parameter (0=baseline, 1=H5A, 2=H5B, 3=H5C)
  - block_bin_final: block-level max of last_ids (shared memory warp reduction)
  - H5A: skip entire batch if frontmost entry > block_bin_final
  - H5B/H5C: per-thread load gate if idx > block_bin_final
  - H5_FRONTIER env var for variant selection at runtime
  - Modified HIGS_LAUNCH_BLEND_BWD_PX macro to pass FRONTIER

Only 3 targeted text replacements in the kernel body + 3 in the launcher.
All patterns are unique to the px_kernel (not the non-px kernel).
"""
import sys, os, re
from pathlib import Path

SRC = Path("/tmp/higs_h3_fwd_1a_source/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu")

code = SRC.read_text()
n_edits = 0

def do_replace(old, new, desc, required=True):
    global code, n_edits
    if old in code:
        code = code.replace(old, new, 1)  # only first occurrence
        n_edits += 1
        print(f"  [OK] {desc}")
        return True
    else:
        if new.strip() in code or desc.endswith("already done"):
            print(f"  [SKIP] {desc} (already applied)")
            return False
        if required:
            print(f"  [FAIL] {desc}")
            # Show context for debugging
            lines = old.split('\n')
            first_line = lines[0].strip()
            for i, line in enumerate(code.split('\n')):
                if first_line in line:
                    print(f"    Found similar at line {i+1}: {line.rstrip()}")
            sys.exit(1)
        return False

# ── Edit 1: Add FRONTIER template parameter ──────────────────────────
print("Edit 1: FRONTIER template parameter")
do_replace(
    "template<uint32_t CDIM, uint32_t PX, bool SCALAR_ADJOINT = false>",
    "template<uint32_t CDIM, uint32_t PX, bool SCALAR_ADJOINT = false, int FRONTIER = 0>",
    "Add FRONTIER template param"
)

# ── Edit 2: Add block_bin_final computation ──────────────────────────
print("Edit 2: block_bin_final computation")
# This pattern is unique to px_kernel: uses warp_bin_final (not bin_final) and has blank line
do_replace(
    """    warp_bin_final = cg::reduce(warp, warp_bin_final, cg::greater<int>());

    for(uint32_t b = 0; b < num_batches; ++b)
    {
        block.sync();
        const int32_t batch_end  = range_end - 1 - (int32_t)(block_size * b);
        const int32_t batch_size = min((int32_t)block_size, batch_end + 1 - range_start);
        const int32_t idx        = batch_end - (int32_t)tr;
        if(idx >= range_start)
        {
            int32_t g            = flatten_ids[idx];""",
    """    warp_bin_final = cg::reduce(warp, warp_bin_final, cg::greater<int>());

    // H5-0R: block_bin_final = max(last_ids[pixel]) over ALL valid pixels in tile
    int32_t block_bin_final = warp_bin_final;
    if constexpr(FRONTIER != 0)
    {
        __shared__ int32_t s_h5_warp_max[8];
        const uint32_t h5_warp_id = warp.meta_group_rank();
        if(warp.thread_rank() == 0) s_h5_warp_max[h5_warp_id] = warp_bin_final;
        block.sync();
        block_bin_final = s_h5_warp_max[0];
        for(uint32_t i = 1; i < block.size() / 32; ++i)
            block_bin_final = max(block_bin_final, s_h5_warp_max[i]);
    }

    for(uint32_t b = 0; b < num_batches; ++b)
    {
        block.sync();
        const int32_t batch_end  = range_end - 1 - (int32_t)(block_size * b);
        const int32_t batch_size = min((int32_t)block_size, batch_end + 1 - range_start);
        const int32_t idx        = batch_end - (int32_t)tr;

        // H5-0R: H5A whole-batch skip
        bool h5_batch_dead = false;
        if constexpr(FRONTIER == 1)
        {
            const int32_t batch_front = batch_end - batch_size + 1;
            h5_batch_dead = (batch_front > block_bin_final);
        }

        // H5-0R: H5B/H5C per-thread load gate (FRONTIER < 2 = no gate)
        if(!h5_batch_dead && idx >= range_start && (FRONTIER < 2 || idx <= block_bin_final))
        {
            int32_t g            = flatten_ids[idx];""",
    "block_bin_final + frontier load gate"
)

# ── Edit 3: Guard processing loop with h5_batch_dead ─────────────────
print("Edit 3: Processing loop guard")
# Unique to px_kernel: blank line between block.sync() and for (non-px has comments)
do_replace(
    """        block.sync();

        for(uint32_t t = (uint32_t)max(0, batch_end - warp_bin_final); t < (uint32_t)batch_size; ++t)""",
    """        block.sync();

        if(!h5_batch_dead)
        for(uint32_t t = (uint32_t)max(0, batch_end - warp_bin_final); t < (uint32_t)batch_size; ++t)""",
    "Processing loop guard"
)

# ── Edit 4: Add H5_FRONTIER env var ──────────────────────────────────
print("Edit 4: H5_FRONTIER env var")
do_replace(
    """    const bool scalar_adjoint = strcmp(scalar_adjoint_name, "scalar_adjoint") == 0;
    TORCH_CHECK(
        scalar_adjoint || strcmp(scalar_adjoint_name, "baseline") == 0,
        "unsupported HIGS_BWD_SCALAR_ADJOINT: ",
        scalar_adjoint_name
    );""",
    """    const bool scalar_adjoint = strcmp(scalar_adjoint_name, "scalar_adjoint") == 0;
    TORCH_CHECK(
        scalar_adjoint || strcmp(scalar_adjoint_name, "baseline") == 0,
        "unsupported HIGS_BWD_SCALAR_ADJOINT: ",
        scalar_adjoint_name
    );
    // H5-0R: Frontier-aware load gating variant
    const char *h5_frontier_env = std::getenv("H5_FRONTIER");
    const int h5_frontier = (h5_frontier_env != nullptr) ? atoi(h5_frontier_env) : 0;
    TORCH_CHECK(
        h5_frontier >= 0 && h5_frontier <= 3,
        "unsupported H5_FRONTIER: expected 0/1/2/3, got ",
        h5_frontier
    );""",
    "H5_FRONTIER env var"
)

# ── Edit 5: Modify launch macro to pass FRONTIER ─────────────────────
print("Edit 5: Launch macro with FRONTIER")
do_replace(
    """#define HIGS_LAUNCH_BLEND_BWD_PX(CDIM, PX_VALUE, SCALAR_VALUE)                                    \\
    do                                                                                             \\
    {                                                                                              \\
        C10_CUDA_CHECK(cudaFuncSetAttribute(higs_blend_bwd_px_kernel<(CDIM), (PX_VALUE),          \\
                                                (SCALAR_VALUE)>,                                   \\
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shmem_size)));          \\
        higs_blend_bwd_px_kernel<(CDIM), (PX_VALUE), (SCALAR_VALUE)><<<grid, threads_px,          \\
            static_cast<size_t>(shmem_size), stream>>>(""",
    """#define HIGS_LAUNCH_BLEND_BWD_PX(CDIM, PX_VALUE, SCALAR_VALUE)                                    \\
    do                                                                                             \\
    {                                                                                              \\
        C10_CUDA_CHECK(cudaFuncSetAttribute(higs_blend_bwd_px_kernel<(CDIM), (PX_VALUE),          \\
                                                (SCALAR_VALUE), h5_frontier>,                       \\
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shmem_size)));          \\
        higs_blend_bwd_px_kernel<(CDIM), (PX_VALUE), (SCALAR_VALUE), h5_frontier><<<grid, threads_px, \\
            static_cast<size_t>(shmem_size), stream>>>(""",
    "Launch macro with FRONTIER"
)

# Write modified source
SRC.write_text(code)
print(f"\n{n_edits} edits applied. Source: {len(code)} bytes, {code.count(chr(10))} lines")

# Verify key markers exist
markers = ["block_bin_final", "h5_batch_dead", "H5_FRONTIER", "h5_frontier>", "FRONTIER < 2"]
for m in markers:
    if m in code:
        print(f"  ✓ Found marker: {m}")
    else:
        print(f"  ✗ MISSING marker: {m}")
        sys.exit(1)
