#!/usr/bin/env python3
"""H5-0R: Patch HigsNativeBackward.cu to add frontier-aware load gating variants.

Modifications:
1. Apply scalar-adjoint patch (if not already applied)
2. Add FRONTIER template parameter (0=baseline, 1=H5A, 2=H5B, 3=H5C)
3. Add block_bin_final computation (block-level max of last_ids)
4. Add frontier load gating in the batch loop
5. Add H5_FRONTIER env var selection in the launcher
6. Rebuild with ninja
"""
import os, sys, subprocess, shutil
from pathlib import Path

SRC = Path("/tmp/higs_h3_fwd_1a_source/gsplat/experimental/render/kernels/cuda/csrc/gaussian_inference/HigsNativeBackward.cu")
BUILD_DIR = Path("/tmp/higs_h3_fwd_1a_build/experimental_gaussian_render_inference_scene_cuda")
PATCH = Path("/tmp/higs-scalar-adjoint.patch")
NINJA = "/home/liaoyuanjun/.local/bin/ninja"

def run(cmd, check=True, capture=False):
    r = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if check and r.returncode != 0:
        print(f"FAILED: {cmd}", file=sys.stderr)
        if capture:
            print(r.stdout, file=sys.stderr)
            print(r.stderr, file=sys.stderr)
        sys.exit(1)
    return r

# Step 1: Apply scalar-adjoint patch
print("=== Step 1: Apply scalar-adjoint patch ===")
code = SRC.read_text()
if "HIGS_BWD_SCALAR_ADJOINT" not in code:
    run(f"cd /tmp/higs_h3_fwd_1a_source && patch -p1 --forward < {PATCH}", check=False)
    code = SRC.read_text()
    if "HIGS_BWD_SCALAR_ADJOINT" not in code:
        print("ERROR: Failed to apply scalar-adjoint patch")
        sys.exit(1)
    print("Scalar-adjoint patch applied")
else:
    print("Scalar-adjoint patch already applied")

# Step 2: Add FRONTIER template parameter
print("\n=== Step 2: Add FRONTIER template parameter ===")
old_tpl = "template<uint32_t CDIM, uint32_t PX, bool SCALAR_ADJOINT = false>"
new_tpl = "template<uint32_t CDIM, uint32_t PX, bool SCALAR_ADJOINT = false, int FRONTIER = 0>"
if old_tpl in code:
    code = code.replace(old_tpl, new_tpl)
    print("Added FRONTIER template parameter")
else:
    if "int FRONTIER = 0" in code:
        print("FRONTIER template parameter already present")
    else:
        print(f"ERROR: Could not find template declaration")
        sys.exit(1)

# Step 3: Add block_bin_final computation after warp_bin_final
print("\n=== Step 3: Add block_bin_final computation ===")
# The warp_bin_final reduction line appears twice (in higs_blend_bwd_kernel and higs_blend_bwd_px_kernel)
# We only want to modify the px_kernel version. Find it by looking for the pattern that's
# followed by the batch loop.
old_block = """    warp_bin_final = cg::reduce(warp, warp_bin_final, cg::greater<int>());

    for(uint32_t b = 0; b < num_batches; ++b)
    {
        block.sync();
        const int32_t batch_end  = range_end - 1 - (int32_t)(block_size * b);
        const int32_t batch_size = min((int32_t)block_size, batch_end + 1 - range_start);
        const int32_t idx        = batch_end - (int32_t)tr;
        if(idx >= range_start)
        {"""

new_block = """    warp_bin_final = cg::reduce(warp, warp_bin_final, cg::greater<int>());

    // H5-0R: Block-level frontier = max(last_ids[pixel]) over ALL valid pixels in tile
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

        // H5-0R: Frontier-aware load gating
        bool h5_batch_dead = false;
        if constexpr(FRONTIER == 1)
        {
            // H5A: skip entire batch if frontmost entry > block_bin_final
            const int32_t batch_front = batch_end - batch_size + 1;
            h5_batch_dead = (batch_front > block_bin_final);
        }

        if(!h5_batch_dead)
        {
            bool h5_thread_load = true;
            if constexpr(FRONTIER == 2 || FRONTIER == 3)
            {
                // H5B/H5C: per-thread load gate — skip load if entry beyond frontier
                h5_thread_load = (idx <= block_bin_final);
            }
            if(h5_thread_load && idx >= range_start)
            {"""

if old_block in code:
    code = code.replace(old_block, new_block)
    print("Added block_bin_final and frontier load gating")
else:
    if "block_bin_final" in code:
        print("block_bin_final already present")
    else:
        print("ERROR: Could not find the batch loop pattern")
        # Try to find a close match
        import re
        matches = re.findall(r'warp_bin_final = cg::reduce.*?for\(uint32_t b = 0;', code, re.DOTALL)
        for i, m in enumerate(matches):
            print(f"  Match {i}: {repr(m[:100])}...")
        sys.exit(1)

# Step 4: Close the if(!h5_batch_dead) block
# The original code after the load block has:
#         }
#         block.sync();
# We need to add a closing brace for the if(!h5_batch_dead) block
print("\n=== Step 4: Close if(!h5_batch_dead) block ===")
# Find the closing of the load block followed by block.sync()
old_close = """        }
        block.sync();

        for(uint32_t t = (uint32_t)max(0, batch_end - warp_bin_final); t < (uint32_t)batch_size; ++t)"""

new_close = """            }
        }
        block.sync();

        if(!h5_batch_dead)
        for(uint32_t t = (uint32_t)max(0, batch_end - warp_bin_final); t < (uint32_t)batch_size; ++t)"""

if old_close in code:
    code = code.replace(old_close, new_close)
    print("Closed if(!h5_batch_dead) block")
else:
    if "h5_batch_dead" in code and code.count("h5_batch_dead") > 2:
        print("if(!h5_batch_dead) closure already present")
    else:
        print("ERROR: Could not find the closing pattern")
        sys.exit(1)

# Step 5: Add H5_FRONTIER env var in the launcher
print("\n=== Step 5: Add H5_FRONTIER env var ===")
# Find the scalar_adjoint env var parsing and add H5_FRONTIER after it
old_env = """    const bool scalar_adjoint = strcmp(scalar_adjoint_name, "scalar_adjoint") == 0;
    TORCH_CHECK(
        scalar_adjoint || strcmp(scalar_adjoint_name, "baseline") == 0,
        "unsupported HIGS_BWD_SCALAR_ADJOINT: ",
        scalar_adjoint_name
    );"""

new_env = """    const bool scalar_adjoint = strcmp(scalar_adjoint_name, "scalar_adjoint") == 0;
    TORCH_CHECK(
        scalar_adjoint || strcmp(scalar_adjoint_name, "baseline") == 0,
        "unsupported HIGS_BWD_SCALAR_ADJOINT: ",
        scalar_adjoint_name
    );
    // H5-0R: Frontier-aware load gating variant selection
    const char *h5_frontier_env = std::getenv("H5_FRONTIER");
    const int h5_frontier = (h5_frontier_env != nullptr) ? atoi(h5_frontier_env) : 0;
    TORCH_CHECK(
        h5_frontier >= 0 && h5_frontier <= 3,
        "unsupported H5_FRONTIER: expected 0/1/2/3, got ",
        h5_frontier
    );"""

if old_env in code:
    code = code.replace(old_env, new_env)
    print("Added H5_FRONTIER env var")
else:
    if "H5_FRONTIER" in code:
        print("H5_FRONTIER env var already present")
    else:
        print("ERROR: Could not find scalar_adjoint env var parsing")
        sys.exit(1)

# Step 6: Modify the launcher to pass FRONTIER template parameter
print("\n=== Step 6: Modify launcher for FRONTIER ===")
# The launcher uses HIGS_LAUNCH_BLEND_BWD_PX(CDIM, PX_VALUE, SCALAR_VALUE)
# We need to add the FRONTIER parameter
old_launch = """#define HIGS_LAUNCH_BLEND_BWD_PX(CDIM, PX_VALUE, SCALAR_VALUE)                                    \\
    do                                                                                             \\
    {                                                                                              \\
        C10_CUDA_CHECK(cudaFuncSetAttribute(higs_blend_bwd_px_kernel<(CDIM), (PX_VALUE),          \\
                                                (SCALAR_VALUE)>,                                   \\
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shmem_size)));          \\
        higs_blend_bwd_px_kernel<(CDIM), (PX_VALUE), (SCALAR_VALUE)><<<grid, threads_px,          \\
            static_cast<size_t>(shmem_size), stream>>>("""

new_launch = """#define HIGS_LAUNCH_BLEND_BWD_PX(CDIM, PX_VALUE, SCALAR_VALUE)                                    \\
    do                                                                                             \\
    {                                                                                              \\
        C10_CUDA_CHECK(cudaFuncSetAttribute(higs_blend_bwd_px_kernel<(CDIM), (PX_VALUE),          \\
                                                (SCALAR_VALUE), h5_frontier>,                       \\
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shmem_size)));          \\
        higs_blend_bwd_px_kernel<(CDIM), (PX_VALUE), (SCALAR_VALUE), h5_frontier><<<grid, threads_px, \\
            static_cast<size_t>(shmem_size), stream>>>("""

if old_launch in code:
    code = code.replace(old_launch, new_launch)
    print("Modified launcher for FRONTIER")
else:
    if "h5_frontier>" in code:
        print("Launcher already modified for FRONTIER")
    else:
        print("ERROR: Could not find launcher macro")
        sys.exit(1)

# Write the modified source
SRC.write_text(code)
print(f"\nModified source written to {SRC}")
print(f"Source size: {len(code)} bytes, {code.count(chr(10))} lines")

# Step 7: Rebuild with ninja
print("\n=== Step 7: Rebuild with ninja ===")
# Touch the source to force recompilation
os.utime(str(SRC), None)
result = subprocess.run(
    f"cd {BUILD_DIR} && {NINJA} -j1 2>&1",
    shell=True, capture_output=True, text=True
)
print("Build stdout (last 2000 chars):", result.stdout[-2000:])
if result.returncode != 0:
    print("Build stderr (last 3000 chars):", result.stderr[-3000:])
    print(f"Build FAILED with exit code {result.returncode}")
    sys.exit(1)
print("Build succeeded!")

# Verify the .so was rebuilt
so_path = BUILD_DIR / "experimental_gaussian_render_inference_scene_cuda.so"
import time
mtime = os.path.getmtime(str(so_path))
print(f".so modified at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}")
