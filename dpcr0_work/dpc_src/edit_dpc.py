import io, sys

path = "/root/epic05_dpc0/dpc_src/HigsNativeBackwardDpc.cu"
with open(path, "r", encoding="utf-8", newline="") as f:
    text = f.read()
lines = text.split("\n")

# 1) add include after HigsNativeBackward.h include
inc_old = '#include "HigsNativeBackward.h"'
inc_new = '#include "HigsNativeBackward.h"\n#include "dpc_stage1.h"'
assert text.count(inc_old) == 1, "include anchor count %d" % text.count(inc_old)
text = text.replace(inc_old, inc_new, 1)

# 2) rename launcher definition (only one occurrence of the bare name + "(" at col 0)
defline = "higs_rasterize_backward("
assert text.count(defline) == 1, "defline count %d" % text.count(defline)
text = text.replace(defline, "higs_rasterize_backward_dpc(", 1)

# 3) replace Stage 1 section by locating markers
lines = text.split("\n")
start = None
end = None
for i, ln in enumerate(lines):
    if ln.strip() == "// ---- Stage 1: pixel-blend backward ----":
        start = i
    if ln.strip() == "// ---- Stage 2: projection VJP ----" and start is not None and end is None:
        end = i
assert start is not None and end is not None, "markers start=%s end=%s" % (start, end)
# the Stage-1 block runs from `start` to the line before `end`; keep one blank line before Stage 2
# find the closing brace of the stage-1 block: it is the last '}' before end
# We replace lines[start:end] (exclusive of the Stage-2 comment) with new content + blank line.
new_stage1 = '''    // ---- Stage 1 (DPC0): deterministic tile-primitive partial accumulation
    //      + Gaussian-major segmented reduction (no float atomicAdd to any
    //      final gradient). Produces the same flat [I*N] raster-gradient
    //      buffers the C0 projection/SH/reduce stages consume. ----
    if(n_isects > 0)
    {
        dpc_stage1(
            means2d, conics, colors_eval, opacities,
            backgrounds, tile_offsets, flatten_ids, active_tiles,
            render_alphas, last_ids, v_render_colors, v_render_alphas,
            I, N, n_isects, width, height, tile_size, tile_h, tile_w,
            static_cast<int64_t>(color_dim), scalar_adjoint, h8_mr,
            v_means2d, v_conics_out, v_colors_flat, v_opacities_flat);
    }
    if(backgrounds.has_value())
    {
        // Background gradient (d(render)/d(background) = T_final per pixel).
        // Computed with the C0 all-pixel kernel; this is the background INPUT
        // gradient (not a per-Gaussian raster gradient) and is outside the DPC0
        // raster-gradient determinism scope.
        const int64_t n_pixels = I * height * width;
        const uint32_t threads = 256;
        const uint32_t grid    = static_cast<uint32_t>((n_pixels + threads - 1) / threads);
        higs_background_bwd_kernel<<<grid, threads, 0, stream>>>(
            static_cast<uint32_t>(n_pixels), static_cast<uint32_t>(color_dim),
            render_alphas.const_data_ptr<float>(), v_render_colors.const_data_ptr<float>(),
            v_backgrounds.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
'''
# Replace from start (the Stage-1 comment) up to but not including end (Stage-2 comment).
# Keep a blank line separator before the Stage-2 comment.
new_lines = lines[:start] + new_stage1.split("\n")[:-1] + [""] + lines[end:]
text = "\n".join(new_lines)

with open(path, "w", encoding="utf-8", newline="") as f:
    f.write(text)

print("OK start_line(1based)=%d end_line(1based)=%d new_total_lines=%d" % (start+1, end+1, text.count(chr(10))+1))