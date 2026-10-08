# Patch backward.h to add count_valid_g declaration
p = "rasterization/include/backward.h"
s = open(p).read()
# add declaration before closing namespace
add = """
    // BWD-ARITH1: count valid G evaluations in blend_backward_cu (exact N_valid_G).
    // Runs forward (to populate buffers) then a counting backward pass. Returns N_valid_G.
    unsigned int count_valid_g(
        const float* means,
        const float3* scales,
        const float4* rotations,
        const float* opacities,
        const float3* sh_coefficients_0,
        const float3* sh_coefficients_rest,
        const float4* w2c,
        const float3* cam_position,
        const float3* bg_color,
        char* primitive_buffers_blob,
        char* tile_buffers_blob,
        char* instance_buffers_blob,
        char* bucket_buffers_blob,
        const int n_primitives,
        const int n_instances,
        const int n_buckets,
        const int instance_primitive_indices_selector,
        const int active_sh_bases,
        const int total_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const bool proper_antialiasing);
"""
# insert before the final closing brace of namespace
idx = s.rfind("}")
s = s[:idx] + add + s[idx:]
open(p,"w").write(s)
print("backward.h patched")
