# Append count_valid_g implementation to backward.cu
p = "rasterization/src/backward.cu"
s = open(p).read()
impl = """

// BWD-ARITH1: count valid G evaluations. Buffers must be pre-populated by forward().
// Sets the global counter, launches blend_backward_cu, reads back the count.
unsigned int faster_gs::rasterization::count_valid_g(
    const float* grad_image,
    const float* image,
    float3* means,
    float3* scales,
    float4* rotations,
    float* opacities,
    float3* sh_coefficients_0,
    float3* sh_coefficients_rest,
    float2* moments_means,
    float2* moments_scales,
    float2* moments_rotations,
    float2* moments_opacities,
    float2* moments_sh_coefficients_0,
    float2* moments_sh_coefficients_rest,
    const float4* w2c,
    const float3* cam_position,
    const float3* bg_color,
    char* primitive_buffers_blob,
    char* tile_buffers_blob,
    char* instance_buffers_blob,
    char* bucket_buffers_blob,
    float* grad_opacities,
    float3* grad_colors,
    float2* grad_mean2d_helper,
    float* grad_conic_helper,
    float* densification_info,
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
    const float current_mean_lr,
    const int adam_step_count,
    unsigned int* counter_out)
{
    using namespace faster_gs::rasterization::config;
    const dim3 grid(div_round_up(width, tile_width), div_round_up(height, tile_height), 1);
    const int n_tiles = grid.x * grid.y;
    const int end_bit = extract_end_bit(n_tiles - 1);

    PrimitiveBuffers primitive_buffers = PrimitiveBuffers::from_blob(primitive_buffers_blob, n_primitives);
    TileBuffers tile_buffers = TileBuffers::from_blob(tile_buffers_blob, n_tiles);
    InstanceBuffers instance_buffers = InstanceBuffers::from_blob(instance_buffers_blob, n_instances, end_bit);
    BucketBuffers bucket_buffers = BucketBuffers::from_blob(bucket_buffers_blob, n_buckets);
    instance_buffers.primitive_indices.selector = instance_primitive_indices_selector;

    // reset counter
    cudaMemset(counter_out, 0, sizeof(unsigned int));
    kernels::backward::g_bwd1_counter = counter_out;

    // dummy grad buffers (not used by blend for counting, but blend writes grad_*)
    std::vector<float> dummy_grad_opacities(n_primitives, 0.0f);
    std::vector<float3> dummy_grad_colors(n_primitives, make_float3(0.0f,0.0f,0.0f));
    std::vector<float2> dummy_grad_mean2d(n_primitives, make_float2(0.0f,0.0f));
    std::vector<float> dummy_grad_conic(3*n_primitives, 0.0f);
    std::vector<float> dummy_densify(n_primitives, 0.0f);

    kernels::backward::blend_backward_cu<<<n_buckets, 32>>>(
        tile_buffers.instance_ranges,
        tile_buffers.buckets_offset,
        instance_buffers.primitive_indices.Current(),
        primitive_buffers.mean2d,
        primitive_buffers.conic_opacity,
        primitive_buffers.color,
        bg_color,
        grad_image,
        image,
        tile_buffers.final_transmittances,
        tile_buffers.max_n_processed,
        tile_buffers.n_processed,
        bucket_buffers.tile_index,
        bucket_buffers.color_transmittance,
        dummy_grad_mean2d.data(),
        dummy_grad_conic.data(),
        dummy_grad_opacities.data(),
        dummy_grad_colors.data(),
        n_primitives,
        width,
        height,
        grid.x
    );
    CHECK_CUDA(config::debug, "count_valid_g blend_backward");
    kernels::backward::g_bwd1_counter = nullptr;

    unsigned int h_count = 0;
    cudaMemcpy(&h_count, counter_out, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    return h_count;
}
"""
s = s.rstrip() + "\n" + impl
open(p,"w").write(s)
print("backward.cu patched, appended count_valid_g")
