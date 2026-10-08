#pragma once

namespace faster_gs::rasterization {

// ATOM0 diagnostic launcher. Times ONLY the blend_backward_atom0_cu kernel
// (variant A/B/C) over a contiguous bucket run [chunk_first_bucket,
// chunk_first_bucket+chunk_n_buckets). Returns kernel elapsed time in ms.
//
// variant 0 (A): writes primitive-indexed atomicAdd grads -> grad_mean2d,
//   grad_conic, grad_opacity, grad_colors (all pre-allocated by caller).
// variant 1 (B): unique per-instance atomicAdd -> private_grad.
// variant 2 (C): unique per-instance plain store -> private_grad.
// partial_out (optional, caller-allocated [chunk_capacity,9] float) receives
// the per-instance pre-atomic partial gradients for the equivalence check.
double backward_atom0_launcher(
    const float* grad_image,
    const float* image,
    const float3* bg_color,
    char* primitive_buffers_blob,
    char* tile_buffers_blob,
    char* instance_buffers_blob,
    char* bucket_buffers_blob,
    int n_primitives,
    int width,
    int height,
    int n_instances,
    int n_buckets,
    int instance_primitive_indices_selector,
    int variant,
    int chunk_first_bucket,
    int chunk_n_buckets,
    void* grad_mean2d,
    void* grad_conic,
    void* grad_opacity,
    void* grad_colors,
    void* private_grad,
    float* partial_out,
    int capture_partials);

}
