#include <torch/extension.h>
#include "rasterization_api.h"
#include "forward.h"
#include "backward.h"
#include "atom0_api.h"
#include "torch_utils.h"
#include "rasterization_config.h"
#include "helper_math.h"
#include <stdexcept>
#include <functional>
#include <tuple>

namespace rasterization_api = faster_gs::rasterization;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int>
forward_wrapper(
    const torch::Tensor& means, const torch::Tensor& scales, const torch::Tensor& rotations,
    const torch::Tensor& opacities, const torch::Tensor& sh_coefficients_0, const torch::Tensor& sh_coefficients_rest,
    const torch::Tensor& w2c, const torch::Tensor& cam_position, const torch::Tensor& bg_color,
    const int active_sh_bases, const int width, const int height,
    const float focal_x, const float focal_y, const float center_x, const float center_y,
    const float near_plane, const float far_plane, const float current_mean_lr, const int adam_step_count)
{
    const int n_primitives = means.size(0);
    const int total_sh_bases = sh_coefficients_rest.size(1);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor image = torch::empty({3, height, width}, float_options);
    torch::Tensor primitive_buffers = torch::empty({0}, byte_options);
    torch::Tensor tile_buffers = torch::empty({0}, byte_options);
    torch::Tensor instance_buffers = torch::empty({0}, byte_options);
    torch::Tensor bucket_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> resize_primitive_buffers = resize_function_wrapper(primitive_buffers);
    const std::function<char*(size_t)> resize_tile_buffers = resize_function_wrapper(tile_buffers);
    const std::function<char*(size_t)> resize_instance_buffers = resize_function_wrapper(instance_buffers);
    const std::function<char*(size_t)> resize_bucket_buffers = resize_function_wrapper(bucket_buffers);

    auto [n_instances, n_buckets, instance_primitive_indices_selector] = rasterization_api::forward(
        resize_primitive_buffers, resize_tile_buffers, resize_instance_buffers, resize_bucket_buffers,
        reinterpret_cast<float3*>(means.data_ptr<float>()),
        reinterpret_cast<float3*>(scales.data_ptr<float>()),
        reinterpret_cast<float4*>(rotations.data_ptr<float>()),
        opacities.data_ptr<float>(),
        reinterpret_cast<float3*>(sh_coefficients_0.data_ptr<float>()),
        reinterpret_cast<float3*>(sh_coefficients_rest.data_ptr<float>()),
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(bg_color.contiguous().data_ptr<float>()),
        image.data_ptr<float>(), n_primitives, active_sh_bases, total_sh_bases,
        width, height, focal_x, focal_y, center_x, center_y, near_plane, far_plane);

    return { image, primitive_buffers, tile_buffers, instance_buffers, bucket_buffers,
             n_instances, n_buckets, instance_primitive_indices_selector };
}

std::tuple<double, torch::Tensor>
backward_atom0(
    torch::Tensor grad_image, torch::Tensor image, torch::Tensor bg_color,
    torch::Tensor primitive_buffers, torch::Tensor tile_buffers,
    torch::Tensor instance_buffers, torch::Tensor bucket_buffers,
    int n_primitives, int width, int height, int n_instances, int n_buckets,
    int instance_primitive_indices_selector, int variant, int chunk_first_bucket, int chunk_n_buckets,
    torch::Tensor grad_mean2d, torch::Tensor grad_conic, torch::Tensor grad_opacity, torch::Tensor grad_colors,
    torch::Tensor private_grad, torch::Tensor partial_out, int capture_partials)
{
    const float* grad_image_ptr = grad_image.contiguous().data_ptr<float>();
    const float* image_ptr = image.data_ptr<float>();
    const float3* bg_color_ptr = reinterpret_cast<const float3*>(bg_color.contiguous().data_ptr<float>());
    char* primitive_ptr = reinterpret_cast<char*>(primitive_buffers.data_ptr());
    char* tile_ptr = reinterpret_cast<char*>(tile_buffers.data_ptr());
    char* instance_ptr = reinterpret_cast<char*>(instance_buffers.data_ptr());
    char* bucket_ptr = reinterpret_cast<char*>(bucket_buffers.data_ptr());

    void* g_mean2d = (grad_mean2d.defined() && grad_mean2d.numel() > 0) ? reinterpret_cast<void*>(grad_mean2d.data_ptr<float>()) : nullptr;
    void* g_conic = (grad_conic.defined() && grad_conic.numel() > 0) ? reinterpret_cast<void*>(grad_conic.data_ptr<float>()) : nullptr;
    void* g_opacity = (grad_opacity.defined() && grad_opacity.numel() > 0) ? reinterpret_cast<void*>(grad_opacity.data_ptr<float>()) : nullptr;
    void* g_colors = (grad_colors.defined() && grad_colors.numel() > 0) ? reinterpret_cast<void*>(grad_colors.data_ptr<float>()) : nullptr;
    void* priv = (private_grad.defined() && private_grad.numel() > 0) ? reinterpret_cast<void*>(private_grad.data_ptr<float>()) : nullptr;
    float* partial_ptr = (partial_out.defined() && partial_out.numel() > 0) ? partial_out.data_ptr<float>() : nullptr;

    double ms = rasterization_api::backward_atom0_launcher(
        grad_image_ptr, image_ptr, bg_color_ptr, primitive_ptr, tile_ptr, instance_ptr, bucket_ptr,
        n_primitives, width, height, n_instances, n_buckets, instance_primitive_indices_selector,
        variant, chunk_first_bucket, chunk_n_buckets,
        g_mean2d, g_conic, g_opacity, g_colors, priv, partial_ptr, capture_partials);
    return std::make_tuple(ms, partial_out);
}

// implemented in atom0_api.cu (nvcc)
torch::Tensor get_instance_primitive_indices(torch::Tensor instance_buffers, int n_instances, int width, int height, int selector);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_wrapper, "FasterGSFused forward (verbatim)");
    m.def("backward_atom0", &backward_atom0, "ATOM0 diagnostic blend_backward variant timer");
    m.def("get_instance_primitive_indices", &get_instance_primitive_indices, "extract instance->primitive map");
}
