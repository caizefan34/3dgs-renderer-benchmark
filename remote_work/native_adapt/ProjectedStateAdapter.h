#pragma once

#include <ATen/core/Tensor.h>

#include <tuple>
#include <vector>

namespace gsplat
{
namespace gaussian_render_inference_scene
{
// Native P2-1A entry point for the shared F9 projected-primitive contract.
// It deliberately accepts already projected/evaluated FP32 state and never
// invokes the packed inference projection or SH paths.
std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> higs_native_hierarchy_from_projected(
    const at::Tensor &visible_ids,
    const at::Tensor &radii,
    const at::Tensor &means2d,
    const at::Tensor &depths,
    const at::Tensor &conics,
    const at::Tensor &opacities,
    const at::Tensor &colors,
    int64_t width,
    int64_t height,
    int64_t tile_size,
    const at::optional<at::Tensor> &background,
    bool debug
);
} // namespace gaussian_render_inference_scene
} // namespace gsplat
