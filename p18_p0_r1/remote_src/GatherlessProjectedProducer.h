#pragma once

#include <ATen/core/Tensor.h>
#include <vector>

namespace gsplat { namespace gaussian_render_inference_scene {
std::vector<at::Tensor> higs_gatherless_projected_producer(
    const at::Tensor &visible_ids, const at::Tensor &means, const at::Tensor &quats,
    const at::Tensor &scales, const at::Tensor &opacities, const at::Tensor &coeffs,
    const at::Tensor &viewmats, const at::Tensor &Ks, const at::Tensor &cam_positions,
    int64_t width, int64_t height, double eps2d, double near_plane,
    double far_plane, double radius_clip);

at::Tensor higs_camera_positions_from_viewmats(const at::Tensor &viewmats);
}} // namespace gsplat::gaussian_render_inference_scene
