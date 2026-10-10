// Link-time stubs for the RasterizeToPixelsFromWorld3DGS* kernels.
//
// When GSPLAT_SKIP_FROM_WORLD=1 the build drops the three
// RasterizeToPixelsFromWorld3DGS*.cu translation units (they take 1.5-3.5h
// each to compile under CUDA 13.x + sm_120 and are unused by the HiGS
// training path). Rasterization.cpp still references the host helpers and
// launch functions declared in RasterizeToPixelsFromWorld3DGS.h /
// RasterizeCSR.cuh, so this TU provides throwing definitions that keep the
// extension linkable. Any attempt to actually use
// rasterize_to_pixels_from_world_3dgs in such a build fails fast with a
// clear message instead of silently producing wrong output.

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

#include "RasterizeCSR.cuh"
#include "RasterizeToPixelsFromWorld3DGS.h"

namespace gsplat
{

namespace
{
[[noreturn]] void from_world_not_built()
{
    TORCH_CHECK(
        false,
        "rasterize_to_pixels_from_world_3dgs was not compiled into this gsplat build "
        "(built with GSPLAT_SKIP_FROM_WORLD=1)."
    );
}
} // namespace

std::tuple<at::Tensor, at::Tensor, int64_t> compute_batch_csr(
    const at::Tensor &tile_offsets,
    int64_t n_isects,
    uint32_t num_tiles,
    int32_t pixels_per_tile,
    at::TensorOptions dummy_options
)
{
    from_world_not_built();
}

at::Tensor compute_bid_to_slot(
    const at::Tensor &batches_per_tile,
    const at::Tensor &batch_offsets,
    int64_t total_batches,
    at::TensorOptions dummy_options
)
{
    from_world_not_built();
}

void launch_rasterize_to_pixels_from_world_3dgs_serial_batch_fwd_kernel(
    const at::Tensor means,
    const at::Tensor quats,
    const at::Tensor scales,
    const at::Tensor colors,
    const at::Tensor opacities,
    const at::optional<at::Tensor> backgrounds,
    const at::optional<at::Tensor> masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor viewmats0,
    const at::optional<at::Tensor> viewmats1,
    const at::Tensor Ks,
    const CameraModelType camera_model,
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> rays,
    const at::optional<at::Tensor> radial_coeffs,
    const at::optional<at::Tensor> tangential_coeffs,
    const at::optional<at::Tensor> thin_prism_coeffs,
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    const bool use_hit_distance,
    const bool unsafe_masked_tile_outputs,
    const at::Tensor batches_per_tile,
    const at::Tensor batch_offsets,
    const int64_t total_batches,
    at::Tensor renders,
    at::Tensor alphas,
    at::Tensor last_ids,
    at::optional<at::Tensor> sample_counts,
    at::optional<at::Tensor> normals,
    at::Tensor fwd_batch_state
)
{
    from_world_not_built();
}

void launch_rasterize_to_pixels_from_world_3dgs_parallel_batch_fwd_kernel(
    const at::Tensor means,
    const at::Tensor quats,
    const at::Tensor scales,
    const at::Tensor colors,
    const at::Tensor opacities,
    const at::optional<at::Tensor> backgrounds,
    const at::optional<at::Tensor> masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor viewmats0,
    const at::optional<at::Tensor> viewmats1,
    const at::Tensor Ks,
    const CameraModelType camera_model,
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> rays,
    const at::optional<at::Tensor> radial_coeffs,
    const at::optional<at::Tensor> tangential_coeffs,
    const at::optional<at::Tensor> thin_prism_coeffs,
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    const bool use_hit_distance,
    const bool unsafe_masked_tile_outputs,
    const at::Tensor batches_per_tile,
    const at::Tensor batch_offsets,
    const at::Tensor bid_to_slot,
    const int64_t total_batches,
    bool fwd_only,
    at::Tensor renders,
    at::Tensor alphas,
    at::Tensor last_ids,
    at::optional<at::Tensor> sample_counts,
    at::optional<at::Tensor> normals,
    at::Tensor fwd_batch_state,
    at::Tensor partials_meta,
    at::Tensor batch_replay_preamble,
    at::Tensor compose_c_stop,
    at::Tensor priming_state
)
{
    from_world_not_built();
}

void launch_rasterize_to_pixels_from_world_3dgs_parallel_batch_bwd_kernel(
    const at::Tensor means,
    const at::Tensor quats,
    const at::Tensor scales,
    const at::Tensor colors,
    const at::Tensor opacities,
    const at::optional<at::Tensor> backgrounds,
    const at::optional<at::Tensor> masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor viewmats0,
    const at::optional<at::Tensor> viewmats1,
    const at::Tensor Ks,
    const CameraModelType camera_model,
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> rays,
    const at::optional<at::Tensor> radial_coeffs,
    const at::optional<at::Tensor> tangential_coeffs,
    const at::optional<at::Tensor> thin_prism_coeffs,
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    const bool use_hit_distance,
    const at::Tensor render_alphas,
    const at::Tensor last_ids,
    const at::Tensor v_render_colors,
    const at::Tensor v_render_alphas,
    const at::optional<at::Tensor> v_render_normals,
    const at::Tensor batches_per_tile,
    const at::Tensor batch_offsets,
    const int64_t total_batches,
    const at::Tensor fwd_batch_state,
    const at::Tensor compose_c_stop,
    at::Tensor v_means,
    at::Tensor v_quats,
    at::Tensor v_scales,
    at::Tensor v_colors,
    at::Tensor v_opacities,
    at::optional<at::Tensor> v_rays
)
{
    from_world_not_built();
}

} // namespace gsplat
