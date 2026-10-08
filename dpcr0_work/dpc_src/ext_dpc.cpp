/*
 * DPC0 ext.cpp — pybind bindings for the DPC deterministic backward extension.
 * Mirrors the frozen C0 ext.cpp but exposes higs_rasterize_backward_dpc.
 * Built as a SEPARATE .so; the authoritative C0 .so is never modified.
 */
#include <torch/extension.h>
#include "csrc/gaussian_inference/GaussianRenderInferenceScene.h"
#include "csrc/gaussian_inference/HigsNativeBackward.h"
#include "csrc/gaussian_inference/GatherVisible.h"
#include "csrc/gaussian_inference/GatherlessProjectedProducer.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<gsplat::gaussian_render_inference_scene::GaussianInferenceRenderer>(m, "GaussianInferenceRenderer")
        .def(py::init<const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, int64_t>(),
            py::arg("means_planar"), py::arg("qso_packed"), py::arg("colors_packed"),
            py::arg("sh_degree"), py::arg("sh_compression_mode"))
        .def("render", &gsplat::gaussian_render_inference_scene::GaussianInferenceRenderer::render,
            py::arg("means_planar"), py::arg("qso_packed"), py::arg("colors_packed"),
            py::arg("viewmat"), py::arg("K"), py::arg("width"), py::arg("height"),
            py::arg("tile_size"), py::arg("near_plane"), py::arg("far_plane"),
            py::arg("radius_clip"), py::arg("eps2d"), py::arg("sh_degree"),
            py::arg("sh_compression_mode"), py::arg("background"), py::arg("out_rgbt"))
        .def("release", &gsplat::gaussian_render_inference_scene::GaussianInferenceRenderer::release)
        .def("num_gaussians", &gsplat::gaussian_render_inference_scene::GaussianInferenceRenderer::numGaussians)
        .def("is_released", &gsplat::gaussian_render_inference_scene::GaussianInferenceRenderer::isReleased)
        .def("get_visible_mask", &gsplat::gaussian_render_inference_scene::GaussianInferenceRenderer::getVisibleMask);

    m.def("higs_rasterize_backward_dpc",
        &gsplat::gaussian_render_inference_scene::higs_rasterize_backward_dpc,
        "DPC0 deterministic primitive-centric native backward.",
        py::arg("means2d"), py::arg("conics"), py::arg("colors_eval"), py::arg("opacities"),
        py::arg("backgrounds"), py::arg("tile_offsets"), py::arg("flatten_ids"),
        py::arg("active_tiles"), py::arg("render_alphas"), py::arg("last_ids"),
        py::arg("means"), py::arg("quats"), py::arg("scales"), py::arg("radii"),
        py::arg("viewmats"), py::arg("Ks"), py::arg("width"), py::arg("height"),
        py::arg("tile_size"), py::arg("eps2d"), py::arg("camera_model"),
        py::arg("v_render_colors"), py::arg("v_render_alphas"), py::arg("sh_coeffs"),
        py::arg("sh_degree"), py::arg("visible_ids"), py::arg("grad_means"),
        py::arg("grad_quats"), py::arg("grad_scales"), py::arg("grad_opacities"),
        py::arg("grad_colors"));

    m.def("higs_gather_visible",
        &gsplat::gaussian_render_inference_scene::higs_gather_visible,
        py::arg("means"), py::arg("quats"), py::arg("scales"),
        py::arg("opacities"), py::arg("colors"), py::arg("visible_ids"));
    m.def("higs_gatherless_projected_producer",
        &gsplat::gaussian_render_inference_scene::higs_gatherless_projected_producer,
        py::arg("visible_ids"), py::arg("means"), py::arg("quats"),
        py::arg("scales"), py::arg("opacities"), py::arg("coeffs"),
        py::arg("viewmats"), py::arg("Ks"), py::arg("cam_positions"),
        py::arg("width"), py::arg("height"), py::arg("eps2d"),
        py::arg("near_plane"), py::arg("far_plane"), py::arg("radius_clip"));
    m.def("higs_camera_positions_from_viewmats",
        &gsplat::gaussian_render_inference_scene::higs_camera_positions_from_viewmats,
        py::arg("viewmats"));
    m.def("higs_union_visible_mask",
        &gsplat::gaussian_render_inference_scene::higs_union_visible_mask,
        py::arg("radii"));
    m.def("higs_gather_rows",
        &gsplat::gaussian_render_inference_scene::higs_gather_rows,
        py::arg("src"), py::arg("row_ids"), py::arg("zero_on_neg") = false);
}
