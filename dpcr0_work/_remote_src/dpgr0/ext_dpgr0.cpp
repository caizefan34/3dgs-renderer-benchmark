// DPGR0 extension: deterministic direct-primitive-gather 3DGS backward.
//
// Minimal pybind surface. Only the DPGR0 backward op is provided; it is bound
// under the name "higs_rasterize_backward_dpgr0" and dispatches to the
// race-free direct-gather implementation. The C0 forward ops
// (gaussian_render_inference_only, higs_gatherless_projected_producer, ...)
// remain loaded from the authoritative C0 extension and are NOT re-registered
// here (avoids the `experimental` torch-lib clash).

#include <torch/extension.h>

#include "csrc/gaussian_inference/HigsNativeBackwardDpgr0.h"
#include "csrc/gaussian_inference/dpgr0_stage1.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("dpgr0_get_stage_times",
          &gsplat::gaussian_render_inference_scene::dpgr0_get_stage_times,
          "DPGR0 timed build: [range_build_ms, gather_ms, reduce_ms] for the last dpgr0_stage1 call.");
    m.def("dpgr0_get_stats",
          &gsplat::gaussian_render_inference_scene::dpgr0_get_stats,
          "DPGR0 stats: [n_isects, n_seg, n_chunked, total_chunks, pstride, N, records, record_bytes].");

    m.def(
        "higs_rasterize_backward_dpgr0",
        &gsplat::gaussian_render_inference_scene::higs_rasterize_backward_dpgr0,
        "DPGR0 deterministic direct-primitive-gather native backward (race-free).",
        py::arg("means2d"),
        py::arg("conics"),
        py::arg("colors_eval"),
        py::arg("opacities"),
        py::arg("backgrounds"),
        py::arg("tile_offsets"),
        py::arg("flatten_ids"),
        py::arg("active_tiles"),
        py::arg("render_alphas"),
        py::arg("last_ids"),
        py::arg("means"),
        py::arg("quats"),
        py::arg("scales"),
        py::arg("radii"),
        py::arg("viewmats"),
        py::arg("Ks"),
        py::arg("width"),
        py::arg("height"),
        py::arg("tile_size"),
        py::arg("eps2d"),
        py::arg("camera_model"),
        py::arg("v_render_colors"),
        py::arg("v_render_alphas"),
        py::arg("sh_coeffs"),
        py::arg("sh_degree"),
        py::arg("visible_ids"),
        py::arg("grad_means"),
        py::arg("grad_quats"),
        py::arg("grad_scales"),
        py::arg("grad_opacities"),
        py::arg("grad_colors")
    );
}