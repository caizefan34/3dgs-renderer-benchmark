// DPC0 extension: deterministic primitive-centric 3DGS backward.
//
// Minimal pybind surface. Only the backward op is provided; it is bound under
// the SAME name "higs_rasterize_backward" with the SAME keyword arguments as
// the C0 op, but dispatches to the race-free DPC0 implementation
// (higs_rasterize_backward_dpc). The C0 forward ops (gaussian_render_inference_only,
// higs_gatherless_projected_producer, ...) remain loaded from the authoritative C0
// extension and are NOT re-registered here (avoids the `experimental` torch-lib clash).

#include <torch/extension.h>

#include "csrc/gaussian_inference/HigsNativeBackwardDpc.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "higs_rasterize_backward",
        &gsplat::gaussian_render_inference_scene::higs_rasterize_backward_dpc,
        "DPC0 deterministic native CUDA backward (race-free primitive-centric). "
        "Drop-in replacement for the C0 higs_rasterize_backward: identical "
        "signature and return contract, but no float atomicAdd to any final gradient.",
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