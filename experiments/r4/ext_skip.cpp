// ext_skip.cpp
// pybind11 binding for the rasterize_to_pixels_bwd_with_skip CUDA extension.
//
// This file is the C++ binding glue compiled alongside
// rasterize_to_pixels_bwd_with_skip.cu into a single JIT extension module.
// It follows the same pattern as gsplat's own ext.cpp: include bindings.h
// (which pulls in torch/extension.h and all the type casters for
// at::optional<torch::Tensor>), forward-declare our custom function, and
// register it with pybind11.
//
// The function rasterize_to_pixels_bwd_with_skip is a template instantiated
// for COLOR_DIM=3 (RGB) inside the .cu file. We forward-declare the template
// here and bind the <3> specialization.

#include "bindings.h"

namespace gsplat {

// Forward declaration of the host-side wrapper template defined and
// explicitly instantiated for CDIM=3 in rasterize_to_pixels_bwd_with_skip.cu.
//
// Signature mirrors gsplat's rasterize_to_pixels_bwd_tensor with an added
// skip_mask parameter inserted before the absgrad flag:
//   - skip_mask: [n_isects] bool tensor, true = skip gradient work for
//     that intersection (but still update T and buffer for correctness).
template <uint32_t CDIM>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_bwd_with_skip(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &colors,
    const torch::Tensor &opacities,
    const at::optional<torch::Tensor> &backgrounds,
    const at::optional<torch::Tensor> &masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const torch::Tensor &tile_offsets,
    const torch::Tensor &flatten_ids,
    const torch::Tensor &render_alphas,
    const torch::Tensor &last_ids,
    const torch::Tensor &v_render_colors,
    const torch::Tensor &v_render_alphas,
    const at::optional<torch::Tensor> &skip_mask,
    const bool absgrad
);

}  // namespace gsplat

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Custom gsplat backward rasterization with certificate-guided skip mask";

    m.def(
        "rasterize_to_pixels_bwd_with_skip",
        &gsplat::rasterize_to_pixels_bwd_with_skip<3>,
        "Backward rasterization with skip_mask support (RGB, COLOR_DIM=3).\n\n"
        "Args (same as gsplat rasterize_to_pixels_bwd + skip_mask):\n"
        "  means2d:        [C, N, 2] float32\n"
        "  conics:         [C, N, 3] float32\n"
        "  colors:         [C, N, 3] float32\n"
        "  opacities:      [N]      float32\n"
        "  backgrounds:    [C, 3]   float32 or None\n"
        "  masks:          [C, tile_height, tile_width] bool or None\n"
        "  image_width:    uint32\n"
        "  image_height:   uint32\n"
        "  tile_size:      uint32\n"
        "  tile_offsets:   [C, tile_height, tile_width] int32\n"
        "  flatten_ids:    [n_isects] int32\n"
        "  render_alphas:  [C, H, W, 1] float32\n"
        "  last_ids:       [C, H, W] int32\n"
        "  v_render_colors:[C, H, W, 3] float32\n"
        "  v_render_alphas:[C, H, W, 1] float32\n"
        "  skip_mask:      [n_isects] bool or None — true = skip grad work\n"
        "  absgrad:        bool\n\n"
        "Returns: (v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities)"
    );
}
