#include <torch/extension.h>

void evaluate_fragments_cuda(torch::Tensor means, torch::Tensor conics, torch::Tensor opacities,
                             torch::Tensor tiles, int64_t width, int64_t height, int64_t tile_size,
                             torch::Tensor max_alpha, torch::Tensor sigma, torch::Tensor pixel_x,
                             torch::Tensor pixel_y);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("evaluate_fragments", &evaluate_fragments_cuda, "Reference-V1 fragment cutoff audit (CUDA)");
}
