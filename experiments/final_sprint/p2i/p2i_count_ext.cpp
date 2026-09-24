#include <torch/extension.h>

void count_tiles_cuda(
    torch::Tensor means2d,
    torch::Tensor radii,
    torch::Tensor tiles_per_gauss,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("count_tiles", &count_tiles_cuda,
          "P2I baseline-equivalent intersection count pass (CUDA)");
}
