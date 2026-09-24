#include <torch/extension.h>

void serial_emit_cuda(
    torch::Tensor means2d, torch::Tensor radii, torch::Tensor depths,
    torch::Tensor image_ids, torch::Tensor cumulative, int64_t tile_size,
    int64_t tile_width, int64_t tile_height, torch::Tensor isect_ids,
    torch::Tensor flatten_ids
);

void warp_emit_cuda(
    torch::Tensor means2d, torch::Tensor radii, torch::Tensor depths,
    torch::Tensor image_ids, torch::Tensor cumulative, int64_t tile_size,
    int64_t tile_width, int64_t tile_height, int64_t threads_per_block,
    torch::Tensor isect_ids, torch::Tensor flatten_ids
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("serial_emit", &serial_emit_cuda, "Reference row-major serial emit (CUDA)");
    m.def("warp_emit", &warp_emit_cuda, "Warp-cooperative row-major emit (CUDA)");
}
