#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cmath>

namespace {

template <typename scalar_t>
__device__ __forceinline__ void tile_bounds(
    const scalar_t* means2d, const int32_t* radii, uint32_t idx,
    uint32_t tile_size, uint32_t tile_width, uint32_t tile_height,
    uint32_t& min_x, uint32_t& min_y, uint32_t& max_x, uint32_t& max_y
) {
    // This deliberately mirrors Reference V1 IntersectTile.cu:55-77.
    const float radius_x = radii[idx * 2];
    const float radius_y = radii[idx * 2 + 1];
    const float tile_radius_x = radius_x / static_cast<float>(tile_size);
    const float tile_radius_y = radius_y / static_cast<float>(tile_size);
    const float tile_x = static_cast<float>(means2d[idx * 2]) / static_cast<float>(tile_size);
    const float tile_y = static_cast<float>(means2d[idx * 2 + 1]) / static_cast<float>(tile_size);
    min_x = min(max(0u, static_cast<uint32_t>(floorf(tile_x - tile_radius_x))), tile_width);
    min_y = min(max(0u, static_cast<uint32_t>(floorf(tile_y - tile_radius_y))), tile_height);
    max_x = min(max(0u, static_cast<uint32_t>(ceilf(tile_x + tile_radius_x))), tile_width);
    max_y = min(max(0u, static_cast<uint32_t>(ceilf(tile_y + tile_radius_y))), tile_height);
}

template <typename scalar_t>
__device__ __forceinline__ int64_t packed_key(
    const scalar_t* depths, const int64_t* image_ids, uint32_t idx,
    int64_t tile_id, uint32_t tile_n_bits
) {
    const int64_t iid_enc = image_ids[idx] << (32 + tile_n_bits);
    // Reference V1 does an int32 pointer reinterpret then zero-extends it.
    const uint32_t depth_bits = __float_as_uint(static_cast<float>(depths[idx]));
    return iid_enc | (tile_id << 32) | static_cast<int64_t>(depth_bits);
}

template <typename scalar_t>
__global__ void serial_emit_kernel(
    const scalar_t* means2d, const int32_t* radii, const scalar_t* depths,
    const int64_t* image_ids, const int64_t* cumulative, uint32_t nnz,
    uint32_t tile_size, uint32_t tile_width, uint32_t tile_height,
    uint32_t tile_n_bits, int64_t* isect_ids, int32_t* flatten_ids
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz || radii[idx * 2] <= 0 || radii[idx * 2 + 1] <= 0) return;
    uint32_t min_x, min_y, max_x, max_y;
    tile_bounds(means2d, radii, idx, tile_size, tile_width, tile_height, min_x, min_y, max_x, max_y);
    int64_t out = idx == 0 ? 0 : cumulative[idx - 1];
    for (uint32_t y = min_y; y < max_y; ++y) {
        for (uint32_t x = min_x; x < max_x; ++x) {
            const int64_t tile_id = static_cast<int64_t>(y) * tile_width + x;
            isect_ids[out] = packed_key(depths, image_ids, idx, tile_id, tile_n_bits);
            flatten_ids[out] = static_cast<int32_t>(idx);
            ++out;
        }
    }
}

template <typename scalar_t>
__global__ void warp_emit_kernel(
    const scalar_t* means2d, const int32_t* radii, const scalar_t* depths,
    const int64_t* image_ids, const int64_t* cumulative, uint32_t nnz,
    uint32_t tile_size, uint32_t tile_width, uint32_t tile_height,
    uint32_t tile_n_bits, int64_t* isect_ids, int32_t* flatten_ids
) {
    const uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t idx = global_thread >> 5;  // exactly one warp per Gaussian
    const uint32_t lane = threadIdx.x & 31;
    if (idx >= nnz || radii[idx * 2] <= 0 || radii[idx * 2 + 1] <= 0) return;
    uint32_t min_x, min_y, max_x, max_y;
    tile_bounds(means2d, radii, idx, tile_size, tile_width, tile_height, min_x, min_y, max_x, max_y);
    const uint32_t row_width = max_x - min_x;
    const uint32_t count = (max_y - min_y) * row_width;
    const int64_t start = idx == 0 ? 0 : cumulative[idx - 1];
    // k is the baseline's serial position.  Writes can retire out of order,
    // but every physical array element equals baseline's y-major/x-major value.
    for (uint32_t k = lane; k < count; k += 32) {
        const uint32_t row = k / row_width;
        const uint32_t col = k - row * row_width;
        const int64_t tile_id = static_cast<int64_t>(min_y + row) * tile_width + min_x + col;
        isect_ids[start + k] = packed_key(depths, image_ids, idx, tile_id, tile_n_bits);
        flatten_ids[start + k] = static_cast<int32_t>(idx);
    }
}

inline uint32_t tile_bits(uint32_t width, uint32_t height) {
    return static_cast<uint32_t>(floor(log2(static_cast<double>(width) * height))) + 1;
}

} // namespace

void serial_emit_cuda(
    torch::Tensor means2d, torch::Tensor radii, torch::Tensor depths,
    torch::Tensor image_ids, torch::Tensor cumulative, int64_t tile_size,
    int64_t tile_width, int64_t tile_height, torch::Tensor isect_ids,
    torch::Tensor flatten_ids
) {
    const uint32_t nnz = static_cast<uint32_t>(means2d.size(0));
    if (!nnz) return;
    const dim3 threads(256), blocks((nnz + threads.x - 1) / threads.x);
    AT_DISPATCH_FLOATING_TYPES(means2d.scalar_type(), "serial_emit_kernel", [&] {
        serial_emit_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            means2d.data_ptr<scalar_t>(), radii.data_ptr<int32_t>(), depths.data_ptr<scalar_t>(),
            image_ids.data_ptr<int64_t>(), cumulative.data_ptr<int64_t>(), nnz,
            static_cast<uint32_t>(tile_size), static_cast<uint32_t>(tile_width),
            static_cast<uint32_t>(tile_height), tile_bits(tile_width, tile_height),
            isect_ids.data_ptr<int64_t>(), flatten_ids.data_ptr<int32_t>()
        );
    });
}

void warp_emit_cuda(
    torch::Tensor means2d, torch::Tensor radii, torch::Tensor depths,
    torch::Tensor image_ids, torch::Tensor cumulative, int64_t tile_size,
    int64_t tile_width, int64_t tile_height, int64_t threads_per_block,
    torch::Tensor isect_ids, torch::Tensor flatten_ids
) {
    TORCH_CHECK(threads_per_block == 128 || threads_per_block == 256, "WARP_ALL supports 128 or 256 threads/CTA");
    const uint32_t nnz = static_cast<uint32_t>(means2d.size(0));
    if (!nnz) return;
    const uint32_t warps_per_block = static_cast<uint32_t>(threads_per_block / 32);
    const dim3 threads(threads_per_block), blocks((nnz + warps_per_block - 1) / warps_per_block);
    AT_DISPATCH_FLOATING_TYPES(means2d.scalar_type(), "warp_emit_kernel", [&] {
        warp_emit_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            means2d.data_ptr<scalar_t>(), radii.data_ptr<int32_t>(), depths.data_ptr<scalar_t>(),
            image_ids.data_ptr<int64_t>(), cumulative.data_ptr<int64_t>(), nnz,
            static_cast<uint32_t>(tile_size), static_cast<uint32_t>(tile_width),
            static_cast<uint32_t>(tile_height), tile_bits(tile_width, tile_height),
            isect_ids.data_ptr<int64_t>(), flatten_ids.data_ptr<int32_t>()
        );
    });
}
