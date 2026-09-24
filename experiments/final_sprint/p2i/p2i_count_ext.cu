#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

// This is deliberately a verbatim math-level extraction of the first-pass
// footprint code in gsplat/cuda/csrc/IntersectTile.cu.  It is a measurement
// harness only: it neither replaces nor alters the renderer pipeline.
template <typename scalar_t>
__global__ void count_tiles_kernel(
    const scalar_t* __restrict__ means2d,
    const int32_t* __restrict__ radii,
    int32_t* __restrict__ tiles_per_gauss,
    const int64_t n,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float radius_x = radii[idx * 2];
    const float radius_y = radii[idx * 2 + 1];
    if (radius_x <= 0 || radius_y <= 0) {
        tiles_per_gauss[idx] = 0;
        return;
    }

    const float mean_x = static_cast<float>(means2d[idx * 2]);
    const float mean_y = static_cast<float>(means2d[idx * 2 + 1]);
    const float tile_radius_x = radius_x / static_cast<float>(tile_size);
    const float tile_radius_y = radius_y / static_cast<float>(tile_size);
    const float tile_x = mean_x / static_cast<float>(tile_size);
    const float tile_y = mean_y / static_cast<float>(tile_size);

    // tile_min is inclusive and tile_max is exclusive: match IntersectTile.cu.
    const uint32_t tile_min_x = min(max(0, static_cast<int>(floorf(tile_x - tile_radius_x))), static_cast<int>(tile_width));
    const uint32_t tile_min_y = min(max(0, static_cast<int>(floorf(tile_y - tile_radius_y))), static_cast<int>(tile_height));
    const uint32_t tile_max_x = min(max(0, static_cast<int>(ceilf(tile_x + tile_radius_x))), static_cast<int>(tile_width));
    const uint32_t tile_max_y = min(max(0, static_cast<int>(ceilf(tile_y + tile_radius_y))), static_cast<int>(tile_height));

    tiles_per_gauss[idx] = static_cast<int32_t>(
        (tile_max_y - tile_min_y) * (tile_max_x - tile_min_x)
    );
}

void count_tiles_cuda(
    torch::Tensor means2d,
    torch::Tensor radii,
    torch::Tensor tiles_per_gauss,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height
) {
    TORCH_CHECK(means2d.is_cuda() && radii.is_cuda() && tiles_per_gauss.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(means2d.is_contiguous() && radii.is_contiguous() && tiles_per_gauss.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(radii.scalar_type() == at::kInt && tiles_per_gauss.scalar_type() == at::kInt, "radii and count output must be int32");
    const int64_t n = means2d.numel() / 2;
    const dim3 threads(256);
    const dim3 blocks((n + threads.x - 1) / threads.x);
    AT_DISPATCH_FLOATING_TYPES(means2d.scalar_type(), "p2i_count_tiles", [&] {
        count_tiles_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            means2d.data_ptr<scalar_t>(), radii.data_ptr<int32_t>(),
            tiles_per_gauss.data_ptr<int32_t>(), n,
            static_cast<uint32_t>(tile_size), static_cast<uint32_t>(tile_width),
            static_cast<uint32_t>(tile_height));
    });
}
