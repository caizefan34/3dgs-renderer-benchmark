// Diagnostic: count FastGS backward bucket-pixel pairs with no accepted splat.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

__global__ void count_kernel(
    const float* means, const float* conics, const int* ranges,
    const int* point_list, const int* n_contrib,
    const int* bucket_tile, const int* bucket_index,
    int width, int height, int tile_width, int group_size, unsigned char* out) {
    int bucket = blockIdx.x;
    int local = threadIdx.x;
    int tile = bucket_tile[bucket];
    int bx = tile % tile_width, by = tile / tile_width;
    int x = bx * 16 + local % 16, y = by * 16 + local / 16;
    int out_idx = 2 * (bucket * 256 + local);
    if (x >= width || y >= height) {
        out[out_idx] = 0;
        out[out_idx + 1] = 0;
        return;
    }
    int begin = ranges[tile * 2], end = ranges[tile * 2 + 1];
    int start = bucket_index[bucket] * group_size;
    int last = n_contrib[y * width + x];
    if (start >= last || begin + start >= end) {
        out[out_idx] = 0;
        out[out_idx + 1] = 0;
        return;
    }
    int stop = min(min(start + group_size, end - begin), last);
    unsigned char active_count = 0;
    for (int j = start; j < stop; ++j) {
        int id = point_list[begin + j];
        float dx = means[2 * id] - x, dy = means[2 * id + 1] - y;
        const float* co = conics + 4 * id;
        float power = -0.5f * (co[0] * dx * dx + co[2] * dy * dy) - co[1] * dx * dy;
        if (power > 0.f) continue;
        float alpha = fminf(.99f, co[3] * expf(power));
        if (alpha >= 1.f / 255.f) {
            ++active_count;
        }
    }
    out[out_idx] = 1 + (active_count > 0);
    out[out_idx + 1] = active_count;
}

torch::Tensor count_empty(
    torch::Tensor means, torch::Tensor conics, torch::Tensor ranges,
    torch::Tensor point_list, torch::Tensor n_contrib,
    torch::Tensor bucket_tile, torch::Tensor bucket_index,
    int width, int height, int group_size) {
    TORCH_CHECK(means.is_cuda() && conics.is_cuda() && ranges.is_cuda() &&
                point_list.is_cuda() && n_contrib.is_cuda() &&
                bucket_tile.is_cuda() && bucket_index.is_cuda(), "CUDA required");
    TORCH_CHECK(means.is_contiguous() && conics.is_contiguous() &&
                ranges.is_contiguous() && point_list.is_contiguous() &&
                n_contrib.is_contiguous() && bucket_tile.is_contiguous() &&
                bucket_index.is_contiguous(), "contiguous required");
    TORCH_CHECK(means.scalar_type() == at::kFloat && conics.scalar_type() == at::kFloat,
                "float32 geometry required");
    TORCH_CHECK(ranges.scalar_type() == at::kInt && point_list.scalar_type() == at::kInt &&
                n_contrib.scalar_type() == at::kInt && bucket_tile.scalar_type() == at::kInt &&
                bucket_index.scalar_type() == at::kInt, "int32 indices required");
    TORCH_CHECK(group_size >= 1 && group_size <= 32, "group_size must be in [1,32]");
    c10::cuda::CUDAGuard guard(means.device());
    int buckets = bucket_tile.numel();
    auto out = torch::empty({buckets, 256, 2}, means.options().dtype(at::kByte));
    count_kernel<<<buckets, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        means.data_ptr<float>(), conics.data_ptr<float>(), ranges.data_ptr<int>(),
        point_list.data_ptr<int>(), n_contrib.data_ptr<int>(),
        bucket_tile.data_ptr<int>(), bucket_index.data_ptr<int>(),
        width, height, (width + 15) / 16, group_size, out.data_ptr<unsigned char>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("count_empty", &count_empty);
}
