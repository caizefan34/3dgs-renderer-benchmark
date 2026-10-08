// Fixed-support per-pixel work counter for a FastGS split event.
// Diagnostic only: same scalar cutoff/early-stop semantics as event_gate_vjp.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

__global__ void count_work_kernel(
    const float* means, const float* conics, const int* roles,
    const int* ranges, const int* point_list,
    int width, int height, int tile_width, float gate, int* output) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= width * height) return;
    int x = p % width, y = p / width;
    int tile = (y / 16) * tile_width + x / 16;
    int begin = ranges[2 * tile], end = ranges[2 * tile + 1];
    float T = 1.f;
    int visited = 0, raw_pass = 0, shadow_pass = 0, active = 0, accepted = 0;
    for (int j = begin; j < end; ++j) {
        ++visited;
        int id = point_list[j];
        float dx = means[2 * id] - x, dy = means[2 * id + 1] - y;
        const float* co = conics + 4 * id;
        float power = -0.5f * (co[0] * dx * dx + co[2] * dy * dy) - co[1] * dx * dy;
        if (power > 0.f) continue;
        float raw = fminf(.99f, co[3] * expf(power));
        if (raw < 1.f / 255.f) continue;
        ++raw_pass;
        int role = roles[id];
        if (role != 0) ++shadow_pass;
        float alpha = raw;
        if (role == 1) {
            alpha = gate <= 0.f ? raw : gate >= 1.f ? 0.f
                    : -expm1f((1.f - gate) * log1pf(-raw));
        } else if (role == 2) {
            alpha = gate <= 0.f ? 0.f : gate >= 1.f ? raw
                    : -expm1f(gate * log1pf(-raw));
        }
        float next_T = T * (1.f - alpha);
        if (next_T < 1e-4f) break;
        ++accepted;
        if (alpha > 0.f) ++active;
        T = next_T;
    }
    output[5 * p] = visited;
    output[5 * p + 1] = raw_pass;
    output[5 * p + 2] = shadow_pass;
    output[5 * p + 3] = active;
    output[5 * p + 4] = accepted;
}

torch::Tensor count_work(torch::Tensor means, torch::Tensor conics,
                         torch::Tensor roles, torch::Tensor ranges,
                         torch::Tensor point_list, int width, int height,
                         float gate) {
    TORCH_CHECK(means.is_cuda() && conics.is_cuda() && roles.is_cuda()
                && ranges.is_cuda() && point_list.is_cuda(), "CUDA inputs required");
    TORCH_CHECK(means.is_contiguous() && conics.is_contiguous() && roles.is_contiguous()
                && ranges.is_contiguous() && point_list.is_contiguous(), "contiguous inputs required");
    TORCH_CHECK(means.scalar_type() == at::kFloat && conics.scalar_type() == at::kFloat,
                "float32 projection required");
    TORCH_CHECK(roles.scalar_type() == at::kInt && ranges.scalar_type() == at::kInt
                && point_list.scalar_type() == at::kInt, "int32 indices required");
    TORCH_CHECK(means.size(0) == roles.numel(), "role count mismatch");
    TORCH_CHECK(ranges.numel() >= 2 * ((width + 15) / 16) * ((height + 15) / 16),
                "tile range count mismatch");
    c10::cuda::CUDAGuard guard(means.device());
    auto out = torch::empty({height * width, 5}, roles.options());
    int pixels = width * height;
    count_work_kernel<<<(pixels + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        means.data_ptr<float>(), conics.data_ptr<float>(), roles.data_ptr<int>(),
        ranges.data_ptr<int>(), point_list.data_ptr<int>(),
        width, height, (width + 15) / 16, gate, out.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("count_work", &count_work);
}
