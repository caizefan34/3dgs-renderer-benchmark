// Isolated diagnostic: fixed-support zero-gate RGB and one-family VJP.
// Inputs are projected, depth-sorted buffers produced by author FastGS CUDA.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void shadow_kernel(
    const float* means, const float* conics, const float* colors,
    const int* roles, const int* ranges, const int* point_list,
    const int* pixels, const float* bg, const float* adjoint,
    int width, int height, int tile_width, int count, float gate,
    float* out_rgb, float* out_vjp) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= count) return;
    int x = pixels[2 * p], y = pixels[2 * p + 1];
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    int tile = (y / 16) * tile_width + (x / 16);
    int begin = ranges[2 * tile], end = ranges[2 * tile + 1];
    float T = 1.f, dT = 0.f;
    float C[3] = {0.f, 0.f, 0.f}, dC[3] = {0.f, 0.f, 0.f};
    for (int j = begin; j < end; ++j) {
        int id = point_list[j];
        float dx = means[2 * id] - x, dy = means[2 * id + 1] - y;
        const float* co = conics + 4 * id;
        float power = -0.5f * (co[0] * dx * dx + co[2] * dy * dy) - co[1] * dx * dy;
        if (power > 0.f) continue;
        float raw = fminf(0.99f, co[3] * expf(power));
        if (raw < 1.f / 255.f) continue;
        int role = roles[id];
        float alpha = raw, dalpha = 0.f;
        if (role != 0) {
            float log_survival = log1pf(-raw);
            if (role == 1) {
                alpha = gate <= 0.f ? raw : gate >= 1.f ? 0.f
                        : -expm1f((1.f - gate) * log_survival);
                dalpha = (1.f - alpha) * log_survival;
            } else {
                alpha = gate <= 0.f ? 0.f : gate >= 1.f ? raw
                        : -expm1f(gate * log_survival);
                dalpha = -(1.f - alpha) * log_survival;
            }
        }
        float next_T = T * (1.f - alpha);
        if (next_T < 1e-4f) break;  // Exclude current splat, as FastGS does.
        const float* color = colors + 3 * id;
        for (int k = 0; k < 3; ++k) {
            C[k] += T * alpha * color[k];
            dC[k] += (dT * alpha + T * dalpha) * color[k];
        }
        dT = dT * (1.f - alpha) - T * dalpha;
        T = next_T;
    }
    float vjp = 0.f;
    for (int k = 0; k < 3; ++k) {
        out_rgb[3 * p + k] = C[k] + T * bg[k];
        vjp += (dC[k] + dT * bg[k]) * adjoint[3 * p + k];
    }
    out_vjp[p] = vjp;
}

std::vector<torch::Tensor> shadow_forward(
    torch::Tensor means, torch::Tensor conics, torch::Tensor colors,
    torch::Tensor roles, torch::Tensor ranges, torch::Tensor point_list,
    torch::Tensor pixels, torch::Tensor bg, torch::Tensor adjoint,
    int width, int height, float gate) {
    TORCH_CHECK(means.is_cuda() && conics.is_cuda() && colors.is_cuda(), "projected rows must be CUDA");
    TORCH_CHECK(roles.is_cuda() && ranges.is_cuda() && point_list.is_cuda(), "tile lists must be CUDA");
    TORCH_CHECK(pixels.is_cuda() && bg.is_cuda() && adjoint.is_cuda(), "pixel inputs must be CUDA");
    TORCH_CHECK(means.is_contiguous() && conics.is_contiguous() && colors.is_contiguous(), "projected rows must be contiguous");
    TORCH_CHECK(roles.is_contiguous() && ranges.is_contiguous() && point_list.is_contiguous(), "tile lists must be contiguous");
    TORCH_CHECK(pixels.is_contiguous() && bg.is_contiguous() && adjoint.is_contiguous(), "pixel inputs must be contiguous");
    TORCH_CHECK(means.scalar_type() == at::kFloat && conics.scalar_type() == at::kFloat && colors.scalar_type() == at::kFloat, "projected rows must be float32");
    TORCH_CHECK(roles.scalar_type() == at::kInt && ranges.scalar_type() == at::kInt && point_list.scalar_type() == at::kInt && pixels.scalar_type() == at::kInt, "indices must be int32");
    TORCH_CHECK(bg.scalar_type() == at::kFloat && adjoint.scalar_type() == at::kFloat, "adjoint must be float32");
    TORCH_CHECK(means.size(1) == 2 && conics.size(1) == 4 && colors.size(1) == 3, "row shapes invalid");
    TORCH_CHECK(pixels.size(1) == 2 && adjoint.size(1) == 3 && pixels.size(0) == adjoint.size(0), "pixel shapes invalid");
    c10::cuda::CUDAGuard guard(means.device());
    auto rgb = torch::zeros({pixels.size(0), 3}, means.options());
    auto vjp = torch::zeros({pixels.size(0)}, means.options());
    int count = static_cast<int>(pixels.size(0));
    shadow_kernel<<<(count + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        means.data_ptr<float>(), conics.data_ptr<float>(), colors.data_ptr<float>(),
        roles.data_ptr<int>(), ranges.data_ptr<int>(), point_list.data_ptr<int>(),
        pixels.data_ptr<int>(), bg.data_ptr<float>(), adjoint.data_ptr<float>(),
        width, height, (width + 15) / 16, count, gate,
        rgb.data_ptr<float>(), vjp.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {rgb, vjp};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("shadow_forward", &shadow_forward);
}
