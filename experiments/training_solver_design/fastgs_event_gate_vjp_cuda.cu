// Isolated diagnostic: all native split families, one fixed-support full-view VJP.
// Projected rows and sorted tile lists come from unchanged author FastGS CUDA.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void event_gate_kernel(
    const float* means, const float* conics, const float* colors,
    const int* roles, const int* families, const int* ranges,
    const int* point_list, const float* adjoint,
    int width, int height, int tile_width, float gate, int debug_family,
    float* out_rgb, float* out_family_vjp, float* out_debug_pixel) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= width * height) return;
    int x = p % width, y = p / width;
    int tile = (y / 16) * tile_width + (x / 16);
    int begin = ranges[2 * tile], end = ranges[2 * tile + 1];
    float T = 1.f, C[3] = {0.f, 0.f, 0.f};
    for (int j = begin; j < end; ++j) {
        int id = point_list[j];
        float dx = means[2 * id] - x, dy = means[2 * id + 1] - y;
        const float* co = conics + 4 * id;
        float power = -0.5f * (co[0] * dx * dx + co[2] * dy * dy) - co[1] * dx * dy;
        if (power > 0.f) continue;
        float raw = fminf(0.99f, co[3] * expf(power));
        if (raw < 1.f / 255.f) continue;
        int role = roles[id];
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
        const float* color = colors + 3 * id;
        for (int k = 0; k < 3; ++k) C[k] += T * alpha * color[k];
        T = next_T;
    }
    for (int k = 0; k < 3; ++k) out_rgb[3 * p + k] = C[k];

    // Rewalk the accepted prefix. For a layer with incoming transmittance T,
    // dI/dalpha = (T*c - I + C_prefix)/(1-alpha), including the black suffix.
    T = 1.f;
    float prefix[3] = {0.f, 0.f, 0.f};
    float debug_vjp = 0.f;
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
        if (next_T < 1e-4f) break;
        const float* color = colors + 3 * id;
        if (role != 0) {
            float dot = 0.f;
            for (int k = 0; k < 3; ++k) {
                float image_derivative = (T * color[k] - C[k] + prefix[k]) / (1.f - alpha);
                dot += adjoint[3 * p + k] * image_derivative;
            }
            float contribution = dot * dalpha;
            atomicAdd(out_family_vjp + families[id], contribution);
            if (families[id] == debug_family) debug_vjp += contribution;
        }
        for (int k = 0; k < 3; ++k) prefix[k] += T * alpha * color[k];
        T = next_T;
    }
    out_debug_pixel[p] = debug_vjp;
}

std::vector<torch::Tensor> event_gate_forward(
    torch::Tensor means, torch::Tensor conics, torch::Tensor colors,
    torch::Tensor roles, torch::Tensor families, torch::Tensor ranges,
    torch::Tensor point_list, torch::Tensor adjoint,
    int width, int height, int family_count, int debug_family, float gate) {
    TORCH_CHECK(means.is_cuda() && conics.is_cuda() && colors.is_cuda(), "projected rows must be CUDA");
    TORCH_CHECK(roles.is_cuda() && families.is_cuda() && ranges.is_cuda() && point_list.is_cuda(), "indices must be CUDA");
    TORCH_CHECK(adjoint.is_cuda(), "adjoint must be CUDA");
    TORCH_CHECK(means.is_contiguous() && conics.is_contiguous() && colors.is_contiguous(), "projected rows must be contiguous");
    TORCH_CHECK(roles.is_contiguous() && families.is_contiguous() && ranges.is_contiguous() && point_list.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(adjoint.is_contiguous(), "adjoint must be contiguous");
    TORCH_CHECK(means.scalar_type() == at::kFloat && conics.scalar_type() == at::kFloat && colors.scalar_type() == at::kFloat && adjoint.scalar_type() == at::kFloat, "values must be float32");
    TORCH_CHECK(roles.scalar_type() == at::kInt && families.scalar_type() == at::kInt && ranges.scalar_type() == at::kInt && point_list.scalar_type() == at::kInt, "indices must be int32");
    TORCH_CHECK(adjoint.numel() == width * height * 3, "adjoint shape invalid");
    TORCH_CHECK(roles.numel() == means.size(0) && families.numel() == means.size(0), "row count mismatch");
    TORCH_CHECK(ranges.numel() >= 2 * ((width + 15) / 16) * ((height + 15) / 16), "range count invalid");
    TORCH_CHECK(family_count > 0 && debug_family >= 0 && debug_family < family_count, "family arguments invalid");
    c10::cuda::CUDAGuard guard(means.device());
    auto rgb = torch::zeros({height, width, 3}, means.options());
    auto grad = torch::zeros({family_count}, means.options());
    auto debug = torch::zeros({height, width}, means.options());
    int pixels = width * height;
    event_gate_kernel<<<(pixels + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        means.data_ptr<float>(), conics.data_ptr<float>(), colors.data_ptr<float>(),
        roles.data_ptr<int>(), families.data_ptr<int>(), ranges.data_ptr<int>(),
        point_list.data_ptr<int>(), adjoint.data_ptr<float>(),
        width, height, (width + 15) / 16, gate, debug_family,
        rgb.data_ptr<float>(), grad.data_ptr<float>(), debug.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {rgb, grad, debug};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("event_gate_forward", &event_gate_forward);
}
