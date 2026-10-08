// Diagnostic fixed-list RGB compositor. Not a production gsplat replacement.
// The gate constants match the audited P11 gsplat backend (Common.h).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

constexpr float kMaxAlpha = 0.99f;
constexpr float kAlphaThreshold = 1.0f / 255.0f;
constexpr float kTransmittanceThreshold = 1e-4f;

struct Weight {
    float a, raw, vis, dx, dy;
    bool valid;
};

__device__ Weight weight(const float* row, float px, float py) {
    Weight w;
    w.dx = row[0] - px;
    w.dy = row[1] - py;
    float sigma = 0.5f * (row[2] * w.dx * w.dx + row[4] * w.dy * w.dy)
                  + row[3] * w.dx * w.dy;
    w.vis = __expf(-sigma);
    w.raw = row[8] * w.vis;
    w.valid = sigma >= 0.f;
    w.a = w.valid ? fminf(kMaxAlpha, w.raw) : 0.f;
    return w;
}

__device__ float child_alpha(float a) {
    // Clamp is applied to the parent response before factorization. Its upper
    // bound also keeps the derivative finite, including sigmoid saturation.
    return a / (1.f + sqrtf(1.f - a));
}

struct Group {
    Weight w1, w2;
    float h1, h2, a, b[3];
    const float* r1;
    const float* r2;
    int pair;
};

__device__ Group group_at(const float* base, const float* children,
                         const int* pairs, int id, float px, float py) {
    Group g;
    g.pair = pairs[id];
    g.r1 = g.pair < 0 ? base + id * 9 : children + g.pair * 18;
    g.w1 = weight(g.r1, px, py);
    if (g.pair < 0) {
        g.h1 = g.w1.a;
        g.h2 = 0.f;
        g.a = g.h1;
        for (int k = 0; k < 3; ++k) g.b[k] = g.h1 * g.r1[5 + k];
    } else {
        g.r2 = g.r1 + 9;
        g.w2 = weight(g.r2, px, py);
        g.h1 = child_alpha(g.w1.a);
        g.h2 = child_alpha(g.w2.a);
        float root1 = sqrtf(1.f - g.w1.a), root2 = sqrtf(1.f - g.w2.a);
        // 1-sqrt((1-a1)(1-a2)), written to preserve a1==a2 exactly.
        // The naive h1+(1-h1)*h2 can round across a native alpha gate.
        g.a = g.w1.a + root1 / (root1 + root2) * (g.w2.a - g.w1.a);
        float w2 = g.a - g.h1;
        for (int k = 0; k < 3; ++k)
            // Same-color children recover the ordinary premultiplied color
            // without a separate equality branch or altered derivative.
            g.b[k] = g.a * g.r1[5 + k] + w2 * (g.r2[5 + k] - g.r1[5 + k]);
    }
    return g;
}

__global__ void pair_fwd_kernel(
    const float* base, const float* children, const int* pairs,
    const int64_t* offsets, const int* ids, const float* bg,
    int width, int height, int tile, int64_t nids,
    float* rgb, float* alpha, int64_t* last) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= width * height) return;
    int x = p % width, y = p / width;
    int tw = (width + tile - 1) / tile, th = (height + tile - 1) / tile;
    int t = (y / tile) * tw + x / tile;
    int64_t begin = offsets[t], end = t + 1 == tw * th ? nids : offsets[t + 1];
    float trans = 1.f, color[3] = {0.f, 0.f, 0.f};
    int64_t final_id = -1;
    for (int64_t j = begin; j < end; ++j) {
        Group g = group_at(base, children, pairs, ids[j], x + 0.5f, y + 0.5f);
        if (g.a < kAlphaThreshold) continue;
        float next = trans * (1.f - g.a);
        // Atomic group decision: neither child contributes if the complete
        // pair would trigger the native exclusive early-termination rule.
        if (next <= kTransmittanceThreshold) break;
        for (int k = 0; k < 3; ++k) color[k] += trans * g.b[k];
        trans = next;
        final_id = j;
    }
    for (int k = 0; k < 3; ++k) rgb[p * 3 + k] = color[k] + trans * bg[k];
    alpha[p] = 1.f - trans;
    last[p] = final_id;
}

__device__ void scatter(const float* row, const Weight& w, float va,
                        float color_weight, const float* vc, float* grad) {
    for (int k = 0; k < 3; ++k) atomicAdd(grad + 5 + k, color_weight * vc[k]);
    if (!w.valid || w.raw > kMaxAlpha) return;
    float vs = -w.raw * va;
    atomicAdd(grad + 0, vs * (row[2] * w.dx + row[3] * w.dy));
    atomicAdd(grad + 1, vs * (row[3] * w.dx + row[4] * w.dy));
    atomicAdd(grad + 2, 0.5f * vs * w.dx * w.dx);
    atomicAdd(grad + 3, vs * w.dx * w.dy);
    atomicAdd(grad + 4, 0.5f * vs * w.dy * w.dy);
    atomicAdd(grad + 8, w.vis * va);
}

__global__ void pair_bwd_kernel(
    const float* base, const float* children, const int* pairs,
    const int64_t* offsets, const int* ids, const float* bg,
    int width, int height, int tile, const float* alpha, const int64_t* last,
    const float* v_rgb, const float* v_alpha, float* v_base, float* v_children) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= width * height || last[p] < 0) return;
    int x = p % width, y = p / width;
    int tw = (width + tile - 1) / tile;
    int t = (y / tile) * tw + x / tile;
    float tfinal = 1.f - alpha[p], trans = tfinal;
    float buffer[3] = {0.f, 0.f, 0.f};
    const float* vc = v_rgb + p * 3;
    for (int64_t j = last[p]; j >= offsets[t]; --j) {
        int id = ids[j];
        Group g = group_at(base, children, pairs, id, x + 0.5f, y + 0.5f);
        if (g.a < kAlphaThreshold) continue;
        float inv = 1.f / fmaxf(1e-6f, 1.f - g.a);
        trans *= inv;
        float v_survival = -tfinal * v_alpha[p];
        for (int k = 0; k < 3; ++k)
            v_survival += (buffer[k] + tfinal * bg[k]) * vc[k];
        v_survival *= inv;
        if (g.pair < 0) {
            float va = -v_survival;
            for (int k = 0; k < 3; ++k) va += trans * g.r1[5 + k] * vc[k];
            scatter(g.r1, g.w1, va, trans * g.a, vc, v_base + id * 9);
        } else {
            float v1 = -(1.f - g.h2) * v_survival;
            float v2 = -(1.f - g.h1) * v_survival;
            for (int k = 0; k < 3; ++k) {
                v1 += trans * (g.r1[5 + k] - g.h2 * g.r2[5 + k]) * vc[k];
                v2 += trans * (1.f - g.h1) * g.r2[5 + k] * vc[k];
            }
            v1 *= 0.5f / sqrtf(1.f - g.w1.a);
            v2 *= 0.5f / sqrtf(1.f - g.w2.a);
            scatter(g.r1, g.w1, v1, trans * g.h1, vc, v_children + g.pair * 18);
            scatter(g.r2, g.w2, v2, trans * (g.a - g.h1),
                    vc, v_children + g.pair * 18 + 9);
        }
        for (int k = 0; k < 3; ++k) buffer[k] += trans * g.b[k];
    }
}

std::vector<torch::Tensor> pair_forward(
    torch::Tensor base, torch::Tensor children, torch::Tensor pairs,
    torch::Tensor offsets, torch::Tensor ids, torch::Tensor bg,
    int width, int height, int tile) {
    c10::cuda::CUDAGuard guard(base.device());
    auto rgb = torch::empty({height, width, 3}, base.options());
    auto alpha = torch::empty({height, width, 1}, base.options());
    auto last = torch::empty({height, width}, offsets.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    pair_fwd_kernel<<<(width * height + 127) / 128, 128, 0, stream>>>(
        base.data_ptr<float>(), children.data_ptr<float>(), pairs.data_ptr<int>(),
        offsets.data_ptr<int64_t>(), ids.data_ptr<int>(), bg.data_ptr<float>(),
        width, height, tile, ids.numel(), rgb.data_ptr<float>(),
        alpha.data_ptr<float>(), last.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {rgb, alpha, last};
}

std::vector<torch::Tensor> pair_backward(
    torch::Tensor base, torch::Tensor children, torch::Tensor pairs,
    torch::Tensor offsets, torch::Tensor ids, torch::Tensor bg,
    int width, int height, int tile, torch::Tensor alpha, torch::Tensor last,
    torch::Tensor v_rgb, torch::Tensor v_alpha) {
    c10::cuda::CUDAGuard guard(base.device());
    auto vb = torch::zeros_like(base), vk = torch::zeros_like(children);
    auto stream = at::cuda::getCurrentCUDAStream();
    pair_bwd_kernel<<<(width * height + 127) / 128, 128, 0, stream>>>(
        base.data_ptr<float>(), children.data_ptr<float>(), pairs.data_ptr<int>(),
        offsets.data_ptr<int64_t>(), ids.data_ptr<int>(), bg.data_ptr<float>(),
        width, height, tile, alpha.data_ptr<float>(), last.data_ptr<int64_t>(),
        v_rgb.data_ptr<float>(), v_alpha.data_ptr<float>(), vb.data_ptr<float>(), vk.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {vb, vk};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &pair_forward);
    m.def("backward", &pair_backward);
}
