#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

constexpr float kAlphaThreshold = 1.0f / 255.0f;

__global__ void evaluate_fragments_kernel(
    const float* means, const float* conics, const float* opacities, const int64_t* tiles,
    int64_t width, int64_t height, int64_t tile_size, float* max_alpha, float* max_sigma,
    int32_t* max_x, int32_t* max_y) {
    const int case_id = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float alpha_s[256];
    __shared__ float sigma_s[256];
    __shared__ int x_s[256];
    __shared__ int y_s[256];
    const int64_t tile = tiles[case_id];
    const int64_t tile_w = (width + tile_size - 1) / tile_size;
    const int x0 = (tile % tile_w) * tile_size;
    const int y0 = (tile / tile_w) * tile_size;
    const int valid_w = min((int)tile_size, (int)width - x0);
    const int valid_h = min((int)tile_size, (int)height - y0);
    float alpha = -1.0f, sigma = 0.0f;
    int x = -1, y = -1;
    if (tid < valid_w * valid_h) {
        x = x0 + tid % valid_w;
        y = y0 + tid / valid_w;
        const float dx = means[case_id * 2] - ((float)x + 0.5f);
        const float dy = means[case_id * 2 + 1] - ((float)y + 0.5f);
        const float a = conics[case_id * 3];
        const float b = conics[case_id * 3 + 1];
        const float c = conics[case_id * 3 + 2];
        sigma = 0.5f * (a * dx * dx + 2.0f * b * dx * dy + c * dy * dy);
        alpha = fminf(0.999f, opacities[case_id] * __expf(-sigma));
        if (sigma < 0.0f || alpha < kAlphaThreshold) alpha = -1.0f;
    }
    alpha_s[tid] = alpha; sigma_s[tid] = sigma; x_s[tid] = x; y_s[tid] = y;
    __syncthreads();
    if (tid == 0) {
        int best = 0;
        for (int i = 1; i < 256; ++i) if (alpha_s[i] > alpha_s[best]) best = i;
        max_alpha[case_id] = alpha_s[best]; max_sigma[case_id] = sigma_s[best];
        max_x[case_id] = x_s[best]; max_y[case_id] = y_s[best];
    }
}

void evaluate_fragments_cuda(torch::Tensor means, torch::Tensor conics, torch::Tensor opacities,
                             torch::Tensor tiles, int64_t width, int64_t height, int64_t tile_size,
                             torch::Tensor max_alpha, torch::Tensor sigma, torch::Tensor pixel_x,
                             torch::Tensor pixel_y) {
    const int64_t n = means.size(0);
    evaluate_fragments_kernel<<<n, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        means.data_ptr<float>(), conics.data_ptr<float>(), opacities.data_ptr<float>(), tiles.data_ptr<int64_t>(),
        width, height, tile_size, max_alpha.data_ptr<float>(), sigma.data_ptr<float>(),
        pixel_x.data_ptr<int32_t>(), pixel_y.data_ptr<int32_t>());
}
