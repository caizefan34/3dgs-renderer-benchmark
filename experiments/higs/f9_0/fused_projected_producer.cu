// F9-0 forward-only gatherless projected-primitive producer.
// This TU intentionally does not modify gsplat, F4, F5, or backward.
#include <torch/extension.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Utils.cuh"

namespace f9 {
namespace cg = cooperative_groups;
using gsplat::mat2;
using gsplat::mat3;
using gsplat::vec2;
using gsplat::vec3;
using gsplat::vec4;

// Kept expression-for-expression with gsplat's degree-3 fast SH path.  The
// caller supplies the same world-space direction that _maybe_evaluate_sh
// materializes before invoking spherical_harmonics.
__device__ inline void sh3_color(const vec3 &dir, const float *coeffs, float *out) {
    const float inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    const float x = dir.x * inorm, y = dir.y * inorm, z = dir.z * inorm;
    const float z2 = z * z;
    const float fTmp0B = -1.092548430592079f * z;
    const float fC1 = x * x - y * y;
    const float fS1 = 2.f * x * y;
    const float pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
    const float pSH7 = fTmp0B * x, pSH5 = fTmp0B * y;
    const float pSH8 = 0.5462742152960395f * fC1, pSH4 = 0.5462742152960395f * fS1;
    const float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    const float fTmp1B = 1.445305721320277f * z;
    const float fC2 = x * fC1 - y * fS1, fS2 = x * fS1 + y * fC1;
    const float pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
    const float pSH13 = fTmp0C * x, pSH11 = fTmp0C * y;
    const float pSH14 = fTmp1B * fC1, pSH10 = fTmp1B * fS1;
    const float pSH15 = -0.5900435899266435f * fC2, pSH9 = -0.5900435899266435f * fS2;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        float r = 0.2820947917738781f * coeffs[c];
        r += 0.48860251190292f * (-y * coeffs[3 + c] + z * coeffs[6 + c] - x * coeffs[9 + c]);
        r += pSH4 * coeffs[12 + c] + pSH5 * coeffs[15 + c] + pSH6 * coeffs[18 + c]
           + pSH7 * coeffs[21 + c] + pSH8 * coeffs[24 + c];
        r += pSH9 * coeffs[27 + c] + pSH10 * coeffs[30 + c] + pSH11 * coeffs[33 + c]
           + pSH12 * coeffs[36 + c] + pSH13 * coeffs[39 + c] + pSH14 * coeffs[42 + c]
           + pSH15 * coeffs[45 + c];
        out[c] = fmaxf(r + 0.5f, 0.f);
    }
}

__global__ void f9_projected_producer_kernel(
    const int64_t n_visible, const int64_t * __restrict__ visible_ids,
    const float * __restrict__ means, const float * __restrict__ quats,
    const float * __restrict__ scales, const float * __restrict__ opacities,
    const float * __restrict__ coeffs, const float * __restrict__ viewmat,
    const float * __restrict__ K, const float * __restrict__ cam_position,
    const uint32_t width, const uint32_t height, const float eps2d,
    const float near_plane, const float far_plane, const float radius_clip,
    int32_t * __restrict__ radii, float * __restrict__ means2d,
    float * __restrict__ depths, float * __restrict__ conics,
    float * __restrict__ opacities_eval, float * __restrict__ colors_eval) {
    const int64_t i = cg::this_grid().thread_rank();
    if (i >= n_visible) return;
    const int64_t gid = visible_ids[i];
    const vec3 mean_w = glm::make_vec3(means + gid * 3);
    const mat3 R = mat3(viewmat[0], viewmat[4], viewmat[8], viewmat[1], viewmat[5], viewmat[9], viewmat[2], viewmat[6], viewmat[10]);
    const vec3 t = vec3(viewmat[3], viewmat[7], viewmat[11]);
    vec3 mean_c;
    gsplat::posW2C(R, t, mean_w, mean_c);
    if (mean_c.z < near_plane || mean_c.z > far_plane) { radii[i*2]=0; radii[i*2+1]=0; return; }
    mat3 covar;
    gsplat::quat_scale_to_covar_preci(glm::make_vec4(quats + gid * 4), glm::make_vec3(scales + gid * 3), &covar, nullptr);
    mat3 covar_c;
    gsplat::covarW2C(R, covar, covar_c);
    mat2 covar2d; vec2 mean2d_local;
    gsplat::persp_proj(mean_c, covar_c, K[0], K[4], K[2], K[5], width, height, covar2d, mean2d_local);
    float compensation;
    const float det = gsplat::add_blur(eps2d, covar2d, compensation);
    if (det <= 0.f) { radii[i*2]=0; radii[i*2+1]=0; return; }
    const mat2 covar2d_inv = glm::inverse(covar2d);
    // B2 calls fully_fused_projection with opacities=None.  Preserve that
    // exact fixed-extent contract; opacity is independently forwarded to F4.
    const float opacity = opacities[gid];
    const float extend = GAUSSIAN_EXTEND;
    const float radius_x = ceilf(extend * sqrtf(covar2d[0][0]));
    const float radius_y = ceilf(extend * sqrtf(covar2d[1][1]));
    if (radius_x <= radius_clip && radius_y <= radius_clip
        || mean2d_local.x + radius_x <= 0 || mean2d_local.x - radius_x >= width
        || mean2d_local.y + radius_y <= 0 || mean2d_local.y - radius_y >= height) {
        radii[i*2]=0; radii[i*2+1]=0; return;
    }
    radii[i*2] = static_cast<int32_t>(radius_x); radii[i*2+1] = static_cast<int32_t>(radius_y);
    means2d[i*2] = mean2d_local.x; means2d[i*2+1] = mean2d_local.y;
    depths[i] = mean_c.z;
    conics[i*3] = covar2d_inv[0][0]; conics[i*3+1] = covar2d_inv[0][1]; conics[i*3+2] = covar2d_inv[1][1];
    opacities_eval[i] = opacity;
    const vec3 dir = mean_w - vec3(cam_position[0], cam_position[1], cam_position[2]);
    sh3_color(dir, coeffs + gid * 16 * 3, colors_eval + i * 3);
}

std::vector<at::Tensor> fused_projected_producer(
    const at::Tensor &visible_ids, const at::Tensor &means, const at::Tensor &quats,
    const at::Tensor &scales, const at::Tensor &opacities, const at::Tensor &coeffs,
    const at::Tensor &viewmat, const at::Tensor &K, const at::Tensor &cam_position,
    int64_t width, int64_t height, double eps2d, double near_plane, double far_plane, double radius_clip) {
    TORCH_CHECK(visible_ids.is_cuda() && visible_ids.scalar_type() == at::kLong && visible_ids.dim() == 1, "visible_ids must be CUDA int64 [Nv]");
    for (const auto *t : {&means,&quats,&scales,&opacities,&coeffs,&viewmat,&K,&cam_position}) TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kFloat, "F9-0 requires FP32 CUDA inputs");
    TORCH_CHECK(means.sizes() == at::IntArrayRef({means.size(0),3}) && quats.sizes() == at::IntArrayRef({means.size(0),4}) && scales.sizes() == at::IntArrayRef({means.size(0),3}), "invalid master geometry shapes");
    TORCH_CHECK(coeffs.sizes() == at::IntArrayRef({means.size(0),16,3}), "F9-0 is fixed to current SH degree 3 / K=16");
    const auto n = visible_ids.numel();
    auto fo = means.options().dtype(at::kFloat); auto io = means.options().dtype(at::kInt);
    auto radii = at::zeros({n,2}, io), means2d = at::empty({n,2}, fo), depths = at::empty({n}, fo), conics = at::empty({n,3}, fo);
    auto opacity_eval = at::empty({n}, fo), colors_eval = at::empty({n,3}, fo);
    if (n) {
        constexpr int threads = 256;
        const int blocks = static_cast<int>((n + threads - 1) / threads);
        f9_projected_producer_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            n, visible_ids.contiguous().data_ptr<int64_t>(), means.contiguous().data_ptr<float>(), quats.contiguous().data_ptr<float>(),
            scales.contiguous().data_ptr<float>(), opacities.contiguous().data_ptr<float>(), coeffs.contiguous().data_ptr<float>(),
            viewmat.contiguous().data_ptr<float>(), K.contiguous().data_ptr<float>(), cam_position.contiguous().data_ptr<float>(),
            width, height, static_cast<float>(eps2d), static_cast<float>(near_plane), static_cast<float>(far_plane), static_cast<float>(radius_clip),
            radii.data_ptr<int32_t>(), means2d.data_ptr<float>(), depths.data_ptr<float>(), conics.data_ptr<float>(), opacity_eval.data_ptr<float>(), colors_eval.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {radii, means2d, depths, conics, opacity_eval, colors_eval};
}
} // namespace f9

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused_projected_producer", &f9::fused_projected_producer, "F9-0 gatherless projected primitive producer"); }
