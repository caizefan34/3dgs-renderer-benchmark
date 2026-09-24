/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Compact-copy kernel for the differentiable HiGS forward path.
//
// PyTorch's ``tensor[visible_ids]`` row gather dispatches to a vectorized
// copy whenever the row width is a multiple of four floats, and that path is
// pathologically slow for random row indices on large scenes (bicycle with a
// 2.27 M visible subset: ~1.7 ms for quats [N,4] and ~1.7 ms for colors
// [N,16,3], vs ~0.1 ms for non-multiple-of-4 widths). This single-purpose
// element-wise kernel avoids the bad dispatch: the five gathers together drop
// from ~4.4 ms to ~0.9 ms on the same scene.

#include "Config.h"

#include <ATen/ops/empty.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "GatherVisible.h"

namespace gsplat
{
namespace gaussian_render_inference_scene
{
namespace
{
// ``ZERO_NEG``: a negative row id writes zeros instead of gathering. This is
// used by the Adam-state sync, where ``row_ids[i] == -1`` marks brand-new
// Gaussians whose state must be zero-initialized; fusing the zero-fill into
// the gather removes the ``zeros_like`` memset and the scatter pass.
template<int INNER, bool ZERO_NEG>
__global__ void gather_rows_kernel_t(
    const float *__restrict__ src,
    const int64_t *__restrict__ idx,
    float *__restrict__ dst,
    const int64_t n_rows)
{
    const int64_t total = n_rows * INNER;
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    if(INNER == 1)
    {
        for(; t < total; t += stride)
        {
            const int64_t r = idx[t];
            if(ZERO_NEG && r < 0)
            {
                dst[t] = 0.0f;
            }
            else
            {
                dst[t] = src[r];
            }
        }
        return;
    }
    for(; t < total; t += stride)
    {
        const int64_t row = t / INNER;
        const int64_t r = idx[row];
        if(ZERO_NEG && r < 0)
        {
            dst[t] = 0.0f;
        }
        else
        {
            dst[t] = src[r * INNER + (t - row * INNER)];
        }
    }
}

template<bool ZERO_NEG>
__global__ void gather_rows_kernel_gen(
    const float *__restrict__ src,
    const int64_t *__restrict__ idx,
    float *__restrict__ dst,
    const int64_t n_rows,
    const int64_t inner)
{
    const int64_t total = n_rows * inner;
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for(; t < total; t += stride)
    {
        const int64_t row = t / inner;
        const int64_t r = idx[row];
        if(ZERO_NEG && r < 0)
        {
            dst[t] = 0.0f;
        }
        else
        {
            dst[t] = src[r * inner + (t - row * inner)];
        }
    }
}

template<bool ZERO_NEG>
void launch_gather_rows_impl(
    const at::Tensor &src_flat, // [N, inner] contiguous FP32
    const at::Tensor &ids,      // [N_rows] int64
    at::Tensor &dst,            // [N_rows, inner] contiguous FP32
    cudaStream_t stream)
{
    const int64_t n_rows = ids.numel();
    const int64_t inner = dst.numel() / std::max<int64_t>(n_rows, 1);
    const int64_t total = n_rows * inner;
    if(total == 0)
    {
        return;
    }
    const int threads = 256;
    const int grid    = static_cast<int>(std::min<int64_t>(65535, (total + threads - 1) / threads));
    const float *s    = src_flat.const_data_ptr<float>();
    const int64_t *id = ids.const_data_ptr<int64_t>();
    float *d          = dst.data_ptr<float>();
    switch(inner)
    {
    case 1:
        gather_rows_kernel_t<1, ZERO_NEG><<<grid, threads, 0, stream>>>(s, id, d, n_rows);
        break;
    case 3:
        gather_rows_kernel_t<3, ZERO_NEG><<<grid, threads, 0, stream>>>(s, id, d, n_rows);
        break;
    case 4:
        gather_rows_kernel_t<4, ZERO_NEG><<<grid, threads, 0, stream>>>(s, id, d, n_rows);
        break;
    case 12:
        gather_rows_kernel_t<12, ZERO_NEG><<<grid, threads, 0, stream>>>(s, id, d, n_rows);
        break;
    case 27:
        gather_rows_kernel_t<27, ZERO_NEG><<<grid, threads, 0, stream>>>(s, id, d, n_rows);
        break;
    case 48:
        gather_rows_kernel_t<48, ZERO_NEG><<<grid, threads, 0, stream>>>(s, id, d, n_rows);
        break;
    default:
        gather_rows_kernel_gen<ZERO_NEG><<<grid, threads, 0, stream>>>(s, id, d, n_rows, inner);
        break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_gather_rows(
    const at::Tensor &src_flat, // [N, inner] contiguous FP32
    const at::Tensor &ids,      // [N_rows] int64
    at::Tensor &dst,            // [N_rows, inner] contiguous FP32
    cudaStream_t stream,
    const bool zero_on_neg = false)
{
    if(zero_on_neg)
    {
        launch_gather_rows_impl<true>(src_flat, ids, dst, stream);
    }
    else
    {
        launch_gather_rows_impl<false>(src_flat, ids, dst, stream);
    }
}
} // namespace

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
higs_gather_visible(
    const at::Tensor &means,
    const at::Tensor &quats,
    const at::Tensor &scales,
    const at::Tensor &opacities,
    const at::Tensor &colors,
    const at::Tensor &visible_ids)
{
    TORCH_CHECK(means.is_cuda(), "higs_gather_visible: means must be CUDA");
    TORCH_CHECK(means.scalar_type() == at::kFloat, "higs_gather_visible: FP32 master tensors required");
    TORCH_CHECK(visible_ids.scalar_type() == at::kLong, "higs_gather_visible: int64 visible_ids required");

    const at::cuda::OptionalCUDAGuard device_guard(means.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const at::Tensor means_c     = means.contiguous();
    const at::Tensor quats_c     = quats.contiguous();
    const at::Tensor scales_c    = scales.contiguous();
    const at::Tensor opacities_c = opacities.contiguous();
    const at::Tensor colors_c    = colors.contiguous();
    const at::Tensor ids_c       = visible_ids.contiguous();

    const int64_t n_rows = ids_c.numel();
    const at::TensorOptions opts = means_c.options();

    at::Tensor v_means     = at::empty({n_rows, 3}, opts);
    at::Tensor v_quats     = at::empty({n_rows, 4}, opts);
    at::Tensor v_scales    = at::empty({n_rows, 3}, opts);
    at::Tensor v_opacities = at::empty({n_rows}, opts);
    std::vector<int64_t> color_sizes = colors_c.sizes().vec();
    color_sizes[0] = n_rows;
    at::Tensor v_colors = at::empty(color_sizes, opts);

    if(n_rows == 0)
    {
        return {v_means, v_quats, v_scales, v_opacities, v_colors};
    }

    const at::Tensor colors_flat = colors_c.dim() == 1 ? colors_c.reshape({-1, 1}) : colors_c.flatten(1, -1);

    launch_gather_rows(means_c, ids_c, v_means, stream);
    launch_gather_rows(quats_c, ids_c, v_quats, stream);
    launch_gather_rows(scales_c, ids_c, v_scales, stream);
    at::Tensor v_opacities_2d = v_opacities.reshape({-1, 1});
    launch_gather_rows(opacities_c.reshape({-1, 1}), ids_c, v_opacities_2d, stream);
    at::Tensor v_colors_flat = v_colors.reshape({n_rows, -1});
    launch_gather_rows(colors_flat, ids_c, v_colors_flat, stream);

    return {v_means, v_quats, v_scales, v_opacities, v_colors};
}

at::Tensor higs_gather_rows(
    const at::Tensor &src,
    const at::Tensor &row_ids,
    const bool zero_on_neg)
{
    TORCH_CHECK(src.is_cuda(), "higs_gather_rows: src must be CUDA");
    TORCH_CHECK(src.scalar_type() == at::kFloat, "higs_gather_rows: FP32 tensor required");
    TORCH_CHECK(row_ids.scalar_type() == at::kLong, "higs_gather_rows: int64 row_ids required");

    const at::cuda::OptionalCUDAGuard device_guard(src.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const at::Tensor src_c = src.contiguous();
    const at::Tensor ids_c = row_ids.contiguous();

    const int64_t n_rows = ids_c.numel();
    const at::TensorOptions opts = src_c.options();
    std::vector<int64_t> sizes = src_c.sizes().vec();
    sizes[0] = n_rows;
    at::Tensor dst = at::empty(sizes, opts);
    if(n_rows == 0)
    {
        return dst;
    }
    const at::Tensor src_flat = src_c.dim() == 1 ? src_c.reshape({-1, 1}) : src_c.flatten(1, -1);
    at::Tensor dst_flat = dst.reshape({n_rows, -1});
    launch_gather_rows(src_flat, ids_c, dst_flat, stream, zero_on_neg);
    return dst;
}

// Union visibility over cameras: a Gaussian is visible if BOTH radius
// components are > 0 in at least one camera. One thread per Gaussian.
template<typename T>
__global__ void union_visible_mask_kernel_t(
    const T *__restrict__ radii,        // [C, N, 2]
    bool *__restrict__ mask,            // [N]
    const int64_t C,
    const int64_t N)
{
    const int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(n >= N)
    {
        return;
    }
    bool vis = false;
    for(int64_t c = 0; c < C && !vis; ++c)
    {
        const T *r = radii + (c * N + n) * 2;
        vis = (r[0] > T(0)) && (r[1] > T(0));
    }
    mask[n] = vis;
}

at::Tensor higs_union_visible_mask(
    const at::Tensor &radii) // [C, N, 2] float32 or int32
{
    TORCH_CHECK(radii.is_cuda(), "higs_union_visible_mask: radii must be CUDA");
    TORCH_CHECK(
        radii.scalar_type() == at::kFloat || radii.scalar_type() == at::kInt,
        "higs_union_visible_mask: FP32 or int32 radii required");
    TORCH_CHECK(radii.dim() == 3 && radii.size(2) == 2, "higs_union_visible_mask: expected [C, N, 2]");

    const at::cuda::OptionalCUDAGuard device_guard(radii.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const at::Tensor radii_c = radii.contiguous();
    const int64_t C = radii_c.size(0);
    const int64_t N = radii_c.size(1);

    at::Tensor mask = at::empty({N}, radii_c.options().dtype(at::kBool));
    if(N == 0)
    {
        return mask;
    }

    const int threads = 256;
    const int grid = static_cast<int>(std::min<int64_t>(65535, (N + threads - 1) / threads));
    if(radii_c.scalar_type() == at::kInt)
    {
        union_visible_mask_kernel_t<int><<<grid, threads, 0, stream>>>(
            radii_c.const_data_ptr<int>(), mask.data_ptr<bool>(), C, N);
    }
    else
    {
        union_visible_mask_kernel_t<float><<<grid, threads, 0, stream>>>(
            radii_c.const_data_ptr<float>(), mask.data_ptr<bool>(), C, N);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return mask;
}
} // namespace gaussian_render_inference_scene
} // namespace gsplat
