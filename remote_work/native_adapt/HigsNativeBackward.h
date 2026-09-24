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

#pragma once

#include <ATen/core/Tensor.h>
#include <cstdint>
#include <tuple>

namespace gsplat
{
namespace gaussian_render_inference_scene
{
    /// Native CUDA backward for the differentiable HiGS forward pipeline.
    ///
    /// This is a self-contained backward implementation for the exact forward
    /// math used by ``gsplat.rendering.rasterization`` (classic 3DGS blend):
    ///
    ///   sigma = 0.5 * (conic.x * dx^2 + conic.z * dy^2) + conic.y * dx * dy
    ///   alpha = min(MAX_ALPHA, opacity * exp(-sigma))
    ///   C     = sum_i T_i * alpha_i * color_i + T_N * background
    ///
    /// The kernel consumes the *forward-captured* state (means2d/conics/colors/
    /// opacities, per-tile sorted intersection ids, last contributing ids and
    /// render alphas) so the backward uses the exact same scene/order/visibility
    /// version as the forward. No recomputation of the rasterization pipeline
    /// happens in the backward.
    ///
    /// Gradient outputs are written to FP32 master tensors:
    ///   v_means [N, 3], v_quats [N, 4], v_scales [N, 3], v_opacities [N],
    ///   v_colors_master (evaluated-RGB [N, D] or SH coefficients [N, K, D]),
    ///   v_backgrounds [3].
    std::tuple<
        at::Tensor, // v_means    [N, 3]
        at::Tensor, // v_quats    [N, 4]
        at::Tensor, // v_scales   [N, 3]
        at::Tensor, // v_opacities [N]
        at::Tensor, // v_colors_master
        at::Tensor, // v_backgrounds [3]
        at::Tensor  // v_means2d [I*N, 2]
        >
    higs_rasterize_backward(
        // ---- forward-captured rasterization state (flattened [I*N, ...]) ----
        const at::Tensor &means2d,      // [I*N, 2] float32
        const at::Tensor &conics,       // [I*N, 3] float32
        const at::Tensor &colors_eval,  // [I*N, 3] float32 (RGB actually fed to rasterize)
        const at::Tensor &opacities,    // [I*N] float32 (compensated == raw in classic mode)
        const at::optional<at::Tensor> &backgrounds, // [I, 3] float32 or nullopt
        const at::Tensor &tile_offsets, // [I, tile_h, tile_w] int32
        const at::Tensor &flatten_ids,  // [n_isects] int32
        const at::optional<at::Tensor> &active_tiles, // [AT] int32 selected tile ids or nullopt
        const at::Tensor &render_alphas, // [I, H, W] float32
        const at::Tensor &last_ids,     // [I, H, W] int32
        // ---- projection VJP inputs ----
        const at::Tensor &means,        // [M, 3] master float32
        const at::Tensor &quats,        // [M, 4] master float32
        const at::Tensor &scales,       // [M, 3] master float32
        const at::Tensor &radii,        // [I, N, 2] int32
        const at::Tensor &viewmats,     // [1, C, 4, 4] float32
        const at::Tensor &Ks,           // [1, C, 3, 3] float32
        // ---- forward hyper params ----
        int64_t width,
        int64_t height,
        int64_t tile_size,
        double eps2d,
        int64_t camera_model,
        // ---- output gradients (from the loss) ----
        const at::Tensor &v_render_colors, // [I, H, W, 3] float32
        const at::Tensor &v_render_alphas, // [I, H, W] float32
        // ---- SH evaluation backward inputs (nullopt for pre-activated RGB) ----
        const at::optional<at::Tensor> &sh_coeffs, // [N, K, D] float32
        int64_t sh_degree,
        // ---- master-gradient outputs (pre-zeroed, written in place) ----
        const at::Tensor &visible_ids,    // [N] int64 ascending master indices
        const at::Tensor &grad_means,     // [M, 3] float32
        const at::Tensor &grad_quats,     // [M, 4] float32
        const at::Tensor &grad_scales,    // [M, 3] float32
        const at::Tensor &grad_opacities, // [M] float32
        const at::Tensor &grad_colors     // [M, K, D] (SH) or [M, D] (RGB) float32
    );
} // namespace gaussian_render_inference_scene
} // namespace gsplat
