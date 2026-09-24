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
    /// Compact-copy the visible rows of the FP32 master tensors.
    ///
    /// ``visible_ids`` is the sorted int64 index list produced by the HiGS
    /// culling pass (union over cameras). The five master tensors keep their
    /// original shapes (``colors`` may be ``[N, 3]`` pre-activated RGB or
    /// ``[N, K, D]`` SH coefficients). All tensors must be contiguous FP32 on
    /// the current CUDA device. The outputs are freshly allocated contiguous
    /// tensors with ``[N_visible, ...]`` leading dims.
    ///
    /// This exists because PyTorch's row gather dispatches to a vectorized
    /// path whenever the row width is a multiple of four floats, which is
    /// pathologically slow for random row indices (quats ``[N,4]`` and colors
    /// ``[N,16,3]`` measured ~1.7 ms each on A100 for a 2.27 M visible subset).
    /// A single-purpose element-wise kernel avoids that dispatch entirely.
    std::tuple<
        at::Tensor, // v_means    [N_visible, 3]
        at::Tensor, // v_quats    [N_visible, 4]
        at::Tensor, // v_scales   [N_visible, 3]
        at::Tensor, // v_opacities [N_visible]
        at::Tensor  // v_colors   [N_visible, ...] (same trailing shape as colors)
        >
    higs_gather_visible(
        const at::Tensor &means,      // [N, 3] float32
        const at::Tensor &quats,      // [N, 4] float32
        const at::Tensor &scales,     // [N, 3] float32
        const at::Tensor &opacities,  // [N] float32
        const at::Tensor &colors,     // [N, ...] float32
        const at::Tensor &visible_ids // [N_visible] int64
    );

    /// Compact-copy an arbitrary subset of rows of one FP32 tensor.
    ///
    /// Same purpose as ``higs_gather_visible`` but for a single tensor of any
    /// trailing shape; used by densify/prune and the Adam-state sync, whose
    /// row gathers would otherwise hit the same slow vectorized PyTorch
    /// dispatch (row widths divisible by four floats).
    at::Tensor higs_gather_rows(
        const at::Tensor &src,     // [N, ...] float32
        const at::Tensor &row_ids, // [N_rows] int64
        const bool zero_on_neg = false // negative ids write zeros (Adam-state sync)
    );

    /// Union-visibility mask over cameras from the culling projection radii:
    /// mask[n] = any_c((r[c,n,0] > 0) && (r[c,n,1] > 0)). Output [N] bool.
    at::Tensor higs_union_visible_mask(
        const at::Tensor &radii // [C, N, 2] float32
    );
} // namespace gaussian_render_inference_scene
} // namespace gsplat
