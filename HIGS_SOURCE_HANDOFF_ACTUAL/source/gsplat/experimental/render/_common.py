# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Shared helpers for experimental render wrappers."""

from __future__ import annotations

import torch


def check_inference_grad_mode() -> None:
    """Raise if autograd is enabled and inference mode is off."""
    if torch.is_grad_enabled() and not torch.is_inference_mode_enabled():
        raise RuntimeError(
            "rasterize_gaussian_inference_scene requires "
            "torch.inference_mode() or torch.no_grad()"
        )


def check_trainable_grad_mode(differentiable: bool = False) -> None:
    """Raise if grad mode is incompatible with the requested mode.

    - When ``differentiable=False``: behaves like :func:`check_inference_grad_mode`.
    - When ``differentiable=True``: requires autograd enabled and inference mode OFF.
    """
    if not differentiable:
        check_inference_grad_mode()
    else:
        if torch.is_inference_mode_enabled():
            raise RuntimeError(
                "rasterize_gaussian_higs_trainable(differentiable=True) requires "
                "autograd to be enabled, but inference mode is active."
            )
        if not torch.is_grad_enabled():
            raise RuntimeError(
                "rasterize_gaussian_higs_trainable(differentiable=True) requires "
                "grad to be enabled."
            )


__all__ = ["check_inference_grad_mode", "check_trainable_grad_mode"]
