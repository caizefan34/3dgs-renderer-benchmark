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


__all__ = ["RenderReturn", "rasterize_gaussian_inference_scene", "rasterize_gaussian_higs_trainable", "rasterize_gaussian_higs_frozen", "rasterize_gaussian_higs_dynamic", "create_higs_renderer", "HigsRendererHandle", "sync_optimizer_state_for_topology_change", "render_scene"]


def __getattr__(name):
    if name == "RenderReturn":
        from ..types import RenderReturn

        return RenderReturn
    if name == "rasterize_gaussian_inference_scene":
        from .gaussian_inference import rasterize_gaussian_inference_scene

        return rasterize_gaussian_inference_scene
    if name == "rasterize_gaussian_higs_trainable":
        from .gaussian_inference import rasterize_gaussian_higs_trainable

        return rasterize_gaussian_higs_trainable
    if name == "rasterize_gaussian_higs_frozen":
        from .gaussian_inference import rasterize_gaussian_higs_frozen

        return rasterize_gaussian_higs_frozen
    if name == "rasterize_gaussian_higs_dynamic":
        from .gaussian_inference import rasterize_gaussian_higs_dynamic

        return rasterize_gaussian_higs_dynamic
    if name == "create_higs_renderer":
        from .gaussian_inference import create_higs_renderer

        return create_higs_renderer
    if name == "HigsRendererHandle":
        from .gaussian_inference import HigsRendererHandle

        return HigsRendererHandle
    if name == "sync_optimizer_state_for_topology_change":
        from .gaussian_inference import sync_optimizer_state_for_topology_change

        return sync_optimizer_state_for_topology_change
    if name == "render_scene":
        from .render_scene import render_scene

        return render_scene
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


