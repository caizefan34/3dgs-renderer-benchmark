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


"""Stateless Gaussian Inference render API."""

from __future__ import annotations

import math
import os

from typing import Any, Optional

import torch
from torch import Tensor

from gsplat.scene import GaussianInferenceScene

from .._common import check_inference_grad_mode, check_trainable_grad_mode
from ..kernels.gaussian_inference_ops import (
    gaussian_render_inference_only,
)
from ..types import RenderReturn


# Known unsupported features for the Inference branch (each raises TypeError).
_INFERENCE_UNSUPPORTED_FEATURES = frozenset(
    {
        "with_ut",
        "with_eval3d",
        "absgrad",
        "sparse_grad",
        "distributed",
        "packed",
        "segmented",
        "return_normals",
        "covars",
        "rays",
        "radial_coeffs",
        "tangential_coeffs",
        "thin_prism_coeffs",
        "ftheta_coeffs",
        "lidar_coeffs",
        "external_distortion_coeffs",
        "rolling_shutter",
        "viewmats_rs",
        "extra_signals",
        "extra_signals_sh_degree",
        "rasterize_mode",
        "channel_chunk",
        "global_z_order",
        "ut_params",
        "colors",
    }
)

# Accepted kwargs for the Inference branch.
_INFERENCE_ACCEPTED_KWARGS = frozenset(
    {
        "viewmat",
        "viewmats",
        "K",
        "Ks",
        "width",
        "height",
        "tile_size",
        "near_plane",
        "far_plane",
        "radius_clip",
        "eps2d",
        "background",
        "render_mode",
        "camera_model",
    }
)


_SH_COMPRESSION_NONE = None  # resolved lazily below to avoid import cycles


def _get_sh_compression_none():
    if _SH_COMPRESSION_NONE is None:
        from gsplat.scene.components.gaussian_inference_scene import SHCompressionMode

        globals()["_SH_COMPRESSION_NONE"] = SHCompressionMode.NONE
    return _SH_COMPRESSION_NONE


def _expand_background(background, C, device):
    """Normalize a per-scene background to the rasterization [1, C, D] layout."""
    if background is None:
        return None
    bg = background
    if bg.dim() == 1:
        bg = bg.unsqueeze(0).unsqueeze(0).expand(1, C, -1)
    elif bg.dim() == 2:
        bg = bg.unsqueeze(0)
    return bg.contiguous()

def _render_mode_color_channels(render_mode: str) -> int:
    """Number of color channels produced by a render mode (0 for depth-only)."""
    return 3 if render_mode in ("RGB", "RGB+D", "RGB+ED") else 0


def _render_mode_has_depth(render_mode: str) -> bool:
    """Whether the render mode composites a depth channel."""
    return render_mode in ("D", "ED", "RGB+D", "RGB+ED")


def _render_mode_channel_count(render_mode: str) -> int:
    """Total output channels (color + optional depth)."""
    return _render_mode_color_channels(render_mode) + (
        1 if _render_mode_has_depth(render_mode) else 0
    )


def _check_grad_mode() -> None:
    check_inference_grad_mode()


def _validate_device_consistency(
    scene: GaussianInferenceScene,
    request: dict[str, Any],
    out: Optional[RenderReturn],
) -> None:
    """Ensure all request tensors live on the same CUDA device as the scene."""
    scene_device = scene.means_planar.device
    for name in ("viewmat", "viewmats", "K", "Ks", "background"):
        val = request.get(name)
        if val is not None and isinstance(val, Tensor) and val.device != scene_device:
            raise ValueError(
                f"{name} is on {val.device} but scene is on {scene_device}; "
                f"all tensors must be on the same device"
            )
    if out is not None and out.frame.device != scene_device:
        raise ValueError(
            f"out buffer is on {out.frame.device} but scene is on "
            f"{scene_device}; all tensors must be on the same device"
        )


def _validate_inference_request(request: dict[str, Any]) -> dict[str, Any]:
    """Validate and extract inference kwargs from ``**request``.

    Returns a normalised dict ready for downstream consumption.
    """
    # -- Reject sh_degree / sh_compression_mode overrides ----------------
    for key in ("sh_degree", "sh_compression_mode"):
        if key in request:
            raise TypeError(
                "sh_degree/sh_compression_mode are read from scene; "
                "cannot be overridden via request kwargs"
            )

    # -- Reject ``backgrounds`` (plural) ---------------------------------
    if "backgrounds" in request:
        raise TypeError(
            "rasterize_gaussian_inference_scene got unexpected keyword argument "
            "'backgrounds'"
        )

    # -- Reject known-unsupported features -------------------------------
    for key in _INFERENCE_UNSUPPORTED_FEATURES:
        if key in request:
            raise TypeError(f"Inference branch does not support {key}")

    # -- Reject truly unknown kwargs -------------------------------------
    for key in request:
        if key not in _INFERENCE_ACCEPTED_KWARGS:
            raise TypeError(
                f"rasterize_gaussian_inference_scene got unexpected keyword "
                f"argument '{key}'"
            )

    # -- Constrained values -----------------------------------------------
    render_mode = request.get("render_mode", "RGB")
    if render_mode != "RGB":
        raise TypeError(
            f"Inference branch supports render_mode='RGB' only; got '{render_mode}'"
        )

    camera_model = request.get("camera_model", "pinhole")
    if camera_model != "pinhole":
        raise TypeError(
            f"Inference branch supports camera_model='pinhole' only; "
            f"got '{camera_model}'"
        )

    tile_size = request.get("tile_size", 8)
    if tile_size not in (8, 16):
        raise TypeError(
            f"Inference branch supports tile_size in {{8, 16}}; got {tile_size}"
        )

    # -- Build normalised result -----------------------------------------
    return {
        "viewmat": request.get("viewmat"),
        "viewmats": request.get("viewmats"),
        "K": request.get("K"),
        "Ks": request.get("Ks"),
        "width": request.get("width"),
        "height": request.get("height"),
        "tile_size": tile_size,
        "near_plane": request.get("near_plane", 0.01),
        "far_plane": request.get("far_plane", 1e10),
        "radius_clip": request.get("radius_clip", 0.0),
        "eps2d": request.get("eps2d", 0.3),
        "background": request.get("background"),
    }


def _normalize_cameras(
    kwargs: dict[str, Any],
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Normalize viewmat/K from request kwargs.

    Returns ``(viewmat, K, cleaned_kwargs)`` where ``cleaned_kwargs`` no
    longer contains viewmat/viewmats/K/Ks.
    """
    viewmat_val = kwargs.get("viewmat")
    viewmats_val = kwargs.get("viewmats")
    K_val = kwargs.get("K")
    Ks_val = kwargs.get("Ks")

    # -- viewmat / viewmats -----------------------------------------------
    if viewmat_val is not None and viewmats_val is not None:
        raise RuntimeError("pass exactly one of viewmat or viewmats, not both")
    if viewmat_val is None and viewmats_val is None:
        raise RuntimeError("pass exactly one of viewmat or viewmats")

    if viewmats_val is not None:
        if viewmats_val.shape[0] > 1:
            raise RuntimeError(
                f"Inference branch supports single camera (leading dim == 1); "
                f"got viewmats with leading dim {viewmats_val.shape[0]}"
            )
        viewmat = viewmats_val[0]
    else:
        viewmat = viewmat_val

    # -- K / Ks -----------------------------------------------------------
    if K_val is not None and Ks_val is not None:
        raise RuntimeError("pass exactly one of K or Ks, not both")
    if K_val is None and Ks_val is None:
        raise RuntimeError("pass exactly one of K or Ks")

    if Ks_val is not None:
        if Ks_val.shape[0] > 1:
            raise RuntimeError(
                f"Inference branch supports single camera (leading dim == 1); "
                f"got Ks with leading dim {Ks_val.shape[0]}"
            )
        K = Ks_val[0]
    else:
        K = K_val

    # Build a cleaned copy without the camera keys
    cleaned = {
        k: v for k, v in kwargs.items() if k not in ("viewmat", "viewmats", "K", "Ks")
    }
    return viewmat, K, cleaned


def _validate_out_buffer(
    out: RenderReturn,
    height: int,
    width: int,
    device: torch.device,
) -> None:
    """Validate the ``out=`` buffer contract."""
    # -- frame -----------------------------------------------------------
    _validate_single_buffer(
        out.frame,
        name="out.frame",
        expected_shape=(1, height, width, 3),
        device=device,
    )
    if out.frame.requires_grad:
        raise RuntimeError("out=... buffer must not be grad-tracked")

    # -- optional alpha in metadata --------------------------------------
    alpha = out.metadata.get("alpha")
    if alpha is not None:
        _validate_single_buffer(
            alpha,
            name="out.metadata['alpha']",
            expected_shape=(1, height, width, 1),
            device=device,
        )
        if alpha.requires_grad:
            raise RuntimeError("out=... buffer must not be grad-tracked")


def _validate_single_buffer(
    t: Tensor,
    *,
    name: str,
    expected_shape: tuple[int, ...],
    device: torch.device,
) -> None:
    """Validate shape, dtype, device, contiguity for a single buffer tensor."""
    ok = (
        t.shape == expected_shape
        and t.dtype == torch.float32
        and t.device == device
        and t.is_contiguous()
    )
    if not ok:
        raise RuntimeError(
            f"{name} expected shape {list(expected_shape)}, "
            f"dtype torch.float32, device {device}, contiguous; "
            f"got shape {list(t.shape)}, dtype {t.dtype}, "
            f"device {t.device}, contiguous={t.is_contiguous()}"
        )


# ---------------------------------------------------------------------------
# rasterize_gaussian_inference_scene  (direct inference entry point)
# ---------------------------------------------------------------------------


def rasterize_gaussian_inference_scene(
    scene: GaussianInferenceScene,
    *,
    out: Optional[RenderReturn] = None,
    **request: Any,
) -> RenderReturn:
    """Render a ``GaussianInferenceScene`` via the fused Inference rasterisation op.

    This is the direct, low-level entry point.  For a polymorphic API that
    also handles ``GaussianScene``, use :func:`render_scene`.
    """
    # 1. Type check
    if not isinstance(scene, GaussianInferenceScene):
        raise TypeError(
            f"rasterize_gaussian_inference_scene requires a GaussianInferenceScene; "
            f"got {type(scene).__name__}"
        )

    # 2. Empty-scene guard
    if scene.is_empty():
        raise ValueError(
            "GaussianInferenceScene has been released and contains no packed "
            "tensors. Did you forget to rebuild the snapshot?"
        )

    # 3. Grad-mode gate
    _check_grad_mode()

    # 3b. Device consistency
    _validate_device_consistency(scene, request, out)

    # 4. Request-subset validation
    validated = _validate_inference_request(request)

    # 5. Camera normalisation
    viewmat, K, rest = _normalize_cameras(validated)

    height = rest["height"]
    width = rest["width"]
    if not isinstance(height, int) or height <= 0:
        raise ValueError(f"height must be a positive integer, got {height!r}")
    if not isinstance(width, int) or width <= 0:
        raise ValueError(f"width must be a positive integer, got {width!r}")

    # 6. Buffer validation
    if out is not None:
        _validate_out_buffer(out, height, width, viewmat.device)

    # 7. Build out_renders / out_alphas views
    out_renders: Optional[Tensor] = None
    out_alphas: Optional[Tensor] = None
    saved_alpha: Optional[Tensor] = None
    if out is not None:
        out_renders = out.frame[0]  # [H, W, 3]
        alpha_buf = out.metadata.get("alpha")
        if alpha_buf is not None:
            saved_alpha = alpha_buf
            out_alphas = alpha_buf[0]  # [H, W, 1]

    # 8. Call accelerated backend (stateless one-shot path)
    renders, alphas = gaussian_render_inference_only(
        scene.means_planar,
        scene.qso_packed,
        scene.colors_packed,
        viewmat,
        K,
        width,
        height,
        scene.sh_degree,
        rest["tile_size"],
        rest["near_plane"],
        rest["far_plane"],
        rest["radius_clip"],
        rest["eps2d"],
        scene.sh_compression_mode,
        rest["background"],
        out_renders=out_renders,
        out_alphas=out_alphas,
    )

    # 9. Package result
    if out is None:
        return RenderReturn(
            frame=renders[None],  # [1, H, W, 3]
            metadata={"alpha": alphas[None]},  # [1, H, W, 1]
        )
    else:
        out.metadata.clear()
        if saved_alpha is not None:
            out.metadata["alpha"] = saved_alpha
        else:
            out.metadata["alpha"] = alphas[None]
        return out


__all__ = ["rasterize_gaussian_inference_scene"]

# ---------------------------------------------------------------------------
# rasterize_gaussian_higs_trainable  (trainable entry point, Stage A proxy)
# ---------------------------------------------------------------------------

_TRAINABLE_ACCEPTED_KWARGS = frozenset({
    "viewmat", "viewmats", "K", "Ks",
    "width", "height",
    "tile_size", "near_plane", "far_plane", "radius_clip", "eps2d",
    "background", "render_mode", "camera_model",
})


def _normalize_cameras_trainable(
    kwargs: dict[str, Any],
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Normalize viewmat/K from request kwargs for the trainable path.

    Returns ``(viewmats, Ks, cleaned_kwargs)`` where both have a leading
    batch dim ``[1, ...]`` suitable for :func:`gsplat.rasterization`.
    """
    viewmat_val = kwargs.pop("viewmat", None)
    viewmats_val = kwargs.pop("viewmats", None)
    K_val = kwargs.pop("K", None)
    Ks_val = kwargs.pop("Ks", None)

    if viewmat_val is not None and viewmats_val is not None:
        raise RuntimeError("pass exactly one of viewmat or viewmats, not both")
    if viewmat_val is None and viewmats_val is None:
        raise RuntimeError("pass exactly one of viewmat or viewmats")

    if viewmats_val is not None:
        viewmats = viewmats_val  # keep as-is (may already have batch dim)
    else:
        viewmats = viewmat_val.unsqueeze(0).unsqueeze(0)  # [4,4] -> [1,1,4,4]

    if K_val is not None and Ks_val is not None:
        raise RuntimeError("pass exactly one of K or Ks, not both")
    if K_val is None and Ks_val is None:
        raise RuntimeError("pass exactly one of K or Ks")

    if Ks_val is not None:
        Ks = Ks_val
    else:
        Ks = K_val.unsqueeze(0).unsqueeze(0)  # [3,3] -> [1,1,3,3]

    return viewmats, Ks, kwargs


def _validate_trainable_request(request: dict[str, Any]) -> dict[str, Any]:
    """Validate and extract kwargs for the trainable path."""
    # Reject known-unsupported inference features
    for key in _INFERENCE_UNSUPPORTED_FEATURES:
        if key in request:
            raise TypeError(f"Trainable path does not support {key}")

    for key in request:
        if key not in _TRAINABLE_ACCEPTED_KWARGS:
            raise TypeError(
                f"rasterize_gaussian_higs_trainable got unexpected keyword "
                f"argument '{key}'"
            )
    # Return all validated kwargs with defaults for optional fields
    result = dict(request)
    result.setdefault("tile_size", 16)
    result.setdefault("near_plane", 0.01)
    result.setdefault("far_plane", 1e10)
    result.setdefault("radius_clip", 0.0)
    result.setdefault("eps2d", 0.3)
    result.setdefault("background", None)
    result.setdefault("render_mode", "RGB")
    result.setdefault("camera_model", "pinhole")
    return result


def rasterize_gaussian_higs_trainable(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    *,
    out: Optional[RenderReturn] = None,
    sh_degree: Optional[int] = None,
    differentiable: bool = False,
    **request: Any,
) -> RenderReturn:
    """Render Gaussians with an optional HiGS preview and a differentiable path.

    This is the trainable entry point for the HiGS renderer.

    When ``differentiable=False`` (default): delegates to
    :func:`rasterize_gaussian_inference_scene` for inference-only rendering.
    Requires ``torch.inference_mode()`` or ``torch.no_grad()``.

    When ``differentiable=True``:
    - Skips the inference-only grad guard.
    - Runs the standard differentiable gsplat rasterization pipeline, which
      supports autograd and gradient flow to all Gaussian parameters.
    - Optionally (if the HiGS backend is available) runs the HiGS inference
      path under ``torch.no_grad()`` for preview / forward comparison.
    - The HiGS result, if available, is stored in ``metadata["higs_preview"]``.

    .. note::
        Stage A uses the standard gsplat rasterization for the backward pass.
        The HiGS path is used only for forward preview. No training speed
        improvement is expected from Stage A.

    Args:
        means: Gaussian means. [N, 3] float32 CUDA.
        quats: Quaternions (normalized). [N, 4] float32 CUDA.
        scales: Scales (activated, positive). [N, 3] float32 CUDA.
        opacities: Opacities (activated, in [0,1]). [N] float32 CUDA.
        colors: Colors or SH coefficients. If ``sh_degree`` is None, expected
            shape [N, D] (post-activation RGB). If ``sh_degree`` is set,
            expected shape [N, K, D] (SH coefficients).
        out: Optional output buffer (:class:`RenderReturn`). If provided, the
            result is written in-place.
        sh_degree: SH degree. If set, ``colors`` are SH coefficients shared
            across cameras.
        differentiable: If True, enables gradient flow through rendering.
        **request: Camera parameters and rendering options. See
            :func:`rasterize_gaussian_inference_scene` for details.

    Returns:
        :class:`RenderReturn` with ``frame`` [1, H, W, 3] and alpha in
        ``metadata["alpha"]``. When ``differentiable=True`` and HiGS backend
        is available, ``metadata["higs_preview"]`` contains the HiGS forward
        result.
    """
    check_trainable_grad_mode(differentiable)

    if not isinstance(means, Tensor) or means.device.type != "cuda":
        raise RuntimeError("All tensors must be CUDA tensors")

    if not differentiable:
        # --- Delegate to the existing HiGS inference path ---
        from gsplat.scene import GaussianInferenceScene

        scene = GaussianInferenceScene.from_gaussian_tensors(
            means, quats, scales, opacities, colors,
            sh_degree=sh_degree,
            id="trainable_fallback",
        )
        return rasterize_gaussian_inference_scene(
            scene, out=out, **request,
        )

    # --- Differentiable path: use standard gsplat rasterization ---
    from gsplat.rendering import rasterization

    validated = _validate_trainable_request(request)
    viewmats, Ks, rest = _normalize_cameras_trainable(validated)

    width = rest["width"]
    height = rest["height"]
    if not isinstance(width, int) or width <= 0:
        raise ValueError(f"width must be a positive integer, got {width!r}")
    if not isinstance(height, int) or height <= 0:
        raise ValueError(f"height must be a positive integer, got {height!r}")

    bg = rest.get("background")

    # Expand dims for the standard rasterization API
    # means: [N,3] -> [1, N, 3], etc. (add batch dim)
    means_b = means.unsqueeze(0)
    quats_b = quats.unsqueeze(0)
    scales_b = scales.unsqueeze(0)
    opacities_b = opacities.unsqueeze(0)
    if colors.dim() == 2:
        colors_b = colors.unsqueeze(0)  # [N, D] -> [1, N, D] for pre-activated
    else:
        colors_b = colors  # [N, K, D] shared across batch/cameras

    # Background: expand for batch/camera dims
    backgrounds = None
    if bg is not None:
        backgrounds = bg.unsqueeze(0).unsqueeze(0)  # [D] -> [1, 1, D]

    # Run standard differentiable rasterization
    render_colors, render_alphas, meta = rasterization(
        means=means_b,
        quats=quats_b,
        scales=scales_b,
        opacities=opacities_b,
        colors=colors_b,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        backgrounds=backgrounds,
        packed=True,
        tile_size=rest.get("tile_size", 16),
        near_plane=rest.get("near_plane", 0.01),
        far_plane=rest.get("far_plane", 1e10),
        radius_clip=rest.get("radius_clip", 0.0),
        eps2d=rest.get("eps2d", 0.3),
        camera_model=rest.get("camera_model", "pinhole"),
        render_mode=rest.get("render_mode", "RGB"),
    )

    # Squeeze batch/camera dims for RenderReturn [1, H, W, 3]
    frame = render_colors.squeeze(0).squeeze(0)  # [1, 1, H, W, 3] -> [H, W, 3]
    alpha = render_alphas.squeeze(0).squeeze(0)  # [1, 1, H, W, 1] -> [H, W, 1]

    metadata: dict[str, Any] = {"alpha": alpha[None]}  # [1, H, W, 1]

    # Optional: run HiGS for preview under no_grad
    try:
        from gsplat.scene import GaussianInferenceScene

        with torch.no_grad():
            scene = GaussianInferenceScene.from_gaussian_tensors(
                means, quats, scales, opacities, colors,
                sh_degree=sh_degree,
                id="trainable_preview",
            )
            higs_result = rasterize_gaussian_inference_scene(
                scene, **request,
            )
            metadata["higs_preview"] = higs_result.frame  # [1, H, W, 3]
    except (ImportError, RuntimeError, ValueError, TypeError):
        # HiGS preview is optional; skip when the extension/scene is unavailable
        pass

    if out is not None:
        out.metadata.clear()
        out.metadata.update(metadata)
        out.frame = frame[None]  # [1, H, W, 3]
        return out

    return RenderReturn(
        frame=frame[None],  # [1, H, W, 3]
        metadata=metadata,
    )



def _cull_gaussians(means, quats, scales, viewmats, Ks, width, height, eps2d=0.3):
    """Determine visible Gaussians via projection culling.
    
    Uses fully_fused_projection (under no_grad) to compute per-Gaussian
    2D radii. Gaussians with both radius components > 0 are visible.
    
    Returns:
        visible_ids: 1D int64 tensor of visible Gaussian indices.
        visible_mask: bool tensor [N] where True means visible.
    """
    from gsplat.cuda._wrapper import fully_fused_projection
    with torch.no_grad():
        radii, _, _, _, _ = fully_fused_projection(
            means=means.unsqueeze(0).contiguous(),
            covars=None,
            quats=quats.unsqueeze(0).contiguous(),
            scales=scales.unsqueeze(0).contiguous(),
            viewmats=viewmats.contiguous(),
            Ks=Ks.contiguous(),
            width=width, height=height, eps2d=eps2d,
            packed=False,
        )
    r = radii.squeeze()  # [N] or [N, 2]
    visible_mask = (r > 0).all(dim=-1) if r.dim() > 1 else (r > 0)
    visible_ids = torch.where(visible_mask)[0]
    return visible_ids, visible_mask


def _cull_gaussians_batched(
    means, quats, scales, viewmats, Ks,
    width, height, eps2d=0.3,
    near_plane=0.01, far_plane=1e10, radius_clip=0.0,
    camera_model="pinhole",
):
    """Union-visibility culling over all cameras with ONE batched projection.

    Runs ``fully_fused_projection`` once for every camera in parallel and marks
    a Gaussian visible when any camera projects it with a positive radius. The
    projection uses the exact same near/far/radius_clip/eps2d/camera_model
    criteria as the render path and operates on the FP32 master params, so the
    mask is projection-consistent with the forward (the FP16-packed HiGS
    renderer is not needed for this and would add a full per-camera render per
    step).

    Returns:
        visible_ids: 1D int64 tensor of visible Gaussian indices (ascending).
        visible_mask: bool tensor [N] where True means visible.
        culling_ratio: float fraction of culled (invisible) Gaussians.
    """
    from gsplat.cuda._wrapper import fully_fused_projection

    with torch.no_grad():
        radii, means2d, depths, conics, _ = fully_fused_projection(
            means=means.unsqueeze(0).contiguous(),
            covars=None,
            quats=quats.unsqueeze(0).contiguous(),
            scales=scales.unsqueeze(0).contiguous(),
            viewmats=viewmats.contiguous(),
            Ks=Ks.contiguous(),
            width=width,
            height=height,
            eps2d=eps2d,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            camera_model=camera_model,
            packed=False,
        )
    r = radii.squeeze(0)  # [C, N, 2]
    visible_mask = _union_visible_mask_native(r)  # [N] union over cameras
    visible_ids = torch.where(visible_mask)[0]
    culling_ratio = 1.0 - (visible_ids.numel() / max(radii.shape[-2], 1))
    return visible_ids, visible_mask, culling_ratio


def _cull_visible_cached(
    handle,
    interval: int,
    means, quats, scales, viewmats, Ks,
    width, height, eps2d=0.3,
    near_plane=0.01, far_plane=1e10, radius_clip=0.0,
    camera_model="pinhole",
    cache_key: str = "default",
):
    """Union-visibility culling with a refresh-interval cache.

    The batched full-N projection is the only forward stage that touches every
    Gaussian (isect/rasterize already run on the visible subset), so on the
    dynamic training loop reusing its result for ``interval`` forwards removes
    one full-N projection (and the union-mask bookkeeping) per step. The
    visible set changes slowly under optimizer steps (~lr * grad per step), and
    every topology change (densify/prune -> ``mark_dirty`` / ``rebuild``)
    invalidates the cache, so a fresh cull is guaranteed exactly when the
    Gaussian count changes.

    Args:
        handle: :class:`HigsRendererHandle` owning the per-camera-key cull
            cache ``_cull_cache`` (must be non-None).
        interval: refresh cadence in forwards; 1 reproduces per-step culling.
    Returns:
        visible_ids: 1D int64 tensor of visible Gaussian indices.
    """
    interval = max(1, int(interval))
    slot = handle._cull_cache.get(cache_key)
    if slot is not None and (handle._fwd_count - slot[2]) < interval:
        return slot[0]
    vis_ids, _vis_mask, _ratio = _cull_gaussians_batched(
        means, quats, scales, viewmats, Ks,
        width, height,
        eps2d=eps2d, near_plane=near_plane,
        far_plane=far_plane, radius_clip=radius_clip,
        camera_model=camera_model,
    )
    handle._cull_cache[cache_key] = (vis_ids, _vis_mask, handle._fwd_count)
    return vis_ids


def _gather_rows_fast(t, row_ids):
    """Compact-copy arbitrary row subsets of one FP32 tensor.

    Uses the single-tensor ``higs_gather_rows`` CUDA kernel when available
    (bypasses PyTorch's slow vectorized row gather for row widths divisible
    by four, which the densify/prune and Adam-state-sync paths hit with
    ``colors [N,16,3]``); falls back to ``t[row_ids]`` otherwise.
    """
    try:
        from ..kernels import _backend as _b

        ext = _b._C
    except Exception:
        ext = None
    if (
        ext is not None
        and hasattr(ext, "higs_gather_rows")
        and t.is_cuda
        and t.dtype == torch.float32
    ):
        try:
            return ext.higs_gather_rows(t, row_ids)
        except RuntimeError:
            pass
    return t[row_ids]


def _gather_rows_new_zeros(t, row_ids):
    """Gather rows with ``row_ids[i] == -1`` writing zeros, in one kernel.

    Used by the Adam-state sync: every row of the new state tensor is either
    copied from its source row (``row_ids[i] >= 0``) or zero-initialized
    (``row_ids[i] == -1`` for brand-new Gaussians). Fusing the zero-fill into
    the gather removes the ``zeros_like`` memset plus the scatter pass, and the
    output can be allocated uninitialized. Falls back to zeros + indexed copy
    when the CUDA extension is unavailable.
    """
    try:
        from ..kernels import _backend as _b

        ext = _b._C
    except Exception:
        ext = None
    if (
        ext is not None
        and hasattr(ext, "higs_gather_rows")
        and t.is_cuda
        and t.dtype == torch.float32
    ):
        try:
            return ext.higs_gather_rows(t, row_ids.to(torch.long), True)
        except RuntimeError:
            pass
    valid = row_ids >= 0
    shape = (row_ids.numel(),) + tuple(t.shape[1:])
    out = torch.zeros(shape, dtype=t.dtype, device=t.device)
    if valid.any():
        out.index_copy_(0, valid.nonzero().flatten(), t[row_ids[valid].clamp(min=0)])
    return out


def _refresh_higs_renderer_scene(
    handle, means, quats, scales, opacities, colors,
    N, sh_degree, sh_compression_mode, lightweight=False,
):
    """Rebuild the packed HiGS scene when it is stale, WITHOUT rendering.

    Mirrors the rebuild path of :func:`_cull_gaussians_higs` (which the direct
    culling API still uses) but skips the per-camera render: the training
    forward derives visibility from a batched FP32 projection, so the renderer's
    packed scene only needs to stay version-consistent for the handle's
    scene_version / topology_rebuilt / pending-backward bookkeeping.

    When ``lightweight`` is True (the differentiable training forward), the
    packed buffers are only kept bookkeeping-consistent, not re-packed: the
    native and recompute backends consume the FP32 captured tensors and never
    the packed FP16 scene. A pure parameter drift (optimizer.step) skips the
    expensive ``pack_gaussian_inference_scene`` call, and a real topology
    change (``mark_dirty()`` or an N change) also defers it: only the version
    bookkeeping advances and ``packed_stale`` is set, so the packed scene is
    rebuilt on demand the next time the non-training
    :func:`_cull_gaussians_higs` culling API is used with this handle.
    """
    from gsplat.scene.kernels.gaussian_inference_ops import (
        pack_gaussian_inference_scene,
    )
    from ..kernels.gaussian_inference_ops import (
        create_native_gaussian_inference_renderer,
    )

    sh_deg_i = -1 if sh_degree is None else sh_degree
    comp = _normalize_sh_compression_mode(sh_compression_mode)
    if handle.n_gaussians != N and not handle.dirty:
        raise RuntimeError(
            f"HiGS renderer handle has n_gaussians={handle.n_gaussians} "
            f"but the input has N={N}. Topology changed without "
            "mark_dirty(); call renderer_handle.mark_dirty() after "
            "densify/prune before the next forward."
        )
    if handle.dirty or handle.n_gaussians != N or (
        not handle.pending_backward
        and handle.params_changed(means, quats, scales, opacities, colors)
    ):
        if lightweight:
            if not handle.dirty and handle.n_gaussians == N:
                # Pure parameter drift on the differentiable path: the packed
                # scene is stale but never consumed here. Do not update
                # _param_versions so _cull_gaussians_higs still sees the drift
                # and re-packs when the non-training culling API is used with
                # this handle.
                handle._topology_rebuilt = False
                return
            # Real topology change on the differentiable path: the packed FP16
            # scene is never consumed by the training forward/backward (both
            # consume the FP32 captured tensors), so defer the pack entirely.
            # Advance only the version bookkeeping and mark the packed scene
            # stale; _cull_gaussians_higs re-packs on demand. _param_versions
            # is intentionally left untouched (a densify/prune creates brand-new
            # tensors whose _version can collide with the captured ones, so
            # packed_stale - not params_changed - is the authoritative signal).
            handle._version += 1
            handle.n_gaussians = N
            handle._dirty = False
            handle._topology_rebuilt = True
            handle._packed_stale = True
            return
        means_planar, qso_packed, colors_packed = pack_gaussian_inference_scene(
            means, quats, scales, opacities, colors, sh_deg_i, comp,
        )
        if handle.renderer is not None:
            handle.renderer.release()
        renderer = create_native_gaussian_inference_renderer(
            means_planar, qso_packed, colors_packed, sh_deg_i, int(comp),
        )
        handle.rebuild(
            means_planar, qso_packed, colors_packed, N,
            packed_dtype=str(qso_packed.dtype), renderer=renderer,
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
        )

def _cull_gaussians_higs(
    means, quats, scales, opacities, colors,
    viewmat, K, width, height, sh_degree,
    tile_size=16, near_plane=0.01, far_plane=1e10,
    radius_clip=0.0, eps2d=0.3, sh_compression_mode=0, background=None,
    renderer_handle=None,
):
    """Determine visible Gaussian IDs using HiGS-native culling (get_visible_mask).

    Calls the HiGS renderer under no_grad to populate the visibility bitmask,
    which reflects exactly which Gaussians the HiGS tile-intersect stage
    considers visible.

    When ``renderer_handle`` is provided (a :class:`HigsRendererHandle`), the
    packed FP16 scene and pybind renderer are reused across calls/frames and are
    rebuilt only when the handle was marked dirty (e.g. after densify/prune).
    Otherwise a temporary renderer is created and released before returning.

    Returns:
        visible_ids: 1-D int64 tensor of visible Gaussian indices.
        visible_mask: bool tensor [N] where True means visible.
        culling_ratio: float fraction of culled (invisible) Gaussians.
    """
    from ..kernels.gaussian_inference_ops import (
        create_native_gaussian_inference_renderer,
    )
    from gsplat.scene.kernels.gaussian_inference_ops import (
        pack_gaussian_inference_scene,
    )
    from gsplat.scene.components.gaussian_inference_scene import SHCompressionMode

    if not _higs_backend_available():
        raise RuntimeError(
            "HiGS CUDA extension is not available. Install the CUDA toolkit and "
            "build the experimental extension, or use use_higs_culling=False."
        )

    N = means.shape[0]
    with torch.no_grad():
        sh_deg_i = -1 if sh_degree is None else sh_degree
        comp = _normalize_sh_compression_mode(sh_compression_mode)

        temp_handle = None
        handle = renderer_handle
        if handle is None:
            means_planar, qso_packed, colors_packed = pack_gaussian_inference_scene(
                means, quats, scales, opacities, colors, sh_deg_i, comp,
            )
            renderer = create_native_gaussian_inference_renderer(
                means_planar, qso_packed, colors_packed, sh_deg_i, int(comp),
            )
            handle = HigsRendererHandle(
                means_planar, qso_packed, colors_packed, renderer,
                packed_dtype=str(qso_packed.dtype),
                n_gaussians=N, sh_degree=sh_deg_i,
            )
            temp_handle = handle
        else:
            if handle.n_gaussians != N and not handle.dirty:
                raise RuntimeError(
                    f"HiGS renderer handle has n_gaussians={handle.n_gaussians} "
                    f"but the input has N={N}. Topology changed without "
                    "mark_dirty(); call renderer_handle.mark_dirty() after "
                    "densify/prune before the next forward."
                )
            if handle.dirty or handle.n_gaussians != N or handle.packed_stale or (not handle.pending_backward and handle.params_changed(means, quats, scales, opacities, colors)):
                means_planar, qso_packed, colors_packed = pack_gaussian_inference_scene(
                    means, quats, scales, opacities, colors, sh_deg_i, comp,
                )
                if handle.renderer is not None:
                    handle.renderer.release()
                renderer = create_native_gaussian_inference_renderer(
                    means_planar, qso_packed, colors_packed, sh_deg_i, int(comp),
                )
                handle.rebuild(
                    means_planar, qso_packed, colors_packed, N,
                    packed_dtype=str(qso_packed.dtype), renderer=renderer,
                    means=means, quats=quats, scales=scales,
                    opacities=opacities, colors=colors,
                )

        _ = handle.renderer.render(
            handle.means_planar, handle.qso_packed, handle.colors_packed,
            viewmat, K, width, height, tile_size,
            near_plane, far_plane, radius_clip, eps2d,
            sh_deg_i, int(comp), background, None,
        )

        # Decode the int32 bitmask [(N + 31) / 32] on-device without a Python
        # per-word loop (avoids N/32 device->host syncs for large scenes).
        mask_bits = handle.renderer.get_visible_mask()
        n_words = mask_bits.shape[0]
        word_idx = torch.arange(n_words, device=mask_bits.device)
        bit_idx = torch.arange(32, device=mask_bits.device)
        bits = (mask_bits.long()[:, None] >> bit_idx[None, :]) & 1  # [n_words, 32]
        flat_bits = bits.reshape(-1)[:N]
        visible_mask = flat_bits.bool()

        if temp_handle is not None:
            temp_handle.release()

    visible_ids = torch.where(visible_mask)[0]
    culling_ratio = 1.0 - (visible_ids.numel() / max(N, 1))
    return visible_ids, visible_mask, culling_ratio

def _higs_backend_available() -> bool:
    """Return True when the experimental Inference CUDA extension is usable."""
    from ..kernels import _backend

    ext = _backend._C
    return ext is not None and hasattr(ext, "higs_rasterize_backward")


def _gather_visible_native(
    means, quats, scales, opacities, colors, visible_ids,
):
    """Gather the visible rows of the master FP32 tensors.

    Uses the compact-copy CUDA kernel from the experimental extension when
    available (avoids the pathologically slow PyTorch row gather for row
    widths divisible by four, e.g. quats [N,4] and colors [N,16,3]);
    otherwise falls back to the plain PyTorch gather.
    """
    try:
        from ..kernels import _backend as _b

        ext = _b._C
    except Exception:
        ext = None
    if (
        ext is not None
        and hasattr(ext, "higs_gather_visible")
        and means.is_cuda
        and means.dtype == torch.float32
        and colors.dtype == torch.float32
    ):
        try:
            return ext.higs_gather_visible(
                means, quats, scales, opacities, colors, visible_ids,
            )
        except RuntimeError:
            pass
    return (
        means[visible_ids].contiguous(),
        quats[visible_ids].contiguous(),
        scales[visible_ids].contiguous(),
        opacities[visible_ids].contiguous(),
        colors[visible_ids].contiguous(),
    )


def _union_visible_mask_native(r):
    """Union visibility mask over cameras from ``[C, N, 2]`` projection radii.

    Uses the fused CUDA kernel when the experimental extension is available
    (one thread per Gaussian, ``any_c(both radii > 0)``); otherwise falls back
    to the PyTorch expression, which the culling callers historically used.
    """
    try:
        from ..kernels import _backend as _b

        ext = _b._C
    except Exception:
        ext = None
    if (
        ext is not None
        and hasattr(ext, "higs_union_visible_mask")
        and r.is_cuda
        and r.dtype in (torch.float32, torch.int32)
    ):
        try:
            return ext.higs_union_visible_mask(r.contiguous())
        except RuntimeError:
            pass
    return (r > 0).all(dim=-1).any(dim=0)


def _normalize_sh_compression_mode(mode):
    """Normalize str/int/enum SH compression modes to SHCompressionMode."""
    from gsplat.scene.components.gaussian_inference_scene import SHCompressionMode

    if isinstance(mode, SHCompressionMode):
        return mode
    if isinstance(mode, str):
        return SHCompressionMode[mode.upper()]
    if isinstance(mode, int):
        return SHCompressionMode(mode)
    raise TypeError(
        f"sh_compression_mode must be a SHCompressionMode, str, or int; "
        f"got {type(mode).__name__}"
    )

def _ste_sh_quantize(colors: Tensor) -> Tensor:
    """Differentiable FP16 quantization of SH coefficients (straight-through).

    ``sh_compression_mode="packed_16b"/"packed_32b"`` store SH3 coefficients
    as FP16; the lossy step is the FP16 cast (the layout flatten is
    inference-scene-only). PyTorch dtype casts are differentiable with an
    identity-style gradient, so this is the standard straight-through
    estimator: the forward render sees the quantized coefficients while the
    gradient flows through the cast back to the FP32 master tensors.
    """
    if colors is None or colors.dtype == torch.float16:
        return colors
    return colors.half().float()


class HigsRendererHandle:
    """Explicit handle owning a packed HiGS scene and its pybind renderer.

    The handle owns the FP16 packed buffers (``qso_packed`` / ``colors_packed``)
    plus the pybind :class:`GaussianInferenceRenderer`. It is the versioned
    "scene object" that binds each forward to a unique ``scene_version`` and
    guarantees that a backward cannot run against a stale hierarchy:

    - ``mark_dirty()`` after densify/prune (raises while a backward is pending);
    - the autograd Function captures ``version`` at forward time and validates
      it again in backward (the handle is kept alive by ``ctx`` so the old
      packed buffers are never released/reused before backward completes).
    """

    def __init__(
        self,
        means_planar,
        qso_packed,
        colors_packed,
        renderer,
        packed_dtype: str,
        n_gaussians: int,
        sh_degree: int,
        means=None,
        quats=None,
        scales=None,
        opacities=None,
        colors=None,
    ) -> None:
        self.means_planar = means_planar
        self.qso_packed = qso_packed
        self.colors_packed = colors_packed
        self.renderer = renderer
        self.packed_dtype = packed_dtype
        self.n_gaussians = n_gaussians
        self.sh_degree = sh_degree
        self._version: int = 0
        self._dirty: bool = False
        self._topology_rebuilt: bool = False
        self._pending_backward: bool = False
        self._packed_stale: bool = False
        self._fwd_count: int = 0
        self._cull_cache: dict = {}
        if means is not None:
            self._param_versions = self._capture_param_versions(
                (means, quats, scales, opacities, colors)
            )
        else:
            self._param_versions = None

    # -- read-only state ---------------------------------------------------
    @property
    def version(self) -> int:
        return self._version

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def topology_rebuilt(self) -> bool:
        return self._topology_rebuilt

    @property
    def pending_backward(self) -> bool:
        return self._pending_backward

    @property
    def packed_stale(self) -> bool:
        return self._packed_stale

    # -- lifecycle ---------------------------------------------------------
    def mark_dirty(self) -> None:
        """Mark the packed scene stale (after densify/prune/clone).

        Raises if a forward already captured this handle and its backward has
        not completed yet: mutating topology mid-graph would invalidate the
        saved hierarchy buffers.
        """
        if self._pending_backward:
            raise RuntimeError(
                "Cannot mutate HiGS topology while a backward is still pending. "
                "Run loss.backward() (and optimizer.step()) before calling "
                "mark_dirty() / densify / prune."
            )
        self._dirty = True
        self._topology_rebuilt = False
        self._packed_stale = True
        self._cull_cache = {}

    def on_duplicate(self, _sel: Tensor) -> None:
        self.mark_dirty()

    def on_split(self, _sel: Tensor, _rest: Tensor) -> None:
        self.mark_dirty()

    def on_remove(self, _mask: Tensor) -> None:
        self.mark_dirty()

    def on_relocate(self, _dead_indices: Tensor, _sampled_indices: Tensor) -> None:
        self.mark_dirty()

    def on_sample_add(self, _sampled_indices: Tensor) -> None:
        self.mark_dirty()

    def rebuild(self, means_planar, qso_packed, colors_packed, n_gaussians: int, packed_dtype: str, renderer=None, means=None, quats=None, scales=None, opacities=None, colors=None) -> None:
        """Replace the packed buffers + renderer after a topology change."""
        self.means_planar = means_planar
        self.qso_packed = qso_packed
        self.colors_packed = colors_packed
        self.packed_dtype = packed_dtype
        self.n_gaussians = n_gaussians
        if renderer is not None:
            self.renderer = renderer
        self._dirty = False
        self._topology_rebuilt = True
        self._packed_stale = False
        self._version += 1
        self._cull_cache = {}
        if means is not None:
            self._param_versions = self._capture_param_versions(
                (means, quats, scales, opacities, colors)
            )

    def _capture_param_versions(self, tensors) -> tuple | None:
        """Snapshot master-tensor in-place mutation versions (optimizer.step)."""
        try:
            return tuple(t._version for t in tensors)
        except AttributeError:
            return None

    def params_changed(self, means, quats, scales, opacities, colors) -> bool:
        """True when the FP32 master tensors changed since the last rebuild.

        Uses ``torch.Tensor._version`` (incremented by in-place ops such as
        ``optimizer.step()``), so the culling hierarchy is refreshed exactly
        when the parameters drift, without a heuristic threshold.
        """
        if self._param_versions is None:
            return False
        current = self._capture_param_versions((means, quats, scales, opacities, colors))
        return current != self._param_versions

    def _begin_forward(self) -> None:
        self._pending_backward = True
        self._fwd_count += 1

    def _end_backward(self, expected_version: int) -> None:
        if expected_version != self._version:
            raise RuntimeError(
                "HiGS scene version mismatch in backward: forward captured "
                f"version {expected_version} but the scene is now at "
                f"{self._version}. Topology was mutated before backward "
                "completed; rebuild the graph or call mark_dirty() before the "
                "next forward instead."
            )
        self._pending_backward = False

    def release(self) -> None:
        """Release the pybind renderer (CUDA workspace). Safe to call twice."""
        if self.renderer is not None:
            self.renderer.release()
            self.renderer = None


def create_higs_renderer(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    sh_degree=None,
    sh_compression_mode="none",
) -> HigsRendererHandle:
    """Create an explicit, reusable HiGS renderer handle for training.

    Packs the FP32 master tensors into the HiGS inference layout (FP16
    ``qso_packed`` / ``colors_packed`` for RGB, FP32 for SH coefficients) and
    constructs the pybind renderer. The handle can be passed to
    ``rasterize_gaussian_higs_frozen`` / ``rasterize_gaussian_higs_dynamic``
    via ``scene=...`` and must be ``release()``-d when no longer needed.

    Raises:
        RuntimeError: if the experimental CUDA extension is not available.
    """
    if not _higs_backend_available():
        raise RuntimeError(
            "Cannot create a HiGS renderer handle: the experimental CUDA "
            "extension is not available. Build it with BUILD_EXPERIMENTAL=1 "
            "or use backward_mode='gsplat_recompute'."
        )
    from gsplat.scene.kernels.gaussian_inference_ops import (
        pack_gaussian_inference_scene,
    )
    from ..kernels.gaussian_inference_ops import (
        create_native_gaussian_inference_renderer,
    )

    sh_deg_i = -1 if sh_degree is None else sh_degree
    comp = _normalize_sh_compression_mode(sh_compression_mode)
    means_planar, qso_packed, colors_packed = pack_gaussian_inference_scene(
        means, quats, scales, opacities, colors, sh_deg_i, comp,
    )
    renderer = create_native_gaussian_inference_renderer(
        means_planar, qso_packed, colors_packed, sh_deg_i, int(comp),
    )
    return HigsRendererHandle(
        means_planar, qso_packed, colors_packed, renderer,
        packed_dtype=str(qso_packed.dtype),
        n_gaussians=means.shape[0],
        sh_degree=sh_deg_i,
        means=means, quats=quats, scales=scales,
        opacities=opacities, colors=colors,
    )

class _FrozenSceneTracker:
    """Tracks Gaussian count across renderer calls for freeze_topology validation."""
    def __init__(self) -> None:
        self._n_gaussians = None
        self._render_count = 0

    @property
    def render_count(self) -> int:
        return self._render_count

    def validate(self, n_gaussians: int, freeze_topology: bool) -> None:
        if not freeze_topology:
            self._n_gaussians = None
            self._render_count = 0
            return
        if self._n_gaussians is None:
            self._n_gaussians = n_gaussians
        elif self._n_gaussians != n_gaussians:
            raise RuntimeError(
                "freeze_topology=True but Gaussian count changed: "
                f"{self._n_gaussians} -> {n_gaussians}. "
                "Densification or pruning is not allowed in frozen-topology mode."
            )
        self._render_count += 1

    def reset(self) -> None:
        self._n_gaussians = None
        self._render_count = 0


_HIGS_FROZEN_TRACKER = _FrozenSceneTracker()
class _HigsDynamicScene:
    """Tracks scene version and topology state for dynamic-topology training (Stage C).

    Provides:
    - Monotonically increasing scene_version for each forward call.
    - dirty flag to signal that master tensors were mutated (densify/prune).
    - An owned :class:`HigsRendererHandle` (when the CUDA backend is available)
      that binds every forward/backward to the same scene version and raises on
      topology mutation while a backward is still pending.
    """

    def __init__(self) -> None:
        self._scene_version: int = 0
        self._n_gaussians: int | None = None
        self._dirty: bool = False
        self._forward_version: int = 0
        self.renderer_handle: HigsRendererHandle | None = None

    @property
    def scene_version(self) -> int:
        return self._scene_version

    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark_dirty(self) -> None:
        """Call after densify/prune/clone to signal master-tensor mutation.

        Delegates to the owned renderer handle (which raises if a backward is
        still pending) so the packed hierarchy can never be invalidated while
        a backward still needs it.
        """
        if self.renderer_handle is not None:
            self.renderer_handle.mark_dirty()
        self._dirty = True

    def on_duplicate(self, _sel: Tensor) -> None:
        self.mark_dirty()

    def on_split(self, _sel: Tensor, _rest: Tensor) -> None:
        self.mark_dirty()

    def on_remove(self, _mask: Tensor) -> None:
        self.mark_dirty()

    def on_relocate(self, _dead_indices: Tensor, _sampled_indices: Tensor) -> None:
        self.mark_dirty()

    def on_sample_add(self, _sampled_indices: Tensor) -> None:
        self.mark_dirty()

    def mark_clean(self) -> None:
        self._dirty = False

    def reset(self) -> None:
        """Reset scene version, dirty state and release the renderer handle."""
        if self.renderer_handle is not None:
            self.renderer_handle.release()
            self.renderer_handle = None
        self._scene_version = 0
        self._n_gaussians = None
        self._dirty = False
        self._forward_version = 0

    def ensure_renderer(
        self, means, quats, scales, opacities, colors, sh_degree,
    ) -> HigsRendererHandle | None:
        """Create (once) the owned renderer handle for the current master tensors."""
        if self.renderer_handle is None and _higs_backend_available():
            self.renderer_handle = create_higs_renderer(
                means, quats, scales, opacities, colors, sh_degree=sh_degree,
            )
        return self.renderer_handle

    def next_version(self, n_gaussians: int) -> int:
        """Advance to the next scene version for a new forward pass."""
        self._scene_version += 1
        self._n_gaussians = n_gaussians
        self._dirty = False
        self._forward_version = self._scene_version
        return self._scene_version

    def validate_count(self, n_gaussians: int) -> None:
        """Validate that a topology change is expected (scene was marked dirty)."""
        if self._n_gaussians is not None and self._n_gaussians != n_gaussians:
            if not self._dirty:
                raise RuntimeError(
                    "Gaussian count changed without mark_dirty(). "
                    "Call mark_dirty() after densify/prune before the next forward."
                )
        self._n_gaussians = n_gaussians


_HIGS_DYNAMIC_SCENE = _HigsDynamicScene()


def _densify_gaussians(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    grads: Tensor | None = None,
    threshold: float = 0.0002,
    noise_scale: float = 0.01,
    color_clamp: tuple | None = (0.0, 1.0),
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Duplicate Gaussians with large position gradients (densification)."""
    N = means.shape[0]
    if grads is not None:
        grad_norm = grads.norm(dim=-1)
        mask = grad_norm > threshold
    else:
        mask = torch.rand(N, device=means.device) < 0.05

    n_new = mask.sum().item()
    if n_new == 0:
        return means, quats, scales, opacities, colors

    dup_ids = mask.nonzero().flatten()
    dup_means = _gather_rows_fast(means, dup_ids)
    noise = torch.randn_like(dup_means) * noise_scale
    new_means = torch.cat([means, dup_means + noise])
    new_quats = torch.cat([quats, _gather_rows_fast(quats, dup_ids)])
    new_scales = torch.cat([scales, _gather_rows_fast(scales, dup_ids)])
    new_opacities = torch.cat(
        [opacities, _gather_rows_fast(opacities, dup_ids)]
    ).clamp(0.0, 1.0)
    new_colors = torch.cat([colors, _gather_rows_fast(colors, dup_ids)])
    if colors.dim() == 2 and color_clamp is not None:
        # RGB pre-activated colors are bounded to [0, 1] by default; pass
        # color_clamp=None to keep the raw (possibly unbounded) values. SH
        # coefficient tensors ([N, K, C]) are never clamped.
        new_colors = new_colors.clamp(color_clamp[0], color_clamp[1])
    return new_means, new_quats, new_scales, new_opacities, new_colors


def _prune_gaussians(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    opacity_threshold: float = 0.01,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Remove Gaussians with very low opacity (pruning)."""
    mask = opacities > opacity_threshold
    if mask.all():
        return means, quats, scales, opacities, colors

    keep_ids = mask.nonzero().flatten()
    return (
        _gather_rows_fast(means, keep_ids),
        _gather_rows_fast(quats, keep_ids),
        _gather_rows_fast(scales, keep_ids),
        _gather_rows_fast(opacities, keep_ids),
        _gather_rows_fast(colors, keep_ids),
    )


def sync_optimizer_state_for_topology_change(
    optimizer, old_to_new_map, **param_pairs,
):
    """Update an optimizer's param groups + Adam state after densify/prune.

    ``param_pairs`` maps a parameter name to ``(old_tensor, new_tensor)``.
    ``old_to_new_map`` is an int64 tensor of length ``N_new``: row ``i`` of the
    new tensors was produced from row ``old_to_new_map[i]`` of the old tensors,
    or ``-1`` for brand-new Gaussians (zero-initialized state).

    - duplicated Gaussians: Adam state rows are copied from the source row;
    - brand-new Gaussians: Adam state is zero-initialized;
    - pruned Gaussians: their state rows are simply dropped with the old tensors.

    The optimizer's ``param_groups`` are rewritten in place so ``optimizer.step()``
    keeps working on the new master tensors.
    """
    old_by_name = {name: pair[0] for name, pair in param_pairs.items()}
    new_by_name = {name: pair[1] for name, pair in param_pairs.items()}
    index_map = old_to_new_map.to(torch.long)
    n_new = index_map.numel()

    for group in optimizer.param_groups:
        params = group["params"]
        for i, param in enumerate(params):
            for name, old_t in old_by_name.items():
                if param is not old_t:
                    continue
                new_t = new_by_name[name]
                state = optimizer.state.get(param)
                if state is not None:
                    new_state = {}
                    for key, value in state.items():
                        if (
                            isinstance(value, torch.Tensor)
                            and value.dim() >= 1
                            and value.shape[0] == old_t.shape[0]
                        ):
                            # One fused kernel: row i is copied from
                            # index_map[i] (valid rows) or zero-filled
                            # (index_map[i] == -1, brand-new Gaussians).
                            new_state[key] = _gather_rows_new_zeros(
                                value, index_map
                            )
                        else:
                            new_state[key] = value
                    del optimizer.state[param]
                    optimizer.state[new_t] = new_state
                params[i] = new_t
    return optimizer


class _HigsAutogradFunction(torch.autograd.Function):
    """autograd Function wrapping HiGS culling + diff. rasterization (Stage B).

    Forward:
      - culls invisible Gaussians under no_grad (HiGS-native bitmask or
        standard projection culling), unioned across all cameras;
      - rasterizes the visible subset with the standard gsplat CUDA kernels
        (projection -> tile intersect -> blend) and captures the exact forward
        state (means2d/conics/colors_eval/opacities/tile offsets/flatten ids/
        render alphas/last ids/radii) for the native backward.

    Backward:
      - ``backward_mode="higs_native"``: computes gradients with the native
        HiGS CUDA kernels from the forward-captured state (no recomputation),
        writing FP32 master gradients. Invisible Gaussians get exactly zero
        gradient (stop-gradient visibility semantics).
      - ``backward_mode="gsplat_recompute"``: explicit fallback that re-runs
        the standard gsplat forward under autograd and uses
        ``torch.autograd.grad``.

    The visibility mask itself is never differentiated; it is discrete culling
    and the mask is a plain boolean index.
    """

    #: Metadata of the most recent forward, consumed by the public wrappers.
    last_forward_metadata: dict[str, Any] = {}

    @staticmethod
    def forward(
        ctx,
        means, quats, scales, opacities, colors,
        means2d_proxy,
        viewmats, Ks,
        width, height, sh_degree, tile_size,
        near_plane, far_plane, radius_clip, eps2d, background,
        render_mode, camera_model,
        enable_culling, use_higs_culling,
        backward_mode="higs_native", renderer_handle=None,
        sh_compression_mode="none",
        tile_sampling_ratio=1.0,
        sampling_mode="uniform",
        tile_mask=None,
        cull_refresh_interval=1,
        cull_cache_key="default",
    ):
        if backward_mode not in ("higs_native", "gsplat_recompute"):
            raise ValueError(
                f"backward_mode must be 'higs_native' or 'gsplat_recompute', "
                f"got {backward_mode!r}"
            )

        # NOTE: torch.autograd.Function.forward always runs with grad disabled,
        # so ``torch.is_grad_enabled()`` cannot be used here. Any input that
        # requires grad means a backward is expected for this call.
        training = any(
            t is not None and t.requires_grad
            for t in (means, quats, scales, opacities, colors)
        )
        comp = _normalize_sh_compression_mode(sh_compression_mode)
        ctx.sh_compression_mode = comp.name.lower()
        if comp != _get_sh_compression_none() and training:
            # PACKED_16B/32B quantize SH3 coefficients to FP16. The cast is
            # applied with a straight-through estimator (see _ste_sh_quantize),
            # so gradients flow back to the FP32 master SH tensors.
            if colors.dim() != 3:
                raise ValueError(
                    f"sh_compression_mode={comp.name} requires SH3 colors "
                    f"([N, K, 3]); got colors.dim()={colors.dim()}. Pass "
                    "sh_compression_mode='none' with RGB colors."
                )

        native = backward_mode == "higs_native" and _higs_backend_available()
        ctx.native = native
        ctx.backward_backend = "higs_native" if native else "gsplat_recompute"
        ctx.native_available = _higs_backend_available()
        if not (0.0 < float(tile_sampling_ratio) <= 1.0):
            raise ValueError(
                "tile_sampling_ratio must be in (0, 1], got %r" % (tile_sampling_ratio,)
            )
        ctx.tile_sampling_ratio = float(tile_sampling_ratio)
        if sampling_mode not in ("uniform", "stratified"):
            raise ValueError(
                f"sampling_mode must be 'uniform' or 'stratified', got {sampling_mode!r}"
            )
        ctx.sampling_mode = sampling_mode
        if tile_mask is not None:
            if not native:
                raise ValueError(
                    "tile_mask requires backward_mode='higs_native' "
                    "(masked isect filtering is implemented in the native capture path)."
                )
            if tile_mask.dtype != torch.bool or tile_mask.dim() != 3:
                raise ValueError(
                    "tile_mask must be a bool tensor of shape [C, tile_h, tile_w], "
                    "got dtype=%s dim=%d" % (tile_mask.dtype, tile_mask.dim())
                )
            n_tiles_check = math.ceil(width / tile_size) * math.ceil(height / tile_size)
            if tile_mask.shape[1] * tile_mask.shape[2] != n_tiles_check:
                raise ValueError(
                    "tile_mask shape %s does not match [C, th=%d, tw=%d]"
                    % (tuple(tile_mask.shape), math.ceil(height / tile_size),
                       math.ceil(width / tile_size))
                )
        ctx.tile_mask_external = tile_mask
        ctx.cull_refresh_interval = max(1, int(cull_refresh_interval))
        ctx.cull_cache_key = cull_cache_key
        if ctx.tile_sampling_ratio < 1.0 and not native:
            raise ValueError(
                "tile_sampling_ratio < 1.0 requires backward_mode='higs_native' "
                "(tile sampling is implemented in the native capture path)."
            )

        if native:
            if render_mode not in ("RGB", "D", "ED", "RGB+D", "RGB+ED"):
                raise ValueError(
                    "higs_native backward supports render_mode in "
                    f"('RGB', 'D', 'ED', 'RGB+D', 'RGB+ED'), got {render_mode!r}. "
                    "Hit-distance modes (d/Ed/RGB-d/RGB-Ed) require the ray-based "
                    "eval3d path; use backward_mode='gsplat_recompute'."
                )
            if camera_model not in ("pinhole", "ortho", "fisheye"):
                raise ValueError(
                    "higs_native backward supports camera_model in "
                    f"('pinhole', 'ortho', 'fisheye'), got {camera_model!r}. "
                    "Use backward_mode='gsplat_recompute' for ftheta/lidar."
                )
            if background is not None and background.dim() != 1:
                raise ValueError(
                    "higs_native backward expects a per-scene background of "
                    f"shape [D]; got {tuple(background.shape)}. "
                    "Use backward_mode='gsplat_recompute' for per-camera "
                    "backgrounds."
                )

        # -- normalize camera tensors to [1, C, ...] -----------------------
        viewmats = viewmats.contiguous()
        Ks = Ks.contiguous()
        if viewmats.dim() == 2:  # [4, 4]
            viewmats = viewmats.unsqueeze(0).unsqueeze(0)
            Ks = Ks.unsqueeze(0).unsqueeze(0)
        elif viewmats.dim() == 3:  # [C, 4, 4]
            viewmats = viewmats.unsqueeze(0)
            Ks = Ks.unsqueeze(0)
        C = viewmats.shape[-3]
        N_total = means.shape[0]
        if means2d_proxy.shape != (C, N_total, 2):
            raise ValueError(
                "means2d_proxy must have shape [C, N, 2], got "
                f"{tuple(means2d_proxy.shape)} for C={C}, N={N_total}"
            )

        # F9-1 deliberately has a narrow, explicit production contract.  The
        # baseline branch remains authoritative for every unsupported mode.
        from ..kernels import _backend as _inference_backend
        _f9_backend = _inference_backend._C
        f9_enabled = bool(
            native
            and _f9_backend is not None
            and hasattr(_f9_backend, "higs_gatherless_projected_producer")
            and sh_degree == 3
            and colors.dim() == 3 and colors.shape[1:] == (16, 3)
            and camera_model == "pinhole"
            and render_mode == "RGB"
            and comp == _get_sh_compression_none()
            and os.getenv("HIGS_DISABLE_F9", "0") != "1"
        )
        ctx.f9_enabled = f9_enabled

        # -- culling (discrete, stop-gradient) -----------------------------
        visible_ids = None
        if enable_culling:
            with torch.no_grad():
                # Keep the packed HiGS scene version-consistent when a handle
                # is present (dynamic mode bookkeeping) without running the
                # expensive per-camera render that was only used for the
                # visibility mask. The differentiable path never consumes the
                # packed FP16 scene (both backends consume the FP32 captured
                # tensors), so pure parameter drift updates the version
                # bookkeeping without re-packing (lightweight=True); real
                # topology changes (mark_dirty / N change) defer the pack and
                # set packed_stale so the non-training culling API rebuilds on
                # demand.
                if renderer_handle is not None:
                    _refresh_higs_renderer_scene(
                        renderer_handle, means, quats, scales, opacities,
                        colors, N_total, sh_degree, sh_compression_mode,
                        lightweight=True,
                    )
                # One batched FP32 projection over all cameras, unioned: same
                # near/far/radius_clip/eps2d/camera_model criteria as the
                # render path, but ~C full HiGS renders cheaper (no
                # intersect/rasterize pass). The camera model is passed through
                # so the mask is projection-consistent for ortho/fisheye too.
                # With a renderer handle (dynamic training) the result is
                # cached for cull_refresh_interval forwards; the cache is
                # invalidated by any topology change (mark_dirty/rebuild).
                if renderer_handle is not None:
                    vis_ids = _cull_visible_cached(
                        renderer_handle, ctx.cull_refresh_interval,
                        means, quats, scales, viewmats, Ks,
                        width, height,
                        eps2d=eps2d, near_plane=near_plane,
                        far_plane=far_plane, radius_clip=radius_clip,
                        camera_model=camera_model,
                        cache_key=ctx.cull_cache_key,
                    )
                else:
                    vis_ids, _vis_mask, _ratio = _cull_gaussians_batched(
                        means, quats, scales, viewmats, Ks,
                        width, height,
                        eps2d=eps2d, near_plane=near_plane,
                        far_plane=far_plane, radius_clip=radius_clip,
                        camera_model=camera_model,
                    )
                visible_ids = vis_ids

        if f9_enabled:
            N_visible = 0 if visible_ids is not None and visible_ids.numel() == 0 else (
                visible_ids.numel() if visible_ids is not None else N_total
            )
            v_means = v_quats = v_scales = v_opacities = v_colors = None
        elif visible_ids is not None and visible_ids.numel() > 0:
            N_visible = visible_ids.numel()
            v_means, v_quats, v_scales, v_opacities, v_colors = _gather_visible_native(
                means, quats, scales, opacities, colors, visible_ids,
            )
        else:
            N_visible = 0 if visible_ids is not None else N_total
            if N_visible == 0:
                v_means = means.new_empty((0, 3))
                v_quats = quats.new_empty((0, 4))
                v_scales = scales.new_empty((0, 3))
                v_opacities = opacities.new_empty((0,))
                v_colors = (
                    colors.new_empty((0, colors.shape[-1]))
                    if colors.dim() == 2
                    else colors.new_empty((0, colors.shape[1], colors.shape[2]))
                )
            else:
                v_means, v_quats, v_scales = means, quats, scales
                v_opacities, v_colors = opacities, colors

        # -- bind forward/backward to the same scene version ---------------
        if renderer_handle is not None:
            renderer_handle._begin_forward()
            ctx.scene_version = renderer_handle.version
            ctx.packed_dtype = renderer_handle.packed_dtype
            ctx.topology_rebuilt = renderer_handle.topology_rebuilt
        else:
            ctx.scene_version = None
            ctx.packed_dtype = None
            ctx.topology_rebuilt = False
        ctx.renderer_handle = renderer_handle

        backgrounds_b = _expand_background(background, C, means.device)

        if N_visible == 0:
            # Pure-background image; every Gaussian is invisible -> zero grads.
            nch = _render_mode_channel_count(render_mode)
            with torch.no_grad():
                frame = torch.zeros(
                    (C, height, width, nch), device=means.device, dtype=means.dtype
                )
                if backgrounds_b is not None and nch >= 3:
                    bg = backgrounds_b.reshape(C, 3)
                    if nch == 4:
                        bg = torch.cat([bg, bg.new_zeros((C, 1))], dim=-1)
                    frame = frame + bg.reshape(C, 1, 1, nch)
                alpha_ch = torch.zeros(
                    (C, height, width, 1), device=means.device, dtype=means.dtype
                )
            compact_saved = (None, None, None, None, None) if f9_enabled else (
                v_means, v_quats, v_scales, v_opacities,
                v_colors if sh_degree is not None else None,
            )
            ctx.save_for_backward(
                means, quats, scales, opacities, colors,
                means2d_proxy,
                viewmats, Ks,
                torch.empty(0, dtype=torch.long, device=means.device),
                None, None, None, None, None, None, None, None, None, None,
                *compact_saved,
                backgrounds_b,
            )
            ctx.N_total = N_total
            ctx.N_visible = 0
            ctx.C = C
            ctx.width = width
            ctx.height = height
            ctx.sh_degree = sh_degree
            ctx.tile_size = tile_size
            ctx.near_plane = near_plane
            ctx.far_plane = far_plane
            ctx.radius_clip = radius_clip
            ctx.eps2d = eps2d
            ctx.render_mode = render_mode
            ctx.camera_model = camera_model
            ctx.culling_ratio = 1.0
            ctx.use_higs_culling = bool(use_higs_culling)
            _HigsAutogradFunction.last_forward_metadata = {
                "backward_backend": ctx.backward_backend,
                "native_available": ctx.native_available,
                "n_gaussians": N_total,
                "n_visible": 0,
                "culling_ratio": 1.0,
                "topology_rebuilt": ctx.topology_rebuilt,
                "packed_dtype": ctx.packed_dtype,
                "sh_compression_mode": getattr(ctx, "sh_compression_mode", "none"),
                "render_mode": render_mode,
                "scene_version": ctx.scene_version,
                "sampled_tile_ratio": 1.0,
                "sampling_mode": "uniform",
                "tile_mask": None,
                "n_isects": 0,
                "n_isects_full": 0,
                "densification_radii": torch.zeros(
                    (C, N_total, 2), dtype=torch.int32, device=means.device
                ),
                "visible_gaussian_ids": torch.empty(
                    0, dtype=torch.long, device=means.device
                ),
            }
            if C == 1:
                return frame.squeeze(0)[None], alpha_ch.squeeze(0)[None]  # [1,H,W,3], [1,H,W,1]
            return frame[None], alpha_ch[None]  # [1,C,H,W,3], [1,C,H,W,1]

        # -- differentiable forward on the visible subset -------------------
        if not f9_enabled and sh_degree is not None and getattr(ctx, "sh_compression_mode", "none") != "none":
            v_colors = _ste_sh_quantize(v_colors)
        if not f9_enabled:
            v_means_b = v_means.unsqueeze(0)  # [1, N, 3]
            v_quats_b = v_quats.unsqueeze(0)
            v_scales_b = v_scales.unsqueeze(0)
            v_opacities_b = v_opacities.unsqueeze(0)
            v_colors_input = v_colors.unsqueeze(0) if v_colors.dim() == 2 else v_colors

        if native:
            render_colors, render_alphas, captured = (
                _HigsAutogradFunction._native_forward_capture(
                    ctx,
                    (means if f9_enabled else v_means_b),
                    (quats if f9_enabled else v_quats_b),
                    (scales if f9_enabled else v_scales_b),
                    (opacities if f9_enabled else v_opacities_b),
                    (colors if f9_enabled else v_colors_input),
                    viewmats, Ks, width, height, sh_degree,
                    tile_size, near_plane, far_plane, radius_clip, eps2d,
                    backgrounds_b, camera_model, render_mode,
                    ctx.tile_sampling_ratio,
                    ctx.sampling_mode,
                    ctx.tile_mask_external,
                    ctx.cull_refresh_interval,
                    visible_ids=(visible_ids if visible_ids is not None else torch.arange(N_total, device=means.device)),
                    gatherless=f9_enabled,
                )
            )
        else:
            from gsplat.rendering import rasterization

            render_colors, render_alphas, std_info = rasterization(
                means=v_means_b, quats=v_quats_b, scales=v_scales_b,
                opacities=v_opacities_b, colors=v_colors_input,
                viewmats=viewmats, Ks=Ks, width=width, height=height,
                sh_degree=sh_degree, backgrounds=backgrounds_b,
                packed=False, tile_size=tile_size,
                near_plane=near_plane, far_plane=far_plane,
                radius_clip=radius_clip, eps2d=eps2d,
                camera_model=camera_model, render_mode=render_mode,
            )
            captured = None

        frame = render_colors.squeeze(0).squeeze(0)
        alpha_ch = render_alphas.squeeze(0).squeeze(0)

        # -- save state for backward ---------------------------------------
        master_visible_ids = (
            visible_ids if visible_ids is not None
            else torch.arange(N_total, device=means.device)
        )
        visible_radii = (
            captured[8]
            if captured is not None
            else std_info["radii"].reshape(C, N_visible, 2)
        )
        densification_radii = torch.zeros(
            (C, N_total, 2), dtype=visible_radii.dtype, device=means.device
        )
        densification_radii.index_copy_(1, master_visible_ids, visible_radii)

        compact_saved = (None, None, None, None, None) if f9_enabled else (
            v_means, v_quats, v_scales, v_opacities,
            v_colors if sh_degree is not None else None,
        )
        ctx.save_for_backward(
            means, quats, scales, opacities, colors,
            means2d_proxy,
            viewmats, Ks,
            master_visible_ids,
            *(captured if captured is not None else (None,) * 10),
            *compact_saved,
            backgrounds_b,
        )
        ctx.N_total = N_total
        ctx.N_visible = N_visible
        ctx.C = C
        ctx.width = width
        ctx.height = height
        ctx.sh_degree = sh_degree
        ctx.tile_size = tile_size
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        ctx.radius_clip = radius_clip
        ctx.eps2d = eps2d
        ctx.render_mode = render_mode
        ctx.camera_model = camera_model
        ctx.culling_ratio = 1.0 - (N_visible / max(N_total, 1))
        ctx.use_higs_culling = bool(use_higs_culling)
        _HigsAutogradFunction.last_forward_metadata = {
            "backward_backend": ctx.backward_backend,
            "native_available": ctx.native_available,
            "n_gaussians": N_total,
            "n_visible": N_visible,
            "f9_enabled": f9_enabled,
            "culling_ratio": ctx.culling_ratio,
            "topology_rebuilt": ctx.topology_rebuilt,
            "packed_dtype": ctx.packed_dtype,
            "sh_compression_mode": getattr(ctx, "sh_compression_mode", "none"),
            "render_mode": render_mode,
            "scene_version": ctx.scene_version,
            "sampled_tile_ratio": getattr(ctx, "sampled_tile_ratio", 1.0),
            "sampling_mode": getattr(ctx, "sampling_mode", "uniform"),
            "tile_mask": getattr(ctx, "tile_mask", None),
            "n_isects": getattr(ctx, "n_isects_sampled", 0),
            "n_isects_full": getattr(ctx, "n_isects_full", 0),
            "densification_radii": densification_radii,
            "visible_gaussian_ids": master_visible_ids,
        }
        return frame[None], alpha_ch[None]

    @staticmethod
    def _native_forward_capture(
        ctx,
        means_b, quats_b, scales_b, opacities_b, colors_input,
        viewmats, Ks, width, height, sh_degree,
        tile_size, near_plane, far_plane, radius_clip, eps2d,
        backgrounds_b, camera_model, render_mode,
        tile_sampling_ratio=1.0,
        sampling_mode="uniform",
        tile_mask=None,
        cull_refresh_interval=1,
        visible_ids=None,
        gatherless=False,
    ):
        """Run the standard gsplat CUDA forward on the visible subset and
        capture every tensor the native backward consumes, in ONE pass."""
        from gsplat.cuda._wrapper import (
            fully_fused_projection,
            isect_tiles,
            isect_offset_encode,
            _make_lazy_cuda_func,
        )
        from gsplat.rendering import _maybe_evaluate_sh
        from ..kernels import _backend as _inference_backend

        C = viewmats.shape[-3]
        N = visible_ids.numel() if gatherless else means_b.shape[-2]
        tile_width = math.ceil(width / tile_size)
        tile_height = math.ceil(height / tile_size)

        with torch.no_grad():
            if gatherless:
                # The extension indexes master rows through visible_ids and
                # directly writes the state consumed by the unchanged F4/F5.
                backend = _inference_backend._C
                cam_positions = backend.higs_camera_positions_from_viewmats(
                    viewmats[0].contiguous()
                )
                radii, means2d, depths, conics, opacities_bc, colors_eval = (
                    backend.higs_gatherless_projected_producer(
                        visible_ids.contiguous(), means_b.contiguous(),
                        quats_b.contiguous(), scales_b.contiguous(),
                        opacities_b.contiguous(), colors_input.contiguous(),
                        viewmats[0].contiguous(), Ks[0].contiguous(),
                        cam_positions, width, height, eps2d, near_plane,
                        far_plane, radius_clip,
                    )
                )
                radii = radii.reshape(1, C, N, 2)
                means2d = means2d.reshape(1, C, N, 2)
                depths = depths.reshape(1, C, N)
                conics = conics.reshape(1, C, N, 3)
                opacities_bc = opacities_bc.reshape(1, C, N)
                colors_eval = colors_eval.reshape(1, C, N, 3)
            else:
                radii, means2d, depths, conics, compensations = fully_fused_projection(
                    means=means_b.contiguous(),
                    covars=None,
                    quats=quats_b.contiguous(),
                    scales=scales_b.contiguous(),
                    viewmats=viewmats,
                    Ks=Ks,
                    width=width,
                    height=height,
                    eps2d=eps2d,
                    near_plane=near_plane,
                    far_plane=far_plane,
                    radius_clip=radius_clip,
                    packed=False,
                    calc_compensations=False,
                    camera_model=camera_model,
                )
                opacities_bc = torch.broadcast_to(
                    opacities_b[..., None, :], (1, C, N)
                ).contiguous()  # [1, C, N]
            n_tiles = tile_width * tile_height
            if tile_mask is not None:
                # Explicit harness-provided mask (e.g. error-guided sampling).
                # The mask is applied verbatim; the harness is responsible for
                # the estimator (e.g. importance weights on the loss).
                mask = tile_mask.reshape(C, n_tiles)
                sampled_ratio = float(mask.float().mean())
            elif 0.0 < tile_sampling_ratio < 1.0:
                # Tile sampling without replacement. The mean over the sampled
                # tiles is an unbiased estimator of the full-frame mean loss,
                # so the harness needs no 1/r rescale -- only a mask over the
                # sampled tiles.
                if sampling_mode == "stratified":
                    # One tile per stratum of size round(1/r) (consecutive
                    # tiles in row-major order). Every region of the frame
                    # keeps a sample, which reduces gradient-estimator
                    # variance vs iid uniform sampling at the same r.
                    stratum_size = max(1, int(round(1.0 / tile_sampling_ratio)))
                    n_strata = (n_tiles + stratum_size - 1) // stratum_size
                    rnd = torch.rand((C, n_strata), device=means_b.device)
                    pick = (rnd * stratum_size).long().clamp(max=stratum_size - 1)
                    base = (
                        torch.arange(n_strata, device=means_b.device)[None, :]
                        * stratum_size
                    )
                    sel = (base + pick).clamp(max=n_tiles - 1)  # [C, n_strata]
                    mask = torch.zeros(
                        (C, n_tiles), dtype=torch.bool, device=means_b.device
                    )
                    mask.scatter_(1, sel, True)
                else:
                    k = max(1, int(round(n_tiles * tile_sampling_ratio)))
                    rnd = torch.rand((C, n_tiles), device=means_b.device)
                    _, top_idx = torch.topk(rnd, k, dim=1)  # [C, k]
                    mask = torch.zeros(
                        (C, n_tiles), dtype=torch.bool, device=means_b.device
                    )
                    mask.scatter_(1, top_idx, True)
                sampled_ratio = float(mask.float().mean())
            else:
                mask = torch.ones(
                    (C, n_tiles), dtype=torch.bool, device=means_b.device
                )
                sampled_ratio = 1.0
            ctx.tile_mask = mask.reshape(C, tile_height, tile_width)
            ctx.sampled_tile_ratio = sampled_ratio
            # Round 38: pass the mask into isect_tiles so the CUDA op restricts
            # the per-Gaussian counts AND the emitted isects to the selected
            # tiles and sorts only the compacted list (the full-N radix sort
            # and the Python-level mask filtering are both removed from the
            # sampled path). The mask is computed before the intersection so
            # the op never builds the full isect list.
            if sampled_ratio < 1.0:
                raise RuntimeError(
                    "tile_sampling_ratio < 1.0 requires the optional core "
                    "tile_mask intersection ABI, which is intentionally absent "
                    "from the frozen B1A base."
                )
            # Frozen B2 Python wrapper predates the immutable core ABI's
            # trailing tile_mask slot.  Full AccuTile semantics are unchanged:
            # pass its explicit null value to the exact same operator.
            tiles_per_gauss, isect_ids, flatten_ids = torch.ops.gsplat.intersect_tile(
                means2d,
                radii,
                depths,
                conics,
                opacities_bc,
                None, None, C,
                tile_size, tile_width, tile_height,
                True, False, None,
            )
            ctx.n_isects_sampled = int(isect_ids.numel())
            # The op returns per-Gaussian counts restricted to the selected
            # tiles, so the exact full-N count is not available from its
            # outputs; the sampled isect fraction is recovered exactly at the
            # tile level (isect_frac ~ sampled_tile_ratio).
            ctx.n_isects_full = int(round(ctx.n_isects_sampled / sampled_ratio))
            isect_offsets = isect_offset_encode(
                isect_ids, C, tile_width, tile_height
            ).reshape((1, C, tile_height, tile_width))
            has_color = render_mode in ("RGB", "RGB+D", "RGB+ED")
            has_depth = _render_mode_has_depth(render_mode)
            is_expected = render_mode in ("ED", "RGB+ED")
            if gatherless:
                # F9-1's contract is RGB SH3, so colors_eval was emitted by
                # the producer.  Do not re-materialize or re-evaluate SH.
                pass
            elif has_color:
                colors_eval = _maybe_evaluate_sh(
                    sh_degree, colors_input, means_b, radii, viewmats,
                    (1,), C, N, True,
                )
                colors_eval = colors_eval.contiguous()  # [1, C, N, 3]
            else:
                # depth-only modes ignore the color input entirely.
                colors_eval = torch.empty(
                    (1, C, N, 0), device=means_b.device, dtype=means_b.dtype
                )
            if has_depth:
                # projection depth (camera-space z) is composited like a color
                # channel; the native backward chains its gradient to v_means.
                colors_eval = torch.cat(
                    [colors_eval, depths[..., None]], dim=-1
                )  # [1, C, N, D]
            colors_eval = colors_eval.contiguous()
            if backgrounds_b is not None:
                if has_color:
                    bg_kernel = backgrounds_b.reshape(1, C, 3)
                    if has_depth:
                        # depth channel has a zero background (not an input).
                        bg_kernel = torch.cat(
                            [bg_kernel, bg_kernel.new_zeros((1, C, 1))], dim=-1
                        )
                else:
                    # depth-only modes ignore the RGB background input and
                    # use a zero background, matching standard gsplat.
                    bg_kernel = backgrounds_b.new_zeros((1, C, 1))
                bg_kernel = bg_kernel.contiguous()
            else:
                bg_kernel = None
            render_colors, render_alphas, _absgrad, last_ids = (
                _make_lazy_cuda_func("rasterize_to_pixels_3dgs")(
                    means2d.contiguous(),
                    conics.contiguous(),
                    colors_eval.contiguous(),
                    opacities_bc.contiguous(),
                    bg_kernel,
                    None,
                    width,
                    height,
                    tile_size,
                    isect_offsets.contiguous(),
                    flatten_ids.contiguous(),
                    False,
                    False,
                )
            )
        D = colors_eval.shape[-1]
        if is_expected:
            # expected depth: render_depth = acc_depth / alpha (same as the
            # standard gsplat post-processing). Keep the raw accumulation for
            # the backward chain through the normalization.
            depth_acc = render_colors[..., -1:]  # [1, C, H, W, 1]
            render_depth = depth_acc / render_alphas.clamp(min=1e-10)
            if has_color:
                render_colors = torch.cat(
                    [render_colors[..., :3], render_depth], dim=-1
                )
            else:
                render_colors = render_depth
        else:
            depth_acc = None

        captured = (
            means2d.reshape(-1, 2).contiguous(),              # [I*N, 2]
            conics.reshape(-1, 3).contiguous(),               # [I*N, 3]
            colors_eval.reshape(-1, D).contiguous(),          # [I*N, D]
            opacities_bc.reshape(-1).contiguous(),            # [I*N]
            isect_offsets.reshape(C, tile_height, tile_width).contiguous(),  # [I, th, tw]
            flatten_ids.contiguous(),                          # [n_isects]
            render_alphas.reshape(C, height, width).contiguous(),  # [I, H, W]
            last_ids.contiguous(),                             # [I, H, W]
            radii.reshape(C, N, 2).contiguous(),               # [I, N, 2]
            (depth_acc.reshape(C, height, width).contiguous()
             if depth_acc is not None else None),              # [I, H, W] or None
        )
        return render_colors, render_alphas, captured

    @staticmethod
    def backward(ctx, grad_frame, grad_alpha):
        if ctx.renderer_handle is not None:
            try:
                ctx.renderer_handle._end_backward(ctx.scene_version)
            except RuntimeError:
                # The handle may have been released by an explicit
                # ``release()`` before backward; only version mismatches
                # indicate a real topology-mutation bug, and those raise below
                # in the native path too. Keep the pending flag cleared.
                if ctx.renderer_handle.version != ctx.scene_version:
                    raise
        if ctx.native:
            return _HigsAutogradFunction._native_backward(ctx, grad_frame, grad_alpha)
        return _HigsAutogradFunction._recompute_backward(ctx, grad_frame, grad_alpha)

    # ------------------------------------------------------------------
    @staticmethod
    def _native_backward(ctx, grad_frame, grad_alpha):
        from ..kernels import _backend as _inference_backend

        backend = _inference_backend._C
        if backend is None or not hasattr(backend, "higs_rasterize_backward"):
            raise RuntimeError(
                "higs_native backward requested but the experimental CUDA "
                "extension is not available; rerun with "
                "backward_mode='gsplat_recompute'."
            )

        saved = ctx.saved_tensors
        (
            means, quats, scales, opacities, colors, means2d_proxy,
            viewmats, Ks, visible_ids,
            means2d_f, conics_f, colors_eval_f, opacities_f,
            tile_offsets_f, flatten_ids_f, render_alphas_f, last_ids_f,
            radii_f, depth_acc,
            v_means, v_quats, v_scales, v_opacities,
            sh_coeffs, backgrounds_b,
        ) = saved

        I = ctx.C
        H, W = ctx.height, ctx.width
        device = means.device
        nch = _render_mode_channel_count(ctx.render_mode)

        if grad_frame is None:
            v_render_colors = torch.zeros((I, H, W, nch), device=device)
        else:
            v_render_colors = grad_frame.reshape(I, H, W, nch).contiguous()
        if grad_alpha is None:
            v_render_alphas = torch.zeros((I, H, W), device=device)
        else:
            v_render_alphas = grad_alpha.reshape(I, H, W).contiguous()

        if depth_acc is not None:
            # Expected-depth modes: render_depth = depth_acc / alpha_safe
            # is a post-rasterization normalization. Chain the depth-channel
            # gradient back to the raw accumulation and to the render alphas.
            g_depth = v_render_colors[..., -1].contiguous()  # [I, H, W]
            alpha_safe = render_alphas_f.clamp(min=1e-10)
            v_acc = g_depth / alpha_safe
            v_alpha_from_depth = -g_depth * depth_acc / alpha_safe.square()
            # clamp backward: d(alpha_safe)/d(alpha) = 1 where alpha >= 1e-10
            v_alpha_from_depth = v_alpha_from_depth * (render_alphas_f >= 1e-10)
            v_render_colors = torch.cat(
                [v_render_colors[..., : nch - 1], v_acc[..., None]], dim=-1
            ).contiguous()
            v_render_alphas = v_render_alphas + v_alpha_from_depth

        if ctx.N_visible == 0:
            # All Gaussians invisible -> zero gradients, background still gets
            # the constant-image gradient if it was provided.
            grad_means = torch.zeros_like(means)
            grad_quats = torch.zeros_like(quats)
            grad_scales = torch.zeros_like(scales)
            grad_opacities = torch.zeros_like(opacities)
            grad_colors = torch.zeros_like(colors)
            grad_background = None
            if (
                ctx.needs_input_grad[16]
                and backgrounds_b is not None
                and nch >= 3
            ):
                grad_background = v_render_colors[..., :3].sum(dim=(0, 1, 2))
            return _HigsAutogradFunction._assemble_grads(
                ctx, grad_means, grad_quats, grad_scales, grad_opacities,
                grad_colors, torch.zeros_like(means2d_proxy), grad_background,
            )

        bg_kernel = None
        if backgrounds_b is not None:
            if _render_mode_color_channels(ctx.render_mode) > 0:
                bg_kernel = backgrounds_b.reshape(I, 3).contiguous()
                if nch == 4:
                    # depth channel has a zero background (not an input).
                    bg_kernel = torch.cat(
                        [bg_kernel, bg_kernel.new_zeros((I, 1))], dim=-1
                    )
            else:
                # depth-only modes: zero background, matching the forward.
                bg_kernel = backgrounds_b.new_zeros((I, 1))
            bg_kernel = bg_kernel.contiguous()

        # Master FP32 gradient accumulators: pre-zeroed here, then written in
        # place by the native kernels through `visible_ids` (strictly
        # increasing, duplicate-free), which removes the per-tensor
        # zeros_like + index_copy_ scatter from the hot path.
        grad_means = torch.zeros_like(means)
        grad_quats = torch.zeros_like(quats)
        grad_scales = torch.zeros_like(scales)
        grad_opacities = torch.zeros_like(opacities)
        grad_colors = torch.zeros_like(colors)
        # Round 39: backward tile sampling - compact the pixel-blend grid to
        # the selected tiles only. active_tiles are the global (image, tile)
        # ids of ctx.tile_mask, so the blend backward launches one block per
        # selected tile instead of the full tile grid; the per-pixel
        # background gradient moves to a separate all-pixel kernel inside the
        # launcher. The full-coverage (all-ones) case keeps the dense path.
        active_tiles = None
        if getattr(ctx, "sampled_tile_ratio", 1.0) < 1.0:
            tm_flat = ctx.tile_mask.reshape(-1)
            active_tiles = torch.nonzero(tm_flat, as_tuple=False).squeeze(1).to(torch.int32)
        out = backend.higs_rasterize_backward(
            means2d=means2d_f,
            conics=conics_f,
            colors_eval=colors_eval_f,
            opacities=opacities_f,
            backgrounds=bg_kernel,
            tile_offsets=tile_offsets_f,
            flatten_ids=flatten_ids_f,
            active_tiles=active_tiles,
            render_alphas=render_alphas_f,
            last_ids=last_ids_f,
            # F9-1 C++ VJPs resolve compact projected rows through
            # visible_ids and directly load these master parameters.
            means=means,
            quats=quats,
            scales=scales,
            radii=radii_f,
            viewmats=viewmats,
            Ks=Ks,
            width=ctx.width,
            height=ctx.height,
            tile_size=ctx.tile_size,
            eps2d=ctx.eps2d,
            camera_model=(
                0 if ctx.camera_model == "pinhole"
                else 1 if ctx.camera_model == "ortho"
                else 2  # CameraModelType: PINHOLE=0, ORTHO=1, FISHEYE=2
            ),
            v_render_colors=v_render_colors,
            v_render_alphas=v_render_alphas,
            sh_coeffs=colors,
            sh_degree=-1 if ctx.sh_degree is None else ctx.sh_degree,
            visible_ids=visible_ids,
            grad_means=grad_means,
            grad_quats=grad_quats,
            grad_scales=grad_scales,
            grad_opacities=grad_opacities,
            grad_colors=grad_colors,
        )
        (
            _v_means_g, _v_quats_g, _v_scales_g, _v_opacities_g,
            _v_colors_master_g, v_backgrounds_g, v_means2d_g,
        ) = out

        grad_means2d = torch.zeros_like(means2d_proxy)
        grad_means2d.index_copy_(
            1, visible_ids, v_means2d_g.reshape(I, ctx.N_visible, 2)
        )

        grad_background = None
        if ctx.needs_input_grad[16] and backgrounds_b is not None and nch >= 3:
            # the kernel accumulates a depth-channel background gradient too;
            # the RGB background input only receives the color-channel part.
            grad_background = v_backgrounds_g if nch == 3 else v_backgrounds_g[..., :3]

        return _HigsAutogradFunction._assemble_grads(
            ctx, grad_means, grad_quats, grad_scales, grad_opacities,
            grad_colors, grad_means2d, grad_background,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _recompute_backward(ctx, grad_frame, grad_alpha):
        from gsplat.rendering import rasterization

        saved = ctx.saved_tensors
        (
            means, quats, scales, opacities, colors, means2d_proxy,
            viewmats, Ks, visible_ids,
            _m2d, _con, _ce, _opa, _to, _fi, _ra, _li, _rad, _depth_acc,
            _v_m, _v_q, _v_s, _v_o, _sh, backgrounds_b,
        ) = saved

        I = ctx.C
        H, W = ctx.width, ctx.height
        device = means.device
        nch = _render_mode_channel_count(ctx.render_mode)
        # Gradients reshaped to the exact output shapes of forward().
        if ctx.C == 1:
            gf_out = grad_frame if grad_frame is not None else torch.zeros((1, H, W, nch), device=device)
            ga_out = grad_alpha if grad_alpha is not None else torch.zeros((1, H, W, 1), device=device)
        else:
            gf_out = grad_frame if grad_frame is not None else torch.zeros((1, I, H, W, nch), device=device)
            ga_out = grad_alpha if grad_alpha is not None else torch.zeros((1, I, H, W, 1), device=device)
        gf = gf_out.reshape(I, H, W, nch)
        ga = ga_out.reshape(I, H, W)

        with torch.enable_grad():
            needs = ctx.needs_input_grad
            tracked = {}
            for name, t, idx in (
                ("means", means, 0),
                ("quats", quats, 1),
                ("scales", scales, 2),
                ("opacities", opacities, 3),
                ("colors", colors, 4),
            ):
                if needs[idx]:
                    tracked[name] = t.detach().requires_grad_(True)

            if not tracked:
                return (None,) * 29

            alpha_final = None
            if ctx.N_visible == 0:
                grads = {name: torch.zeros_like(t) for name, t in tracked.items()}
            else:
                v_means = tracked["means"][visible_ids]
                v_quats = tracked["quats"][visible_ids]
                v_scales = tracked["scales"][visible_ids]
                v_opacities = tracked["opacities"][visible_ids]
                v_colors = (
                    tracked["colors"][visible_ids]
                    if tracked["colors"].dim() == 2
                    else tracked["colors"][visible_ids]
                )
                v_means_b = v_means.unsqueeze(0)
                v_quats_b = v_quats.unsqueeze(0)
                v_scales_b = v_scales.unsqueeze(0)
                v_opacities_b = v_opacities.unsqueeze(0)
                v_colors_input = v_colors.unsqueeze(0) if v_colors.dim() == 2 else v_colors

                render_colors, render_alphas, recompute_info = rasterization(
                    means=v_means_b, quats=v_quats_b, scales=v_scales_b,
                    opacities=v_opacities_b, colors=v_colors_input,
                    viewmats=viewmats, Ks=Ks,
                    width=ctx.width, height=ctx.height,
                    sh_degree=ctx.sh_degree, backgrounds=backgrounds_b,
                    packed=False, tile_size=ctx.tile_size,
                    near_plane=ctx.near_plane, far_plane=ctx.far_plane,
                    radius_clip=ctx.radius_clip, eps2d=ctx.eps2d,
                    camera_model=ctx.camera_model, render_mode=ctx.render_mode,
                )
                frame = render_colors.squeeze(0).squeeze(0)
                alpha_ch = render_alphas.squeeze(0).squeeze(0)
                outputs = (frame[None], alpha_ch[None])
                out_grads = torch.autograd.grad(
                    outputs,
                    tuple(tracked.values()) + (recompute_info["means2d"],),
                    grad_outputs=(gf_out, ga_out),
                    allow_unused=True,
                )
                alpha_final = render_alphas
                grads = dict(zip(tracked.keys(), out_grads[:-1]))
                visible_means2d_grad = out_grads[-1]
                for k, g in list(grads.items()):
                    if g is None:
                        grads[k] = torch.zeros_like(tracked[k])

        grad_means = grads.get("means")
        grad_quats = grads.get("quats")
        grad_scales = grads.get("scales")
        grad_opacities = grads.get("opacities")
        grad_colors = grads.get("colors")
        grad_means2d = torch.zeros_like(means2d_proxy)
        if ctx.N_visible > 0:
            grad_means2d.index_copy_(
                1,
                visible_ids,
                visible_means2d_grad.reshape(I, ctx.N_visible, 2),
            )

        grad_background = None
        if (
            ctx.needs_input_grad[16]
            and backgrounds_b is not None
            and grad_frame is not None
            and nch >= 3
        ):
            # dL/dbg = sum_pixels v_render_colors * (1 - render_alpha)
            with torch.no_grad():
                if alpha_final is not None:
                    alpha_im = alpha_final.reshape(I, H, W)
                else:
                    alpha_im = torch.zeros((I, H, W), device=device)
                gb = (gf * (1.0 - alpha_im[..., None])).sum(dim=(0, 1, 2))
                grad_background = gb if nch == 3 else gb[..., :3]

        return _HigsAutogradFunction._assemble_grads(
            ctx, grad_means, grad_quats, grad_scales, grad_opacities,
            grad_colors, grad_means2d, grad_background,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _assemble_grads(
        ctx, grad_means, grad_quats, grad_scales, grad_opacities,
        grad_colors, grad_means2d, grad_background,
    ):
        needs = ctx.needs_input_grad
        return (
            grad_means if needs[0] else None,
            grad_quats if needs[1] else None,
            grad_scales if needs[2] else None,
            grad_opacities if needs[3] else None,
            grad_colors if needs[4] else None,
            grad_means2d if needs[5] else None,
            None, None,  # viewmats, Ks
            None, None, None, None,  # width, height, sh_degree, tile_size
            None, None, None, None,  # near_plane, far_plane, radius_clip, eps2d
            grad_background,
            None, None,  # render_mode, camera_model
            None, None,  # enable_culling, use_higs_culling
            None, None, None,  # backward_mode, renderer_handle, sh_compression_mode
            None,  # tile_sampling_ratio (non-tensor input)
            None,  # sampling_mode (non-tensor input)
            None,  # tile_mask (bool tensor, non-differentiable input)
            None,  # cull_refresh_interval (non-tensor input)
            None,  # cull_cache_key (non-tensor input)
        )


def _new_means2d_proxy(means: Tensor, viewmats: Tensor, *parameters: Tensor) -> Tensor:
    n_cameras = 1 if viewmats.dim() == 2 else viewmats.shape[-3]
    training = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (means, *parameters)
    )
    return torch.zeros(
        (n_cameras, means.shape[0], 2),
        dtype=means.dtype,
        device=means.device,
        requires_grad=training,
    )

def _slice_colors_for_sh_degree(colors: Tensor, sh_degree: int | None) -> Tensor:
    """Restrict master SH colors to the coefficients active at ``sh_degree``.

    Training uses progressive SH: ``colors`` stays at the full ``[N, K_max, 3]``
    master shape while each forward runs at ``min(step // interval, K_max)``.
    HiGS renderer packing and the differentiable forward require the active
    coefficient count, so a superset is sliced here. The slice is
    differentiable, so gradients flow back to the full master tensor; inactive
    coefficients simply receive zero gradient at this step.
    """
    if sh_degree is None:
        return colors
    k_active = (int(sh_degree) + 1) ** 2
    if colors.dim() == 3 and colors.shape[1] > k_active:
        return colors[:, :k_active, :].contiguous()
    return colors


def _higs_frozen_forward(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    *,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    sh_degree=None,
    tile_size: int = 16,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    background=None,
    render_mode: str = "RGB",
    camera_model: str = "pinhole",
    enable_culling: bool = True,
    use_higs_culling: bool = False,
    return_higs_preview: bool = False,
    freeze_topology: bool = True,
    backward_mode: str = "higs_native",
    sh_compression_mode="none",
    tile_sampling_ratio: float = 1.0,
    sampling_mode: str = "uniform",
    tile_mask=None,
    cull_refresh_interval: int = 1,
    scene=None,
) -> dict:
    """Forward pass for frozen-topology HiGS diff. path (Stage B).

    Delegates to _HigsAutogradFunction which handles culling + diff.
    rasterization. Handles HiGS preview and metadata outside the autograd
    Function.

    Args:
        backward_mode: "higs_native" (default, native CUDA backward) or
            "gsplat_recompute" (explicit standard-gsplat recomputation
            fallback; metadata then reports ``backward_backend="gsplat_recompute"``).
        sh_compression_mode: "none" (default) or a PACKED_16B/PACKED_32B mode;
            lossy SH quantization is applied with a straight-through estimator
            in the training path.
        scene: optional :class:`HigsRendererHandle` reused across calls so the
            packed FP16 buffers and renderer are not rebuilt per frame.
    """

    _HIGS_FROZEN_TRACKER.validate(means.shape[0], freeze_topology)

    means2d_proxy = _new_means2d_proxy(
        means, viewmats, quats, scales, opacities, colors
    )
    frame, alpha = _HigsAutogradFunction.apply(
        means, quats, scales, opacities, colors,
        means2d_proxy,
        viewmats, Ks,
        width, height, sh_degree, tile_size,
        near_plane, far_plane, radius_clip, eps2d, background,
        render_mode, camera_model,
        enable_culling, use_higs_culling,
        backward_mode, scene, sh_compression_mode, tile_sampling_ratio,
        sampling_mode, tile_mask, cull_refresh_interval, "default",
    )
    fwd_meta = dict(_HigsAutogradFunction.last_forward_metadata)
    N_total = fwd_meta["n_gaussians"]

    higs_preview = None
    if return_higs_preview:
        try:
            from gsplat.scene import GaussianInferenceScene
            with torch.no_grad():
                scene_obj = GaussianInferenceScene.from_gaussian_tensors(
                    means, quats, scales, opacities, colors,
                    sh_degree=sh_degree,
                    sh_compression="none",
                    id="frozen_preview",
                )
                req = dict(
                    viewmat=viewmats[0, 0] if viewmats.dim() > 2 else viewmats,
                    K=Ks[0, 0] if Ks.dim() > 2 else Ks,
                    width=width, height=height,
                    tile_size=tile_size, near_plane=near_plane,
                    far_plane=far_plane, radius_clip=radius_clip,
                    eps2d=eps2d, render_mode=render_mode,
                    camera_model=camera_model,
                )
                if background is not None:
                    req["background"] = background
                higs_result = rasterize_gaussian_inference_scene(scene_obj, **req)
                higs_preview = higs_result.frame
        except (ImportError, RuntimeError, ValueError, TypeError):
            # Preview is best-effort; the differentiable path is unaffected.
            higs_preview = None

    return {
        "frame": frame,
        "alpha": alpha,
        "densification_info": {
            "means2d": means2d_proxy,
            "radii": fwd_meta["densification_radii"],
            "gaussian_ids": torch.arange(N_total, device=means.device),
            "visible_gaussian_ids": fwd_meta["visible_gaussian_ids"],
            "width": width,
            "height": height,
            "n_cameras": means2d_proxy.shape[0],
        },
        "metadata": {
            "n_gaussians": N_total,
            "n_visible": fwd_meta["n_visible"],
            "f9_enabled": fwd_meta.get("f9_enabled", False),
            "culling_ratio": fwd_meta["culling_ratio"],
            "backward_backend": fwd_meta["backward_backend"],
            "native_available": fwd_meta["native_available"],
            "scene_version": fwd_meta["scene_version"],
            "topology_rebuilt": fwd_meta["topology_rebuilt"],
            "packed_dtype": fwd_meta["packed_dtype"],
            "sh_compression_mode": fwd_meta.get("sh_compression_mode"),
            "render_mode": render_mode,
            "use_higs_culling": use_higs_culling,
            "higs_preview": higs_preview,
            "freeze_topology": freeze_topology,
            "render_count": _HIGS_FROZEN_TRACKER.render_count,
            "sampled_tile_ratio": fwd_meta.get("sampled_tile_ratio", 1.0),
            "tile_mask": fwd_meta.get("tile_mask"),
            "n_isects": fwd_meta.get("n_isects", 0),
            "n_isects_full": fwd_meta.get("n_isects_full", 0),
            "cull_refresh_interval": cull_refresh_interval,
        },
    }

def rasterize_gaussian_higs_frozen(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    *,
    sh_degree = None,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    tile_size: int = 16,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    background = None,
    render_mode: str = "RGB",
    camera_model: str = "pinhole",
    enable_culling: bool = True,
    use_higs_culling: bool = False,
    return_higs_preview: bool = False,
    freeze_topology: bool = True,
    backward_mode: str = "higs_native",
    sh_compression_mode="none",
    tile_sampling_ratio: float = 1.0,
    sampling_mode: str = "uniform",
    tile_mask=None,
    cull_refresh_interval: int = 1,
    scene=None,
) -> dict:
    """Differentiable HiGS rendering with per-frame visibility culling (Stage B).

    Stage B improves on Stage A by culling invisible Gaussians before
    the differentiable rendering pass. Only visible Gaussians participate
    in the autograd computation.

    When ``freeze_topology=True`` (default), the Gaussian count is validated
    across renderer calls. If the count changes, RuntimeError is raised.

    Args:
        enable_culling: If True, cull invisible Gaussians via projection.
        freeze_topology: If True, validate Gaussian count consistency.
        use_higs_culling: If True, use HiGS-native culling.
        return_higs_preview: If True, include HiGS inference output.
        backward_mode: "higs_native" (default) or "gsplat_recompute".
        sh_compression_mode: "none" (default); PACKED_16B/32B modes are
            supported via a straight-through FP16 quantization in training.
        scene: optional :class:`HigsRendererHandle` for renderer reuse.

    Returns:
        dict with keys: frame, alpha, metadata
    """
    return _higs_frozen_forward(
        means, quats, scales, opacities, colors,
        viewmats=viewmats, Ks=Ks, width=width, height=height,
        sh_degree=sh_degree, tile_size=tile_size,
        near_plane=near_plane, far_plane=far_plane,
        radius_clip=radius_clip, eps2d=eps2d,
        background=background, render_mode=render_mode,
        camera_model=camera_model, enable_culling=enable_culling,
        use_higs_culling=use_higs_culling,
        return_higs_preview=return_higs_preview,
        freeze_topology=freeze_topology,
        backward_mode=backward_mode,
        sh_compression_mode=sh_compression_mode,
        tile_sampling_ratio=tile_sampling_ratio,
        sampling_mode=sampling_mode,
        tile_mask=tile_mask,
        cull_refresh_interval=cull_refresh_interval,
        scene=scene,
    )

def _higs_dynamic_forward(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    *,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    sh_degree=None,
    tile_size: int = 16,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    background=None,
    render_mode: str = "RGB",
    camera_model: str = "pinhole",
    enable_culling: bool = True,
    use_higs_culling: bool = False,
    return_higs_preview: bool = False,
    backward_mode: str = "higs_native",
    sh_compression_mode="none",
    tile_sampling_ratio: float = 1.0,
    sampling_mode: str = "uniform",
    tile_mask=None,
    cull_refresh_interval: int = 1,
    cull_cache_key: str = "default",
    scene=None,
) -> dict:
    """Forward pass for dynamic-topology HiGS diff. path (Stage C).

    Unlike _higs_frozen_forward, this function does NOT enforce a fixed
    Gaussian count. It uses _HigsDynamicScene to track versions and
    validate that topology mutations (densify/prune) happen at the right time.
    When the CUDA extension is available it also owns a
    :class:`HigsRendererHandle` so every forward/backward is bound to the same
    scene version and topology mutation raises while a backward is pending.
    """

    tracker = scene if isinstance(scene, _HigsDynamicScene) else _HIGS_DYNAMIC_SCENE
    tracker.validate_count(means.shape[0])

    handle = scene if isinstance(scene, HigsRendererHandle) else None
    colors_eff = _slice_colors_for_sh_degree(colors, sh_degree)
    if handle is None and _higs_backend_available():
        handle = tracker.ensure_renderer(
            means, quats, scales, opacities, colors_eff, sh_degree,
        )

    means2d_proxy = _new_means2d_proxy(
        means, viewmats, quats, scales, opacities, colors
    )
    frame, alpha = _HigsAutogradFunction.apply(
        means, quats, scales, opacities, colors_eff,
        means2d_proxy,
        viewmats, Ks,
        width, height, sh_degree, tile_size,
        near_plane, far_plane, radius_clip, eps2d, background,
        render_mode, camera_model,
        enable_culling, use_higs_culling,
        backward_mode, handle, sh_compression_mode, tile_sampling_ratio,
        sampling_mode, tile_mask, cull_refresh_interval,
        cull_cache_key,
    )

    scene_version = tracker.next_version(means.shape[0])
    fwd_meta = dict(_HigsAutogradFunction.last_forward_metadata)
    N_total = means.shape[0]

    higs_preview = None
    if return_higs_preview:
        try:
            from gsplat.scene import GaussianInferenceScene
            with torch.no_grad():
                scene_obj = GaussianInferenceScene.from_gaussian_tensors(
                    means, quats, scales, opacities, colors,
                    sh_degree=sh_degree, sh_compression="none", id="dynamic_preview",
                )
                req = dict(
                    viewmat=viewmats[0, 0] if viewmats.dim() > 2 else viewmats,
                    K=Ks[0, 0] if Ks.dim() > 2 else Ks,
                    width=width, height=height, tile_size=tile_size,
                    near_plane=near_plane, far_plane=far_plane,
                    radius_clip=radius_clip, eps2d=eps2d,
                    render_mode=render_mode, camera_model=camera_model,
                )
                if background is not None:
                    req["background"] = background
                higs_result = rasterize_gaussian_inference_scene(scene_obj, **req)
                higs_preview = higs_result.frame
        except (ImportError, RuntimeError, ValueError, TypeError):
            higs_preview = None

    return {
        "frame": frame,
        "alpha": alpha,
        "densification_info": {
            "means2d": means2d_proxy,
            "radii": fwd_meta["densification_radii"],
            "gaussian_ids": torch.arange(N_total, device=means.device),
            "visible_gaussian_ids": fwd_meta["visible_gaussian_ids"],
            "width": width,
            "height": height,
            "n_cameras": means2d_proxy.shape[0],
        },
        "metadata": {
            "n_gaussians": N_total,
            "scene_version": scene_version,
            "backward_backend": fwd_meta["backward_backend"],
            "native_available": fwd_meta["native_available"],
            "n_visible": fwd_meta["n_visible"],
            "culling_ratio": fwd_meta["culling_ratio"],
            "topology_rebuilt": fwd_meta["topology_rebuilt"],
            "packed_dtype": fwd_meta["packed_dtype"],
            "sh_compression_mode": fwd_meta.get("sh_compression_mode"),
            "render_mode": render_mode,
            "use_higs_culling": use_higs_culling,
            "higs_preview": higs_preview,
            "sampled_tile_ratio": fwd_meta.get("sampled_tile_ratio", 1.0),
            "tile_mask": fwd_meta.get("tile_mask"),
            "n_isects": fwd_meta.get("n_isects", 0),
            "n_isects_full": fwd_meta.get("n_isects_full", 0),
            "cull_refresh_interval": cull_refresh_interval,
        },
    }


def rasterize_gaussian_higs_dynamic(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    *,
    sh_degree=None,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    tile_size: int = 16,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    background=None,
    render_mode: str = "RGB",
    camera_model: str = "pinhole",
    enable_culling: bool = True,
    use_higs_culling: bool = False,
    return_higs_preview: bool = False,
    backward_mode: str = "higs_native",
    sh_compression_mode="none",
    tile_sampling_ratio: float = 1.0,
    sampling_mode: str = "uniform",
    tile_mask=None,
    cull_refresh_interval: int = 1,
    cull_cache_key: str = "default",
    scene=None,
) -> dict:
    """Differentiable HiGS rendering with dynamic-topology support (Stage C).

    This is the Stage C entry point. Removes the freeze_topology restriction
    present in rasterize_gaussian_higs_frozen. Gaussian count may change
    between forward calls via _densify_gaussians / _prune_gaussians.

    Before the next forward after a topology mutation, call
    _HIGS_DYNAMIC_SCENE.mark_dirty() (or ``scene.mark_dirty()`` on an explicit
    :class:`HigsRendererHandle`) to indicate the change.

    Args:
        backward_mode: "higs_native" (default) or "gsplat_recompute".
        sh_compression_mode: "none" (default); PACKED_16B/32B modes are
            supported via a straight-through FP16 quantization in training.
        scene: optional :class:`HigsRendererHandle`; when omitted, the
            module-level :class:`_HigsDynamicScene` singleton owns one.
    """
    return _higs_dynamic_forward(
        means, quats, scales, opacities, colors,
        viewmats=viewmats, Ks=Ks, width=width, height=height,
        sh_degree=sh_degree, tile_size=tile_size,
        near_plane=near_plane, far_plane=far_plane,
        radius_clip=radius_clip, eps2d=eps2d,
        background=background, render_mode=render_mode,
        camera_model=camera_model, enable_culling=enable_culling,
        use_higs_culling=use_higs_culling,
        return_higs_preview=return_higs_preview,
        backward_mode=backward_mode,
        sh_compression_mode=sh_compression_mode,
        tile_sampling_ratio=tile_sampling_ratio,
        sampling_mode=sampling_mode,
        tile_mask=tile_mask,
        cull_refresh_interval=cull_refresh_interval,
        cull_cache_key=cull_cache_key,
        scene=scene,
    )
