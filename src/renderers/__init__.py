"""Renderer adapter registry."""
from __future__ import annotations

from typing import Type

from .base import RendererAdapter

_RENDERER_REGISTRY: dict[str, Type[RendererAdapter]] = {}


def register_renderer(name: str, cls: Type[RendererAdapter]) -> None:
    """Register a renderer adapter class."""
    _RENDERER_REGISTRY[name] = cls


def get_renderer(name: str, device: str = "cuda") -> RendererAdapter | None:
    """Retrieve a renderer instance by name."""
    if name not in _RENDERER_REGISTRY:
        print(f"  Renderer '{name}' not registered")
        return None
    renderer = _RENDERER_REGISTRY[name](device=device)
    if not renderer.is_available():
        print(f"  Renderer '{name}' not available on this system")
        return None
    return renderer


def get_renderer_class(name: str) -> Type[RendererAdapter] | None:
    """Return the registered renderer class without availability filtering."""
    return _RENDERER_REGISTRY.get(name)


def list_renderers() -> list[str]:
    """Return all registered renderer names."""
    return list(_RENDERER_REGISTRY.keys())


def list_available(device: str = "cuda") -> list[str]:
    """Return names of renderers that are currently available."""
    available: list[str] = []
    for name, cls in _RENDERER_REGISTRY.items():
        r = cls(device=device)
        if r.is_available():
            available.append(name)
    return available

from .gsplat_renderer import (
    GsplatRenderer,
    GsplatDenseRenderer,
    GsplatHiGSRenderer,
    GsplatHiGSTile16Renderer,
    GsplatHiGSSH32Renderer,
    GsplatHiGSSH16Renderer,
    GsplatHiGSTile16SH32Renderer,
    GsplatHiGSTile16SH16Renderer,
    GsplatHiGSAutoRenderer,
)
from .diff_gaussian_renderer import DiffGaussianRenderer
from .fast_gauss_renderer import FastGaussRenderer
from .speedy_splat_renderer import SpeedySplatRenderer, SpeedySplatRawRenderer
from .tcgs_renderer import TCGSRenderer
from .flashgs_renderer import FlashGSRenderer
from .local_gs_renderer import LocalGSRenderer
from .gemm_gs_renderer import GemmGSRenderer
from .experimental_renderer import (
    StopThePopRenderer,
)

register_renderer("gsplat", GsplatRenderer)
register_renderer("gsplat_dense", GsplatDenseRenderer)
register_renderer("gsplat_higs", GsplatHiGSRenderer)
register_renderer("gsplat_higs_tile16", GsplatHiGSTile16Renderer)
register_renderer("gsplat_higs_sh32", GsplatHiGSSH32Renderer)
register_renderer("gsplat_higs_sh16", GsplatHiGSSH16Renderer)
register_renderer("gsplat_higs_tile16_sh32", GsplatHiGSTile16SH32Renderer)
register_renderer("gsplat_higs_tile16_sh16", GsplatHiGSTile16SH16Renderer)
register_renderer("gsplat_higs_auto", GsplatHiGSAutoRenderer)
register_renderer("diff_gaussian", DiffGaussianRenderer)
register_renderer("original_3dgs", DiffGaussianRenderer)
register_renderer("fast_gauss", FastGaussRenderer)
register_renderer("speedy_splat", SpeedySplatRenderer)
register_renderer("speedy_splat_raw", SpeedySplatRawRenderer)
register_renderer("tcgs", TCGSRenderer)
register_renderer("flashgs", FlashGSRenderer)
register_renderer("local_gs", LocalGSRenderer)
register_renderer("gemm_gs", GemmGSRenderer)
register_renderer("stopthepop", StopThePopRenderer)
