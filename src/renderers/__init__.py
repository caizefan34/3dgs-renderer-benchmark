"""
Renderer adapter registry.

Provides a registration mechanism and lookup functions for renderer adapters.
Each adapter is registered by name and can be queried for availability.
"""
from .base import RendererAdapter

_RENDERER_REGISTRY = {}

def register_renderer(name, cls):
    """Register a renderer adapter class.

    Args:
        name: String identifier for the renderer (e.g., 'diff_gaussian').
        cls: RendererAdapter subclass to register.
    """
    _RENDERER_REGISTRY[name] = cls

def get_renderer(name, device="cuda"):
    """Retrieve a renderer instance by name.

    Args:
        name: Renderer identifier string.
        device: Target device for tensor allocation.

    Returns:
        RendererAdapter instance, or None if not registered or unavailable.
    """
    if name not in _RENDERER_REGISTRY:
        print(f"  Renderer '{name}' not registered")
        return None
    renderer = _RENDERER_REGISTRY[name](device=device)
    if not renderer.is_available():
        print(f"  Renderer '{name}' not available on this system")
        return None
    return renderer

def list_renderers():
    """Return all registered renderer names."""
    return list(_RENDERER_REGISTRY.keys())

def list_available(device="cuda"):
    """Return names of renderers that are currently available.

    Args:
        device: Target device for availability check.

    Returns:
        List of available renderer name strings.
    """
    available = []
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
