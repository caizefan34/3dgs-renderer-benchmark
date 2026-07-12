"""
Renderer adapter registry.
"""
from .base import RendererAdapter

_RENDERER_REGISTRY = {}

def register_renderer(name, cls):
    _RENDERER_REGISTRY[name] = cls

def get_renderer(name, device="cuda"):
    if name not in _RENDERER_REGISTRY:
        print(f"  Renderer '{name}' not registered")
        return None
    renderer = _RENDERER_REGISTRY[name](device=device)
    if not renderer.is_available():
        print(f"  Renderer '{name}' not available")
        return None
    return renderer

def list_renderers():
    return list(_RENDERER_REGISTRY.keys())

def list_available(device="cuda"):
    available = []
    for name, cls in _RENDERER_REGISTRY.items():
        r = cls(device=device)
        if r.is_available():
            available.append(name)
    return available

from .gsplat_renderer import GsplatRenderer
from .diff_gaussian_renderer import DiffGaussianRenderer
from .fast_gauss_renderer import FastGaussRenderer
from .speedy_splat_renderer import SpeedySplatRenderer

register_renderer("gsplat", GsplatRenderer)
register_renderer("diff_gaussian", DiffGaussianRenderer)
register_renderer("fast_gauss", FastGaussRenderer)
register_renderer("speedy_splat", SpeedySplatRenderer)
