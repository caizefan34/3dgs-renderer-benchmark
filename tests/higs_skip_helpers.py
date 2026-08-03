"""Shared skip helpers for HiGS tests that need the patched gsplat backend.

The HiGS rasterization APIs live in a patched build of gsplat (see
``patches/higs-differentiable.patch``); the stock pip release does not ship
them.  These helpers let the suite skip cleanly whenever the patched gsplat
(and, for CUDA tests, a CUDA device) is unavailable, instead of failing with
``ImportError``/``ModuleNotFoundError``.
"""
from __future__ import annotations

import importlib.util

import pytest
import torch

_HIGS_APIS = (
    "rasterize_gaussian_higs_trainable",
    "rasterize_gaussian_higs_frozen",
    "rasterize_gaussian_higs_dynamic",
)


def higs_module_available() -> bool:
    """True when the patched gsplat exposing the HiGS APIs is importable."""
    # ``gsplat.rendering`` exists in the patched/newer gsplat but not in the
    # stock pip release, so it is a cheap first-line check.
    if importlib.util.find_spec("gsplat.rendering") is None:
        return False
    try:
        import gsplat.experimental as experimental

        for name in _HIGS_APIS:
            getattr(experimental, name)
        return True
    except (ImportError, AttributeError):
        return False


def higs_backend_available() -> bool:
    """True when the HiGS CUDA backend can actually run on this machine."""
    return torch.cuda.is_available() and higs_module_available()


# Skip when the HiGS CUDA backend cannot run (no CUDA or patched gsplat absent).
skipif_higs_unavailable = pytest.mark.skipif(
    not higs_backend_available(), reason="HiGS CUDA backend unavailable"
)

# Skip when the patched gsplat module is missing, but keep running on machines
# that have it even without CUDA (used by static/API-surface tests).
skipif_higs_module_unavailable = pytest.mark.skipif(
    not higs_module_available(), reason="Patched gsplat (HiGS APIs) not installed"
)