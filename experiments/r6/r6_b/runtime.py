"""Small runtime surface for the R6-B patched gsplat extension."""
from __future__ import annotations

from functools import wraps


def extension():
    from gsplat.cuda._backend import _C
    required = ("r6b_set_mode", "r6b_mode", "r6b_invalidate", "r6b_last_prepare_ms")
    missing = [name for name in required if not hasattr(_C, name)]
    if missing:
        raise RuntimeError(
            "R6-B gsplat extension is not loaded; run prepare_r6b_source.py and "
            "put its parent directory before the baseline gsplat on PYTHONPATH"
        )
    return _C


def configure(mode: str) -> None:
    values = {"baseline": 0, "b0": 1, "b1": 2}
    try:
        extension().r6b_set_mode(values[mode])
    except KeyError as exc:
        raise ValueError(f"unknown R6-B mode: {mode}") from exc


def invalidate() -> None:
    """Require a full clear on the next backward after a topology edit."""
    extension().r6b_invalidate()


def last_prepare_ms() -> float:
    """CUDA-event time for the rasterizer buffer preparation in the last backward."""
    return float(extension().r6b_last_prepare_ms())


def install_topology_guard(model) -> None:
    """Invalidate after every operation that can change row identity.

    The wrapper is deliberately instance-local and leaves model math, optimizer
    migration, pruning, and scheduling untouched.  Repeated invalidation is
    safe: it only makes the next raster backward take B0's full-clear path.
    """
    if getattr(model, "_r6b_topology_guard", False):
        return
    for name in ("densification_postfix", "prune_points"):
        original = getattr(model, name)

        @wraps(original)
        def guarded(*args, __original=original, **kwargs):
            result = __original(*args, **kwargs)
            invalidate()
            return result

        setattr(model, name, guarded)
    model._r6b_topology_guard = True
