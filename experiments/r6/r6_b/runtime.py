"""Small runtime surface for the R6-B patched gsplat extension."""
from __future__ import annotations

from functools import wraps


def extension():
    from gsplat.cuda._backend import _C
    required = ("r6b_set_mode", "r6b_mode", "r6b_invalidate",
                "r6b_last_prepare_ms", "r6b_last_scatter_ms",
                "r6b_last_clear_ms", "r6b_metadata_bytes", "r6b_prev_n_rows")
    missing = [name for name in required if not hasattr(_C, name)]
    if missing:
        raise RuntimeError(
            "R6-B gsplat extension is not loaded (missing: %s); run "
            "prepare_r6b_source.py and put its parent directory before the "
            "baseline gsplat on PYTHONPATH" % ", ".join(missing)
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


def last_scatter_ms() -> float:
    """B1-v2: CUDA-event time for the touched-mask scatter phase (in finish)."""
    return float(extension().r6b_last_scatter_ms())


def last_clear_ms() -> float:
    """B1-v2: CUDA-event time for the selective-clear / full-clear phase.

    For B0 this equals last_prepare_ms (the only work in prepare is the
    full active-range zero-fill).  For B1-v2 this is the mask-scan + gradient-
    zero kernel, separate from the scatter phase that runs in finish().
    """
    return float(extension().r6b_last_clear_ms())


def metadata_bytes() -> int:
    """Extra persistent metadata: the [C*N] bool touched mask in bytes."""
    return int(extension().r6b_metadata_bytes())


def prev_n_rows() -> int:
    """Number of rows (C*N) in the retained touched mask, or 0 if none."""
    return int(extension().r6b_prev_n_rows())


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
