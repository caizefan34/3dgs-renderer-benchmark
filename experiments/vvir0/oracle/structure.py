"""Deterministic tracked-view selection for VRIR0.

Selects n views across camera indices at quantiles, including endpoints only if
they are valid (0 <= idx < n_cameras). Deterministic and reproducible.
"""
from __future__ import annotations
from typing import List
import numpy as np

def select_tracked_views(n_cameras: int, n_tracked: int) -> List[int]:
    if n_cameras <= 0:
        return []
    n_tracked = min(n_tracked, n_cameras)
    if n_tracked == 1:
        return [0]
    # quantile positions in [0,1], endpoints included
    qs = np.linspace(0.0, 1.0, n_tracked)
    idxs = [int(round(q * (n_cameras - 1))) for q in qs]
    # dedupe preserving order, clamp to valid range
    seen = set(); out = []
    for i in idxs:
        i = max(0, min(n_cameras - 1, i))
        if i not in seen:
            seen.add(i); out.append(i)
    # if dedup collapsed too many (tiny n_cameras), fill with stride
    if len(out) < n_tracked:
        stride = max(1, n_cameras // n_tracked)
        for i in range(0, n_cameras, stride):
            if i not in seen:
                seen.add(i); out.append(i)
            if len(out) >= n_tracked:
                break
    return sorted(out)
