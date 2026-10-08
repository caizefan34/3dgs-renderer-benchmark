"""VRIR1 incremental-raster-update oracle: state drift + membership delta.

Pure CPU/numpy. Consumes per-checkpoint raster structures captured from the
VRIR0-compatible baseline rasterizer and measures, between two consecutive
training checkpoints:

  G3 state drift:  xyz / radius / depth / projected-mean (means2d) deltas.
  G4 structure delta: (gaussian, tile) intersection membership difference
                      (unchanged / added / removed), computed EXACTLY from the
                      per-gaussian tile AABBs.

No approximate equality anywhere (fail-closed). The membership model is the same
bit-matched gsplat 1.5.3 tile-AABB model validated against fresh
flatten_ids/isect_offsets in VRIR0 (validate_membership_model, rate 1.0).
"""
from __future__ import annotations
import numpy as np


def _pct(arr, ps=(50, 95, 99)):
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        d = {"p%s" % int(p): 0.0 for p in ps}
        d.update({"max": 0.0, "mean": 0.0, "n": 0})
        return d
    d = {"p%s" % int(p): float(np.percentile(a, p)) for p in ps}
    d.update({"max": float(a.max()), "mean": float(a.mean()), "n": int(a.size)})
    return d


def state_drift(xyz_p, xyz_c, m2d_p, m2d_c, r_p, r_c, dep_p, dep_c, act_p, act_c):
    """Per-gaussian state deltas between two checkpoints (frozen topology: same N).

    xyz is camera-independent (over all N). radius/depth/means2d are per-camera
    projected quantities, measured over the union of active gaussians.
    """
    out = {}
    dxyz = np.linalg.norm(xyz_c - xyz_p, axis=1)
    out["xyz"] = _pct(dxyz)
    act = np.asarray(act_p, dtype=bool) | np.asarray(act_c, dtype=bool)
    out["n_active_union"] = int(np.count_nonzero(act))
    if np.any(act):
        dm2d = np.linalg.norm(m2d_c[act] - m2d_p[act], axis=1)
        ddep = np.abs(dep_c[act] - dep_p[act])
        dr = np.abs(np.asarray(r_c, dtype=np.float64)[act]
                    - np.asarray(r_p, dtype=np.float64)[act])  # [n,2]
        dr_mag = np.linalg.norm(dr, axis=1)
        out["means2d"] = _pct(dm2d)
        out["depth"] = _pct(ddep)
        out["radius_l2"] = _pct(dr_mag)
        out["radius_x_p95"] = float(np.percentile(dr[:, 0], 95)) if dr.shape[0] else 0.0
        out["radius_y_p95"] = float(np.percentile(dr[:, 1], 95)) if dr.shape[0] else 0.0
    else:
        for k in ("means2d", "depth", "radius_l2"):
            out[k] = _pct(np.array([]))
        out["radius_x_p95"] = 0.0
        out["radius_y_p95"] = 0.0
    return out


def membership_delta(bmn_p, bpl_p, bmn_c, bpl_c):
    """Exact (gaussian, tile) membership delta from per-gaussian tile AABBs.

    Each active gaussian contributes every tile in its half-open AABB
    [b_minus, b_plus) (the gsplat isect_tiles assignment, validated in VRIR0).
    Returns (unchanged, added, removed) intersection counts. Inactive gaussians
    have empty AABBs (b_minus == b_plus) and contribute nothing.
    """
    bmn_p = np.asarray(bmn_p, dtype=np.int64); bpl_p = np.asarray(bpl_p, dtype=np.int64)
    bmn_c = np.asarray(bmn_c, dtype=np.int64); bpl_c = np.asarray(bpl_c, dtype=np.int64)
    ox0 = np.maximum(bmn_p[:, 0], bmn_c[:, 0]); ox1 = np.minimum(bpl_p[:, 0], bpl_c[:, 0])
    oy0 = np.maximum(bmn_p[:, 1], bmn_c[:, 1]); oy1 = np.minimum(bpl_p[:, 1], bpl_c[:, 1])
    ox = np.clip(ox1 - ox0, 0, None); oy = np.clip(oy1 - oy0, 0, None)
    inter = ox * oy
    area_p = (bpl_p[:, 0] - bmn_p[:, 0]) * (bpl_p[:, 1] - bmn_p[:, 1])
    area_c = (bpl_c[:, 0] - bmn_c[:, 0]) * (bpl_c[:, 1] - bmn_c[:, 1])
    unchanged = int(inter.sum())
    added = int((area_c - inter).sum())
    removed = int((area_p - inter).sum())
    return unchanged, added, removed


def unaffected_fraction(unchanged, added, removed):
    denom = unchanged + added + removed
    return (unchanged / denom) if denom > 0 else 1.0