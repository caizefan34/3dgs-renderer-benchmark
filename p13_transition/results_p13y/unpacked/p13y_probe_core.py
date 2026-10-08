from __future__ import annotations
import hashlib, json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
import torch


def _cpu_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _single_camera(x: np.ndarray) -> np.ndarray:
    # P12 trains one camera per iteration. Strip singleton batch/camera dims.
    while x.ndim > 1 and x.shape[0] == 1:
        x = x[0]
    return x


def _sha256_bytes(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
        h.update(a.tobytes())
    return h.hexdigest()


def snapshot_from_meta(meta: Dict[str, Any], iteration: int, scene: str,
                       anchor_id: int, out_dir: str | Path) -> Dict[str, Any]:
    """Persist exact X1/X2 state from gsplat rasterization() meta.

    Evidence source is renderer-produced state only. No CPU footprint reconstruction.
    Assumes one diagnostic camera per call and P12 packed=False.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    radii = _single_camera(_cpu_np(meta["radii"]))
    depths = _single_camera(_cpu_np(meta["depths"]))
    offsets = _single_camera(_cpu_np(meta["isect_offsets"])).astype(np.int64, copy=False)
    flatten_ids = _cpu_np(meta["flatten_ids"]).reshape(-1).astype(np.int64, copy=False)
    isect_ids = _cpu_np(meta["isect_ids"]).reshape(-1)

    if radii.ndim != 1:
        raise ValueError(f"Expected single-camera radii [N], got {radii.shape}")
    N = int(radii.shape[0])
    tile_h, tile_w = int(offsets.shape[-2]), int(offsets.shape[-1])
    starts = offsets.reshape(-1)
    M = int(flatten_ids.size)
    ends = np.empty_like(starts)
    if starts.size:
        ends[:-1] = starts[1:]
        ends[-1] = M
    if starts.size and (starts.min() < 0 or starts.max() > M or np.any(ends < starts)):
        raise ValueError("Invalid isect_offsets/flatten_ids relationship")

    # For packed=False, one-camera flatten_ids are Gaussian IDs. Guard against drift.
    if M and (flatten_ids.min() < 0 or flatten_ids.max() >= N):
        raise ValueError(
            f"flatten_ids not direct Gaussian IDs for single-camera packed=False: "
            f"range=[{flatten_ids.min()},{flatten_ids.max()}], N={N}"
        )

    visible_ids = np.flatnonzero(radii > 0).astype(np.int64)
    tile_ids = np.repeat(np.arange(starts.size, dtype=np.int64), ends - starts)
    if tile_ids.size != M:
        raise ValueError(f"Expanded tile IDs {tile_ids.size} != intersections {M}")

    stem = f"{scene}_a{anchor_id:02d}_i{iteration:05d}"
    npz_path = out_dir / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        visible_ids=visible_ids,
        radii=radii,
        depths=depths,
        tile_starts=starts,
        tile_ends=ends,
        tile_ids=tile_ids,
        gaussian_ids=flatten_ids,
        isect_ids=isect_ids,
        N=np.asarray([N], dtype=np.int64),
        tile_h=np.asarray([tile_h], dtype=np.int64),
        tile_w=np.asarray([tile_w], dtype=np.int64),
    )

    summary = {
        "scene": scene,
        "anchor_id": int(anchor_id),
        "iteration": int(iteration),
        "N": N,
        "n_visible": int(visible_ids.size),
        "n_intersections": M,
        "tile_h": tile_h,
        "tile_w": tile_w,
        "visible_sha256": _sha256_bytes(visible_ids),
        "membership_order_sha256": _sha256_bytes(tile_ids, flatten_ids),
        "raw_npz": str(npz_path),
    }
    with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _jaccard_sorted(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 and b.size == 0:
        return 1.0
    inter = np.intersect1d(a, b, assume_unique=False).size
    union = np.union1d(a, b).size
    return float(inter / union) if union else 1.0


def _inv_count(seq: np.ndarray) -> int:
    # O(n log n) merge-sort inversion count.
    if seq.size < 2:
        return 0
    arr = seq.astype(np.int64, copy=True)
    tmp = np.empty_like(arr)
    def rec(lo, hi):
        if hi - lo <= 1:
            return 0
        mid = (lo + hi) // 2
        inv = rec(lo, mid) + rec(mid, hi)
        i, j, k = lo, mid, lo
        while i < mid and j < hi:
            if arr[i] <= arr[j]:
                tmp[k] = arr[i]; i += 1
            else:
                tmp[k] = arr[j]; j += 1
                inv += mid - i
            k += 1
        while i < mid: tmp[k] = arr[i]; i += 1; k += 1
        while j < hi: tmp[k] = arr[j]; j += 1; k += 1
        arr[lo:hi] = tmp[lo:hi]
        return inv
    return int(rec(0, len(arr)))


def _tile_sequences(z) -> List[np.ndarray]:
    starts, ends, gids = z["tile_starts"], z["tile_ends"], z["gaussian_ids"]
    return [gids[int(s):int(e)] for s, e in zip(starts, ends)]


def compare_snapshots(path_a: str | Path, path_b: str | Path) -> Dict[str, Any]:
    za, zb = np.load(path_a), np.load(path_b)
    va, vb = za["visible_ids"], zb["visible_ids"]
    Na, Nb = int(za["N"][0]), int(zb["N"][0])
    if Na != Nb:
        raise ValueError(f"Topology differs: N {Na} vs {Nb}")
    sa, sb = _tile_sequences(za), _tile_sequences(zb)
    if len(sa) != len(sb):
        raise ValueError("Tile grid differs")

    # Unique (tile, Gaussian) key. N fixed after 15K by X0.
    ka = za["tile_ids"].astype(np.int64) * Na + za["gaussian_ids"].astype(np.int64)
    kb = zb["tile_ids"].astype(np.int64) * Nb + zb["gaussian_ids"].astype(np.int64)
    memb_j = _jaccard_sorted(ka, kb)

    unchanged_membership = 0
    unchanged_order = 0
    inv_rates = []
    weighted_inv_num = 0
    weighted_inv_den = 0
    for a, b in zip(sa, sb):
        if np.array_equal(np.sort(a), np.sort(b)):
            unchanged_membership += 1
        if np.array_equal(a, b):
            unchanged_order += 1
        if a.size < 2 or b.size < 2:
            inv_rates.append(0.0)
            continue
        common = np.intersect1d(a, b, assume_unique=False)
        if common.size < 2:
            inv_rates.append(0.0)
            continue
        common_set = set(map(int, common.tolist()))
        a_common = [int(g) for g in a.tolist() if int(g) in common_set]
        pos_b = {int(g): i for i, g in enumerate(b.tolist()) if int(g) in common_set}
        perm = np.asarray([pos_b[g] for g in a_common], dtype=np.int64)
        inv = _inv_count(perm)
        den = len(perm) * (len(perm) - 1) // 2
        rate = float(inv / den) if den else 0.0
        inv_rates.append(rate)
        weighted_inv_num += inv
        weighted_inv_den += den

    ntiles = max(1, len(sa))
    return {
        "visible_jaccard": _jaccard_sorted(va, vb),
        "visible_retained_fraction": float(np.intersect1d(va, vb).size / max(1, va.size)),
        "visible_entered_fraction": float(np.setdiff1d(vb, va).size / max(1, vb.size)),
        "membership_jaccard": memb_j,
        "unchanged_membership_tile_fraction": float(unchanged_membership / ntiles),
        "unchanged_full_order_tile_fraction": float(unchanged_order / ntiles),
        "order_inversion_rate_median_tile": float(np.median(inv_rates)) if inv_rates else 0.0,
        "order_inversion_rate_weighted": float(weighted_inv_num / weighted_inv_den) if weighted_inv_den else 0.0,
        "n_tiles": len(sa),
        "N": Na,
        "n_intersections_a": int(za["gaussian_ids"].size),
        "n_intersections_b": int(zb["gaussian_ids"].size),
    }


# --------------------------------------------------------------------------
# P13-Y addition (byte copy of the verified P13-X p13x_probe_core.py above this
# line). Y1 must reuse the VERIFIED P13-X renderer-meta instrumentation, so the
# classic tile-intersection call is reproduced verbatim from
# publication_trainer_p13x_probe.py:
#     p13x_call_kwargs = {"tile_size": 16, "packed": False, "radius_clip": 0.0,
#                         "eps2d": eps2d_eff, "render_mode": "RGB",
#                         "rasterize_mode": "classic", "camera_model": "pinhole"}
#     _r, _alpha, meta = gsplat.rendering.rasterization(..., **p13x_call_kwargs)
#     probe_meta = {"radii": meta["radii"].reshape(-1, 2).min(-1).values,
#                   "depths": meta["depths"], "isect_offsets": meta["isect_offsets"],
#                   "flatten_ids": meta["flatten_ids"], "isect_ids": meta["isect_ids"]}
# The C0 training renderer (HIGS dynamic) does not return isect/tile meta; this
# standard rasterization call on the identical frozen parameter state is the
# preregistered identity block (packed=false, tile=16, eps2d=0.3, classic).
P13Y_CLASSIC_CALL_KWARGS = {
    "tile_size": 16, "packed": False, "radius_clip": 0.0, "eps2d": 0.3,
    "render_mode": "RGB", "rasterize_mode": "classic", "camera_model": "pinhole",
}


def gsplat_file():
    import gsplat
    return gsplat.__file__


def classic_rasterize(means, quats, scales, opacities, colors, viewmats, Ks,
                      width, height, sh_degree, eps2d=0.3):
    """P13-X-identical classic meta render; returns (image, alpha, meta)."""
    from gsplat.rendering import rasterization
    kwargs = dict(P13Y_CLASSIC_CALL_KWARGS)
    kwargs["eps2d"] = eps2d
    return rasterization(means=means, quats=quats, scales=scales,
                         opacities=opacities, colors=colors,
                         viewmats=viewmats, Ks=Ks, width=width, height=height,
                         sh_degree=sh_degree, **kwargs)
