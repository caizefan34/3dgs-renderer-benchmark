"""Pure-Python/NumPy reference helpers for the VRIR0 oracle.

This file is deliberately NOT an optimized implementation.  It is intended to
make the certificate auditable. DSH must source-audit the installed gsplat fork
before trusting the assumptions encoded here.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

@dataclass
class BBoxSignature:
    active: np.ndarray   # bool [N]
    tmin: np.ndarray     # int64 [N,2]
    tmax: np.ndarray     # int64 [N,2]


def tile_bbox_signature(means2d, radii, width:int, height:int, tile_size:int=16):
    m = np.asarray(means2d, dtype=np.float64).reshape(-1,2)
    r = np.asarray(radii, dtype=np.float64).reshape(-1,2)
    if not np.isfinite(m).all() or not np.isfinite(r).all():
        raise ValueError("non-finite means2d/radii")
    tw = math.ceil(width / tile_size)
    th = math.ceil(height / tile_size)
    tmin = np.floor((m-r)/tile_size).astype(np.int64)
    tmax = np.ceil((m+r)/tile_size).astype(np.int64)
    tmin[:,0] = np.clip(tmin[:,0], 0, tw); tmax[:,0] = np.clip(tmax[:,0], 0, tw)
    tmin[:,1] = np.clip(tmin[:,1], 0, th); tmax[:,1] = np.clip(tmax[:,1], 0, th)
    active = (r > 0.0).all(axis=1)
    return BBoxSignature(active, tmin, tmax)


def membership_equal(a:BBoxSignature, b:BBoxSignature) -> bool:
    return (np.array_equal(a.active,b.active) and
            np.array_equal(a.tmin,b.tmin) and
            np.array_equal(a.tmax,b.tmax))


def float32_depth_bits(depths):
    d = np.asarray(depths, dtype=np.float32).reshape(-1)
    if not np.isfinite(d).all():
        raise ValueError("non-finite depth")
    # VRIR0 freezes the positive-depth path. Negative values make raw IEEE bits
    # non-monotone under unsigned comparison and therefore fail closed.
    if (d <= 0).any():
        raise ValueError("non-positive depth in certificate domain")
    return d.view(np.uint32)


def order_certificate(cached_flatten_ids, cached_offsets, current_depths,
                      tile_height:int, tile_width:int, n_gaussians:int):
    """Check cached per-tile lists under current (depth_bits, emit_rank) order.

    Assumptions frozen for VRIR0 and independently source-audited by DSH:
    - one image, dense mode;
    - flatten_id == dense Gaussian emission rank;
    - stable equal-key order follows that emission rank.
    """
    flat = np.asarray(cached_flatten_ids, dtype=np.int64).reshape(-1)
    off = np.asarray(cached_offsets, dtype=np.int64).reshape(-1)
    if len(off) != tile_height * tile_width:
        raise ValueError("unexpected offset table shape")
    if len(flat) and (flat.min() < 0 or flat.max() >= n_gaussians):
        raise ValueError("flatten id outside dense single-image domain")
    bits = float32_depth_bits(current_depths)
    violations = 0; pairs = 0
    for tid, start in enumerate(off):
        end = off[tid+1] if tid+1 < len(off) else len(flat)
        ids = flat[int(start):int(end)]
        if len(ids) < 2:
            continue
        k = bits[ids].astype(np.uint64)
        emit = ids.astype(np.int64)
        # Adjacent lexicographic inversion if depth decreases, or equal depth
        # reverses source emission order.
        bad = (k[:-1] > k[1:]) | ((k[:-1] == k[1:]) & (emit[:-1] > emit[1:]))
        violations += int(np.count_nonzero(bad))
        pairs += len(ids)-1
    return violations == 0, violations, pairs


def structure_equal(cache_flat, cache_offsets, fresh_flat, fresh_offsets):
    return (np.array_equal(np.asarray(cache_flat), np.asarray(fresh_flat)) and
            np.array_equal(np.asarray(cache_offsets), np.asarray(fresh_offsets)))


def build_key_slack_budgets(cached_flatten_ids, cached_offsets, cached_depths,
                            tile_height:int, tile_width:int, n_gaussians:int):
    """Build per-Gaussian uint32-key drift budgets from fresh cached structure.

    Returns (key0, budget, tie_pairs). Budget max means unconstrained/no adjacency.
    Each strict adjacent gap G contributes floor((G-1)/2) to both endpoints.
    Equal-key adjacent pairs are returned separately and are NOT certified here.
    """
    flat=np.asarray(cached_flatten_ids,dtype=np.int64).reshape(-1)
    off=np.asarray(cached_offsets,dtype=np.int64).reshape(-1)
    key0=float32_depth_bits(cached_depths).astype(np.uint64)
    budget=np.full(n_gaussians, np.iinfo(np.uint32).max, dtype=np.uint64)
    ties=[]
    for tid,start in enumerate(off):
        end=off[tid+1] if tid+1<len(off) else len(flat)
        ids=flat[int(start):int(end)]
        if len(ids)<2: continue
        a=ids[:-1]; b=ids[1:]
        ka=key0[a]; kb=key0[b]
        if np.any(ka>kb):
            raise ValueError('cached structure is not sorted by cached depth keys')
        for ia,ib,x,y in zip(a,b,ka,kb):
            g=int(y-x)
            if g==0:
                ties.append((int(ia),int(ib)))
                continue
            q=(g-1)//2
            if q < budget[ia]: budget[ia]=q
            if q < budget[ib]: budget[ib]=q
    return key0, budget, ties


def slack_order_certificate(current_depths, key0, budget, tie_pairs=(), fail_on_ties=True):
    cur=float32_depth_bits(current_depths).astype(np.uint64)
    key0=np.asarray(key0,dtype=np.uint64); budget=np.asarray(budget,dtype=np.uint64)
    if cur.shape != key0.shape or budget.shape != key0.shape:
        raise ValueError('key/budget shape mismatch')
    delta=np.where(cur>=key0,cur-key0,key0-cur)
    ok=bool(np.all(delta <= budget))
    if tie_pairs:
        if fail_on_ties:
            return False, int(np.count_nonzero(delta > budget)), len(tie_pairs)
        # Optional audited tie check: cached emit-rank order must remain lexicographic.
        for ia,ib in tie_pairs:
            if cur[ia] > cur[ib] or (cur[ia] == cur[ib] and ia > ib):
                ok=False
                break
    return ok, int(np.count_nonzero(delta > budget)), len(tie_pairs)
