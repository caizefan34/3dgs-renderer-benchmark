"""
VRIR0 structural-coherence certificates.

Implements the verified gsplat 1.4.0 semantics (see P1_SOURCE_AUDIT.md):
  - accutile=False -> exact tile AABB membership:
      b- = clamp(floor((m - r)/T), 0, tile_dim); b+ = clamp(ceil((m + r)/T), 0, tile_dim)
      radius <= 0 -> Gaussian contributes no tiles.
  - intersection key = camera_id | (tile_id << 32) | float32_bits(depth)  (low 32 bits).
  - flatten_ids[cur] = dense Gaussian index (single-camera packed=False path).
  - depths = mean_c.z (view-space z); sort is stable CUB radix over the full key.

All structure tensors are handled on CPU int64/int32 to allow exact comparison.
No approximate equality is used anywhere (fail-closed).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

TILE_SIZE = 16

def float32_bits(x: np.ndarray) -> np.ndarray:
    """Reinterpret float32 array as uint32 bit patterns (monotone for positive depth)."""
    x32 = x.astype('<f4')
    return x32.view('<u4').astype(np.uint32)

def tile_bounds(means2d: np.ndarray, radii: np.ndarray, width: int, height: int,
                tile_size: int = TILE_SIZE) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-Gaussian tile AABB bounds, bit-matching gsplat isect_tiles.cu.

    gsplat (OpT=float32) computes per axis:
        tile_min = min(max(0, (uint32)floor((m - r)/T)), dim)
        tile_max = min(max(0, (uint32)ceil ((m + r)/T)), dim)
    The (uint32) cast of a negative floor saturates to 0, so this is clamp(floor/ceil,0,dim).
    A Gaussian contributes no tiles iff radius <= 0. Inactive gaussians' means2d are never
    written by the projection kernel (uninitialized memory), so their bounds are
    canonicalized to the empty range [0,0)x[0,0): they contribute zero tiles and never
    affect equality. Computed in float32 to bit-match the kernel (no approximate equality).
    """
    tw = (width + tile_size - 1) // tile_size
    th = (height + tile_size - 1) // tile_size
    N = means2d.shape[0]
    # radii layout is gsplat-version dependent:
    #   gsplat 1.4.0: meta['radii'] is [N]   -> single scalar radius applied to both axes.
    #   gsplat 1.5.3: meta['radii'] is [N, 2] -> per-axis (radius_x, radius_y) integers.
    # IntersectTile.cu (1.5.3) culls a Gaussian iff radius_x <= 0 OR radius_y <= 0,
    # then bounds each axis independently. The 1.4.0 scalar path is the special case
    # radius_x == radius_y, so both are handled here without changing membership semantics.
    if radii.ndim == 2:
        active = (radii[:, 0] > 0) & (radii[:, 1] > 0)
    else:
        active = radii > 0
    b_minus = np.zeros((N, 2), dtype=np.int64)
    b_plus = np.zeros((N, 2), dtype=np.int64)
    if np.any(active):
        m = means2d[active].astype('<f4')
        if radii.ndim == 2:
            rx = radii[active][:, 0].astype('<f4'); ry = radii[active][:, 1].astype('<f4')
        else:
            rx = radii[active].astype('<f4'); ry = rx
        T = np.float32(tile_size)
        tx = m[:, 0] / T; ty = m[:, 1] / T
        trx = rx / T; try_ = ry / T
        bmx = np.floor(tx - trx); bMy = np.floor(ty - try_)
        bMx = np.ceil(tx + trx);  bMy_ = np.ceil(ty + try_)
        bmx = np.clip(bmx, 0, tw); bMy = np.clip(bMy, 0, th)
        bMx = np.clip(bMx, 0, tw); bMy_ = np.clip(bMy_, 0, th)
        b_minus[active] = np.stack([bmx.astype(np.int64), bMy.astype(np.int64)], axis=1)
        b_plus[active] = np.stack([bMx.astype(np.int64), bMy_.astype(np.int64)], axis=1)
    return b_minus, b_plus, active

@dataclass
class Structure:
    """Fresh structure captured from a baseline rasterization visit."""
    camera_id: int
    iteration: int
    width: int
    height: int
    n_gaussians: int
    flatten_ids: np.ndarray          # [M] int32 dense indices, sorted by key
    isect_offsets: np.ndarray        # [n_tiles] int32, prefix offsets per (cam,tile)
    tile_width: int
    tile_height: int
    means2d: np.ndarray              # [N, 2] float64 (CPU)
    radii: np.ndarray                # [N] float64 (CPU)
    depths: np.ndarray               # [N] float64 (CPU)
    b_minus: np.ndarray              # [N, 2] int64
    b_plus: np.ndarray               # [N, 2] int64
    active: np.ndarray               # [N] bool
    depth_bits: np.ndarray           # [N] uint32
    topology_epoch: int
    n_isects: int

    @property
    def tile_lists(self) -> Dict[int, np.ndarray]:
        """Return {tile_id: ordered flatten_ids array} from offsets."""
        off = self.isect_offsets
        out = {}
        for tid in range(len(off) - 1):
            s, e = int(off[tid]), int(off[tid + 1])
            if e > s:
                out[tid] = self.flatten_ids[s:e]
        return out

def build_key_slack_budgets(cached: Structure, n_gaussians: int):
    """C-ORD-B: per-Gaussian uint32 depth-key drift budgets from cached structure (vectorized).

    Returns (key0 [N] uint64, budget [N] uint64, tie_pairs list[(i,j)]).
    Each strict adjacent gap G contributes floor((G-1)/2) to both endpoints (min).
    Equal-key adjacent pairs (G=0) are returned separately and NOT certified as slack.
    """
    key0 = cached.depth_bits.astype(np.uint64)
    budget = np.full(n_gaussians, np.iinfo(np.uint64).max, dtype=np.uint64)
    ties = []
    flat = cached.flatten_ids.astype(np.int64)
    off = cached.isect_offsets.astype(np.int64)
    M = flat.shape[0]
    if M < 2:
        return key0, budget, ties
    tile_per_isect = np.repeat(np.arange(len(off) - 1), np.diff(off))
    same_tile = tile_per_isect[:-1] == tile_per_isect[1:]
    a = flat[:-1]; b = flat[1:]
    ka = key0[a]; kb = key0[b]
    g = kb.astype(np.int64) - ka.astype(np.int64)
    strict = same_tile & (g >= 1)
    tie_mask = same_tile & (g == 0)
    if np.any(tie_mask):
        ta = a[tie_mask]; tb = b[tie_mask]
        ties = list(zip(ta.tolist(), tb.tolist()))
    if np.any(strict):
        q = ((g[strict] - 1) // 2).astype(np.uint64)
        sa = a[strict]; sb = b[strict]
        # scatter-min
        np.minimum.at(budget, sa, q)
        np.minimum.at(budget, sb, q)
    return key0, budget, ties

@dataclass
class CertificateResult:
    config: bool
    topology: bool
    numeric: bool
    membership: bool
    order_exact_scan: bool
    order_slack_n: bool
    accept: bool
    reject_reason: Optional[str] = None
    adjacent_pairs_checked: int = 0
    adjacent_order_violations: int = 0
    tile_truth_same_fraction: float = float('nan')
    false_accept: bool = False
    false_reject: bool = False

def membership_equal(cached: Structure, fresh: Structure) -> bool:
    """C-MEM: exact active-mask and tile-bounds equality for every Gaussian identity."""
    if cached.topology_epoch != fresh.topology_epoch:
        return False
    if cached.n_gaussians != fresh.n_gaussians:
        return False
    if not np.array_equal(cached.active, fresh.active):
        return False
    # only active gaussians have meaningful bounds; inactive have b-=b+=clamp
    if not np.array_equal(cached.b_minus, fresh.b_minus):
        return False
    if not np.array_equal(cached.b_plus, fresh.b_plus):
        return False
    return True

def order_exact_scan(cached: Structure, fresh: Structure):
    """C-ORD exact scan (vectorized): every cached tile list nondecreasing under current ordering.

    Ordering key = (depth_bits[i], emit_rank=i) since flatten_ids == dense index
    and stable sort preserves emission order for equal keys. Adjacent entries in the
    sorted flatten_ids belong to the same tile except at tile boundaries (detected via
    isect_offsets). Returns (ok, pairs_checked, violations).
    """
    flat = cached.flatten_ids.astype(np.int64)
    off = cached.isect_offsets.astype(np.int64)
    M = flat.shape[0]
    if M < 2:
        return True, 0, 0
    # tile id per intersection via offsets
    tile_per_isect = np.repeat(np.arange(len(off) - 1), np.diff(off))
    same_tile = tile_per_isect[:-1] == tile_per_isect[1:]
    cur_bits = fresh.depth_bits.astype(np.int64)
    kb = cur_bits[flat]                      # [M]
    dj = kb[:-1]; dn = kb[1:]
    ej = flat[:-1]; en = flat[1:]
    viol_mask = same_tile & ((dj > dn) | ((dj == dn) & (ej > en)))
    pairs = int(np.count_nonzero(same_tile))
    viol = int(np.count_nonzero(viol_mask))
    return (viol == 0), pairs, viol

def order_slack_n(cached: Structure, fresh: Structure,
                  key0: np.ndarray, budget: np.ndarray,
                  tie_pairs: List[Tuple[int, int]], fail_on_ties: bool = True):
    """C-ORD-B O(N) slack certificate. Returns (ok, n_violations, n_ties)."""
    cur = fresh.depth_bits.astype(np.uint64)
    delta = np.where(cur >= key0, cur - key0, key0 - cur)
    over = delta > budget
    n_over = int(np.count_nonzero(over))
    ok = (n_over == 0)
    if tie_pairs and fail_on_ties:
        # any unresolved equal-key adjacency rejects slack certificate
        ok = False
        n_over = max(n_over, len(tie_pairs))
    return ok, n_over, len(tie_pairs)

def truth_same(cached: Structure, fresh: Structure) -> Tuple[bool, float]:
    """Independent ground-truth: exact equality of cached vs fresh flatten_ids+offsets.

    Returns (exact_equal, tile_truth_same_fraction). Compares per-tile ordered lists.
    """
    if cached.n_isects != fresh.n_isects:
        return False, 0.0
    if not np.array_equal(cached.isect_offsets, fresh.isect_offsets):
        # compute per-tile same fraction
        pass
    exact_flat = np.array_equal(cached.flatten_ids, fresh.flatten_ids)
    exact_off = np.array_equal(cached.isect_offsets, fresh.isect_offsets)
    # per-tile fraction
    n_same = 0; n_tot = 0
    off = fresh.isect_offsets
    for tid in range(len(off) - 1):
        s, e = int(off[tid]), int(off[tid + 1])
        n_tot += 1
        if e > s:
            cf = cached.flatten_ids[s:e]
            ff = fresh.flatten_ids[s:e]
            if len(cf) == len(ff) and np.array_equal(cf, ff):
                n_same += 1
    frac = n_same / n_tot if n_tot > 0 else 1.0
    return (exact_flat and exact_off), frac

def validate_membership_model(flat: np.ndarray, off: np.ndarray, b_minus: np.ndarray,
                              b_plus: np.ndarray, active: np.ndarray,
                              tile_width: int, n_gaussians: int) -> Tuple[bool, int]:
    """Self-validate tile_bounds against the actual per-Gaussian tile sets implied by the
    fresh baseline structure (flatten_ids + isect_offsets). P3 stop criterion: the computed
    membership descriptor must match the fresh baseline structure.

    Reconstructs, for each Gaussian, the set of tiles gsplat assigned it from (offsets,
    flatten_ids) using tile_id = ty*tile_width + tx, and checks it equals the AABB
    [b_minus, b_plus) predicted by tile_bounds. Returns (ok, n_mismatched_gaussians).
    """
    flat = np.asarray(flat).astype(np.int64).reshape(-1)
    off = np.asarray(off).astype(np.int64).reshape(-1)
    n_tiles = len(off) - 1
    if n_tiles <= 0 or flat.size == 0:
        return True, 0
    tile_per_isect = np.repeat(np.arange(n_tiles), np.diff(off))
    tx = tile_per_isect % tile_width
    ty = tile_per_isect // tile_width
    gauss = flat
    within = ((tx >= b_minus[gauss, 0]) & (tx < b_plus[gauss, 0]) &
              (ty >= b_minus[gauss, 1]) & (ty < b_plus[gauss, 1]))
    area = (b_plus[:, 0] - b_minus[:, 0]) * (b_plus[:, 1] - b_minus[:, 1])
    counts = np.bincount(gauss, minlength=n_gaussians)
    count_ok = np.array_equal(counts, area)
    within_ok = bool(np.all(within))
    n_bad = int(np.count_nonzero(~within)) + int(np.count_nonzero(counts != area))
    return (within_ok and count_ok), n_bad


def evaluate_certificate(cached: Optional[Structure], fresh: Structure,
                         config_same: bool, epoch: int) -> CertificateResult:
    """Full CERT = C0 & C1 & C2 & C3 & C4. Fail-closed."""
    if cached is None:
        return CertificateResult(config=config_same, topology=False, numeric=True,
                                 membership=False, order_exact_scan=False,
                                 order_slack_n=False, accept=False,
                                 reject_reason='NO_CACHE')
    # C1 topology
    topo = (cached.topology_epoch == fresh.topology_epoch)
    # C2 numeric sanity, scoped to ACTIVE gaussians. gsplat's projection kernel returns
    # early for culled gaussians (radii<=0) WITHOUT writing means2d/depths, so inactive
    # entries hold uninitialized memory (frequently NaN/Inf). Inactive gaussians
    # contribute zero tiles, so finiteness is checked on the active subset only. Active
    # depths must be finite and strictly positive (positive-depth path, CERTIFICATION C).
    act = fresh.active
    numeric = bool(np.all(np.isfinite(fresh.radii)))
    if numeric and np.any(act):
        numeric = bool(np.all(np.isfinite(fresh.means2d[act]))
                       and np.all(np.isfinite(fresh.depths[act]))
                       and np.all(fresh.depths[act] > 0))
    # C3 membership
    mem = membership_equal(cached, fresh) if (topo and numeric) else False
    # C4 order (only meaningful if membership unchanged)
    if mem:
        try:
            ord_ok, pairs, viol = order_exact_scan(cached, fresh)
        except Exception as exc:
            ord_ok, pairs, viol = False, 0, 0
        key0, budget, ties = build_key_slack_budgets(cached, fresh.n_gaussians)
        slack_ok, n_over, n_ties = order_slack_n(cached, fresh, key0, budget, ties)
    else:
        ord_ok, pairs, viol = False, 0, 0
        slack_ok = False
        key0 = budget = None
        ties = []
    accept = bool(config_same and topo and numeric and mem and ord_ok and slack_ok)
    # ground truth
    tsame, frac = truth_same(cached, fresh)
    false_accept = bool(accept and not tsame)
    false_reject = bool((not accept) and tsame and cached is not None)
    reason = None
    if not accept:
        if not config_same: reason = 'CONFIG'
        elif not topo: reason = 'TOPOLOGY'
        elif not numeric: reason = 'NUMERIC'
        elif not mem: reason = 'MEMBERSHIP'
        elif not ord_ok: reason = 'ORDER_EXACT'
        elif not slack_ok: reason = 'ORDER_SLACK'
    return CertificateResult(config=config_same, topology=topo, numeric=numeric,
                             membership=mem, order_exact_scan=ord_ok,
                             order_slack_n=slack_ok, accept=accept,
                             reject_reason=reason, adjacent_pairs_checked=pairs,
                             adjacent_order_violations=viol, tile_truth_same_fraction=frac,
                             false_accept=false_accept, false_reject=false_reject)
