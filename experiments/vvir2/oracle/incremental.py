"""VRIR2 minimal incremental raster-update oracle: membership kernel.

Pure GPU (torch). Reuses the VRIR0-verified tile_bounds / membership model
(bit-matching gsplat 1.5.3 isect_tiles, validated rate 1.0 in VRIR0/VRIR1).

Membership is fully determined by the per-Gaussian descriptor
  D = (b_minus[N,2], b_plus[N,2], active[N], depth_bits[N])
because the gsplat intersection key is (tile_id << 32) | float32_bits(depth)
and flatten_ids is the dense Gaussian index. Within a tile the stable CUB radix
sort orders by depth_bits ascending, ties by emission order = Gaussian index
ascending. So membership depends ONLY on D, not on means2d beyond the AABB.

Baseline (ground truth) = the real frozen gsplat path:
  isect_tiles(means2d, radii, depths, tile_size, tw, th, sort=True)
  isect_offset_encode(isect_ids, 1, tw, th)

Candidate (incremental) = M_old + Delta_G -> M_new, repairing ONLY affected
tiles (union of old/new tiles of changed Gaussians). Unaffected tiles are copied
verbatim. No global rebuild fallback, no semantics change.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch

TILE_SIZE = 16


def tile_bounds_torch(means2d, radii, width, height, tile_size=TILE_SIZE):
    """Port of certificate.tile_bounds to torch (float32, bit-matching gsplat)."""
    means2d = means2d.contiguous()
    if radii.ndim == 2:
        active = (radii[:, 0] > 0) & (radii[:, 1] > 0)
    else:
        active = radii > 0
    N = means2d.shape[0]
    tw = (width + tile_size - 1) // tile_size
    th = (height + tile_size - 1) // tile_size
    b_minus = torch.zeros((N, 2), dtype=torch.int64, device=means2d.device)
    b_plus = torch.zeros((N, 2), dtype=torch.int64, device=means2d.device)
    if bool(active.any()):
        m = means2d[active].to(torch.float32)
        if radii.ndim == 2:
            rx = radii[active][:, 0].to(torch.float32)
            ry = radii[active][:, 1].to(torch.float32)
        else:
            rx = radii[active].to(torch.float32)
            ry = rx
        T = float(tile_size)
        tx = m[:, 0] / T; ty = m[:, 1] / T
        trx = rx / T; try_ = ry / T
        bmx = torch.floor(tx - trx).clamp(0, tw)
        bmy = torch.floor(ty - try_).clamp(0, th)
        bMx = torch.ceil(tx + trx).clamp(0, tw)
        bMy = torch.ceil(ty + try_).clamp(0, th)
        idx = active.nonzero(as_tuple=True)[0]
        b_minus[idx, 0] = bmx.to(torch.int64)
        b_minus[idx, 1] = bmy.to(torch.int64)
        b_plus[idx, 0] = bMx.to(torch.int64)
        b_plus[idx, 1] = bMy.to(torch.int64)
    return b_minus, b_plus, active


def depth_bits_torch(depths):
    """float32 depth -> int64 bit pattern (low 32), matching gsplat key low bits."""
    d = depths.to(torch.float32).contiguous()
    return d.view(torch.int32).to(torch.int64)


@dataclass
class Descriptor:
    b_minus: torch.Tensor   # [N,2] int64
    b_plus: torch.Tensor    # [N,2] int64
    active: torch.Tensor    # [N] bool
    depth_bits: torch.Tensor  # [N] int64
    n_tiles: int
    tile_width: int
    tile_height: int

    def to(self, device):
        return Descriptor(self.b_minus.to(device), self.b_plus.to(device),
                          self.active.to(device), self.depth_bits.to(device),
                          self.n_tiles, self.tile_width, self.tile_height)


def make_descriptor(means2d, radii, depths, width, height, tile_size=TILE_SIZE):
    b_minus, b_plus, active = tile_bounds_torch(means2d, radii, width, height, tile_size)
    tw = (width + tile_size - 1) // tile_size
    th = (height + tile_size - 1) // tile_size
    return Descriptor(b_minus, b_plus, active, depth_bits_torch(depths),
                      tw * th, tw, th)


@dataclass
class Membership:
    flatten_ids: torch.Tensor  # [M] int32
    isect_ids: torch.Tensor    # [M] int64
    offsets: torch.Tensor      # [n_tiles+1] int32 (prefix; offsets[0]=0, offsets[-1]=M)
    n_tiles: int

    def tile_of_entry(self):
        counts = self.offsets[1:] - self.offsets[:-1]
        return torch.repeat_interleave(torch.arange(self.n_tiles, device=self.offsets.device),
                                       counts)


def baseline_membership(means2d, radii, depths, width, height, tile_size=TILE_SIZE):
    """Ground-truth membership via the real frozen gsplat path (single camera)."""
    from gsplat import isect_tiles, isect_offset_encode
    tw = (width + tile_size - 1) // tile_size
    th = (height + tile_size - 1) // tile_size
    m2 = means2d.to(torch.float32).unsqueeze(0)
    r = radii.contiguous().unsqueeze(0)
    d = depths.to(torch.float32).unsqueeze(0)
    tpg, isect_ids, flatten_ids = isect_tiles(
        m2, r, d, tile_size, tw, th, sort=True, segmented=False,
        packed=False, n_images=1)
    isect_off = isect_offset_encode(isect_ids, 1, tw, th)  # [1, th, tw]
    off1d = isect_off.reshape(-1).to(torch.int32)
    flat = flatten_ids.reshape(-1).contiguous()
    isect = isect_ids.reshape(-1).contiguous()
    M = int(flat.shape[0])
    offsets = torch.cat([off1d, torch.tensor([M], dtype=torch.int32, device=off1d.device)])
    return Membership(flat, isect, offsets, tw * th)


def _expand_aabb(b_minus, b_plus, active, tw):
    """Expand per-Gaussian tile AABBs -> (tile_id [K], gid [K]) for active gaussians."""
    idx = active.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        z = torch.zeros(0, dtype=torch.int64, device=b_minus.device)
        return z, z
    bmx = b_minus[idx, 0]; bmy = b_minus[idx, 1]
    bMx = b_plus[idx, 0]; bMy = b_plus[idx, 1]
    nx = (bMx - bmx).clamp(min=0)
    ny = (bMy - bmy).clamp(min=0)
    cnt = nx * ny
    total = int(cnt.sum().item())
    if total == 0:
        z = torch.zeros(0, dtype=torch.int64, device=b_minus.device)
        return z, z
    off = torch.zeros(idx.numel() + 1, dtype=torch.int64, device=b_minus.device)
    torch.cumsum(cnt, 0, out=off[1:])
    lin = torch.arange(total, dtype=torch.int64, device=b_minus.device) - torch.repeat_interleave(off[:-1], cnt)
    nx_rep = torch.repeat_interleave(nx, cnt)
    bmx_rep = torch.repeat_interleave(bmx, cnt)
    bmy_rep = torch.repeat_interleave(bmy, cnt)
    tx = bmx_rep + torch.remainder(lin, nx_rep)
    ty = bmy_rep + torch.div(lin, nx_rep, rounding_mode="floor")
    tile = ty * tw + tx
    gid = torch.repeat_interleave(idx, cnt)
    return tile, gid


def changed_gaussians(d_old, d_new):
    """Delta_G: gaussians whose membership contribution changed."""
    ch = (d_old.b_minus != d_new.b_minus).any(1) | (d_old.b_plus != d_new.b_plus).any(1) \
        | (d_old.active != d_new.active) | (d_old.depth_bits != d_new.depth_bits)
    return ch


@dataclass
class IncrementalResult:
    membership: Membership
    r_g: float          # changed gaussian ratio |Delta_G| / N
    r_t: float          # changed tile ratio |affected tiles| / n_tiles
    n_changed: int
    n_affected_tiles: int


def incremental_membership(m_old, d_old, d_new):
    """Candidate: M_old + Delta_G -> M_new, repairing only affected tiles."""
    dev = m_old.flatten_ids.device
    N = d_new.active.shape[0]
    n_tiles = d_new.n_tiles
    ch = changed_gaussians(d_old, d_new)
    n_changed = int(ch.sum().item())
    r_g = n_changed / N if N else 1.0

    # old tiles of changed gaussians (old descriptor)
    ot_tile, ot_gid = _expand_aabb(d_old.b_minus, d_old.b_plus, d_old.active & ch, d_new.tile_width)
    nt_tile, nt_gid = _expand_aabb(d_new.b_minus, d_new.b_plus, d_new.active & ch, d_new.tile_width)

    if ot_tile.numel() + nt_tile.numel() > 0:
        affected = torch.unique(torch.cat([ot_tile, nt_tile]))
    else:
        affected = torch.zeros(0, dtype=torch.int64, device=dev)
    n_aff = int(affected.numel())
    r_t = n_aff / n_tiles if n_tiles else 1.0

    # per-tile counts for new layout
    counts = m_old.offsets[1:] - m_old.offsets[:-1]  # [n_tiles] int32
    new_counts = counts.clone()

    if n_aff == 0:
        return Membership(m_old.flatten_ids.clone(),
                          m_old.isect_ids.clone(), m_old.offsets.clone(), n_tiles), \
            IncrementalResult(Membership(m_old.flatten_ids.clone(), m_old.isect_ids.clone(),
                                         m_old.offsets.clone(), n_tiles), r_g, r_t, 0, 0)

    aff_mask = torch.zeros(n_tiles, dtype=torch.bool, device=dev)
    aff_mask[affected] = True

    # old entries located in affected tiles (kept if their gaussian is unchanged)
    tile_old = m_old.tile_of_entry()  # [M_old]
    in_aff = aff_mask[tile_old]
    keep = in_aff & (~ch[m_old.flatten_ids.long()])
    kept_gid = m_old.flatten_ids[keep]
    kept_tile = tile_old[keep]
    kept_depth = m_old.isect_ids[keep]  # full key unchanged (depth_bits + tile)

    # new entries for changed gaussians (their new tiles)
    new_key = (nt_tile.to(torch.int64) << 32) | (d_new.depth_bits[nt_gid] & 0xFFFFFFFF)

    rep_tile = torch.cat([kept_tile, nt_tile])
    rep_gid = torch.cat([kept_gid, nt_gid])
    rep_key = torch.cat([kept_depth, new_key])

    # stable sort: primary (tile<<32|depth_bits), tie-break gid ascending
    o0 = torch.argsort(rep_gid.to(torch.int64), stable=True)
    rep_tile = rep_tile[o0]; rep_key = rep_key[o0]; rep_gid = rep_gid[o0]
    o1 = torch.argsort(rep_key, stable=True)
    rep_tile = rep_tile[o1]; rep_key = rep_key[o1]; rep_gid = rep_gid[o1]

    # per-tile counts of the rebuilt affected tiles
    rep_counts = torch.bincount(rep_tile, minlength=n_tiles)[:n_tiles]
    new_counts[affected] = rep_counts.to(torch.int32)
    new_offsets = torch.zeros(n_tiles + 1, dtype=torch.int32, device=dev)
    torch.cumsum(new_counts, 0, out=new_offsets[1:])
    M_new = int(new_offsets[-1].item())

    new_flatten = torch.empty(M_new, dtype=torch.int32, device=dev)
    new_isect = torch.empty(M_new, dtype=torch.int64, device=dev)

    # scatter unaffected old entries (verbatim) into their new positions
    rank_in_tile = torch.arange(m_old.flatten_ids.shape[0], device=dev) - m_old.offsets[tile_old]
    pos_all = new_offsets[tile_old] + rank_in_tile.to(torch.int32)
    sel = (~in_aff).nonzero(as_tuple=True)[0]
    new_flatten[pos_all[sel]] = m_old.flatten_ids[sel]
    new_isect[pos_all[sel]] = m_old.isect_ids[sel]

    # scatter rebuilt affected entries
    rep_off = torch.zeros(n_tiles + 1, dtype=torch.int64, device=dev)
    torch.cumsum(rep_counts.to(torch.int64), 0, out=rep_off[1:])
    rep_rank = torch.arange(rep_tile.shape[0], device=dev) - rep_off[rep_tile]
    rep_pos = (new_offsets[rep_tile].to(torch.int64) + rep_rank).to(torch.int64)
    new_flatten[rep_pos] = rep_gid.to(torch.int32)
    new_isect[rep_pos] = rep_key

    m_new = Membership(new_flatten, new_isect, new_offsets, n_tiles)
    return m_new, IncrementalResult(m_new, r_g, r_t, n_changed, n_aff)


def exact_match(m_a, m_b):
    """Bit-exact comparison of two memberships (same M required)."""
    if m_a.flatten_ids.shape[0] != m_b.flatten_ids.shape[0]:
        return False, {"reason": "M_mismatch",
                       "M_a": int(m_a.flatten_ids.shape[0]),
                       "M_b": int(m_b.flatten_ids.shape[0])}
    flat_ok = bool(torch.equal(m_a.flatten_ids, m_b.flatten_ids))
    isect_ok = bool(torch.equal(m_a.isect_ids, m_b.isect_ids))
    off_ok = bool(torch.equal(m_a.offsets, m_b.offsets))
    return (flat_ok and isect_ok and off_ok), {
        "flatten_ids_ok": flat_ok, "isect_ids_ok": isect_ok, "offsets_ok": off_ok,
        "n_flatten_mismatch": int((m_a.flatten_ids != m_b.flatten_ids).sum().item()) if not flat_ok else 0,
        "n_isect_mismatch": int((m_a.isect_ids != m_b.isect_ids).sum().item()) if not isect_ok else 0,
        "n_offset_mismatch": int((m_a.offsets != m_b.offsets).sum().item()) if not off_ok else 0,
    }