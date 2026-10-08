#!/usr/bin/env python3
"""ET0 core: frozen primitives (reuse a0_lib A0/IR0 instrumentation) + churn +
hellinger + stats. NO production CUDA modified; renderer driven read-only via
the same runtime capture patch A0/IR0 used. See ET0_PREREG.md (FROZEN)."""
import json, math, sys
import numpy as np
import torch

sys.path.insert(0, "/root/epic05_moonshot/a0")
import a0_lib as L

BETA1, BETA2, EPS_ADAM = 0.9, 0.999, 1e-15
GROUPS = ["xyz", "shs", "scaling", "rotation", "opacity"]
SH_DEGREE = 3
ALPHAS = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00]
TAU_C = 0.10


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True) if False else None
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=L.json_default)
    print(f"[wrote] {path}", flush=True)


# ---- camera semantics (ET0 S2: corrected SB0 train pool, exclude i%8==0) ----
def stratified(pool, k):
    N = len(pool); used = set(); out = []
    for i in range(k):
        p = int(math.floor((i + 0.5) * N / k))
        while p in used or p >= N:
            p += 1
        used.add(p); out.append(p)
    return out


def camera_semantics(ds):
    n = len(ds)
    holdout = [i for i in range(n) if i % 8 == 0]
    train = [i for i in range(n) if i % 8 != 0]
    assert set(train).isdisjoint(set(holdout)), "train/holdout overlap"
    return n, train, holdout


def source_and_holdout(ds):
    n, train, holdout = camera_semantics(ds)
    src_cams = [train[p] for p in stratified(train, 8)]
    hold_cams = [holdout[p] for p in stratified(holdout, 8)]
    return n, src_cams, hold_cams, train, holdout


# ---- exact native Adam displacement (trusted IR0 path) ----
def adam_update(grad, og):
    ea = og["exp_avg"]; eas = og["exp_avg_sq"]; lr = og["lr"]; step = og["step"] + 1
    mm = BETA1 * ea + (1.0 - BETA1) * grad
    vv = BETA2 * eas + (1.0 - BETA2) * grad * grad
    m_hat = mm / (1.0 - BETA1 ** step)
    v_hat = vv / (1.0 - BETA2 ** step)
    return -lr * m_hat / (v_hat.sqrt() + EPS_ADAM)


# ---- renderer-state capture -> per-tile ordered Gaussian-id sequences ----
def tile_state(cap, base):
    """Extract, for the single captured image (image_idx=0):
       codes  int64 [n_isect] encoded (tile*base + master) with FIXED base
       tile   int64 [n_isect] tile id per intersection
       master int64 [n_isect] master gaussian id per intersection
       order  int64 [n_isect] within-tile raster order (0..k-1)
       n_tiles. base must be a FIXED constant across compared captures."""
    C = cap["C"]; H = cap["height"]; W = cap["width"]; ts = cap["tile_size"]
    th = (H + ts - 1) // ts; tw = (W + ts - 1) // ts; n_tiles = th * tw
    offset = cap["tile_offsets_f"].reshape(C, -1)[0].to(torch.long).cpu()
    flat = cap["flatten_ids_f"].reshape(-1).to(torch.long).cpu()
    master_all = cap["visible_ids"].reshape(-1).to(torch.long).cpu()[flat]
    n_isect = master_all.numel()
    if n_isect == 0:
        z = torch.zeros(0, dtype=torch.int64)
        return z, z, z, z, n_tiles
    gid = torch.arange(n_isect)
    tile = torch.searchsorted(offset, gid, right=True) - 1
    tile = tile.clamp(0, n_tiles - 1)
    order = gid - offset[tile]
    codes = tile * base + master_all
    return codes, tile, master_all, order, n_tiles


def c_tile(codes0, codesA):
    if codes0.numel() == 0 and codesA.numel() == 0:
        return 0.0
    u0 = torch.unique(codes0); uA = torch.unique(codesA)
    inter = torch.isin(uA, u0).sum().item()
    union = u0.numel() + uA.numel() - inter
    return float(union - inter) / float(union) if union > 0 else 0.0


def c_order(ts0, tsA, base):
    """Vectorized normalized adjacent-order disagreement + insertion/deletion.
    ts0/tsA are tile_state tuples; base is the FIXED key multiplier."""
    _, tile0, m0, o0, n_tiles = ts0
    _, tileA, mA, oA, _ = tsA
    key0 = tile0 * base + m0
    keyA = tileA * base + mA
    u0 = torch.unique(key0); uA = torch.unique(keyA)
    inter = u0[torch.isin(u0, uA)]
    union = u0.numel() + uA.numel() - inter.numel()
    insdel = float(u0.numel() + uA.numel() - 2 * inter.numel()) / float(union) \
        if union > 0 else 0.0
    if inter.numel() == 0:
        return 0.5 * 0.0 + 0.5 * insdel, insdel
    ia = torch.searchsorted(u0, inter); ib = torch.searchsorted(uA, inter)
    o0c = o0[ia]; oAc = oA[ib]; tileC = tile0[ia]
    maxorder = int(max(o0.max().item(), oA.max().item())) + 2
    perm = torch.argsort(tileC * maxorder + o0c)
    tile_s = tileC[perm]; oA_s = oAc[perm]
    if tile_s.numel() < 2:
        return 0.5 * 0.0 + 0.5 * insdel, insdel
    same = tile_s[1:] == tile_s[:-1]
    inverted = oA_s[1:] < oA_s[:-1]
    disagg_num = int((same & inverted).sum().item())
    disagg_den = int(same.sum().item())
    A = float(disagg_num) / disagg_den if disagg_den > 0 else 0.0
    return 0.5 * A + 0.5 * insdel, insdel


# ---- squared Hellinger between N(mu0,S0) and N(mu1,S1), fp64 ----
def quat_to_R(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.empty((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def cov3(quat, scaling):
    R = quat_to_R(quat).double()
    s = torch.exp(scaling.double())
    S = torch.diag_embed(s * s)
    return R @ S @ R.transpose(1, 2)


def hellinger_sq(xyz0, quat0, scaling0, xyz1, quat1, scaling1):
    mu0 = xyz0.double(); mu1 = xyz1.double()
    S0 = cov3(quat0, scaling0); S1 = cov3(quat1, scaling1)
    M = 0.5 * (S0 + S1)
    detS0 = torch.det(S0).clamp_min(0); detS1 = torch.det(S1).clamp_min(0)
    detM = torch.det(M).clamp_min(1e-30)
    dmu = mu1 - mu0
    sol = torch.linalg.solve(M, dmu.unsqueeze(-1)).squeeze(-1)
    quad = (dmu * sol).sum(-1)
    det_term = detS0.pow(0.25) * detS1.pow(0.25) / detM.pow(0.5)
    bc = det_term * torch.exp(-0.125 * quad)
    return (1.0 - bc).clamp_min(0.0)


# ---- stats ----
def _rank_avg(v):
    v = np.asarray(v, float); order = np.argsort(v)
    ranks = np.empty(len(v), float); ranks[order] = np.arange(1, len(v) + 1)
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt)); np.add.at(s, inv, ranks)
    return (s / cnt)[inv]


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(_rank_avg(x), _rank_avg(y))[0, 1])


def auc(labels, scores):
    labels = np.asarray(labels); scores = np.asarray(scores, float)
    if len(np.unique(labels)) < 2:
        return float("nan")
    ranks = _rank_avg(scores)
    n1 = int((labels == 1).sum()); n0 = int((labels == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
