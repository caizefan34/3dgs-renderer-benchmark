#!/usr/bin/env python3
"""CG0 deterministic depth+RGB compiler (prereg sections 5-8).

Forbidden: reading source xyz/cov/opacity/SH/optimizer/per-Gaussian IDs/RGB.
Allowed: oracle expected depth maps (rendered from source), GT training images,
calibration cameras, and the scalar source Gaussian COUNT N_source.

Fusion is fully vectorized: candidates are sorted by (voxel_key, camera, pixel);
within a voxel, normal-consistent clusters use consecutive single-linkage with the
frozen cosine threshold (deterministic, reproducible). Per-cluster statistics are
segmented reductions (scatter), so millions of candidates/clusters stay on-GPU.
"""
import math
import torch
import torch.nn.functional as F
from cg0_prereg import (STRIDE, NORMAL_COS_THRESH, DEPTH_DISCONT_FRAC,
                        sh_basis, sh_degree_for_count, logit, mat_to_quat)

EPS = 1e-12


def backproject(cam, us, vs, ds):
    K = cam.K
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (us - cx) / fx * ds
    y = (vs - cy) / fy * ds
    cam_pt = torch.stack([x, y, ds], dim=-1)
    R = cam.viewmatrix[:3, :3]
    C = cam.camera_center
    return cam_pt @ R + C.unsqueeze(0)


def view_dirs(cam, X):
    d = X - cam.camera_center.unsqueeze(0)
    return F.normalize(d, dim=-1, eps=1e-12)


def generate_candidates(depth_maps, cams, compile_idx, gt_getter, device):
    """depth_maps: list[i] is [H,W] depth for source camera i (0=invalid).
    Returns candidate dict with global deterministic ordering keys."""
    H, W = depth_maps[compile_idx[0]].shape
    cols = torch.arange(0, W, STRIDE, device=device)
    rows = torch.arange(0, H, STRIDE, device=device)
    grid_c, grid_r = torch.meshgrid(cols, rows, indexing="xy")
    flat_c = grid_c.reshape(-1).long()
    flat_r = grid_r.reshape(-1).long()
    P = flat_c.shape[0]
    all_X, all_tu, all_tv, all_n, all_rgb, all_vdir = [], [], [], [], [], []
    all_cidx, all_pix = [], []

    for ci, cam in zip(compile_idx, cams):
        D = depth_maps[ci]
        Ds = D[flat_r, flat_c]
        c_up = (flat_c + 1).clamp(0, W - 1); c_dn = (flat_c - 1).clamp(0, W - 1)
        r_up = (flat_r + 1).clamp(0, H - 1); r_dn = (flat_r - 1).clamp(0, H - 1)
        dcp = D[flat_r, c_up]; dcm = D[flat_r, c_dn]
        drp = D[r_up, flat_c]; drm = D[r_dn, flat_c]

        def ok(d): return torch.isfinite(d) & (d > 1e-6)
        ocp, ocm, orp, orm = ok(dcp), ok(dcm), ok(drp), ok(drm)
        valid = ok(Ds)
        base_ok = valid & ocp & ocm & orp & orm
        rel = DEPTH_DISCONT_FRAC
        disc = ((dcp - Ds).abs() > rel * Ds) | ((dcm - Ds).abs() > rel * Ds) | \
               ((drp - Ds).abs() > rel * Ds) | ((drm - Ds).abs() > rel * Ds)
        keep = base_ok & (~disc)
        if int(keep.sum()) == 0:
            continue
        idx = keep.nonzero(as_tuple=False).reshape(-1)
        uc = flat_c[idx].float(); vc = flat_r[idx].float()
        dc = Ds[idx]
        Xc = backproject(cam, uc, vc, dc)
        Xu = backproject(cam, c_up[idx].float(), flat_r[idx].float(), dcp[idx])
        Xd = backproject(cam, c_dn[idx].float(), flat_r[idx].float(), dcm[idx])
        Xv = backproject(cam, uc, r_up[idx].float(), drp[idx])
        Xt = backproject(cam, uc, r_dn[idx].float(), drm[idx])
        tu = Xu - Xd; tv = Xv - Xt
        cross = torch.cross(tu, tv, dim=1)
        cn = cross.norm(dim=1)
        keep2 = cn >= 1e-10
        if int(keep2.sum()) == 0:
            continue
        idx2 = keep2.nonzero(as_tuple=False).reshape(-1)
        Xc = Xc[idx2]; tu = tu[idx2]; tv = tv[idx2]
        cross = cross[idx2]; cn = cn[idx2]
        n = cross / cn.unsqueeze(-1)
        gt = gt_getter(ci)
        # orient normal toward the observing camera (visible side of a depth-map surface);
        # the cross-product sign is winding-dependent, so enforce outward orientation
        to_cam = cam.camera_center.unsqueeze(0) - Xc
        sgn = torch.sign((n * to_cam).sum(dim=1, keepdim=True))
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        n = n * sgn
        rgb = gt[flat_r[idx][idx2], flat_c[idx][idx2]]
        vdir = view_dirs(cam, Xc)
        all_X.append(Xc); all_tu.append(tu); all_tv.append(tv)
        all_n.append(n); all_rgb.append(rgb); all_vdir.append(vdir)
        all_cidx.append(torch.full((Xc.shape[0],), ci, dtype=torch.long, device=device))
        all_pix.append(torch.stack([flat_c[idx][idx2], flat_r[idx][idx2]], dim=1).long())

    return {
        "X": torch.cat(all_X), "tu": torch.cat(all_tu), "tv": torch.cat(all_tv),
        "n": torch.cat(all_n), "rgb": torch.cat(all_rgb), "vdir": torch.cat(all_vdir),
        "cidx": torch.cat(all_cidx), "pix": torch.cat(all_pix),
        "M": sum(t.shape[0] for t in all_X),
    }


NORMAL_Q = 0.2   # normal quantization step; guarantees within-bin pairwise cosine >= 0.94 >= 0.90


def _cluster_ids(X, n, h):
    """Collision-free cluster id per candidate = (voxel, normal-quantum).
    Monotone non-increasing in h. Vectorized."""
    vox = torch.floor(X / h).long()
    vox = vox - vox.min(dim=0).values
    Rv = int(vox.max().item()) + 1
    vid = vox[:, 0] * (Rv * Rv) + vox[:, 1] * Rv + vox[:, 2]
    qn = torch.round(n / NORMAL_Q).long()
    qn = qn - qn.min(dim=0).values
    Rq = int(qn.max().item()) + 1
    qid = qn[:, 0] * (Rq * Rq) + qn[:, 1] * Rq + qn[:, 2]
    return vid * (Rq ** 3) + qid


def fuse_count(cand, N_target, scene_scale, iters=60):
    """Vectorized binary search on voxel width h via (voxel, normal-quantum)
    cluster ids. Returns (h, count, labels, final_order)."""
    X = cand["X"]; n = cand["n"]
    M = X.shape[0]; device = X.device
    base_order = torch.argsort(cand["cidx"] * 100000 + cand["pix"][:, 0] * 1000 + cand["pix"][:, 1], stable=True)
    Xb = X[base_order]; nb = n[base_order]

    def compute(h):
        key = _cluster_ids(Xb, nb, h)
        uk, inv = torch.unique(key, return_inverse=True)
        return int(uk.shape[0]), inv

    extent = float((Xb.max(dim=0).values - Xb.min(dim=0).values).max().item())
    extent = max(extent, 1e-6)
    h_lo = extent * 1e-6
    h_hi = extent * 2.0
    best = (h_lo, M + 1, None, None)
    for _ in range(iters):
        h_mid = math.sqrt(h_lo * h_hi)
        cnt, inv = compute(h_mid)
        if abs(cnt - N_target) < abs(best[1] - N_target):
            best = (h_mid, cnt, inv, None)
        if cnt > N_target:
            h_lo = h_mid
        else:
            h_hi = h_mid
        if abs(cnt - N_target) <= 0.05 * N_target:
            break
    h_best, cnt_best, inv_best, _ = best
    cnt, inv = compute(h_best)
    final_order = base_order
    return h_best, cnt, inv, final_order


def _batched_eigh(Sigma, floor, chunk=60000):
    """Chunked eigh with CPU fallback to avoid cuSOLVER batch limits."""
    N = Sigma.shape[0]
    evals = torch.empty(N, 3, device=Sigma.device, dtype=Sigma.dtype)
    evecs = torch.empty(N, 3, 3, device=Sigma.device, dtype=Sigma.dtype)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        blk = Sigma[i:j]
        try:
            ev, evec = torch.linalg.eigh(blk)
        except Exception:
            ev, evec = torch.linalg.eigh(blk.cpu())
            ev = ev.to(Sigma.device); evec = evec.to(Sigma.device)
        evals[i:j] = torch.nan_to_num(ev, nan=floor)
        evecs[i:j] = evec
    return evals, evecs


def cluster_gaussians(cand, labels, final_order, h, scene_scale):
    """Vectorized per-cluster Gaussian emission. labels are cluster ids in
    final_order (= base_order) space."""
    X = cand["X"][final_order]; tu = cand["tu"][final_order]; tv = cand["tv"][final_order]; n = cand["n"][final_order]
    M = X.shape[0]; device = X.device
    n_clust = int(labels.max().item()) + 1
    counts = torch.bincount(labels, minlength=n_clust).float()
    mu = torch.zeros(n_clust, 3, device=device)
    mu.scatter_add_(0, labels.unsqueeze(1).expand(-1, 3), X); mu = mu / counts.unsqueeze(1).clamp_min(1)
    tu_m = torch.zeros(n_clust, 3, device=device); tu_m.scatter_add_(0, labels.unsqueeze(1).expand(-1, 3), tu); tu_m = tu_m / counts.unsqueeze(1).clamp_min(1)
    tv_m = torch.zeros(n_clust, 3, device=device); tv_m.scatter_add_(0, labels.unsqueeze(1).expand(-1, 3), tv); tv_m = tv_m / counts.unsqueeze(1).clamp_min(1)
    n_c = torch.zeros(n_clust, 3, device=device); n_c.scatter_add_(0, labels.unsqueeze(1).expand(-1, 3), n); n_c = F.normalize(n_c / counts.unsqueeze(1).clamp_min(1), dim=1, eps=1e-8)
    tu_norm = tu.norm(dim=1); tv_norm = tv.norm(dim=1)
    su_c = 0.5 * (torch.zeros(n_clust, device=device).scatter_add_(0, labels, tu_norm) / counts.clamp_min(1)) * STRIDE
    sv_c = 0.5 * (torch.zeros(n_clust, device=device).scatter_add_(0, labels, tv_norm) / counts.clamp_min(1)) * STRIDE
    sn_c = 0.15 * torch.sqrt((su_c * sv_c).clamp_min(1e-12))
    Xc = X - mu[labels]
    cov = torch.zeros(n_clust, 3, 3, device=device)
    for a in range(3):
        for b in range(3):
            t = Xc[:, a] * Xc[:, b]
            s = torch.zeros(n_clust, device=device); s.scatter_add_(0, labels, t)
            cov[:, a, b] = s / counts.clamp_min(1)
    cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    eu = F.normalize(tu_m, dim=1, eps=1e-8)
    bad_u = eu.norm(dim=1) < 1e-6
    if bool(bad_u.any()):
        eu[bad_u] = torch.tensor([1.0, 0.0, 0.0], device=device)
    # Gram-Schmidt: orthonormalize ev against eu so R is a rotation (orthogonal)
    ev = tv_m - (tv_m * eu).sum(dim=1, keepdim=True) * eu
    ev = F.normalize(ev, dim=1, eps=1e-8)
    bad_v = ev.norm(dim=1) < 1e-6
    if bool(bad_v.any()):
        ev[bad_v] = torch.tensor([0.0, 1.0, 0.0], device=device)
    en = F.normalize(torch.cross(eu, ev, dim=1), dim=1, eps=1e-8)
    R = torch.stack([eu, ev, en], dim=2)
    sig = torch.stack([su_c, sv_c, sn_c], dim=1).clamp_min(1e-8)
    Sigma_t = R @ torch.diag_embed(sig * sig) @ R.transpose(1, 2)
    Sigma = Sigma_t + cov
    Sigma = torch.nan_to_num(Sigma, nan=0.0, posinf=0.0, neginf=0.0)
    Sigma = 0.5 * (Sigma + Sigma.transpose(1, 2))
    floor = 1e-8 * scene_scale * scene_scale
    evals, evecs = _batched_eigh(Sigma, floor)
    scaling = torch.sqrt(evals.clamp_min(floor))
    quat = _batch_mat_to_quat(evecs)
    return {"mu": mu, "scaling": scaling, "rotation": quat, "Sigma": Sigma,
            "n": n_c, "n_clust": n_clust, "counts": counts}


def _batch_mat_to_quat(R):
    """Vectorized 3x3 rotation-matrix -> quaternion [w,x,y,z]. R: [N,3,3]."""
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)
    # case trace>0
    s = torch.sqrt((tr + 1.0).clamp_min(1e-12)) * 2
    w = 0.25 * s
    x = (m21 - m12) / s.clamp_min(1e-12)
    y = (m02 - m20) / s.clamp_min(1e-12)
    z = (m10 - m01) / s.clamp_min(1e-12)
    q0 = torch.stack([w, x, y, z], dim=1)
    # case m00 max
    s1 = torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(1e-12)) * 2
    w1 = (m21 - m12) / s1.clamp_min(1e-12); x1 = 0.25 * s1
    y1 = (m01 + m10) / s1.clamp_min(1e-12); z1 = (m02 + m20) / s1.clamp_min(1e-12)
    q1 = torch.stack([w1, x1, y1, z1], dim=1)
    # case m11 max
    s2 = torch.sqrt((1.0 + m11 - m00 - m22).clamp_min(1e-12)) * 2
    w2 = (m02 - m20) / s2.clamp_min(1e-12); x2 = (m01 + m10) / s2.clamp_min(1e-12)
    y2 = 0.25 * s2; z2 = (m12 + m21) / s2.clamp_min(1e-12)
    q2 = torch.stack([w2, x2, y2, z2], dim=1)
    # case m22 max
    s3 = torch.sqrt((1.0 + m22 - m00 - m11).clamp_min(1e-12)) * 2
    w3 = (m10 - m01) / s3.clamp_min(1e-12); x3 = (m02 + m20) / s3.clamp_min(1e-12)
    y3 = (m12 + m21) / s3.clamp_min(1e-12); z3 = 0.25 * s3
    q3 = torch.stack([w3, x3, y3, z3], dim=1)
    cond0 = tr > 0
    cond1 = (~cond0) & (m00 > m11) & (m00 > m22)
    cond2 = (~cond0) & (m11 > m00) & (m11 > m22)
    cond3 = (~cond0) & (~cond0) & (~cond1) & (~cond2)
    q = torch.where(cond0.unsqueeze(1), q0, q)
    q = torch.where(cond1.unsqueeze(1), q1, q)
    q = torch.where(cond2.unsqueeze(1), q2, q)
    q = torch.where(cond3.unsqueeze(1), q3, q)
    return F.normalize(q, dim=1, eps=1e-12)
