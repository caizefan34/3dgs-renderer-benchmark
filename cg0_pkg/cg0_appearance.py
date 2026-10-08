#!/usr/bin/env python3
"""CG0 appearance compiler (section 7) + analytic opacity/calibration (section 8).
Forward-only, no gradients. Vectorized degree assignment + batched solve."""
import math
import torch
import torch.nn.functional as F
from cg0_prereg import (sh_basis, n_coeffs_for_degree, C0, SH_DC_OFFSET)

DEPTH_CONSISTENCY = 0.01
KMAX = n_coeffs_for_degree(3)


def degree_vec(n_obs):
    """Vectorized frozen SH degree by observation count."""
    d = torch.zeros(n_obs.shape, dtype=torch.long, device=n_obs.device)
    d[n_obs >= 4] = 1
    d[n_obs >= 9] = 2
    d[n_obs >= 20] = 3
    return d


def _project(cam, X):
    R = cam.viewmatrix[:3, :3]; t = cam.viewmatrix[:3, 3]
    cam_pt = X @ R.t() + t.unsqueeze(0)
    z = cam_pt[:, 2]
    K = cam.K
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * cam_pt[:, 0] / z.clamp_min(1e-12) + cx
    v = fy * cam_pt[:, 1] / z.clamp_min(1e-12) + cy
    W, H = cam.image_width, cam.image_height
    in_frame = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1) & (z > 1e-6)
    return u, v, z, in_frame


def compile_sh(cand, clusters, cams, compile_idx, depth_maps, gt_getter, device,
               chunk=200000):
    mu = clusters["mu"]; nrm = clusters["n"]; N = mu.shape[0]
    Ata = torch.zeros(N, KMAX, KMAX, device=device)
    Atb = torch.zeros(N, KMAX, 3, device=device)
    traceYtWY = torch.zeros(N, device=device)
    n_obs = torch.zeros(N, device=device)
    mean_rgb = cand["rgb"].mean(dim=0) if cand["rgb"].numel() else torch.zeros(3, device=device)

    for ci, cam, D in zip(compile_idx, cams, depth_maps):
        u, v, z, in_frame = _project(cam, mu)
        if int(in_frame.sum()) == 0:
            continue
        cl_all = in_frame.nonzero(as_tuple=False).reshape(-1)
        ui = u[cl_all].round().long().clamp(0, cam.image_width - 1)
        vi = v[cl_all].round().long().clamp(0, cam.image_height - 1)
        zi = z[cl_all]
        Dpix = D[vi, ui]
        consist = (Dpix - zi).abs() / zi.clamp_min(1e-12) <= DEPTH_CONSISTENCY
        if int(consist.sum()) == 0:
            continue
        sel = consist.nonzero(as_tuple=False).reshape(-1)
        cl_idx = cl_all[sel]
        pts = mu[cl_idx]
        vdir = F.normalize(pts - cam.camera_center.unsqueeze(0), dim=1, eps=1e-12)
        Y = sh_basis(vdir)
        gt = gt_getter(ci)
        rgb = gt[vi[sel], ui[sel]]
        w = (nrm[cl_idx] * (-vdir)).sum(dim=1).clamp_min(0) ** 2
        Yw = Y * w.unsqueeze(1)
        tgt = rgb - SH_DC_OFFSET
        for c0 in range(0, sel.shape[0], chunk):
            c1 = min(c0 + chunk, sel.shape[0])
            Yc = Y[c0:c1]; Ywc = Yw[c0:c1]; tc = tgt[c0:c1]; cc = cl_idx[c0:c1]
            gram = torch.einsum("pk,pl->pkl", Ywc, Yc)
            Ata.index_add_(0, cc, gram)
            atb = torch.einsum("pk,pc->pkc", Ywc, tc)
            Atb.index_add_(0, cc, atb)
            traceYtWY.index_add_(0, cc, (Ywc * Yc).sum(1))
            n_obs.index_add_(0, cc, torch.ones(cc.shape[0], device=device))

    degree = degree_vec(n_obs)
    shs = torch.zeros(N, KMAX, 3, device=device)
    cond = torch.full((N,), float("nan"), device=device)
    fallback = torch.zeros(N, dtype=torch.long, device=device)
    eye_cache = {K: torch.eye(K, device=device) for K in (1, 4, 9, 16)}

    deg_to_K = {3: 16, 2: 9, 1: 4, 0: 1}
    for d, K in deg_to_K.items():
        idx = (degree == d).nonzero(as_tuple=False).reshape(-1)
        if idx.shape[0] == 0:
            continue
        Ai = Ata[idx][:, :K, :K]
        bi = Atb[idx][:, :K, :]
        lam = 1e-4 * traceYtWY[idx] / max(K, 1) + 1e-8
        Areg = Ai + lam.unsqueeze(-1).unsqueeze(-1) * eye_cache[K].unsqueeze(0)
        try:
            L = torch.linalg.cholesky(Areg)
            sol = torch.cholesky_solve(bi, L)
            shs[idx, :K, :] = sol
            cond[idx] = torch.linalg.cond(Areg)
        except Exception:
            sol = torch.linalg.lstsq(Ai, bi).solution
            shs[idx, :K, :] = sol
            fallback[idx] = 1
    # unobserved clusters (n_obs==0): global-mean DC fallback (no data to solve)
    d0 = (n_obs == 0).nonzero(as_tuple=False).reshape(-1)
    if d0.shape[0] > 0:
        shs[d0, 0, :] = (mean_rgb - SH_DC_OFFSET) / C0
        fallback[d0] = 1
    dc_rgb = (SH_DC_OFFSET + C0 * shs[:, 0, :]).clamp(0.0, 1.0)
    shs[:, 0, :] = (dc_rgb - SH_DC_OFFSET) / C0
    return {"shs": shs, "n_obs": n_obs, "degree": degree, "cond": cond,
            "fallback": fallback, "N": N}


def analytic_opacity(n_obs):
    a = 1.0 - torch.exp(-n_obs.clamp(max=8.0) / 4.0)
    a = a.clamp(1e-4, 1 - 1e-4)
    return torch.log(a / (1.0 - a))


CALIB_GRID_S = (0.75, 1.00, 1.25, 1.50)
CALIB_GRID_B = (-2.0, -1.0, 0.0, 1.0, 2.0)


def apply_calibration(clusters, opacity_logit, s, b):
    return clusters["scaling"] * math.sqrt(s), clusters["rotation"], opacity_logit + b
