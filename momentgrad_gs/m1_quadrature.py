"""M1 — Quadrature oracle for MomentGrad-GS.

Tests the first falsifiable gate: can a Gaussian's per-pixel backward field be
approximated by K Gaussian-adaptive quadrature samples instead of summing over
all covered pixels?

For each Gaussian the backward field is
    H(p) = dL/dI(p) . C_i . T_i(p)
and the first-order gradient terms are Gaussian-weighted moments of H:
    g_mean2d = Sigma^{-1} sum_p H(p) alpha(p) delta_p
    g_opac   = (1-opa) sum_p H(p) alpha(p)
    g_color  = sum_p dL/dI(p) T_i(p) alpha(p)
We approximate the footprint sum with K-point Gauss-Hermite quadrature against
the screen Gaussian, evaluating H (incl. exact transmittance) at the K points.

We compare against:
  * first-order exact gradient (oracle moments, exact T at footprint pixels)
  * full autograd gradient (renderer-exact, includes dT/dtheta background term)
and report cosine / rel-L2 / sign agreement per parameter family.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .oracle import project_gaussians, rasterize


SQRT2PI = math.sqrt(2 * math.pi)


def gauss_hermite_2d(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tensor-product Gauss-Hermite (probabilists') for N(0,I_2).

    Returns U [K,2] sample offsets (unit Gaussian space), W [K] weights summing to 1.
    Exact for total polynomial degree <= 2n-1. K = n^2.
    """
    x, w = np.polynomial.hermite_e.hermegauss(n)   # nodes/weights for exp(-x^2/2)
    w = w / SQRT2PI                                  # -> E[f] under N(0,1)
    Ux, Uy = np.meshgrid(x, x, indexing="ij")
    U = np.stack([Ux.reshape(-1), Uy.reshape(-1)], axis=1)
    Wx, Wy = np.meshgrid(w, w, indexing="ij")
    W = (Wx.reshape(-1) * Wy.reshape(-1))
    return torch.from_numpy(U).float(), torch.from_numpy(W).float()


def _chol(inv2d: torch.Tensor, cov2d: torch.Tensor) -> torch.Tensor:
    """L such that L L^T = cov2d (use Cholesky; cov2d passed in)."""
    L = torch.linalg.cholesky(cov2d)
    return L


def first_order_gradients(proj: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Exact first-order gradients from oracle moments (exact T at footprint px)."""
    inv2d = proj["inv2d"]; opa = proj["opacity"]
    mom = proj["_moments"]
    # g_mean2d [N,2] = inv2d @ m1  (m1 = sum H alpha delta)
    g_mean = torch.einsum("nij,nj->ni", inv2d, mom["m1"])
    g_opa = (1.0 - opa) * mom["m0"]
    g_color = mom["mcolor"]
    return {"mean2d": g_mean, "opacity": g_opa, "color": g_color}


def quadrature_gradients(proj: Dict[str, torch.Tensor], padj: torch.Tensor,
                         U: torch.Tensor, W: torch.Tensor, chunk: int = 128) -> Dict[str, torch.Tensor]:
    """K-point Gauss-Hermite quadrature estimate of the first-order gradients.

    padj: [H,W,3] image adjoint dL/dI.
    """
    device = proj["mu2d"].device
    mu2d = proj["mu2d"]; inv2d = proj["inv2d"]; cov2d = proj["cov2d"]
    opa = proj["opacity"]; colors = proj["colors"]; radius = proj["radius"]
    depth = proj["depth"]; keep = proj["keep"]
    idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
    U = U.to(device); W = W.to(device)
    K = U.shape[0]
    H, Wd = padj.shape[0], padj.shape[1]
    padj_flat = padj.reshape(H * Wd, 3)

    g_mean = torch.zeros(mu2d.shape[0], 2, device=device)
    g_opa = torch.zeros(mu2d.shape[0], device=device)
    g_color = torch.zeros(mu2d.shape[0], 3, device=device)

    # sample points in pixel space per gaussian: P_i = mu2d_i + (L_i U^T)^T
    L = torch.linalg.cholesky(cov2d)                       # [N,2,2]
    det_L = torch.det(L).clamp_min(1e-12)                  # sqrt(det Sigma)
    norm = (2.0 * math.pi * det_L)                         # integral-vs-expectation factor
    # offsets [N,K,2] = U @ L_i^T
    offsets = torch.einsum("kj,nij->nki", U, L)            # [N,K,2]
    P = mu2d[:, None, :] + offsets                          # [N,K,2]

    for s in range(0, idx.shape[0], chunk):
        sel = idx[s:s+chunk]
        Ps = P[sel]                                         # [b,K,2]
        # nearest-pixel adjoint lookup
        px = Ps[..., 0].round().long().clamp(0, Wd-1)
        py = Ps[..., 1].round().long().clamp(0, H-1)
        padj_k = padj_flat[py * Wd + px]                    # [b,K,3]
        # alpha of every gaussian j at these sample points: [b, m, K]
        d = Ps[:, None, :, :] - mu2d[None, :, None, :]      # [b,m,K,2]
        inb = (d.abs() <= radius[None, :, None, None]).all(-1)
        q = torch.einsum("bmki,mij,bmkj->bmk", d, inv2d, d)
        alpha_j = torch.where(inb, opa[None, :, None] * torch.exp(-0.5 * q), torch.zeros_like(q))
        # transmittance for gaussian i: product over front j (depth_j < depth_i)
        depth_i = depth[sel][:, None, None]                 # [b,1,1]
        front = (depth[None, :, None] < depth_i).float()    # [b,m,1]
        logT = (torch.log((1.0 - alpha_j).clamp_min(1e-10)) * front).sum(dim=1)  # [b,K]
        T = torch.exp(logT)                                 # [b,K]
        col_i = colors[sel]                                 # [b,3]
        # H(p_k) = (padj_k . col_i) * T   [b,K]
        Hc = torch.einsum("bkc,bc->bk", padj_k, col_i) * T  # [b,K]
        # g_mean2d_i = inv2d_i @ (opa_i * sum_k W_k H_k (L_i u_k))
        Lu = offsets[sel]                                   # [b,K,2] = L_i U_k
        integrand_mean = (W[None, :, None] * Hc[:, :, None] * Lu)  # [b,K,2]
        g_mean[sel] = torch.einsum("nij,bj->bi", inv2d[sel], (opa[sel][:, None] * integrand_mean.sum(dim=1)) * norm[sel][:, None])
        g_opa[sel] = (1.0 - opa[sel]) * opa[sel] * (W * Hc).sum(dim=1) * norm[sel]
        # g_color_i[c] = sum_k W_k padj_k[c] T_k opa_i
        g_color[sel] = opa[sel][:, None] * (W[None, :, None] * padj_k * T[:, :, None]).sum(dim=1) * norm[sel][:, None]

    return {"mean2d": g_mean, "opacity": g_opa, "color": g_color}


def full_autograd_gradients(means, quats, scales, opac, colors_dc, worldtocam, K,
                            H, W, GT, tile=16):
    """Renderer-exact gradients via autograd through the oracle forward."""
    params = {"means": means.detach().clone().requires_grad_(True),
              "quats": quats.detach().clone().requires_grad_(True),
              "scales": scales.detach().clone().requires_grad_(True),
              "opac": opac.detach().clone().requires_grad_(True),
              "colors_dc": colors_dc.detach().clone().requires_grad_(True)}
    proj = project_gaussians(params["means"], params["quats"], params["scales"],
                             torch.sigmoid(params["opac"]), worldtocam, K,
                             colors_dc=params["colors_dc"])
    res = rasterize(proj, H, W, pixel_loss_grad=None, tile=tile)
    image = res.image
    loss = 0.5 * ((image - GT) ** 2).sum() / (H * W)
    grads = torch.autograd.grad(loss, list(params.values()))
    return {k: g.detach() for k, g in zip(params.keys(), grads)}, proj, res
