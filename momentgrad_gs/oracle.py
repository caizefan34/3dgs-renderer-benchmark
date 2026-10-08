"""Pure-PyTorch oracle rasterizer for the MomentGrad-GS lane.

gsplat 1.4.0's fused CUDA backend cannot JIT-compile on this machine (no MSVC),
so we implement a self-contained, correct 3DGS rasterizer. It exposes the
per-Gaussian screen-space kernel and the compositing transmittance/adjoint fields
that the compressed-backward hypothesis is about, and it is the M0 canonical
reference for every later stage.

The compositing is per-pixel front-to-back alpha blending. Each pixel receives
contributions exactly from the Gaussians whose 3-sigma screen bbox covers it,
all of which are assigned to that pixel's tile, so per-tile compositing is exact
and self-contained (no cross-tile transmittance dependency).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import torch

SH_C0 = 0.28209479177387814


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-12)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w),
        2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def project_gaussians(means, quats, scales, opacities, worldtocam, K,
                      near=0.01, eps2d=0.3, colors_dc=None, max_radius=1024):
    device = means.device
    N = means.shape[0]
    Rwg = quat_to_rotmat(quats)
    var = torch.exp(2.0 * scales)                 # variances [N,3]
    Rs = Rwg * var[:, None, :]                     # R diag(s^2): scale columns
    cov_w = Rs @ Rwg.transpose(-1, -2)             # R diag(s^2) R^T
    Rcw = worldtocam[:3, :3].to(device)
    tcw = worldtocam[:3, 3].to(device)
    t = (Rcw @ means.unsqueeze(-1)).squeeze(-1) + tcw
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z = t[:, 2:3].clamp(min=near)
    J = torch.zeros(N, 2, 3, device=device)
    J[:, 0, 0] = fx / z[:, 0]
    J[:, 0, 2] = -fx * t[:, 0] / (z[:, 0] ** 2)
    J[:, 1, 1] = fy / z[:, 0]
    J[:, 1, 2] = -fy * t[:, 1] / (z[:, 0] ** 2)
    M = J @ Rcw
    cov_c = M @ cov_w @ M.transpose(-1, -2)
    eye2 = torch.eye(2, device=device)[None]
    cov2d = cov_c + eps2d * eye2
    a = cov2d[:, 0, 0]; b = cov2d[:, 0, 1]; d = cov2d[:, 1, 1]
    det = (a * d - b * b).clamp(min=1e-10)
    inv2d = torch.stack([d, -b, -b, a], dim=-1).reshape(N, 2, 2) / det[:, None, None]
    mu2d = torch.stack([fx * t[:, 0] / z[:, 0] + cx, fy * t[:, 1] / z[:, 0] + cy], dim=-1)
    tr = a + d
    disc = torch.clamp((tr * tr) / 4 - det, min=0.0)
    lam_max = tr / 2 + torch.sqrt(disc)
    radius = torch.ceil(3.0 * torch.sqrt(lam_max)).long().clamp(min=1, max=max_radius)
    if colors_dc is None:
        colors = torch.full((N, 3), 0.5, device=device)
    else:
        colors = (SH_C0 * colors_dc + 0.5).clamp(min=0.0)
    keep = (t[:, 2] > near) & (radius > 0) & (radius < max_radius)
    return {"mu2d": mu2d, "cov2d": cov2d, "inv2d": inv2d, "det": det,
            "radius": radius, "depth": t[:, 2], "opacity": opacities,
            "colors": colors, "keep": keep, "t": t}


@dataclass
class RasterResult:
    image: torch.Tensor
    alpha: torch.Tensor
    moments: dict
    n_interactions: int


def rasterize(proj, H, W, pixel_loss_grad: Optional[torch.Tensor] = None, tile: int = 16):
    device = proj["mu2d"].device
    mu2d = proj["mu2d"]; inv2d = proj["inv2d"]; colors = proj["colors"]
    opa = proj["opacity"]; radius = proj["radius"]; depth = proj["depth"]
    keep = proj["keep"]
    n_full = mu2d.shape[0]
    idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return RasterResult(torch.zeros(H, W, 3, device=device),
                            torch.zeros(H, W, device=device), {}, 0)
    mu2d = mu2d[idx]; inv2d = inv2d[idx]; colors = colors[idx]
    opa = opa[idx]; radius = radius[idx]; depth = depth[idx]
    ng = idx.numel()
    order = torch.argsort(depth)
    mu2d = mu2d[order]; inv2d = inv2d[order]; colors = colors[order]
    opa = opa[order]; radius = radius[order]
    orig_idx = idx[order]

    image = torch.zeros(H, W, 3, device=device)
    alpha_img = torch.zeros(H, W, device=device)
    mom = {"m0": torch.zeros(ng, device=device), "m1": torch.zeros(ng, 2, device=device),
           "m2": torch.zeros(ng, 2, 2, device=device), "count": torch.zeros(ng, device=device),
           "mcolor": torch.zeros(ng, 3, device=device)}
    n_interactions = 0
    has_grad = pixel_loss_grad is not None
    padj = pixel_loss_grad.to(device) if has_grad else None
    ntx = (W + tile - 1) // tile; nty = (H + tile - 1) // tile
    tiny = 1e-10

    for ty in range(nty):
        for tx in range(ntx):
            x0, y0 = tx * tile, ty * tile
            x1, y1 = min(W, x0 + tile), min(H, y0 + tile)
            pw, ph = x1 - x0, y1 - y0
            if pw <= 0 or ph <= 0:
                continue
            gy, gx = torch.meshgrid(torch.arange(y0, y1, device=device),
                                    torch.arange(x0, x1, device=device), indexing="ij")
            px = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).float()
            P = px.shape[0]
            in_tile = (mu2d[:, 0] + radius > x0) & (mu2d[:, 0] - radius < x1) & \
                      (mu2d[:, 1] + radius > y0) & (mu2d[:, 1] - radius < y1)
            gids = torch.nonzero(in_tile, as_tuple=False).squeeze(-1)
            if gids.numel() == 0:
                continue
            dx = px[:, None, :] - mu2d[gids][None, :, :]
            r = radius[gids][None, :, None]
            inb = (dx.abs() <= r).all(dim=-1)
            inv_g = inv2d[gids]
            q = torch.einsum("pmi,mij,pmj->pm", dx, inv_g, dx)
            alpha_g = torch.where(inb, opa[gids][None, :] * torch.exp(-0.5 * q), torch.zeros_like(opa[gids][None, :]))
            log_om = torch.log((1.0 - alpha_g).clamp(min=tiny))
            Texcl = torch.exp(torch.cumsum(log_om, dim=1) - log_om)
            col_g = colors[gids]  # [m,3]
            weight = Texcl * alpha_g
            image[y0:y1, x0:x1] = (weight[:, :, None] * col_g[None, :, :]).sum(dim=1).reshape(ph, pw, 3)
            alpha_img[y0:y1, x0:x1] = weight.sum(dim=1).reshape(ph, pw)
            n_interactions += int(inb.sum().item())
            if has_grad:
                padj_3 = padj[y0:y1, x0:x1].reshape(-1, 3)
                Hc = torch.einsum("pc,pmc->pm", padj_3, col_g[None, :, :].expand(P, -1, -1))
                fa = (Hc * Texcl) * alpha_g
                mcol = torch.einsum("pc,pm->mc", padj_3, weight)  # [m,3]
                mom["mcolor"].index_add_(0, gids, mcol)
                mom["m0"].index_add_(0, gids, fa.sum(dim=0))
                mom["m1"].index_add_(0, gids, torch.einsum("pm,pmi->mi", fa, dx))
                mom["m2"].index_add_(0, gids, torch.einsum("pm,pmi,pmj->mij", fa, dx, dx))
                mom["count"].index_add_(0, gids, inb.sum(dim=0).float())

    mom_full = {}
    mom_full["m0"] = torch.zeros(n_full, device=device)
    mom_full["m0"][orig_idx] = mom["m0"]
    mom_full["count"] = torch.zeros(n_full, device=device)
    mom_full["count"][orig_idx] = mom["count"]
    mom_full["mcolor"] = torch.zeros(n_full, 3, device=device)
    mom_full["mcolor"][orig_idx] = mom["mcolor"]
    for k, shp in [("m1", (2,)), ("m2", (2, 2))]:
        t = torch.zeros(n_full, *shp, device=device)
        t[orig_idx] = mom[k]
        mom_full[k] = t
    return RasterResult(image, alpha_img, mom_full, n_interactions)


def _add(dst, gids, src, ng):
    out = torch.zeros_like(dst)
    out.index_add_(0, gids, src)
    return out
