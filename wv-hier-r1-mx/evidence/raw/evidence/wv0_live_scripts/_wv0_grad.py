#!/usr/bin/env python3
"""WV0 gradient oracle: dense reference + tile-granularity sparse estimator.

Math (per spec 搂8, 搂11):
- Loss is L2 per pixel: L = sum_p w_p * ||r_p - t_p||^2, r_p in R^3.
  Per-pixel gradient contributions dL_p/dtheta are additive, so a uniform
  tile-subset with inverse-probability weighting gives an UNBIASED estimator:
      E[ghat] = sum_p E[w_p] * dL_p/dtheta = sum_p (pi * 1/pi) * dL_p = g*.
- Granularity G: image partitioned into GxG tiles. Each tile Bernoulli-included
  with prob pi (uniform). G=1 -> iid pixel sparsity (each pixel its own tile).
  Boundary tiles have fewer pixels but same inclusion prob -> per-pixel inclusion
  probability is exactly pi for all pixels -> unbiased.
- Budget p: target E[selected pixels]/total = p. With equal-size tiles, pi = p.
  We use pi = p exactly (the inclusion probability), and IPW weight = 1/pi.
"""
import math, numpy as np, torch


def tile_pixel_mask_fast(H, W, G, seed, pi, device="cuda"):
    """Vectorized tile-pixel mask. Tile-id grid cached per (H,W,G)."""
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    th = math.ceil(H/G); tw = math.ceil(W/G)
    inc = (torch.rand(th*tw, generator=g) < pi).float()  # [th*tw]
    cache_key = (H, W, G, device)
    ti = _TILEID_CACHE.get(cache_key)
    if ti is None or ti.shape != (H, W):
        ti = torch.arange(H).div(G, rounding_mode="floor").clamp(max=th-1)[:,None].expand(H, W)
        tj = torch.arange(W).div(G, rounding_mode="floor").clamp(max=tw-1)[None,:].expand(H, W)
        _TILEID_CACHE[cache_key] = ti*tw + tj
    mask = _TILEID_CACHE[cache_key]
    m = inc[mask] > 0.5
    return m.to(device), int(inc.sum())

_TILEID_CACHE = {}


def dense_reference_grad(gparams, cam, sh_degree):
    """Full-supervision L2 gradient. Returns dict group->grad tensor, and render."""
    from _wv0_core import render_full
    render, alpha, meta = render_full(gparams, cam, sh_degree)
    gt = cam["gt"]
    # render shape [1,H,W,3] -> [3,H,W]
    r = render[0].permute(2,0,1)
    diff = r - gt
    loss = (diff*diff).sum()
    grads = torch.autograd.grad(loss, [gparams[k] for k in
                     ["means","quats","scales","opacities","colors"]],
                     retain_graph=False, allow_unused=True)
    gdict = {k: (grads[i].detach().clone() if grads[i] is not None else torch.zeros_like(gparams[k]))
             for i,k in enumerate(["means","quats","scales","opacities","colors"])}
    return gdict, render, alpha, meta, float(loss)


def sparse_tile_grad(gparams, cam, sh_degree, G, seed, pi):
    """Tile-granularity sparse gradient with IPW. Returns gdict + work counters."""
    from _wv0_core import render_full
    render, alpha, meta = render_full(gparams, cam, sh_degree)
    gt = cam["gt"]
    r = render[0].permute(2,0,1)
    mask, n_tiles_sel = tile_pixel_mask_fast(cam["H"], cam["W"], G, seed, pi)
    w = torch.where(mask, torch.tensor(1.0/pi, device=r.device), torch.tensor(0.0, device=r.device))
    diff = r - gt
    loss = (diff*diff * w[None]).sum()
    grads = torch.autograd.grad(loss, [gparams[k] for k in
                     ["means","quats","scales","opacities","colors"]],
                     retain_graph=False, allow_unused=True)
    gdict = {k: (grads[i].detach().clone() if grads[i] is not None else torch.zeros_like(gparams[k]))
             for i,k in enumerate(["means","quats","scales","opacities","colors"])}
    # work counters
    n_sel_px = int(mask.sum())
    counters = {
        "selected_pixels": n_sel_px,
        "total_pixels": cam["H"]*cam["W"],
        "selected_tiles": n_tiles_sel,
        "n_cameras": 1,
    }
    return gdict, render, alpha, meta, counters, float(loss), mask
