"""M1 runner: quadrature vs canonical footprint-integral gradients.

Builds a controlled per-view test:
  * on-screen Gaussians with moderate, varied footprints,
  * a realistic residual/adjoint field from a perturbed-parameter render,
  * canonical first-order gradient (exact footprint sum via oracle moments),
  * K-point Gauss-Hermite quadrature gradient,
and reports cosine / rel-L2 / sign agreement per parameter family plus the
estimated Gaussian-pixel work reduction.
"""
from __future__ import annotations
import json, sys, time, math
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from momentgrad_gs.scene import load_scene, pick_views
from momentgrad_gs.oracle import project_gaussians, rasterize
from momentgrad_gs.m1_quadrature import (gauss_hermite_2d, first_order_gradients,
                                         quadrature_gradients)

BASE = Path(__file__).resolve().parents[1] / "data" / "datasets" / "mipnerf360"
OUT = Path(__file__).resolve().parent / "out"


def select_onscreen(init, cam, device, n=500, sigma_px=6.0, seed=0, rscale=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        proj = project_gaussians(init.means.to(device), init.quats.to(device),
            init.scales.to(device), torch.sigmoid(init.opacities.to(device)),
            cam.worldtocam.to(device), cam.K.to(device), colors_dc=init.colors.to(device), max_radius=256)
        mu = proj["mu2d"]; kp = proj["keep"]
        rad = math.ceil(3.0 * sigma_px) + 2
        on = kp & (mu[:, 0] > rad) & (mu[:, 0] < cam.width - rad) & (mu[:, 1] > rad) & (mu[:, 1] < cam.height - rad)
        idx = torch.nonzero(on, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return None
        sel = idx[torch.randperm(idx.numel(), generator=g)[:n]].to(device)
        # moderate isotropic world scale -> target screen sigma
        depth = proj["depth"][sel]
        fx = float(cam.K[0, 0])
        world_sigma = sigma_px * depth / fx  # sigma_px in render pixels
        scales = torch.log(world_sigma.clamp_min(1e-3))[:, None].repeat(1, 3)
        return sel, scales


def run_scene(scene, view_idx, n=500, sigma_px=15.0, ks=(2, 3, 4), device="cuda", rscale=0.25, tile=32):
    cams, init = load_scene(str(BASE / scene), split="train")
    cam = cams[view_idx]
    import copy
    cam = copy.copy(cam)
    cam.height = max(16, int(cam.height * rscale)); cam.width = max(16, int(cam.width * rscale))
    cam.K = cam.K * rscale
    sel_scales = select_onscreen(init, cam, device, n=n, sigma_px=sigma_px, rscale=rscale)
    if sel_scales is None:
        return None
    sel, scales = sel_scales
    wt = cam.worldtocam.to(device); K = cam.K.to(device)
    means = init.means[sel].to(device); quats = init.quats[sel].to(device)
    opac = init.opacities[sel].to(device); colors_dc = init.colors[sel].to(device)

    # canonical render (current params)
    proj = project_gaussians(means, quats, scales, torch.sigmoid(opac), wt, K, colors_dc=colors_dc)
    with torch.no_grad():
        res = rasterize(proj, cam.height, cam.width, pixel_loss_grad=None)
        I = res.image
    # perturbed render -> GT (realistic residual field)
    scales2 = scales * 1.15
    opac2 = opac + 0.5
    proj2 = project_gaussians(means, quats, scales2, torch.sigmoid(opac2), wt, K, colors_dc=colors_dc)
    with torch.no_grad():
        res2 = rasterize(proj2, cam.height, cam.width, pixel_loss_grad=None)
        GT = res2.image
    padj = (I - GT) / (cam.height * cam.width)   # dL/dI for L=0.5*sum(I-GT)^2/(HW)

    # canonical first-order gradients (exact footprint sum)
    resg = rasterize(proj, cam.height, cam.width, pixel_loss_grad=padj, tile=tile)
    proj["_moments"] = resg.moments
    can = first_order_gradients(proj)
    n_inter = resg.n_interactions
    n_vis = int((resg.moments["count"] > 0).sum())

    out = {"scene": scene, "view": view_idx, "n_gauss": int(sel.numel()),
           "n_visible": n_vis, "n_interactions": n_inter, "sigma_px": sigma_px,
           "configs": {}}
    for npt in ks:
        U, W = gauss_hermite_2d(npt)
        quad = quadrature_gradients(proj, padj, U, W)
        K = U.shape[0]
        cfg = {"K": K}
        for fam in ["mean2d", "opacity", "color"]:
            a = can[fam]; b = quad[fam]
            gcos = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
            # per-gaussian cosine (robust to outlier magnitude)
            a2 = a.reshape(a.shape[0], -1); b2 = b.reshape(b.shape[0], -1)
            cos_i = torch.nn.functional.cosine_similarity(a2, b2, dim=1)
            cos_i = cos_i[torch.isfinite(cos_i)]
            rel = (a - b).norm() / (a.norm() + 1e-12)
            sign = ((a.sign() * b.sign()) > 0).float().mean()
            cfg[fam] = {
                "global_cosine": gcos,
                "median_cos": float(cos_i.median()) if cos_i.numel() else float("nan"),
                "easy_frac": float((cos_i >= 0.99).float().mean()) if cos_i.numel() else float("nan"),
                "rel_l2": float(rel), "sign_agree": float(sign),
            }
        work = K * max(n_vis, 1)
        cfg["work_reduction"] = n_inter / work
        out["configs"][str(K)] = cfg
    return out


if __name__ == "__main__":
    scenes = ["room", "garden", "bicycle"]
    views = {"room": 0, "garden": 0, "bicycle": 0}
    results = []
    for sp in scenes:
        t0 = time.time()
        r = run_scene(sp, views[sp])
        if r is None:
            print(sp, "no on-screen gaussians"); continue
        results.append(r)
        print(f"\n=== {sp} (view {views[sp]}) n={r['n_gauss']} vis={r['n_visible']} interactions={r['n_interactions']} ===")
        for K, cfg in r["configs"].items():
            print(f"  K={K:>2}: mean2d gcos={cfg['mean2d']['global_cosine']:.4f} med={cfg['mean2d']['median_cos']:.4f} easy={cfg['mean2d']['easy_frac']:.3f} | "
                  f"opa gcos={cfg['opacity']['global_cosine']:.4f} med={cfg['opacity']['median_cos']:.4f} | "
                  f"color gcos={cfg['color']['global_cosine']:.4f} med={cfg['color']['median_cos']:.4f} | "
                  f"workRed={cfg['work_reduction']:.1f}x")
        print(f"  ({time.time()-t0:.1f}s)")
    OUT.mkdir(exist_ok=True)
    (OUT / "m1_quadrature.json").write_text(json.dumps(results, indent=2))
    print("\nwrote", OUT / "m1_quadrature.json")
