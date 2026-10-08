"""Smooth-field control for M1: isolates whether quadrature failure is caused by
adjoint-field non-smoothness. A smooth low-frequency field should give near-perfect,
monotonic convergence; the real compositing field does not."""
from __future__ import annotations
import json, sys, copy
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from momentgrad_gs.scene import load_scene
from momentgrad_gs.oracle import project_gaussians, rasterize
from momentgrad_gs.m1_quadrature import gauss_hermite_2d, first_order_gradients, quadrature_gradients
from momentgrad_gs.m1_run import select_onscreen, BASE
OUT = Path(__file__).resolve().parent / "out"


def run(scene, view_idx=0, n=500, sigma_px=15.0, rscale=0.25, device="cuda"):
    torch.manual_seed(0)
    cams, init = load_scene(str(BASE / scene), split="train")
    cam = copy.copy(cams[view_idx])
    cam.height = max(16, int(cam.height * rscale)); cam.width = max(16, int(cam.width * rscale))
    cam.K = cam.K * rscale
    sel, scales = select_onscreen(init, cam, device, n=n, sigma_px=sigma_px, rscale=rscale)
    means = init.means[sel].to(device); quats = init.quats[sel].to(device)
    opac = init.opacities[sel].to(device); colors = init.colors[sel].to(device)
    wt = cam.worldtocam.to(device); K = cam.K.to(device)
    proj = project_gaussians(means, quats, scales, torch.sigmoid(opac), wt, K, colors_dc=colors, max_radius=256)
    H, W = cam.height, cam.width
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=device), torch.linspace(-1, 1, W, device=device), indexing="ij")
    padj = (0.02 + 0.01 * xx + 0.01 * yy)[..., None].expand(H, W, 3).contiguous()
    resg = rasterize(proj, H, W, pixel_loss_grad=padj, tile=32); proj["_moments"] = resg.moments
    can = first_order_gradients(proj)
    out = {"scene": scene, "field": "smooth_lowfreq", "configs": {}}
    for npt in [1, 2, 3, 4]:
        U, Wt = gauss_hermite_2d(npt); quad = quadrature_gradients(proj, padj, U, Wt); Kk = U.shape[0]
        cfg = {}
        for fam in ["mean2d", "opacity", "color"]:
            a = can[fam].reshape(can[fam].shape[0], -1); b = quad[fam].reshape(quad[fam].shape[0], -1)
            g = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
            m = torch.nn.functional.cosine_similarity(a, b, dim=1); m = float(m[torch.isfinite(m)].median())
            cfg[fam] = {"global_cosine": g, "median_cos": m}
        out["configs"][str(Kk)] = cfg
    return out


if __name__ == "__main__":
    res = [run(sp) for sp in ["room", "garden", "bicycle"]]
    OUT.mkdir(exist_ok=True)
    (OUT / "M1_SMOOTH_CONTROL.json").write_text(json.dumps(res, indent=2))
    for r in res:
        print(r["scene"], {K: round(v["mean2d"]["global_cosine"], 5) for K, v in r["configs"].items()})
    print("wrote", OUT / "M1_SMOOTH_CONTROL.json")
