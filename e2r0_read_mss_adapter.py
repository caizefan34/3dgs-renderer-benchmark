"""MSS0 native adapter (DSH production file).

Guard-first: require_guarded_process() MUST be called before importing torch/gsplat.
The native path shares gsplat's exact projection/SH/alpha compositor (no reimplementation
of covariance/SH math). This module exposes:

  * native_rasterize / native_forward -- gsplat.rasterization with the FROZEN
    render_with_meta parameters (eps2d=0.1, radius_clip=0.0, tile_size=16,
    near_plane=0.01, far_plane=1e10, absgrad=True, backgrounds=None,
    rasterize_mode="classic"). Returns the RAW (unclamped) composite + gsplat meta.
    The frozen trainer clamps rgb to [0,1] only for the loss; MSS keeps the raw
    composite as z and applies kappa=clamp(0,1) inside the image objective.
  * capture_rays -- reconstruct the EXACT per-ray accepted (alpha, color) sequence
    gsplat composites, using gsplat's own meta means2d/conics/opacities/isect for the
    accepted set + composite (matches gsplat to ~1e-7), and a differentiable
    re-projection (_project_and_sh) for the accepted gaussians so Jacobian gates can
    differentiate alpha_i / color_i w.r.t. the Gaussian parameters.

Camera->viewmat/K is replicated from cameras.json (benchmark_framework is absent in
this env); the convention is validated by render-vs-GT PSNR in run_e1.py.
"""
from __future__ import annotations
import math
import os
import json
from pathlib import Path


# ---- guard FIRST ----
from mx_guard import require_guarded_process  # noqa: E402  (stdlib only, safe pre-torch)
require_guarded_process()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import gsplat  # noqa: E402

CKPT_DEFAULT = "/mnt/storage_pool/liaoyuanjun/retiregs/RETIRE0_room_source/checkpoints/iter_15000.pt"
CAMS_DEFAULT = "/home/liaoyuanjun/3dgs-renderer-benchmark/data/official/mipnerf360/room/cameras.json"
IMAGES_DEFAULT = "/home/liaoyuanjun/3dgs-renderer-benchmark/data/datasets/mipnerf360/room/images"

RESOLUTIONS = {"720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160)}

# Frozen render_with_meta parameters (baseline/reference_v1/trainer.py).
NATIVE_RASTER_KW = dict(
    tile_size=16, packed=False, radius_clip=0.0, eps2d=0.1,
    near_plane=0.01, far_plane=1e10, render_mode="RGB", absgrad=True,
)

# Frozen gsplat composite constants (RasterizeToPixels3DGSFwd.cu + Common.h).
ALPHA_CAP = 0.999
ALPHA_THRESHOLD = 1.0 / 255.0
T_MIN = 1e-4


def load_cameras(cam_path, resolution="1080p"):
    """Replicate benchmark_framework.load_cameras_from_json + resize_cameras.

    cameras.json entries: fx, fy, width, height, position (camera center, world),
    rotation (3x3). Standard 3DGS/gsplat convention:
      c2w = [[R, C], [0, 1]]  (R = camera-to-world rotation)
      viewmatrix (w2c) = inv(c2w)
      K = [[fx,0,cx],[0,fy,cy],[0,0,1]], cx=width/2, cy=height/2
    resize scales intrinsics to target resolution (same as resize_cameras).
    """
    cams = json.loads(Path(cam_path).read_text())
    tw, th = RESOLUTIONS.get(resolution, (1920, 1080))
    out = []
    for c in cams:
        W, H = float(c["width"]), float(c["height"])
        sx, sy = tw / W, th / H
        fx = float(c["fx"]) * sx
        fy = float(c["fy"]) * sy
        cx = tw / 2.0
        cy = th / 2.0
        R = torch.tensor(c["rotation"], dtype=torch.float32)  # 3x3, camera-to-world
        C = torch.tensor(c["position"], dtype=torch.float32)  # camera center, world
        c2w = torch.eye(4, dtype=torch.float32)
        c2w[:3, :3] = R
        c2w[:3, 3] = C
        w2c = torch.linalg.inv(c2c2w := c2w)
        K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)
        out.append({
            "viewmatrix": w2c, "K": K, "image_width": tw, "image_height": th,
            "camera_center": C, "image_name": c.get("img_name"),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        })
    return out


def load_checkpoint(path=CKPT_DEFAULT):
    """Load room checkpoint raw params + apply GaussianModel activations."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    p = {
        "xyz": obj["xyz"].float(),
        "scaling": obj["scaling"].float(),
        "rotation": obj["rotation"].float(),
        "opacity": obj["opacity"].float(),
        "shs": obj["shs"].float(),
        "active_sh_degree": int(obj.get("active_sh_degree", 3)),
    }
    return p


def gaussian_params(p):
    """Apply activations to get native render inputs (matches GaussianModel getters)."""
    return {
        "means": p["xyz"],
        "quats": F.normalize(p["rotation"], dim=-1),
        "scales": torch.exp(p["scaling"]),
        "opacities": torch.sigmoid(p["opacity"]),
        "colors": p["shs"],
    }


def _sh_slice(shs, sh_degree):
    n_active = (int(sh_degree) + 1) ** 2
    return shs[..., :n_active, :]


def native_rasterize(gparams, cam, sh_degree=None, device="cuda"):
    """Frozen-native render. Returns (raw_rgb[H,W,3] UNCLAMPED, meta).

    Uses gsplat.rasterization with the exact render_with_meta parameters so the
    composite is identical to the frozen trainer (which clamps to [0,1] for the
    loss). MSS keeps the raw composite as z; kappa=clamp(0,1) is applied in the
    image objective, matching the frozen L(kappa(z), Y).
    """
    if sh_degree is None:
        sh_degree = 3
    dev = torch.device(device)
    gp = {k: v.to(dev) for k, v in gparams.items()}
    viewmats = cam["viewmatrix"].to(dev).unsqueeze(0)
    Ks = cam["K"].to(dev).unsqueeze(0)
    r, _alphas, meta = gsplat.rasterization(
        means=gp["means"], quats=gp["quats"], scales=gp["scales"],
        opacities=gp["opacities"], colors=_sh_slice(gp["colors"], sh_degree),
        viewmats=viewmats, Ks=Ks,
        width=cam["image_width"], height=cam["image_height"],
        sh_degree=sh_degree, **NATIVE_RASTER_KW,
    )
    return r[0], meta  # raw, unclamped, shape [H,W,3]


def native_forward(gparams, cam, sh_degree=None, device="cuda"):
    """Back-compatible wrapper returning the clamped image (frozen loss input)."""
    raw, _meta = native_rasterize(gparams, cam, sh_degree=sh_degree, device=device)
    return raw.clamp(0.0, 1.0)


def _sh_colors(gparams, cam, sh_degree, device, radii_mask=None):
    """Differentiable SH colors = clamp_min(SH(dirs)+0.5, 0), matching gsplat."""
    dev = torch.device(device)
    means = gparams["means"].to(dev)
    viewmats = cam["viewmatrix"].to(dev).unsqueeze(0)
    campos = torch.inverse(viewmats)[..., :3, 3]  # [1,3]
    dirs = means.unsqueeze(0) - campos.unsqueeze(1)  # [1,N,3]
    shs = _sh_slice(gparams["colors"], sh_degree).to(dev).unsqueeze(0)  # [1,N,K,3]
    batch_dims = dirs.shape[:-1]
    if radii_mask is None:
        radii_mask = torch.ones(batch_dims, dtype=torch.bool, device=dev)
    masks = radii_mask.bool().reshape(batch_dims)
    colors = gsplat.spherical_harmonics(sh_degree, dirs, shs, masks=masks)
    return torch.clamp_min(colors + 0.5, 0.0)[0]  # [N,3]


def _project_and_sh(gparams, cam, sh_degree, device):
    """Differentiable gsplat projection (source-identical to rasterization).

    Mirrors gsplat rasterization(): rasterize_mode="classic" -> calc_compensations=False
    (no opacity compensation); colors = clamp_min(SH(dirs)+0.5, 0). The opacities
    argument is intentionally NOT passed to fully_fused_projection: means2d/conics do
    not depend on it, and omitting it keeps the projection bit-deterministic. Returns
    per-gaussian tensors on device, differentiable w.r.t. gparams leaves.
    """
    dev = torch.device(device)
    means = gparams["means"].to(dev)
    quats = gparams["quats"].to(dev)
    scales = gparams["scales"].to(dev)
    viewmats = cam["viewmatrix"].to(dev).unsqueeze(0)
    Ks = cam["K"].to(dev).unsqueeze(0)
    width, height = cam["image_width"], cam["image_height"]
    radii, means2d, depths, conics, _comp = gsplat.fully_fused_projection(
        means, None, quats, scales, viewmats, Ks, width, height,
        eps2d=NATIVE_RASTER_KW["eps2d"], near_plane=NATIVE_RASTER_KW["near_plane"],
        far_plane=NATIVE_RASTER_KW["far_plane"], radius_clip=NATIVE_RASTER_KW["radius_clip"],
        calc_compensations=False,
    )
    colors = _sh_colors(gparams, cam, sh_degree, device, radii_mask=(radii > 0).all(dim=-1))
    return {
        "means2d": means2d[0], "conics": conics[0], "depths": depths[0],
        "radii": radii[0], "opac": gparams["opacities"].to(dev), "colors": colors, "means": means,
    }


def _alpha_sigma(means2d, conics, opac, px, py, gids):
    """alpha_i for accepted gaussians at pixel center (px,py). Differentiable."""
    m = means2d[gids]
    c = conics[gids]
    dx = px - m[:, 0]
    dy = py - m[:, 1]
    sigma = 0.5 * (c[:, 0] * dx * dx + c[:, 2] * dy * dy) + c[:, 1] * dx * dy
    alpha = torch.clamp_max(opac[gids] * torch.exp(-sigma), ALPHA_CAP)
    return alpha, sigma


def capture_rays(gparams, cam, sh_degree, pixels, device="cuda", light=False):
    """Reconstruct the exact per-ray accepted (alpha, color) sequence gsplat composites.

    pixels: LongTensor [P,2] of (x, y) integer pixel coords.
    Returns dict:
      gids   : LongTensor [P,K]  accepted gaussian indices per ray (padded, -1 pad)
      K      : LongTensor [P]    accepted length per ray (after threshold + early-stop)
      alpha  : [P,K]             accepted alpha_i (differentiable, fixed accepted set)
      color  : [P,K,3]           accepted SH color c_i (differentiable)
      u1     : [P,3]             composite from gsplat META values (== native raw pixel)
      ref    : [P,3]             native raw pixel value (ground truth for u1)
      Kmax   : int
    The accepted set + u1 are taken from gsplat's own rasterization meta (exact, ~1e-7);
    alpha/color are recomputed differentiably via _project_and_sh for the SAME accepted
    set so Jacobian gates can differentiate them w.r.t. the Gaussian parameters.
    """
    dev = torch.device(device)
    width, height = cam["image_width"], cam["image_height"]
    tile_size = NATIVE_RASTER_KW["tile_size"]
    pixels = torch.as_tensor(pixels, dtype=torch.long, device=dev)
    if pixels.ndim == 1:
        pixels = pixels.unsqueeze(0)
    P = pixels.shape[0]

    with torch.no_grad():
        ref_img, meta = native_rasterize(gparams, cam, sh_degree=sh_degree, device=device)
        ref_img = ref_img.detach()
        m2d_m = meta["means2d"][0].detach()
        con_m = meta["conics"][0].detach()
        op_m = meta["opacities"][0].detach()
        flat = meta["flatten_ids"].detach()
        tile_key = meta["isect_ids"].detach() >> 32
        tw = int(meta["tile_width"])
        col_m = _sh_colors(gparams, cam, sh_degree, device,
                           radii_mask=(meta["radii"][0] > 0).all(dim=-1)).detach()
        px = pixels[:, 0]
        py = pixels[:, 1]
        tid = (py // tile_size) * tw + (px // tile_size)
        lo = torch.searchsorted(tile_key, tid)
        hi = torch.searchsorted(tile_key, tid + 1)
        gids_list = []
        branch_list = []
        Klen = torch.zeros(P, dtype=torch.long, device=dev)
        u1 = torch.zeros(P, 3, device=dev)
        for p in range(P):
            l, h = int(lo[p]), int(hi[p])
            g = flat[l:h]
            cx = float(px[p]) + 0.5
            cy = float(py[p]) + 0.5
            branch_p = False
            if g.numel() == 0:
                gids_list.append(g)
                branch_list.append(False)
                continue
            m = m2d_m[g]; c = con_m[g]
            dx = cx - m[:, 0]; dy = cy - m[:, 1]
            sigma = 0.5 * (c[:, 0] * dx * dx + c[:, 2] * dy * dy) + c[:, 1] * dx * dy
            a = torch.clamp_max(op_m[g] * torch.exp(-sigma), ALPHA_CAP)
            T = 1.0
            acc = []
            for i in range(a.numel()):
                if float(sigma[i]) < 0.0 or float(a[i]) < ALPHA_THRESHOLD:
                    continue
                if float(a[i]) >= ALPHA_CAP - 1e-7:
                    branch_p = True
                next_T = T * (1.0 - float(a[i]))
                if next_T <= T_MIN:
                    branch_p = True
                    break
                acc.append(i)
                T = next_T
            if not acc:
                gids_list.append(g[:0])
                branch_list.append(branch_p)
                continue
            idx = torch.tensor(acc, device=dev)
            gids_list.append(g[idx])
            Klen[p] = len(acc)
            branch_list.append(branch_p)
            gg = g[idx]
            mm = m2d_m[gg]; cc = con_m[gg]
            ddx = cx - mm[:, 0]; ddy = cy - mm[:, 1]
            ss = 0.5 * (cc[:, 0] * ddx * ddx + cc[:, 2] * ddy * ddy) + cc[:, 1] * ddx * ddy
            aa = torch.clamp_max(op_m[gg] * torch.exp(-ss), ALPHA_CAP)
            TT = 1.0
            for i in range(aa.numel()):
                u1[p] = u1[p] + TT * aa[i] * col_m[gg][i]
                TT = TT * (1.0 - aa[i])
        Kmax = int(Klen.max().item()) if P else 0

    # alpha/color/gids for the FIXED accepted sets. light=True reuses gsplat meta
    # (no full differentiable projection of all Gaussians); used by the macro theta
    # step which recomputes per-Gaussian Jacobians independently.
    alpha = torch.zeros(P, Kmax, device=dev)
    color = torch.zeros(P, Kmax, 3, device=dev)
    gids_pad = torch.full((P, Kmax), -1, dtype=torch.long, device=dev)
    if light:
        with torch.no_grad():
            for p in range(P):
                g = gids_list[p]; k = int(g.numel())
                if k == 0:
                    continue
                gids_pad[p, :k] = g
                cx = float(px[p]) + 0.5; cy = float(py[p]) + 0.5
                mm = m2d_m[g]; cc = con_m[g]
                ddx = cx - mm[:, 0]; ddy = cy - mm[:, 1]
                ss = 0.5 * (cc[:, 0] * ddx * ddx + cc[:, 2] * ddy * ddy) + cc[:, 1] * ddx * ddy
                aa = torch.clamp_max(op_m[g] * torch.exp(-ss), ALPHA_CAP)
                alpha[p, :k] = aa
                color[p, :k] = col_m[g]
    else:
        proj = _project_and_sh(gparams, cam, sh_degree, device)
        for p in range(P):
            g = gids_list[p]
            k = int(g.numel())
            if k == 0:
                continue
            gids_pad[p, :k] = g
            cx = float(pixels[p, 0]) + 0.5
            cy = float(pixels[p, 1]) + 0.5
            a, _sig = _alpha_sigma(proj["means2d"], proj["conics"], proj["opac"], cx, cy, g)
            alpha[p, :k] = a
            color[p, :k] = proj["colors"][g]

    return {"gids": gids_pad, "K": Klen, "alpha": alpha, "color": color, "u1": u1,
            "ref": ref_img[pixels[:, 1], pixels[:, 0]], "Kmax": Kmax,
            "branch": torch.tensor(branch_list, dtype=torch.bool, device=dev)}


def load_scene(ckpt=CKPT_DEFAULT, cam_path=CAMS_DEFAULT, resolution="1080p"):
    p = load_checkpoint(ckpt)
    cams = load_cameras(cam_path, resolution)
    return {"params": p, "gparams": gaussian_params(p), "cameras": cams, "resolution": resolution}
