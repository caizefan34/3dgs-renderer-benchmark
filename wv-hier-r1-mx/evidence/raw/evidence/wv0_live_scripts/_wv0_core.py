#!/usr/bin/env python3
"""WV-HIER-ORACLE0-MX core library (MX, gsplat 1.5.3 substrate).
State + camera loading, dense reference gradient, tile-granularity sparse
estimator with IPW, structural-work counters.
"""
import math, json, os, sys, time, hashlib
import numpy as np
import torch

REPO = "/home/liaoyuanjun/3dgs-renderer-benchmark"
RV1 = os.path.join(REPO, "results/reference_v1")
DATA = "/mnt/storage_pool/liaoyuanjun/wv0_data"

GRANULARITIES = [16, 8, 4, 1]
BUDGETS = [0.25, 0.40]
N_MASKS = 32
EPS2D = 0.3
TILE_BASE = 16

# state_id -> (ckpt_relpath under RV1, scene, iteration)
STATE_PATHS = {
 "room__iter5000":  ("room_30k/checkpoints/iter_5000.pt", "room", 5000),
 "room__iter15000": ("room_30k/checkpoints/iter_15000.pt", "room", 15000),
 "room__iter30000": ("room_30k/checkpoints/iter_30000.pt", "room", 30000),
 "garden__iter5000":  ("s22/garden/checkpoints/baseline_iter_5000.pt", "garden", 5000),
 "garden__iter15000": ("s22/garden/checkpoints/baseline_iter_15000.pt", "garden", 15000),
 "garden__iter30000": ("s22/garden/checkpoints/baseline_iter_30000.pt", "garden", 30000),
 "bicycle__iter5000":  ("s22/bicycle/checkpoints/baseline_iter_5000.pt", "bicycle", 5000),
 "bicycle__iter15000": ("s22/bicycle/checkpoints/baseline_iter_15000.pt", "bicycle", 15000),
 "bicycle__iter30000": ("s22/bicycle/checkpoints/baseline_iter_30000.pt", "bicycle", 30000),
}


def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()


def load_cameras(scene, resolution="540p", device="cuda"):
    """Load subset cameras from synced data dir. Returns list of dicts."""
    from PIL import Image
    cj = os.path.join(DATA, "official", "mipnerf360", scene, "cameras.json")
    cams = json.load(open(cj))
    H0 = cams[0]["height"]; W0 = cams[0]["width"]
    if resolution == "1080p": scale = 1080.0/H0
    elif resolution == "720p": scale = 720.0/H0
    elif resolution == "540p": scale = 540.0/H0
    else: scale = 1.0
    img_dir = os.path.join(DATA, "datasets", "mipnerf360", scene, "images")
    out = []
    for c in cams:
        W = int(round(c["width"]*scale)); H = int(round(c["height"]*scale))
        R = np.asarray(c["rotation"], dtype=np.float32)
        t = np.asarray(c["position"], dtype=np.float32)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3,:3] = R
        c2w[:3,3] = t
        w2c = np.linalg.inv(c2w)
        K = np.array([[c["fx"]*scale, 0, W/2.0],
                      [0, c["fy"]*scale, H/2.0],
                      [0, 0, 1.0]], dtype=np.float32)
        cam = {"w2c": torch.from_numpy(w2c).to(device),
               "K": torch.from_numpy(K).to(device),
               "W": W, "H": H, "img_name": c.get("img_name",""), "id": c.get("id",0)}
        # GT
        gt = None
        for ext in (".JPG",".jpg",".png",".jpeg"):
            p = os.path.join(img_dir, c["img_name"]+ext)
            if os.path.exists(p):
                im = Image.open(p).convert("RGB")
                im = im.resize((W,H), Image.BILINEAR)
                gt = torch.from_numpy(np.asarray(im)).float().permute(2,0,1).to(device)/255.0
                break
        if gt is not None:
            cam["gt"] = gt
        out.append(cam)
    return out


def load_state_params(ckpt_path, device="cuda"):
    o = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    def pick(*names):
        for n in names:
            if n in o: return o[n]
        return None
    raw = {"xyz": pick("xyz","means").float(),
           "shs": pick("shs").float(),
           "scaling": pick("scaling").float(),
           "rotation": pick("rotation").float(),
           "opacity": pick("opacity").float()}
    active_sh = int(o.get("active_sh_degree", o.get("sh_degree", 3)))
    N = raw["xyz"].shape[0]
    act = {"means": raw["xyz"].clone(),
           "quats": torch.nn.functional.normalize(raw["rotation"], dim=-1),
           "scales": torch.exp(raw["scaling"]),
           "opacities": (torch.sigmoid(raw["opacity"]).squeeze(-1) if raw["opacity"].dim()==2 and raw["opacity"].shape[-1]==1 else torch.sigmoid(raw["opacity"])),
           "colors": raw["shs"][:, :(active_sh+1)**2, :].contiguous(),
           "shs_raw": raw["shs"], "N": N, "active_sh": active_sh}
    return raw, act


def make_grad_enabled(act, device="cuda"):
    g = {}
    g["means"] = act["means"].clone().to(device).requires_grad_(True)
    g["quats"] = act["quats"].clone().to(device).requires_grad_(True)
    g["scales"] = act["scales"].clone().to(device).requires_grad_(True)
    g["opacities"] = act["opacities"].clone().to(device).requires_grad_(True)
    g["colors"] = act["colors"].clone().to(device).requires_grad_(True)
    return g


# parameter group mapping for gradient reporting
PARAM_GROUPS = {
    "means": "geometry", "quats": "geometry", "scales": "geometry",
    "opacities": "colors", "colors": "colors",
}
ALL_GROUPS = ["means","quats","scales","opacities","colors"]


def render_full(gparams, cam, sh_degree, tile_size=16, backgrounds=None, absgrad=False):
    from gsplat import rasterization
    viewmats = cam["w2c"].unsqueeze(0)
    Ks = cam["K"].unsqueeze(0)
    render, alpha, meta = rasterization(
        means=gparams["means"], quats=gparams["quats"], scales=gparams["scales"],
        opacities=gparams["opacities"], colors=gparams["colors"],
        viewmats=viewmats, Ks=Ks, width=cam["W"], height=cam["H"],
        tile_size=tile_size, packed=False, sh_degree=sh_degree,
        backgrounds=backgrounds, render_mode="RGB", eps2d=EPS2D,
        radius_clip=0.0, absgrad=absgrad)
    return render, alpha, meta
