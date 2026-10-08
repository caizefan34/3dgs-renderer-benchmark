#!/usr/bin/env python
"""Bounded per-step timing probe for E1 feasibility. NOT a scientific result.
Self-contained: inlines helpers (build_optimizer/position_lr/eta_at) verbatim from
run_e1.py and uses the CHECKPOINTED comax twin (comax_render_eta_ckpt) exactly as
run_e1_ckpt.py monkeypatches it. Measures wall per training step for baseline (gsplat)
vs comax (ckpt reference) on the frozen room checkpoint at 1080p."""
import os, sys, time, json
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/comax0_mx/package/COMAX0_MX_DSH/scripts")
sys.path.insert(0, "/mnt/storage_pool/liaoyuanjun/comax0_mx/package/COMAX0_MX_DSH/scripts/comax_gs")
import torch, numpy as np
import torch.nn.functional as F
import common
from comax_eta_ckpt import comax_render_eta_ckpt


def build_optimizer(model, lr):
    return torch.optim.Adam([
        {"params": [model._xyz], "lr": lr["position"], "name": "xyz"},
        {"params": [model._shs], "lr": lr["feature"], "name": "feature"},
        {"params": [model._opacity], "lr": lr["opacity"], "name": "opacity"},
        {"params": [model._scaling], "lr": lr["scaling"], "name": "scaling"},
        {"params": [model._rotation], "lr": lr["rotation"], "name": "rotation"},
    ])


def position_lr(step, cfg):
    init, fin, mx = cfg["position_lr_init"], cfg["position_lr_final"], cfg["position_lr_max_steps"]
    return init * (fin / init) ** (min(step, mx) / mx)


def eta_at(step, ramp_steps):
    if ramp_steps <= 0:
        return 1.0
    return min(1.0, step / float(ramp_steps))


def render_comax_ckpt(model, cam, groups, eta=1.0, background=None, tile=16):
    if background is None:
        background = torch.zeros(3, device=model.get_xyz.device)
    cam_center = cam.viewmatrix.inverse()[:3, 3]
    dirs = F.normalize(model.get_xyz - cam_center, dim=-1)
    colors = (common.eval_sh(model.get_sh_degree, dirs, model.get_features) + 0.5).clamp(0, 1).contiguous()
    from gsplat.rendering import fully_fused_projection
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means=model.get_xyz, covars=None, quats=model.get_rotation, scales=model.get_scaling,
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height, eps2d=0.3, near_plane=0.01,
        far_plane=1e10, radius_clip=0.0, packed=False, opacities=model.get_opacity)
    chunk_tiles = int(os.environ.get("COMAX0_CKPT_CHUNK_TILES", "8"))
    out = comax_render_eta_ckpt(means2d[0].contiguous(), conics[0].contiguous(), depths[0].contiguous(),
                                radii[0].contiguous(), model.get_opacity.contiguous(), colors,
                                groups.contiguous(), cam.image_width, cam.image_height, eta=eta,
                                tile=tile, backgrounds=background, chunk_tiles=chunk_tiles)
    return out["rgb"]


dev = "cuda"
scene, seed, ramp_steps = "room", 17, 200
cfg = {"position": 0.00016, "feature": 0.0025, "opacity": 0.025,
       "scaling": 0.005, "rotation": 0.001, "position_lr_init": 0.00016,
       "position_lr_final": 0.0000016, "position_lr_max_steps": 30000,
       "lambda_dssim": 0.2}
_CFG = json.load(open("/mnt/storage_pool/liaoyuanjun/comax0_mx/package/COMAX0_MX_DSH/config/mx_target.json"))
ck = common.load_checkpoint(_CFG["checkpoint"], device=dev)
groups = common.load_groups(scene, seed, device=dev)
sys.path.insert(0, common.REPO)
from scripts.epic05.phase7.dataset import GTDataset
from scripts.epic05.phase7.loss import combined_loss
ds = GTDataset(scene=scene, repo_root=common.REPO, resolution="1080p", device=dev, background="black")
train_idx, test_idx = common.frozen_split(len(ds))
rng = np.random.RandomState(seed)
order = list(train_idx)
rng.shuffle(order)
bg = torch.zeros(3, device=dev)
lam = cfg["lambda_dssim"]
NSTEP = 6
results = {}
for branch in ("baseline", "comax"):
    torch.manual_seed(seed); np.random.seed(seed)
    m = common.params_from_ckpt(ck, device=dev)
    for _p in (m._xyz, m._shs, m._opacity, m._scaling, m._rotation):
        _p.requires_grad_(True)
    opt = build_optimizer(m, cfg)
    times = []
    for step in range(NSTEP):
        cam_idx = order[step % len(order)]
        cam, gt = ds.get_item(cam_idx)
        abs_iter = 15000 + step
        for g in opt.param_groups:
            if g.get("name") == "xyz":
                g["lr"] = position_lr(abs_iter, cfg)
        eta = eta_at(step, ramp_steps) if branch == "comax" else 1.0
        if branch == "baseline":
            pred = common.render_baseline(m, cam, background=bg)
        else:
            pred = render_comax_ckpt(m, cam, groups, eta=eta, background=bg)
        loss = combined_loss(pred, gt, lambda_dssim=lam)["loss"]
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.time()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)
    torch.cuda.synchronize()
    results[branch] = {"step_times_s": [round(x, 2) for x in times], "mean_s": round(float(np.mean(times[1:])), 2), "std_s": round(float(np.std(times[1:])), 2)}
    del m, opt
    torch.cuda.empty_cache()
print(json.dumps({"schema": "comax0.e1stepcost.probe.v2", "scene": scene, "seed": seed, "n_steps": NSTEP, "resolution": "1080p", "results": results}, indent=2))
