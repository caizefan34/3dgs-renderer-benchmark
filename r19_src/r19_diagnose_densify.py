#!/usr/bin/env python3
"""Diagnose densification divergence: compare means2d.absgrad distributions
between std_ll and higs at the first densify step (iter 500). If HiGS absgrad is
systematically smaller, the adapter is under-triggering densification."""
from __future__ import annotations
import os, sys, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
REFV1 = REPO / "baseline" / "reference_v1"
PHASE7 = REPO / "scripts" / "epic05" / "phase7"
SRCDIR = REPO / "src"
PROJECT = Path(os.environ.get("HIGS_PROJECT",
    "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx"))
for p in (str(SRCDIR), str(PHASE7), str(REFV1), str(PROJECT), str(HERE)):
    sys.path.insert(0, p)
from gaussian_model import GaussianModel
from config import ReferenceV1Config
from dataset import GTDataset
from r19_trainer import attach_absgrad_hook, sfm_from_ply, SepSSIM

THRESH = 0.0008


def make_render(backend):
    if backend == "std_ll":
        from gsplat import rasterization
        def render(model, cam, shd):
            r, _, meta = rasterization(
                means=model.get_xyz, quats=model.get_rotation,
                scales=model.get_scaling, opacities=model.get_opacity,
                colors=model.get_features,
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=16, packed=False, sh_degree=shd,
                radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True)
            m2d = meta["means2d"]
            if m2d.requires_grad:
                m2d.retain_grad()
            return r[0].clamp(0,1), meta["radii"][0], m2d
    else:
        from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen
        def render(model, cam, shd):
            out = rasterize_gaussian_higs_frozen(
                model.get_xyz, model.get_rotation, model.get_scaling,
                model.get_opacity, model.get_features,
                viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
                Ks=cam.K.unsqueeze(0).unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                sh_degree=shd, enable_culling=True, use_higs_culling=True,
                freeze_topology=False, render_mode="RGB")
            info = out["densification_info"]
            m2d = info["means2d"]
            if m2d.requires_grad:
                m2d.retain_grad()
                attach_absgrad_hook(m2d)
            frame = out["frame"].reshape(cam.image_height, cam.image_width, 3)
            radii = info["radii"]
            while radii.dim() > 2:
                radii = radii[0]
            return frame.clamp(0,1), radii, m2d
    return render


def collect_absgrad(backend, scene, iters=500):
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    render = make_render(backend)
    dataset = GTDataset(scene=scene, repo_root=REPO, resolution="1080p",
                        device="cuda", background="black")
    n_cam = len(dataset)
    centers = np.array([dataset.get_camera(i).camera_center.cpu().numpy()
                        for i in range(min(50, n_cam))])
    ext = 0.1
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            ext = max(ext, float(np.linalg.norm(centers[i]-centers[j])))
    ext = max(ext, 0.1)
    pcd = sfm_from_ply(str(REPO/"data"/"datasets"/"mipnerf360"/scene/"point_cloud_sparse.ply"), sh_degree=3)
    model = GaussianModel(max_sh_degree=3)
    model.create_from_pcd(pcd, spatial_lr_scale=ext)
    model.training_setup({"position_lr_init":0.00016,"position_lr_final":0.0000016,
        "position_lr_delay_mult":0.01,"position_lr_max_steps":30000,
        "feature_lr":0.0025,"opacity_lr":0.025,"scaling_lr":0.005,"rotation_lr":0.001,"percent_dense":0.01})
    ssim_fn = SepSSIM(device="cuda")
    rng = random.Random(42); stack=list(range(n_cam)); seq=[]
    for _ in range(iters):
        if not stack: stack=list(range(n_cam))
        seq.append(stack.pop(rng.randint(0,len(stack)-1)))
    absgrad_samples = []
    signed_samples = []
    for it in range(1, iters+1):
        cam_idx = seq[it-1]; cam, gt = dataset.get_item(cam_idx)
        image, radii, means2d = render(model, cam, model.active_sh_degree)
        L1 = F.l1_loss(image, gt); dssim = ssim_fn(image, gt)
        loss = 0.8*L1 + 0.2*dssim
        model.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if it >= 100:  # after warmup
            ag = means2d.absgrad if hasattr(means2d, "absgrad") else means2d.grad.abs()
            sg = means2d.grad
            vis = (radii > 0).any(dim=-1)  # [N]
            if ag is not None:
                agf = ag.detach().flatten(0, 1) if ag.dim() == 3 else ag.detach()
                visf = vis.flatten() if vis.dim() > 1 else vis
                absgrad_samples.append(agf[visf].flatten().abs().cpu().numpy())
            if sg is not None:
                sgf = sg.detach().flatten(0, 1) if sg.dim() == 3 else sg.detach()
                visf = vis.flatten() if vis.dim() > 1 else vis
                signed_samples.append(sgf[visf].flatten().cpu().numpy())
        model.optimizer.step()
    ag = np.concatenate(absgrad_samples) if absgrad_samples else np.array([0.0])
    sg = np.concatenate(signed_samples) if signed_samples else np.array([0.0])
    return {
        "backend": backend,
        "absgrad_mean": float(ag.mean()), "absgrad_median": float(np.median(ag)),
        "absgrad_p90": float(np.percentile(ag, 90)), "absgrad_max": float(ag.max()),
        "frac_above_thresh": float((ag > THRESH).mean()),
        "n_visible": int(ag.size),
        "signed_mean_abs": float(np.abs(sg).mean()),
    }


def main():
    scene = "bicycle"
    res = []
    for be in ("std_ll", "higs"):
        print(f"  [diag] {be} collecting absgrad over 500 iters...", flush=True)
        r = collect_absgrad(be, scene, iters=500)
        res.append(r)
        print(f"    absgrad mean={r['absgrad_mean']:.3e} median={r['absgrad_median']:.3e} "
              f"p90={r['absgrad_p90']:.3e} frac_above_0.0008={r['frac_above_thresh']:.4f}", flush=True)
    import json
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
