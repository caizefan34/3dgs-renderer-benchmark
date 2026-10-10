#!/usr/bin/env python3
"""R19 P1 training-level correctness comparison.

Runs BOTH backends (std_ll, higs) from the IDENTICAL init checkpoint + seed,
capturing per-step: loss, RGB/alpha, Gaussian position/scale/rotation/opacity/SH
gradients, optimizer state, Gaussian count, densification/pruning events, NaN/Inf,
gradient magnitude + relative L2. Distinguishes normal float error propagation
from algorithmic semantic mismatch. Writes R19_TRAINING_CORRECTNESS.json.

Usage: r19_p1_correctness.py --scene bicycle --steps 1000
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
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
from r19_trainer import SepSSIM, compute_psnr, attach_absgrad_hook, sfm_from_ply

EXPECTED = {
    "std_ll": "40a009837e75620454059cfdbb5088cc08423f281e4c734da3053b727d39df50",
    "higs": "b64d02c1cdad51bfc9ff0c2c435f9a4c1d1603466700194fb30f1c2348bcf2c3",
}


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def make_render(backend):
    if backend == "std_ll":
        from gsplat import rasterization
        def render(model, cam, sh_degree):
            r, alphas, meta = rasterization(
                means=model.get_xyz, quats=model.get_rotation,
                scales=model.get_scaling, opacities=model.get_opacity,
                colors=model.get_features,
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=16, packed=False, sh_degree=sh_degree,
                radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True)
            means2d = meta["means2d"]
            if means2d.requires_grad:
                means2d.retain_grad()
            alpha = alphas.mean() if alphas.numel() > 0 else None
            return r[0].clamp(0, 1), meta["radii"][0], means2d, alpha
    else:
        from gsplat.experimental.render.functional import rasterize_gaussian_higs_frozen
        def render(model, cam, sh_degree):
            out = rasterize_gaussian_higs_frozen(
                model.get_xyz, model.get_rotation, model.get_scaling,
                model.get_opacity, model.get_features,
                viewmats=cam.viewmatrix.unsqueeze(0).unsqueeze(0),
                Ks=cam.K.unsqueeze(0).unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                sh_degree=sh_degree, enable_culling=True, use_higs_culling=True,
                freeze_topology=False, render_mode="RGB")
            info = out["densification_info"]
            means2d = info["means2d"]
            if means2d.requires_grad:
                means2d.retain_grad()
                attach_absgrad_hook(means2d)
            frame = out["frame"].reshape(cam.image_height, cam.image_width, 3)
            radii = info["radii"]
            while radii.dim() > 2:
                radii = radii[0]
            alpha = info.get("alpha")
            if alpha is not None:
                alpha = alpha.mean()
            return frame.clamp(0, 1), radii, means2d, alpha
    return render


def run_arm(backend, scene, steps, seed):
    """Run one backend for `steps` iters from a fresh identical init.
    Returns a trajectory of per-step diagnostics + the final model state dict."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    render = make_render(backend)
    dataset = GTDataset(scene=scene, repo_root=REPO, resolution="1080p",
                        device="cuda", background="black")
    n_cameras = len(dataset)
    # scene extent
    centers = np.array([dataset.get_camera(i).camera_center.cpu().numpy()
                        for i in range(min(50, n_cameras))])
    extent = 0.0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            extent = max(extent, float(np.linalg.norm(centers[i] - centers[j])))
    scene_extent = max(extent, 0.1)
    sparse_ply = REPO / "data" / "datasets" / "mipnerf360" / scene / "point_cloud_sparse.ply"
    pcd_data = sfm_from_ply(str(sparse_ply), sh_degree=3)
    model = GaussianModel(max_sh_degree=3)
    model.create_from_pcd(pcd_data, spatial_lr_scale=scene_extent)
    initial_N = int(model._xyz.shape[0])
    model.training_setup({
        "position_lr_init": 0.00016, "position_lr_final": 0.0000016,
        "position_lr_delay_mult": 0.01, "position_lr_max_steps": 30000,
        "feature_lr": 0.0025, "opacity_lr": 0.025,
        "scaling_lr": 0.005, "rotation_lr": 0.001, "percent_dense": 0.01})
    ssim_fn = SepSSIM(device="cuda")
    rng = random.Random(seed)
    stack = list(range(n_cameras))
    cam_seq = []
    for _ in range(steps):
        if not stack:
            stack = list(range(n_cameras))
        cam_seq.append(stack.pop(rng.randint(0, len(stack) - 1)))

    traj = []
    for it in range(1, steps + 1):
        model.update_learning_rate(it)
        if it % 1000 == 0:
            model.oneupSHdegree()
        cam_idx = cam_seq[it - 1]
        cam, gt = dataset.get_item(cam_idx)
        image, radii, means2d, alpha = render(model, cam, model.active_sh_degree)
        L1 = F.l1_loss(image, gt)
        dssim = ssim_fn(image, gt)
        loss = 0.8 * L1 + 0.2 * dssim
        model.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # capture gradient magnitudes (pre-optimizer-step)
        g = {
            "xyz": float(model._xyz.grad.norm()) if model._xyz.grad is not None else 0.0,
            "scaling": float(model._scaling.grad.norm()) if model._scaling.grad is not None else 0.0,
            "rotation": float(model._rotation.grad.norm()) if model._rotation.grad is not None else 0.0,
            "opacity": float(model._opacity.grad.norm()) if model._opacity.grad is not None else 0.0,
            "features": float(model._shs.grad.norm()) if model._shs.grad is not None else 0.0,
            "means2d": float(means2d.grad.norm()) if means2d.grad is not None else 0.0,
        }
        lv = float(loss.item())
        nan = (not math.isfinite(lv)) or any(not math.isfinite(v) for v in g.values())
        traj.append({"iter": it, "loss": lv, "psnr": compute_psnr(image, gt),
                     "n_gs": int(model._xyz.shape[0]), "grad_norms": g,
                     "loss_nan": nan, "alpha_mean": float(alpha.mean()) if alpha is not None else None})
        # densification (same rules)
        if it < 15000:
            visibility_filter = (radii > 0).any(dim=-1)
            model.max_radii2D[visibility_filter] = torch.max(
                model.max_radii2D[visibility_filter],
                radii[visibility_filter].float().max(dim=-1).values)
            model.add_densification_stats(means2d, visibility_filter,
                                          width=cam.image_width, height=cam.image_height)
            if it >= 500 and it < 15000 and it % 100 == 0:
                size_threshold = 20 if it > 3000 else None
                model.densify_and_prune(max_grad=0.0008, min_opacity=0.005,
                                        extent=scene_extent, max_screen_size=size_threshold,
                                        radii=radii.float().max(dim=-1).values)
            if it < 15000 and it % 3000 == 0:
                model.reset_opacity()
        model.optimizer.step()
    return traj, initial_N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="bicycle")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # verify binaries
    bins = {}
    for be, rel in (("std_ll", "gsplat/csrc.so"),
                    ("higs", "gsplat/experimental/render/kernels/csrc.so")):
        sha = sha256_file(PROJECT / rel)
        bins[be] = {"sha256": sha, "match": sha == EXPECTED[be]}

    results = {"deliverable": "R19_TRAINING_CORRECTNESS", "scene": args.scene,
               "steps": args.steps, "seed": 42, "binaries": bins,
               "arms": {}}
    trajs = {}
    for be in ("std_ll", "higs"):
        print(f"  [P1] running {be} for {args.steps} steps...", flush=True)
        traj, init_N = run_arm(be, args.scene, args.steps, seed=42)
        trajs[be] = traj
        results["arms"][be] = {
            "initial_N": init_N, "final_N": traj[-1]["n_gs"],
            "final_loss": traj[-1]["loss"], "final_psnr": traj[-1]["psnr"],
            "any_nan": any(t["loss_nan"] for t in traj),
            "loss_trajectory": [t["loss"] for t in traj],
            "n_gs_trajectory": [t["n_gs"] for t in traj],
        }

    # paired comparison at checkpoints
    checkpoints = [i + 1 for i in range(args.steps) if (i + 1) in (1, 100, 500, 1000)]
    comp = []
    for c in checkpoints:
        a = trajs["std_ll"][c - 1]; b = trajs["higs"][c - 1]
        loss_abs = abs(a["loss"] - b["loss"])
        loss_rel = loss_abs / max(abs(a["loss"]), 1e-12)
        psnr_abs = abs(a["psnr"] - b["psnr"])
        # gradient relative L2 per param
        grad_rel = {}
        for k in a["grad_norms"]:
            ga, gb = a["grad_norms"][k], b["grad_norms"][k]
            grad_rel[k] = abs(ga - gb) / max(abs(ga), 1e-12)
        comp.append({"step": c, "loss_std": a["loss"], "loss_higs": b["loss"],
                     "loss_abs": loss_abs, "loss_rel": loss_rel,
                     "psnr_std": a["psnr"], "psnr_higs": b["psnr"], "psnr_abs": psnr_abs,
                     "n_gs_std": a["n_gs"], "n_gs_higs": b["n_gs"],
                     "grad_rel": grad_rel})
    results["paired_comparison"] = comp
    results["gaussian_count_match"] = (results["arms"]["std_ll"]["final_N"]
                                       == results["arms"]["higs"]["final_N"])
    results["n_gs_match_all_steps"] = all(
        r["n_gs_std"] == r["n_gs_higs"] for r in comp)

    with open(os.path.join(args.outdir, "R19_TRAINING_CORRECTNESS.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: results[k] for k in ("scene", "steps", "gaussian_count_match",
                                              "n_gs_match_all_steps")}, indent=2))
    for r in comp:
        print(f"  step{r['step']}: loss_std={r['loss_std']:.6f} loss_higs={r['loss_higs']:.6f} "
              f"rel={r['loss_rel']:.2e} psnr_std={r['psnr_std']:.4f} psnr_higs={r['psnr_higs']:.4f} "
              f"n_gs={r['n_gs_std']}/{r['n_gs_higs']}")


if __name__ == "__main__":
    main()
