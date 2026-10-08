"""ClockGS core: geometry-clock training on the canonical graphdeco 3DGS stack.

Implements the C0 convergence oracle. The canonical 3DGS loop is patched so
that geometry parameter families (xyz, scaling, rotation) are UPDATED only
once every k_geo iterations, while appearance (opacity, SH DC/rest) updates
every iteration.

Skipping is implemented by setting geometry `.grad = None` on skipped steps.
PyTorch Adam skips any parameter whose `.grad is None`, so the geometry
optimizer state (exp_avg / exp_avg_sq) is provably frozen on skipped steps
(no momentum drift). This is a training-dynamics oracle: the full backward is
still computed every iteration (the "can we skip computing it" question is
C2/C3), isolating "does slowing geometry updates hurt convergence?".
"""
import os
import sys
import json
import time
import random
import argparse
from argparse import ArgumentParser
from dataclasses import dataclass, field

import torch
import numpy as np
from tqdm import tqdm

# --- wire canonical stack ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: F401  (sets sys.path, shims simple_knn, fused_ssim)

from scene import Scene, GaussianModel
from arguments import ModelParams, OptimizationParams, PipelineParams
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0,
           separate_sh=False, override_color=None, use_trained_exp=False):
    # Local renderer matching the installed (older) Inria rasterizer:
    # no antialiasing field; forward returns (rendered_image, radii).
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                          requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx, tanfovy=tanfovy, bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False, debug=pipe.debug)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = None
    rotations = None
    cov3D_precomp = None
    shs = colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            from utils.sh_utils import eval_sh
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    if cov3D_precomp is None:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    rendered_image, radii, _depth, _alpha = rasterizer(
        means3D=means3D, means2D=means2D, shs=shs, colors_precomp=colors_precomp,
        opacities=opacity, scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp)
    rendered_image = rendered_image.clamp(0, 1)
    return {"render": rendered_image, "viewspace_points": screenspace_points,
            "visibility_filter": (radii > 0).nonzero(), "radii": radii}

# geometry vs appearance parameter group names (canonical optimizer groups)
GEOM_GROUPS = ["xyz", "scaling", "rotation"]
APPEAR_GROUPS = ["f_dc", "f_rest", "opacity"]
SH_GROUPS = ["f_dc", "f_rest"]


def set_geom_grad_none(gaussians: GaussianModel):
    """Zero+detach geometry grads so Adam skips them (state frozen)."""
    for name, p in [("xyz", gaussians._xyz),
                    ("scaling", gaussians._scaling),
                    ("rotation", gaussians._rotation)]:
        if p.grad is not None:
            p.grad = None


def geom_update_norm(gaussians: GaussianModel):
    """L2 norm of the geometry optimizer step that *would* be applied."""
    # We approximate the update magnitude from current grads * lr (Adam is
    # near-unit step direction); used only for logging, not gating.
    return None


def train_clock(scene_name, source_path, model_path, k_geo, k_opacity=1, k_sh=1,
                 start_iter=15000, end_iter=30000, images="images_8",
                 eval_split=True, seed=0, log_every=500, extra_params=None,
                 densify_until=15000, position_lr_max_steps=30000,
                 out_log=None, record_drift=True):
    """Run one geometry-clock variant from a fresh (densified) training.

    For C0 we train the full run but only *apply* geometry updates every
    k_geo iters after densification has stopped (iter >= densify_until).
    Opacity updates every k_opacity iters; SH every k_sh iters.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Build arg namespaces
    lp = ModelParams(ArgumentParser())
    lp.source_path = source_path
    lp.model_path = model_path
    lp.images = images
    lp.eval = eval_split
    lp.train_test_exp = False
    lp.sh_degree = 3
    lp.white_background = False
    lp.resolution = -1
    lp.depths = ""
    lp.data_device = "cuda"

    op = OptimizationParams(ArgumentParser())
    op.iterations = end_iter
    op.densify_until_iter = densify_until
    op.densify_from_iter = 500
    op.densification_interval = 100
    op.opacity_reset_interval = 3000
    op.densify_grad_threshold = 0.0002
    op.position_lr_init = 0.00016
    op.position_lr_final = 0.0000016
    op.position_lr_delay_mult = 0.01
    op.position_lr_max_steps = position_lr_max_steps
    op.feature_lr = 0.0025
    op.opacity_lr = 0.025
    op.scaling_lr = 0.005
    op.rotation_lr = 0.001
    op.lambda_dssim = 0.2
    op.percent_dense = 0.01
    op.random_background = False
    op.optimizer_type = "default"
    op.depth_l1_weight_init = 1.0
    op.depth_l1_weight_final = 0.01
    if extra_params:
        for k, v in extra_params.items():
            setattr(op, k, v)

    pp = PipelineParams(ArgumentParser())
    pp.convert_SHs_python = False
    pp.compute_cov3D_python = False
    pp.debug = False
    pp.antialiasing = False

    os.makedirs(model_path, exist_ok=True)
    gaussians = GaussianModel(lp.sh_degree, op.optimizer_type)
    scene = Scene(lp, gaussians, load_iteration=None, shuffle=True,
                  resolution_scales=[1.0])
    gaussians.training_setup(op)

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras(scale=1.0).copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # clocks: map group -> period
    clocks = {"xyz": k_geo, "scaling": k_geo, "rotation": k_geo,
              "opacity": k_opacity, "f_dc": k_sh, "f_rest": k_sh}

    drift_log = []
    ema_loss = 0.0
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    t0 = time.time()

    pbar = tqdm(range(1, end_iter + 1), desc=f"{scene_name} k_geo={k_geo}")
    for iteration in pbar:
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(scale=1.0).copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = random.randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        render_pkg = render(viewpoint_cam, gaussians, pp, background,
                            use_trained_exp=False, separate_sh=False)
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss = (1.0 - op.lambda_dssim) * Ll1 + op.lambda_dssim * (1.0 - ssim_value)

        loss.backward()
        iter_end.record()
        torch.cuda.synchronize()
        elapsed = iter_start.elapsed_time(iter_end)
        ema_loss = 0.4 * loss.item() + 0.6 * ema_loss

        # --- densification (only while iteration < densify_until_iter) ---
        if iteration < op.densify_until_iter:
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.add_densification_stats(render_pkg["viewspace_points"], visibility_filter)
            if iteration > op.densify_from_iter and iteration % op.densification_interval == 0:
                gaussians.densify_and_prune(op.densify_grad_threshold, 0.005,
                                            scene.cameras_extent, None, radii)
            if iteration % op.opacity_reset_interval == 0:
                gaussians.reset_opacity()

        # --- ClockGS: selective optimizer stepping ---
        geom_now = (iteration % k_geo == 0) or (iteration < op.densify_until_iter)
        # during densification, update everything every iter (topology changes)
        step_geom = geom_now or (iteration < op.densify_until_iter)
        step_opacity = (iteration % k_opacity == 0) or (iteration < op.densify_until_iter)
        step_sh = (iteration % k_sh == 0) or (iteration < op.densify_until_iter)

        if not step_geom:
            set_geom_grad_none(gaussians)
        if not step_opacity:
            if gaussians._opacity.grad is not None:
                gaussians._opacity.grad = None
        if not step_sh:
            if gaussians._features_dc.grad is not None:
                gaussians._features_dc.grad = None
            if gaussians._features_rest.grad is not None:
                gaussians._features_rest.grad = None

        # record geometry drift on applied steps
        if record_drift and step_geom and iteration >= op.densify_until_iter:
            with torch.no_grad():
                drift = {
                    "iter": iteration,
                    "xyz_norm": float(gaussians._xyz.grad.norm()) if gaussians._xyz.grad is not None else 0.0,
                    "scaling_norm": float(gaussians._scaling.grad.norm()) if gaussians._scaling.grad is not None else 0.0,
                    "rotation_norm": float(gaussians._rotation.grad.norm()) if gaussians._rotation.grad is not None else 0.0,
                }
            drift_log.append(drift)

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        if iteration % log_every == 0 or iteration == end_iter:
            pbar.set_postfix({"loss": f"{ema_loss:.4f}",
                              "N": gaussians._xyz.shape[0],
                              "ms": f"{elapsed:.1f}"})
            if out_log:
                with open(out_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "scene": scene_name, "k_geo": k_geo, "iter": iteration,
                        "loss": ema_loss, "N": int(gaussians._xyz.shape[0]),
                        "ms_iter": elapsed, "step_geom": bool(step_geom),
                        "step_opacity": bool(step_opacity), "step_sh": bool(step_sh),
                        "time_s": time.time() - t0,
                    }) + "\n")

    # --- final eval on test cameras ---
    torch.cuda.empty_cache()
    test_cams = scene.getTestCameras(scale=1.0)
    psnrs = []
    ssims = []
    with torch.no_grad():
        for cam in test_cams:
            img = torch.clamp(render(cam, gaussians, pp, background,
                                     use_trained_exp=False, separate_sh=False)["render"], 0, 1)
            gt = torch.clamp(cam.original_image.cuda(), 0, 1)
            psnrs.append(float(psnr(img, gt).mean().item()))
            ssims.append(float(ssim(img, gt).mean().item()))
    metrics = {
        "scene": scene_name, "k_geo": k_geo, "k_opacity": k_opacity, "k_sh": k_sh,
        "start_iter": start_iter, "end_iter": end_iter,
        "psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)),
        "psnr_std": float(np.std(psnrs)), "ssim_std": float(np.std(ssims)),
        "n_test": len(test_cams), "N_final": int(gaussians._xyz.shape[0]),
        "train_time_s": time.time() - t0,
        "seed": seed, "images": images,
    }
    if out_log:
        with open(out_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "EVAL", **metrics}) + "\n")
    return metrics, drift_log


class NamespaceMock:
    """Minimal argparse.Namespace-like stand-in for ParamGroup constructors."""
    pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--k_geo", type=int, default=1)
    ap.add_argument("--k_opacity", type=int, default=1)
    ap.add_argument("--k_sh", type=int, default=1)
    ap.add_argument("--start_iter", type=int, default=15000)
    ap.add_argument("--end_iter", type=int, default=30000)
    ap.add_argument("--images", default="images_8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--densify_until", type=int, default=15000)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--out_log", default=None)
    args = ap.parse_args()
    m, d = train_clock(args.scene, args.source, args.model_path, args.k_geo,
                       args.k_opacity, args.k_sh, args.start_iter, args.end_iter,
                       args.images, True, args.seed, args.log_every,
                       densify_until=args.densify_until, out_log=args.out_log)
    print("FINAL", json.dumps(m))
