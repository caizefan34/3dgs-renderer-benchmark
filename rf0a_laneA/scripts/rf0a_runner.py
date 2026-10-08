#!/usr/bin/env python
# rf0a_runner.py - ReproFast0 Lane A instrumented runner (EPIC-05 / A100)
# Arm A (fastgs)  : VERBATIM official FastGS train.py loop (commit 44e02a5) + instrumentation only.
# Arm B (vanilla) : faithful original-3DGS densification + per-iteration Adam, SAME rasterizer backend.
# No FastGS algorithm change in arm A; arm B is the same-backend faithful comparator.
import argparse, json, os, sys, time, random
import numpy as np
import torch
from random import randint
from argparse import Namespace

from scene import Scene, GaussianModel
from gaussian_renderer import render_fastgs
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from utils.general_utils import safe_state, build_rotation
from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras
from arguments import ModelParams, PipelineParams, OptimizationParams
import torchvision

SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ---------------- faithful original-3DGS densification (arm B only) ----------------
def v_densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
    n_init_points = self.get_xyz.shape[0]
    padded_grad = torch.zeros((n_init_points), dtype=torch.float, device="cuda")
    padded_grad[:grads.shape[0]] = grads.squeeze()
    selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
    selected_pts_mask = torch.logical_and(selected_pts_mask,
        torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
    stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
    means = torch.zeros((stds.size(0), 3), device="cuda")
    samples = torch.normal(mean=means, std=stds)
    rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
    new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
    new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
    new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
    new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
    new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
    new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
    new_tmp_radii = torch.zeros(N * int(selected_pts_mask.sum()), device="cuda")
    self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)
    prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * int(selected_pts_mask.sum()), device="cuda", dtype=bool)))
    self.prune_points(prune_filter)

def v_densify_and_clone(self, grads, grad_threshold, scene_extent):
    selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
    selected_pts_mask = torch.logical_and(selected_pts_mask,
        torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
    new_xyz = self._xyz[selected_pts_mask]
    new_features_dc = self._features_dc[selected_pts_mask]
    new_features_rest = self._features_rest[selected_pts_mask]
    new_opacities = self._opacity[selected_pts_mask]
    new_scaling = self._scaling[selected_pts_mask]
    new_rotation = self._rotation[selected_pts_mask]
    new_tmp_radii = self.tmp_radii[selected_pts_mask]
    self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

def v_densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
    grads = self.xyz_gradient_accum / self.denom
    grads[grads.isnan()] = 0.0
    self.v_densify_and_clone(grads, max_grad, extent)
    self.v_densify_and_split(grads, max_grad, extent)
    prune_mask = (self.get_opacity < min_opacity).squeeze()
    if max_screen_size:
        big_points_vs = self.max_radii2D > max_screen_size
        big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    self.prune_points(prune_mask)
    torch.cuda.empty_cache()

def v_add_densification_stats(self, viewspace_point_tensor, update_filter):
    self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
    self.denom[update_filter] += 1

# ---------------- evaluation (matches metrics.py) ----------------
@torch.no_grad()
def eval_split(gaussians, scene, pipe, background, cams, mult):
    psnrs, ssims, lpipss = [], [], []
    for cam in cams:
        out = render_fastgs(cam, gaussians, pipe, background, mult)["render"]
        gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
        image = torch.clamp(out, 0.0, 1.0)
        psnrs.append(psnr(image, gt).mean().double())
        ssims.append(fast_ssim(image.unsqueeze(0), gt.unsqueeze(0)).mean().double())
        lpipss.append(lpips(image, gt, net_type='vgg').mean().double())
    return {"psnr": float(torch.stack(psnrs).mean().item()) if psnrs else None,
            "ssim": float(torch.stack(ssims).mean().item()) if ssims else None,
            "lpips": float(torch.stack(lpipss).mean().item()) if lpipss else None,
            "n": len(cams)}

def training(dataset, opt, pipe, saving_iterations, method, eval_mult):
    if dataset.model_path:
        os.makedirs(dataset.model_path, exist_ok=True)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.tmp_radii = torch.zeros(0, device="cuda")
    gaussians.training_setup(opt)
    if method == "vanilla":
        gaussians.v_densify_and_prune = v_densify_and_prune.__get__(gaussians)
        gaussians.v_densify_and_clone = v_densify_and_clone.__get__(gaussians)
        gaussians.v_densify_and_split = v_densify_and_split.__get__(gaussians)
        gaussians.add_densification_stats = v_add_densification_stats.__get__(gaussians)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = scene.getTrainCameras().copy()
    ema_loss_for_log = 0.0
    torch.cuda.reset_peak_memory_stats()
    wall_start = time.time()
    total_time = 0.0
    traj = []
    n_train = len(viewpoint_stack)
    visibility_filter = None
    radii = None

    for iteration in range(1, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        rand_idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)

        iter_start.record()
        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)
        image = render_pkg["render"]
        radii = render_pkg["radii"]
        visibility_filter = render_pkg["visibility_filter"]
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration in saving_iterations:
                scene.save(iteration)
            if iteration % 1000 == 0 or iteration == opt.iterations:
                traj.append({"iter": int(iteration), "n_gaussians": int(gaussians._xyz.shape[0])})

            optim_start.record()
            if iteration < opt.iterations:
                if method == "fastgs":
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(render_pkg["viewspace_points"], visibility_filter)
                    if iteration < opt.densify_until_iter:
                        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            my_viewpoint_stack = scene.getTrainCameras().copy()
                            camlist = sampling_cameras(my_viewpoint_stack)
                            importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True)
                            gaussians.densify_and_prune_fastgs(max_screen_size=size_threshold, min_opacity=0.005,
                                extent=scene.cameras_extent, radii=radii, args=opt,
                                importance_score=importance_score, pruning_score=pruning_score)
                        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                            gaussians.reset_opacity()
                    if iteration % 3000 == 0 and iteration > 15000 and iteration < 30000:
                        my_viewpoint_stack = scene.getTrainCameras().copy()
                        camlist = sampling_cameras(my_viewpoint_stack)
                        _, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt)
                        gaussians.final_prune_fastgs(min_opacity=0.1, pruning_score=pruning_score)
                    if opt.optimizer_type == "default":
                        gaussians.optimizer_step(iteration)
                    elif opt.optimizer_type == "sparse_adam":
                        visible = radii > 0
                        gaussians.optimizer.step(visible, radii.shape[0])
                        gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.tmp_radii = radii
                    gaussians.add_densification_stats(render_pkg["viewspace_points"], visibility_filter)
                    if iteration < opt.densify_until_iter:
                        if iteration % opt.densification_interval == 0:
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            gaussians.v_densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                            gaussians.reset_opacity()
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    gaussians.shoptimizer.step()
                    gaussians.shoptimizer.zero_grad(set_to_none=True)
            optim_end.record()
            torch.cuda.synchronize()
            optim_time = optim_start.elapsed_time(optim_end)
            iter_time = iter_start.elapsed_time(iter_end)
            total_time += (iter_time + optim_time) / 1e3

    wall = time.time() - wall_start
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    final_count = int(gaussians._xyz.shape[0])
    train_cams = scene.getTrainCameras().copy()
    step = max(1, len(train_cams) // 10)
    train_sample = train_cams[0:len(train_cams):step][:10]
    train_metrics = eval_split(gaussians, scene, pipe, background, train_sample, eval_mult)
    test_metrics = eval_split(gaussians, scene, pipe, background, scene.getTestCameras(), eval_mult)
    return {
        "method": method, "seed": SEED, "iterations": int(opt.iterations),
        "final_gaussian_count": final_count,
        "training_wall_seconds": round(wall, 3),
        "internal_training_time_seconds": round(total_time, 3),
        "peak_vram_gb": round(peak_vram, 4),
        "train_psnr_diagnostic": train_metrics,
        "holdout_metrics": test_metrics,
        "gaussian_trajectory": traj,
        "n_train_cams": int(n_train), "n_test_cams": int(len(scene.getTestCameras())),
    }

def main():
    global SEED
    parser = argparse.ArgumentParser(description="RF0A runner")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--method", choices=["fastgs", "vanilla"], required=True)
    parser.add_argument("--eval_mult", type=float, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    SEED = args.seed
    safe_state(False)
    set_seed(SEED)
    opt = op.extract(args)
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    eval_mult = args.eval_mult if args.eval_mult is not None else opt.mult
    res = training(dataset, opt, pipe, [30000], args.method, eval_mult)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("RESULT_JSON=" + args.out)
    print(json.dumps({k: res[k] for k in ["final_gaussian_count","training_wall_seconds","internal_training_time_seconds","peak_vram_gb","holdout_metrics"]}, indent=2))

if __name__ == "__main__":
    main()
