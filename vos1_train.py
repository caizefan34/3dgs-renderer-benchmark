import os, sys, json, time, math, random, argparse
import numpy as np
import torch
from random import randint

sys.path.insert(0, "/root/epic05_frozen/graphdeco-b0-pub")

from scene import Scene, GaussianModel
from gaussian_renderer import GaussianRasterizationSettings, GaussianRasterizer
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.general_utils import safe_state, get_expon_lr_func
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips


def vos_render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0):
    # Faithful copy of graphdeco gaussian_renderer.render, but:
    #  - drops the unsupported `antialiasing` kwarg (env mismatch)
    #  - forces separate_sh=False (installed rasterizer has no dc/shs split)
    # Same rasterizer, same gradients, same loss inputs.
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = pc.get_features
    colors_precomp = None

    rendered_image, radii, is_used = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    rendered_image = rendered_image.clamp(0, 1)
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0),
        "radii": radii,
        "depth": is_used,
    }


class VOS1GaussianModel(GaussianModel):
    """GaussianModel that tracks per-Gaussian invisible_age in lockstep with
    densification (append zeros) and pruning (filter mask). Does NOT alter any
    densification/pruning decision; only maintains a parallel counter."""

    def __init__(self, sh_degree, optimizer_type="sparse_adam", k=1):
        super().__init__(sh_degree, optimizer_type)
        self.k = int(k)
        self.invisible_age = None

    def init_age(self):
        self.invisible_age = torch.zeros(self.get_xyz.shape[0], dtype=torch.int32, device="cuda")

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                              new_opacities, new_scaling, new_rotation, new_tmp_radii):
        n_new = int(new_xyz.shape[0])
        super().densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                      new_opacities, new_scaling, new_rotation, new_tmp_radii)
        self.invisible_age = torch.cat(
            [self.invisible_age, torch.zeros(n_new, dtype=torch.int32, device="cuda")], dim=0)

    def prune_points(self, mask):
        super().prune_points(mask)
        self.invisible_age = self.invisible_age[~mask]


@torch.no_grad()
def evaluate_test(scene, gaussians, pipe, background):
    test_cams = scene.getTestCameras()
    ssims, psnrs, lpipss = [], [], []
    for cam in test_cams:
        image = torch.clamp(vos_render(cam, gaussians, pipe, background)["render"], 0.0, 1.0)
        gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
        ssims.append(ssim(image, gt).mean().double())
        psnrs.append(psnr(image, gt).mean().double())
        lpipss.append(lpips(image, gt, net_type="vgg").mean().double())
    return {
        "psnr": float(torch.stack(psnrs).mean().item()),
        "ssim": float(torch.stack(ssims).mean().item()),
        "lpips": float(torch.stack(lpipss).mean().item()),
        "n_test": len(test_cams),
    }


def train_one(args, lp, op, pipe):
    scene_name = args.scene_name
    k = int(args.k)
    iterations = int(args.iterations)
    seed = int(args.seed)

    os.makedirs(args.model_path, exist_ok=True)

    gaussians = VOS1GaussianModel(lp.sh_degree, optimizer_type=op.optimizer_type, k=k)
    scene = Scene(lp, gaussians)
    gaussians.init_age()
    gaussians.training_setup(op)

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    use_sparse_adam = op.optimizer_type == "sparse_adam"

    # Timers
    wall_start = time.time()
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    opt_start = torch.cuda.Event(enable_timing=True)
    opt_end = torch.cuda.Event(enable_timing=True)

    total_opt_time = 0.0
    total_iter_time = 0.0
    total_updates = 0
    total_skipped = 0
    total_opt_calls = 0
    trace = []

    for iteration in range(1, iterations + 1):
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        iter_start.record()

        render_pkg = vos_render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss = (1.0 - op.lambda_dssim) * Ll1 + op.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            visibility_filter = render_pkg["visibility_filter"]  # radii > 0
            radii = render_pkg["radii"]
            N = int(radii.shape[0])

            # ---- VOS1 invisible_age schedule (ONLY optimizer frequency changes) ----
            visible = radii > 0
            gaussians.invisible_age[visible] = 0
            gaussians.invisible_age[~visible] += 1
            if k == 1:
                should_update = torch.ones_like(visible)
            else:
                should_update = visible | (gaussians.invisible_age % k == 0)
            # -----------------------------------------------------------------------

            # Invariant: invisible_age must track the live Gaussian count exactly.
            assert gaussians.invisible_age.shape[0] == N, (
                f"invisible_age desync: age={gaussians.invisible_age.shape[0]} N={N}")

            n_upd = int(should_update.sum().item())
            n_skip = N - n_upd
            total_updates += n_upd
            total_skipped += n_skip

            opt_start.record()
            if use_sparse_adam:
                gaussians.optimizer.step(should_update, N, iteration)
            else:
                gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            opt_end.record()
            torch.cuda.synchronize()
            total_opt_time += opt_start.elapsed_time(opt_end)
            total_opt_calls += 1

            # Densification (unmodified) — uses TRUE visibility, not should_update
            if iteration < op.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(render_pkg["viewspace_points"], visibility_filter)
                if iteration > op.densify_from_iter and iteration % op.densification_interval == 0:
                    size_threshold = 20 if iteration > op.opacity_reset_interval else None
                    gaussians.densify_and_prune(op.densify_grad_threshold, 0.005,
                                                scene.cameras_extent, size_threshold, radii)
                if iteration % op.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

            if iteration == iterations:
                scene.save(iteration)

            torch.cuda.synchronize()
            total_iter_time += iter_start.elapsed_time(iter_end)

            if iteration % 500 == 0 or iteration == iterations:
                trace.append({
                    "iter": iteration,
                    "N": N,
                    "update_fraction": n_upd / max(N, 1),
                    "invisible_fraction": float((~visible).float().mean().item()),
                    "loss": float(loss.item()),
                })

    wall_time = time.time() - wall_start
    metrics = evaluate_test(scene, gaussians, pipe, background)

    final_N = int(gaussians.get_xyz.shape[0])
    result = {
        "scene": scene_name,
        "k": k,
        "seed": seed,
        "iterations": iterations,
        "final_N": final_N,
        "wall_time_s": wall_time,
        "optimizer_time_s": total_opt_time / 1000.0,
        "iter_time_ms_mean": total_iter_time / iterations,
        "update_count": total_updates,
        "skipped_updates": total_skipped,
        "total_optimizer_calls": total_opt_calls,
        "update_fraction": total_updates / max(total_updates + total_skipped, 1),
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
        "lpips": metrics["lpips"],
        "n_test": metrics["n_test"],
    }
    return result, trace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VOS1 visibility-aware Adam scheduler")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--result_path", type=str, default="")
    parser.add_argument("--trace_path", type=str, default="")
    args = parser.parse_args()

    lp.source_path = args.source_path
    lp.model_path = args.model_path
    lp.images = "images"
    lp.depths = ""
    lp.resolution = -1
    lp.white_background = False
    lp.eval = True
    lp.train_test_exp = False
    lp.data_device = "cuda"
    lp.sh_degree = 3

    safe_state(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    result, trace = train_one(args, lp, op, pp)

    if args.result_path:
        os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
        with open(args.result_path, "w") as f:
            json.dump(result, f, indent=2)
    if args.trace_path:
        os.makedirs(os.path.dirname(args.trace_path), exist_ok=True)
        with open(args.trace_path, "w") as f:
            json.dump(trace, f, indent=2)

    print("VOS1_RESULT " + json.dumps(result))
