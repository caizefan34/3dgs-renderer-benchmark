#!/usr/bin/env python3
"""
B1A 13-Scene 30K Full-Training Validation.

B1  = clean gsplat 1.5.3 baseline (AABB tile enumeration, accutile=False)
B1A = B1 + true strip-based AccuTile (accutile=True)

Both methods share identical seed, initialization, camera sequence, optimizer,
learning-rate schedule, densification, clone/split/prune, opacity reset, SH
progression, loss, iterations, evaluation split, image resolution. The ONLY
independent variable is AccuTile OFF vs ON.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 b1a_13scene_train.py --scene room --method b1  --gpu 0 --output_dir /dev/shm/accutile30k/room/b1
  CUDA_VISIBLE_DEVICES=1 python3 b1a_13scene_train.py --scene room --method b1a --gpu 1 --output_dir /dev/shm/accutile30k/room/b1a
"""
import argparse
import json
import math
import os
import sys
import time
import random
import hashlib
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

# === AccuTile tree selection (must happen BEFORE `import gsplat`) ===
TRUE_ACCUTILE_TREE = "/mnt/storage_pool/liaoyuanjun/gsplat-true-accutile-v153"

def select_gsplat_tree(method: str):
    """For B1A, put the true-accutile source tree first on sys.path so that
    `from gsplat import rasterization` resolves to the accutile-enabled build.
    For B1, leave the installed (pip) gsplat 1.5.3 in place (AABB baseline)."""
    if method == "b1a":
        if TRUE_ACCUTILE_TREE not in sys.path:
            sys.path.insert(0, TRUE_ACCUTILE_TREE)
        # Verify the accutile parameter is available
        import gsplat as _g
        import inspect as _insp
        _sig = _insp.signature(_g.rasterization)
        if "accutile" not in _sig.parameters:
            raise RuntimeError(
                f"B1A selected gsplat at {_g.__file__} has no 'accutile' param "
                f"-- not the true strip-based AccuTile tree."
            )
        print(f"  [gsplat-tree] B1A using TRUE ACCUTILE: {_g.__file__}")
    else:
        import gsplat as _g
        print(f"  [gsplat-tree] B1 using installed baseline: {_g.__file__}")

# === Repo path setup ===
REPO_ROOT = Path(__file__).resolve().parent
REFERENCE_V1_DIR = REPO_ROOT / "baseline" / "reference_v1"
# Insert in reverse priority order so REFERENCE_V1_DIR ends up at index 0
# (highest priority) for gaussian_model/config/colmap_reader, while the
# phase7 dir (which has a DIFFERENT gaussian_model.py) stays lower priority.
sys.path.insert(0, str(REPO_ROOT / "scripts" / "epic05" / "phase7"))   # lowest (has GTDataset + alt GaussianModel)
sys.path.insert(0, str(REPO_ROOT / "src"))                              # benchmark_framework
sys.path.insert(0, str(REFERENCE_V1_DIR))                              # HIGHEST for gaussian_model/config

# === Early method detection: pick the gsplat source tree BEFORE importing gsplat ===
def _detect_method_from_argv():
    for i, a in enumerate(sys.argv):
        if a == "--method" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--method="):
            return a.split("=", 1)[1]
    return "b1"

_PRE_METHOD = _detect_method_from_argv()
if _PRE_METHOD == "b1a":
    # TRUE_ACCUTILE_TREE has no gaussian_model.py, so inserting at 0 only
    # affects `import gsplat` (which REFERENCE_V1_DIR does not provide).
    if TRUE_ACCUTILE_TREE not in sys.path:
        sys.path.insert(0, TRUE_ACCUTILE_TREE)

from gaussian_model import GaussianModel
from config import ReferenceV1Config
from colmap_reader import read_points3D_binary, sfm_to_pcd_data
from dataset import GTDataset
from gsplat import rasterization


class SepSSIM:
    def __init__(self, window_size=11, sigma=1.5, device="cuda"):
        self.C1 = (0.01) ** 2
        self.C2 = (0.03) ** 2
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0, 1, 3, 2).contiguous()
        self.padding = window_size // 2

    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.cat([pred, target, pred ** 2, target ** 2, pred * target], dim=1)
        b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_p, mu_t = b[:, 0:3], b[:, 3:6]
        bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
        mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
        sp2, st2, spt = bp2 - mu_p2, bt2 - mu_t2, bpt - mu_pt
        ssim_map = (2 * mu_pt + self.C1) * (2 * spt + self.C2) / \
                   ((mu_p2 + mu_t2 + self.C1) * (sp2 + st2 + self.C2))
        return 1.0 - ssim_map.mean()


def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    return float(20 * math.log10(1.0 / math.sqrt(mse.item()))) if mse > 1e-10 else 100.0


def render_with_meta(model, cam, sh_degree, accutile: bool):
    data = {
        "xyz": model.get_xyz,
        "rotations": model.get_rotation,
        "scales": model.get_scaling,
        "opacity": model.get_opacity,
        "shs": model.get_features,
    }
    kwargs = dict(
        tile_size=16, packed=False, sh_degree=sh_degree,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
    )
    if accutile:
        kwargs["accutile"] = True
    r, _, meta = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        **kwargs,
    )
    means2d_full = meta["means2d"]
    if means2d_full.requires_grad:
        means2d_full.retain_grad()
    return r[0].clamp(0, 1), meta, means2d_full


def evaluate_model(model, dataset, ssim_fn, sh_degree, cam_indices, accutile, lpips_fn=None):
    psnrs, ssims, l1s, lpips_vals = [], [], [], []
    for ci in cam_indices:
        cam, gt = dataset.get_item(ci)
        with torch.no_grad():
            img, _, _ = render_with_meta(model, cam, sh_degree, accutile)
        psnrs.append(compute_psnr(img, gt))
        ssims.append(float(1.0 - ssim_fn(img, gt).item()))
        l1s.append(float(F.l1_loss(img, gt).item()))
        if lpips_fn is not None:
            pred_lp = img.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
            gt_lp = gt.unsqueeze(0).permute(0, 3, 1, 2) * 2 - 1
            lpips_vals.append(float(lpips_fn(pred_lp, gt_lp).item()))
    result = {
        "psnr": float(np.mean(psnrs)), "ssim": float(np.mean(ssims)),
        "l1": float(np.mean(l1s)), "n_cameras": len(cam_indices),
    }
    if lpips_vals:
        result["lpips"] = float(np.mean(lpips_vals))
    return result


def compute_scene_extent(dataset, n_samples=50):
    centers = []
    n = min(n_samples, len(dataset))
    for i in range(n):
        cam = dataset.get_camera(i)
        centers.append(cam.camera_center.cpu().numpy())
    centers = np.array(centers)
    extent = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            extent = max(extent, d)
    return max(extent, 0.1)


def get_provenance(repo_root, scene, method, accutile):
    prov = {}
    try:
        prov["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
        prov["git_describe"] = subprocess.check_output(["git", "describe", "--tags"], cwd=str(repo_root), text=True).strip()
        prov["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo_root), text=True).strip())
    except Exception:
        prov["git_head"] = "unknown"
        prov["git_describe"] = "unknown"
        prov["git_dirty"] = None

    prov["pytorch_version"] = torch.__version__
    prov["cuda_version"] = torch.version.cuda
    import gsplat
    prov["gsplat_version"] = gsplat.__version__
    prov["gsplat_file"] = gsplat.__file__
    prov["accutile_enabled"] = accutile

    prov["gpu_name"] = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    prov["gpu_uuid"] = props.name
    prov["gpu_sms"] = props.multi_processor_count
    prov["hostname"] = subprocess.check_output(["hostname"], text=True).strip()

    model_path = REFERENCE_V1_DIR / "gaussian_model.py"
    config_path = REFERENCE_V1_DIR / "config.py"
    prov["gaussian_model_hash"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    prov["config_hash"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    prov["trainer_hash"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prov["gaussian_model_path"] = str(model_path.relative_to(repo_root))
    prov["config_path"] = str(config_path.relative_to(repo_root))

    try:
        import lpips
        prov["lpips_version"] = lpips.__version__
    except Exception:
        prov["lpips_version"] = "unknown"

    prov["scene"] = scene
    prov["method"] = method
    prov["iterations"] = 30000
    prov["seed"] = 42
    prov["dataset_resolution"] = "1080p"
    prov["loss_definition"] = "(1-lambda)*L1_fullres + lambda*(1-SepSSIM), lambda=0.2"
    prov["ssim_implementation"] = "SepSSIM window=11 sigma=1.5"
    prov["semantic_label"] = "REFERENCE_V1_ABSGRAD"
    prov["config_base"] = "REFERENCE_V1_ABSGRAD"
    prov["densify_grad_threshold"] = 0.0008
    prov["lambda_dssim"] = 0.2
    prov["absgrad"] = True
    prov["grow_grad2d"] = 0.0008
    prov["tile_size"] = 16
    prov["independent_variable"] = "accutile OFF (B1) vs ON (B1A)"

    return prov


def train(scene, method, output_dir, gpu_id):
    accutile = (method == "b1a")
    print(f"\n{'='*60}")
    print(f"  B1A 13-SCENE 30K FULL-TRAINING VALIDATION")
    print(f"  Scene: {scene}, Method: {method}, AccuTile: {accutile}, GPU: {gpu_id}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    config = ReferenceV1Config()
    config.scene = scene
    config.repo_root = str(REPO_ROOT)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    print(f"Loading dataset: {scene}")
    dataset = GTDataset(scene=scene, repo_root=REPO_ROOT, resolution="1080p",
                        device="cuda", background="black")
    n_cameras = len(dataset)
    print(f"  {n_cameras} cameras loaded")

    scene_extent = compute_scene_extent(dataset)
    print(f"  Scene extent: {scene_extent:.4f}")

    print("Loading SfM point cloud...")
    sfm_path = os.path.join(str(REPO_ROOT), "data", "datasets", "mipnerf360", scene, "sparse", "0", "points3D.bin")
    if os.path.exists(sfm_path):
        print(f"  Reading COLMAP SfM points from {sfm_path}")
        points3d = read_points3D_binary(sfm_path)
        pcd_data = sfm_to_pcd_data(points3d, sh_degree=config.sh_degree)
        print(f"  SfM points: {points3d['num_points']}")
    else:
        print(f"  WARNING: SfM file not found, falling back to PLY")
        from dataset import load_initial_checkpoint
        pcd_data = load_initial_checkpoint(scene, str(REPO_ROOT), device="cuda")

    model = GaussianModel(max_sh_degree=config.sh_degree)
    model.create_from_pcd(pcd_data, spatial_lr_scale=scene_extent)
    initial_N = model._xyz.shape[0]

    model.training_setup({
        "position_lr_init": config.position_lr_init,
        "position_lr_final": config.position_lr_final,
        "position_lr_delay_mult": config.position_lr_delay_mult,
        "position_lr_max_steps": config.position_lr_max_steps,
        "feature_lr": config.feature_lr,
        "opacity_lr": config.opacity_lr,
        "scaling_lr": config.scaling_lr,
        "rotation_lr": config.rotation_lr,
        "percent_dense": config.percent_dense,
    })

    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg").to("cuda")
        lpips_fn.eval()
        print(f"  LPIPS initialized (vgg)")
    except Exception as e:
        print(f"  WARNING: LPIPS not available: {e}")

    ssim_fn = SepSSIM(device="cuda")

    viewpoint_stack = list(range(n_cameras))
    camera_sequence = []
    rng = random.Random(config.seed)
    for iteration in range(1, config.iterations + 1):
        if not viewpoint_stack:
            viewpoint_stack = list(range(n_cameras))
        rand_idx = rng.randint(0, len(viewpoint_stack) - 1)
        cam_idx = viewpoint_stack.pop(rand_idx)
        camera_sequence.append(cam_idx)
    camera_sequence = np.array(camera_sequence, dtype=np.int32)
    np.save(os.path.join(output_dir, "camera_sequence.npy"), camera_sequence)

    eval_indices = list(range(0, n_cameras, max(1, n_cameras // 10)))

    results = {
        "scene": scene, "method": method, "accutile": accutile,
        "seed": config.seed, "num_iters": config.iterations,
        "lambda_dssim": config.lambda_dssim,
        "initial_N": initial_N, "scene_extent": float(scene_extent),
        "training_metrics": {"checkpoints": {}},
        "checkpoints_saved": [],
        "final_eval": None,
        "timing": {},
    }

    total_clones = 0
    total_splits = 0
    total_prunes = 0
    total_opacity_resets = 0
    iter_times = []
    peak_vram = 0.0
    per_event_clones = []
    per_event_splits = []
    per_event_prunes = []
    per_event_iters = []

    eval_iterations = set(config.eval_iterations)
    checkpoint_iterations = {5000, 10000, 15000, 20000, 25000, 30000}

    print(f"\nStarting training: {config.iterations} iterations")
    print(f"  Initial N: {initial_N}")

    t_start = time.perf_counter()

    for iteration in range(1, config.iterations + 1):
        model.update_learning_rate(iteration)

        if iteration % config.sh_progress_interval == 0:
            model.oneupSHdegree()

        cam_idx = camera_sequence[iteration - 1]
        cam, gt_image = dataset.get_item(cam_idx)

        t0 = time.perf_counter()
        image, meta, means2d = render_with_meta(model, cam, model.active_sh_degree, accutile)

        L1 = F.l1_loss(image, gt_image)
        dssim = ssim_fn(image, gt_image)
        loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim

        loss.backward()

        radii = meta["radii"][0]
        visibility_filter = (radii > 0).any(dim=-1)

        if iteration < config.densify_until_iter:
            model.max_radii2D[visibility_filter] = torch.max(
                model.max_radii2D[visibility_filter],
                radii[visibility_filter].float().max(dim=-1).values
            )
            model.add_densification_stats(means2d, visibility_filter,
                                          width=cam.image_width, height=cam.image_height)

        if (iteration > config.densify_from_iter and
            iteration < config.densify_until_iter and
            iteration % config.densification_interval == 0):
            size_threshold = config.max_screen_size if iteration > config.opacity_reset_interval else None
            current_radii = radii.float().max(dim=-1).values
            event_result = model.densify_and_prune(
                max_grad=config.densify_grad_threshold,
                min_opacity=config.min_opacity,
                extent=scene_extent,
                max_screen_size=size_threshold,
                radii=current_radii,
            )
            total_clones += event_result["cloned"]
            total_splits += event_result["split"]
            total_prunes += event_result["pruned_total"]
            per_event_clones.append(event_result["cloned"])
            per_event_splits.append(event_result["split"])
            per_event_prunes.append(event_result["pruned_total"])
            per_event_iters.append(iteration)

        if iteration < config.densify_until_iter and iteration % config.opacity_reset_interval == 0:
            model.reset_opacity()
            total_opacity_resets += 1

        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        iter_ms = (time.perf_counter() - t0) * 1000
        iter_times.append(iter_ms)

        vram = torch.cuda.max_memory_allocated() / (1024**3)
        if vram > peak_vram:
            peak_vram = vram

        if iteration in eval_iterations or iteration == config.iterations:
            eval_result = evaluate_model(model, dataset, ssim_fn, model.active_sh_degree,
                                         eval_indices, accutile, lpips_fn)
            eval_result["n_gaussians"] = model._xyz.shape[0]
            eval_result["mean_iter_ms"] = float(np.mean(iter_times[-500:])) if iter_times else 0
            eval_result["cumulative_wall_s"] = time.perf_counter() - t_start
            results["training_metrics"]["checkpoints"][str(iteration)] = eval_result
            lpips_str = f" LPIPS={eval_result.get('lpips','?')}"
            if isinstance(eval_result.get('lpips'), float):
                lpips_str = f" LPIPS={eval_result['lpips']:.4f}"
            print(f"  [{method}/{scene} iter {iteration:>6d}] PSNR={eval_result['psnr']:.2f} "
                  f"SSIM={eval_result['ssim']:.4f}{lpips_str} GS={model._xyz.shape[0]:,} "
                  f"iter={eval_result['mean_iter_ms']:.1f}ms", flush=True)

        if iteration in checkpoint_iterations:
            ckpt = model.capture()
            ckpt["optimizer_state_dict"] = model.optimizer.state_dict()
            ckpt["iteration"] = iteration
            ckpt["scene"] = scene
            ckpt["method"] = method
            ckpt["accutile"] = accutile
            ckpt_path = os.path.join(ckpt_dir, f"iter_{iteration}.pt")
            torch.save(ckpt, ckpt_path)
            results["checkpoints_saved"].append({
                "iter": iteration, "path": ckpt_path,
                "n_gaussians": model._xyz.shape[0],
            })
            # Keep only the 30K checkpoint to save disk; delete intermediates.
            if iteration != 30000:
                try:
                    os.remove(ckpt_path)
                except OSError:
                    pass

    total_time = time.perf_counter() - t_start

    all_cams = list(range(n_cameras))
    final_eval = evaluate_model(model, dataset, ssim_fn, model.active_sh_degree, all_cams, accutile, lpips_fn)
    final_eval["n_gaussians"] = model._xyz.shape[0]
    final_eval["n_eval_cameras"] = n_cameras
    results["final_eval"] = final_eval
    lpips_str = f" LPIPS={final_eval.get('lpips','?')}"
    if isinstance(final_eval.get('lpips'), float):
        lpips_str = f" LPIPS={final_eval['lpips']:.4f}"
    print(f"  [{method}/{scene} FINAL] PSNR={final_eval['psnr']:.2f} SSIM={final_eval['ssim']:.4f}"
          f"{lpips_str} GS={final_eval['n_gaussians']:,}", flush=True)

    iter_times_arr = np.array(iter_times)
    results["timing"] = {
        "total_wall_s": total_time,
        "total_wall_min": total_time / 60,
        "mean_iter_ms": float(np.mean(iter_times_arr)),
        "median_iter_ms": float(np.median(iter_times_arr)),
        "std_iter_ms": float(np.std(iter_times_arr)),
        "n_iters": config.iterations,
        "steady_state_mean_iter_ms": float(np.mean(iter_times_arr[15000:])) if len(iter_times_arr) > 15000 else 0,
        "steady_state_median_iter_ms": float(np.median(iter_times_arr[15000:])) if len(iter_times_arr) > 15000 else 0,
    }
    results["total_clones"] = total_clones
    results["total_splits"] = total_splits
    results["total_prunes"] = total_prunes
    results["total_opacity_resets"] = total_opacity_resets
    results["final_N"] = model._xyz.shape[0]
    results["peak_vram_gb"] = peak_vram
    results["densification_events"] = {
        "iters": per_event_iters,
        "clones": per_event_clones,
        "splits": per_event_splits,
        "prunes": per_event_prunes,
    }
    results["provenance"] = get_provenance(REPO_ROOT, scene, method, accutile)

    print(f"\n  [{method}/{scene}] Total wall: {total_time/60:.1f} min")
    print(f"  Mean iter: {np.mean(iter_times_arr):.2f} ms  Median: {np.median(iter_times_arr):.2f} ms")
    print(f"  Final N: {model._xyz.shape[0]:,}  Peak VRAM: {peak_vram:.2f} GB")

    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump({
            "scene": scene, "method": method, "accutile": accutile,
            "training_metrics": results["training_metrics"],
            "timing": results["timing"],
            "total_clones": total_clones, "total_splits": total_splits,
            "total_prunes": total_prunes, "total_opacity_resets": total_opacity_resets,
            "final_N": results["final_N"], "initial_N": initial_N,
            "densification_events": results["densification_events"],
            "checkpoints_saved": [c for c in results["checkpoints_saved"] if c["iter"] == 30000],
            "final_eval": final_eval,
            "peak_vram_gb": peak_vram,
        }, f, indent=2)
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump({
            "scene": scene, "method": method, "accutile": accutile,
            "final_eval": final_eval,
            "checkpoint_eval": results["training_metrics"]["checkpoints"],
        }, f, indent=2)
    with open(os.path.join(output_dir, "provenance.json"), "w") as f:
        json.dump(results["provenance"], f, indent=2)
    with open(os.path.join(output_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # MANIFEST.sha256
    manifest_lines = []
    for fname in ["config.json", "provenance.json", "training_metrics.json",
                  "evaluation_metrics.json", "camera_sequence.npy", "training_results.json"]:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            h = hashlib.sha256(open(fpath, "rb").read()).hexdigest()
            manifest_lines.append(f"{h}  {fname}")
    for fname in os.listdir(ckpt_dir):
        fpath = os.path.join(ckpt_dir, fname)
        if os.path.isfile(fpath):
            h = hashlib.sha256(open(fpath, "rb").read()).hexdigest()
            manifest_lines.append(f"{h}  checkpoints/{fname}")
    with open(os.path.join(output_dir, "MANIFEST.sha256"), "w") as f:
        f.write("\n".join(manifest_lines) + "\n")

    print(f"\n  Results saved to {output_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--method", required=True, choices=["b1", "b1a"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    select_gsplat_tree(args.method)
    train(args.scene, args.method, args.output_dir, args.gpu)
