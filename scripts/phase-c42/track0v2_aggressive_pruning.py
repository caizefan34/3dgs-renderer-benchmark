#!/usr/bin/env python3
"""
Track 0v2: C42 Re-validation with AGGRESSIVE continuous pruning.

Uses higher opacity threshold (0.05) and higher grad threshold (0.002)
to keep GS count near 1M, matching old P1 conditions.
"""
import json, math, sys, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/src")))
sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")))

from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint
from loss import d_ssim_loss

DEVICE = "cuda"
SEED = 42
EVAL_INTERVAL = 500
EVAL_CAMERAS = list(range(0, 311, 24))[:13]
TOTAL_ITERS = 5000
DENSIFY_START = 500
DENSIFY_END = 5000
DENSIFY_INTERVAL = 100
PRUNE_THRESHOLD = 0.05   # 10x higher — more aggressive pruning
GRAD_THRESHOLD = 0.002   # 10x higher — less aggressive densification

def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    if mse < 1e-10: return 100.0
    return float(20 * math.log10(1.0 / math.sqrt(mse.item())))

def compute_ssim_metric(pred, gt):
    return float(1.0 - d_ssim_loss(pred, gt))

def render(model, cam, data=None):
    if data is None: data = model.forward()
    r, _, _ = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=model.sh_degree,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB")
    return r[0].clamp(0, 1)

def compute_loss_timed(pred, gt, scale):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    l1 = F.l1_loss(pred, gt)
    torch.cuda.synchronize()
    l1_time = (time.perf_counter() - t0) * 1000
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if scale < 1.0:
        pred_s = F.interpolate(pred.unsqueeze(0).permute(0, 3, 1, 2), scale_factor=scale, mode="area").squeeze(0).permute(1, 2, 0)
        gt_s = F.interpolate(gt.unsqueeze(0).permute(0, 3, 1, 2), scale_factor=scale, mode="area").squeeze(0).permute(1, 2, 0)
    else:
        pred_s, gt_s = pred, gt
    dsim = d_ssim_loss(pred_s, gt_s)
    torch.cuda.synchronize()
    ssim_time = (time.perf_counter() - t0) * 1000
    loss = (1 - 0.2) * l1 + 0.2 * dsim
    return loss, l1_time, ssim_time

def compute_loss_only(pred, gt, scale):
    l1 = F.l1_loss(pred, gt)
    if scale < 1.0:
        pred_s = F.interpolate(pred.unsqueeze(0).permute(0, 3, 1, 2), scale_factor=scale, mode="area").squeeze(0).permute(1, 2, 0)
        gt_s = F.interpolate(gt.unsqueeze(0).permute(0, 3, 1, 2), scale_factor=scale, mode="area").squeeze(0).permute(1, 2, 0)
    else:
        pred_s, gt_s = pred, gt
    dsim = d_ssim_loss(pred_s, gt_s)
    return (1 - 0.2) * l1 + 0.2 * dsim

def evaluate(model, dataset):
    psnrs, ssims, lpipses = [], [], []
    model.eval()
    with torch.no_grad():
        data = model.forward()
        for ci in EVAL_CAMERAS:
            cam = dataset.get_camera(ci)
            gt = dataset.get_gt_image(ci)
            pred = render(model, cam, data)
            psnrs.append(compute_psnr(pred, gt))
            ssims.append(compute_ssim_metric(pred, gt))
            lpipses.append(float(F.l1_loss(pred, gt).item()))
    return float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(lpipses))

def train_scale(dataset, sfm_data, scale, total_iters=TOTAL_ITERS):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    n_gauss = sfm_data["xyz"].shape[0]
    model = GaussianModel(num_points=n_gauss, sh_degree=0, max_sh_degree=3, device=DEVICE)
    model.init_from_sfm(
        xyz=sfm_data["xyz"],
        opacity_logit=torch.logit(torch.full((n_gauss, 1), 0.1, device=DEVICE)),
        scales_log=sfm_data.get("scales"),
        rotations_raw=sfm_data.get("rotations"),
        shs=sfm_data.get("shs"))
    model.set_sh_degree(0)
    lr_params = [
        {"params": [model.xyz], "lr": 1.6e-4, "name": "xyz"},
        {"params": [model.rotations], "lr": 1e-3, "name": "rotations"},
        {"params": [model.scales], "lr": 5e-3, "name": "scales"},
        {"params": [model.opacity], "lr": 5e-2, "name": "opacity"},
        {"params": [model.shs], "lr": 2.5e-3, "name": "shs"},
    ]
    optimizer = torch.optim.Adam(lr_params, eps=1e-15, betas=(0.9, 0.999))
    n_cams = len(dataset)
    cam_indices = list(range(n_cams))
    np.random.shuffle(cam_indices)
    results = {
        "scale": scale, "iters": total_iters,
        "per_iter_times": [], "ssim_times": [], "l1_times": [],
        "eval_points": [], "gs_trajectory": [],
    }
    total_clone, total_split, total_prune = 0, 0, 0
    for iter_idx in range(total_iters):
        model.train()
        ci = cam_indices[iter_idx % n_cams]
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)
        data = model.forward()
        pred = render(model, cam, data)
        if iter_idx % 100 == 0:
            loss, l1_time, ssim_time = compute_loss_timed(pred, gt, scale)
            results["ssim_times"].append(ssim_time)
            results["l1_times"].append(l1_time)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        else:
            loss = compute_loss_only(pred, gt, scale)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Densification with AGGRESSIVE pruning
        if iter_idx >= DENSIFY_START and iter_idx < DENSIFY_END and iter_idx % DENSIFY_INTERVAL == 0:
            model.accumulate_positional_gradient()
            counts = model.densification(grad_threshold=GRAD_THRESHOLD)
            total_clone += counts["cloned"]
            total_split += counts["split"]
            removed = model.prune(opacity_threshold=PRUNE_THRESHOLD)
            total_prune += removed
        if iter_idx % 1000 == 0 and iter_idx > 0:
            if model.sh_degree < 3:
                model.set_sh_degree(model.sh_degree + 1)
        if iter_idx > 0 and iter_idx % 3000 == 0:
            removed = model.prune_and_reset(opacity_threshold=PRUNE_THRESHOLD, current_step=iter_idx)
            total_prune += removed
        optimizer.step()
        if iter_idx % 100 == 0:
            torch.cuda.synchronize()
            iter_time = (time.perf_counter() - t0) * 1000
            results["per_iter_times"].append(iter_time)
            results["gs_trajectory"].append({"iter": iter_idx, "n_gaussians": model.xyz.shape[0]})
        if iter_idx % EVAL_INTERVAL == 0 or iter_idx == total_iters - 1:
            psnr, ssim_v, lpips_v = evaluate(model, dataset)
            gs_count = model.xyz.shape[0]
            results["eval_points"].append({
                "iter": iter_idx, "psnr": psnr, "ssim": ssim_v, "lpips": lpips_v,
                "gaussians": gs_count,
                "clone": total_clone, "split": total_split, "prune": total_prune,
            })
            print(f"  [s={scale}@{iter_idx}] PSNR={psnr:.2f} SSIM={ssim_v:.4f} GS={gs_count:,} (pruned={total_prune:,})")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()
    repo_root = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print("=" * 72)
    print(f"Track 0v2: C42 with aggressive pruning (scale={args.scale})")
    print(f"  prune_threshold={PRUNE_THRESHOLD} grad_threshold={GRAD_THRESHOLD}")
    print("=" * 72)
    dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device=DEVICE)
    sfm_data = load_initial_checkpoint("room", repo_root, device=DEVICE)
    print(f"  SfM points: {sfm_data['xyz'].shape[0]:,}")
    results = train_scale(dataset, sfm_data, args.scale)
    iter_times = results["per_iter_times"]
    ssim_times = results["ssim_times"]
    final = results["eval_points"][-1]
    print(f"\nSUMMARY: scale={args.scale}")
    print(f"  Mean iter: {np.mean(iter_times):.1f} ms  SSIM: {np.mean(ssim_times):.2f} ms")
    print(f"  Final PSNR: {final['psnr']:.2f}  GS: {final['gaussians']:,}")
    print(f"  Clone/Split/Prune: {final['clone']}/{final['split']}/{final['prune']}")
    scale_str = str(args.scale).replace(".", "")
    save_path = repo_root / "results" / "a100" / "phase-c42" / f"track0v2_pruned_{scale_str}.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {save_path}")

if __name__ == "__main__":
    main()
