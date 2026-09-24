#!/usr/bin/env python3
"""
Phase C44 unified training experiment.

Supports:
  Track A: Adaptive SSIM resolution schedule (--track A --schedule A1/A2/A3)
  Track C: Adaptive SSIM frequency (--track C --freq 2/4/8)
  Baseline: --track baseline (scale=1.0, SSIM every iter)

Training config (consistent with Track 0v2):
  5000 iters, aggressive pruning (threshold=0.05, grad=0.002)
  SH degree +1 every 1000 iters, opacity reset every 3000 iters
  Room scene, 1080p, seed=42
"""
import json, math, sys, time, argparse, numpy as np
from pathlib import Path
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/src")))
sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")))
from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint
from loss import d_ssim_loss

DEVICE = "cuda"
SEED = 42
TOTAL_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_CAMERAS = list(range(0, 311, 24))[:13]
DENSIFY_START = 500
DENSIFY_END = 5000
DENSIFY_INTERVAL = 100
PRUNE_THRESHOLD = 0.05
GRAD_THRESHOLD = 0.002

# Adaptive schedules
SCHEDULES = {
    "baseline": [(0, 1.0)],
    "A1": [(0, 0.5), (1000, 0.75)],
    "A2": [(0, 0.5), (2000, 0.75)],
    "A3": [(0, 0.75), (3000, 1.0)],
}

def get_scale(schedule, iter_idx):
    """Get current SSIM scale for given iteration."""
    scale = 1.0
    for threshold, s in SCHEDULES[schedule]:
        if iter_idx >= threshold:
            scale = s
    return scale

def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    return float(20 * math.log10(1.0 / math.sqrt(mse.item()))) if mse > 1e-10 else 100.0

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

def compute_loss_with_ssim(pred, gt, scale):
    """Full loss with SSIM at given scale."""
    l1 = F.l1_loss(pred, gt)
    if scale < 1.0:
        pred_s = F.interpolate(pred.unsqueeze(0).permute(0,3,1,2), scale_factor=scale, mode="area").squeeze(0).permute(1,2,0)
        gt_s = F.interpolate(gt.unsqueeze(0).permute(0,3,1,2), scale_factor=scale, mode="area").squeeze(0).permute(1,2,0)
    else:
        pred_s, gt_s = pred, gt
    dsim = d_ssim_loss(pred_s, gt_s)
    return 0.8 * l1 + 0.2 * dsim

def compute_loss_l1_only(pred, gt):
    """L1 only, no SSIM (for Track C non-SSIM iterations)."""
    return F.l1_loss(pred, gt)

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

def train(args, dataset, sfm_data):
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
    optimizer = torch.optim.Adam(lr_params, eps=1e-15)
    n_cams = len(dataset)
    cam_indices = list(range(n_cams))
    np.random.shuffle(cam_indices)
    results = {
        "track": args.track, "config": args.config,
        "iters": TOTAL_ITERS,
        "per_iter_times": [], "ssim_times": [],
        "eval_points": [], "gs_trajectory": [],
        "total_train_time_s": 0.0,
    }
    total_clone, total_split, total_prune = 0, 0, 0
    train_start = time.perf_counter()

    for iter_idx in range(TOTAL_ITERS):
        model.train()
        ci = cam_indices[iter_idx % n_cams]
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)
        data = model.forward()
        pred = render(model, cam, data)

        # Determine loss computation
        use_ssim = True
        current_scale = 1.0

        if args.track == "A":
            current_scale = get_scale(args.config, iter_idx)
        elif args.track == "C":
            freq = int(args.config.replace("freq", ""))
            use_ssim = (iter_idx % freq == 0)
            current_scale = 1.0  # Track C always uses scale=1.0
        elif args.track == "AC":
            # Combined: adaptive schedule + frequency
            current_scale = get_scale(args.config.split("_")[0], iter_idx)
            freq = int(args.config.split("_")[1].replace("freq", ""))
            use_ssim = (iter_idx % freq == 0)

        # Time SSIM if used
        ssim_time = 0.0
        if iter_idx % 100 == 0:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if use_ssim:
                loss = compute_loss_with_ssim(pred, gt, current_scale)
            else:
                loss = compute_loss_l1_only(pred, gt)
            torch.cuda.synchronize()
            iter_t0 = time.perf_counter()
            ssim_time = (time.perf_counter() - t0) * 1000
        else:
            if use_ssim:
                loss = compute_loss_with_ssim(pred, gt, current_scale)
            else:
                loss = compute_loss_l1_only(pred, gt)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Densification with continuous pruning
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
            iter_time = (time.perf_counter() - iter_t0) * 1000
            results["per_iter_times"].append(iter_time)
            results["ssim_times"].append(ssim_time)
            results["gs_trajectory"].append({"iter": iter_idx, "n_gaussians": model.xyz.shape[0]})

        if iter_idx % EVAL_INTERVAL == 0 or iter_idx == TOTAL_ITERS - 1:
            psnr, ssim_v, lpips_v = evaluate(model, dataset)
            gs_count = model.xyz.shape[0]
            results["eval_points"].append({
                "iter": iter_idx, "psnr": psnr, "ssim": ssim_v, "lpips": lpips_v,
                "gaussians": gs_count,
                "clone": total_clone, "split": total_split, "prune": total_prune,
                "scale": current_scale, "use_ssim": use_ssim,
            })
            print(f"  [{args.track}/{args.config}@{iter_idx}] PSNR={psnr:.2f} SSIM={ssim_v:.4f} "
                  f"GS={gs_count:,} scale={current_scale} ssim={'Y' if use_ssim else 'N'}")

    results["total_train_time_s"] = time.perf_counter() - train_start
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True, choices=["baseline", "A", "C", "AC"])
    parser.add_argument("--config", required=True,
                        help="baseline: 'baseline'; A: 'A1'/'A2'/'A3'; C: 'freq2'/'freq4'/'freq8'")
    args = parser.parse_args()

    repo_root = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 72)
    print(f"Phase C44: track={args.track} config={args.config}")
    print("=" * 72)

    dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device=DEVICE)
    sfm_data = load_initial_checkpoint("room", repo_root, device=DEVICE)
    print(f"  SfM points: {sfm_data['xyz'].shape[0]:,}")

    results = train(args, dataset, sfm_data)

    final = results["eval_points"][-1]
    iter_times = results["per_iter_times"][1:]  # skip first
    total_time = results["total_train_time_s"]

    print(f"\n{'='*72}")
    print(f"SUMMARY: {args.track}/{args.config}")
    print(f"{'='*72}")
    print(f"  Total train time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Mean iter time:   {np.mean(iter_times):.1f} ms")
    print(f"  Mean SSIM time:   {np.mean(results['ssim_times']):.2f} ms")
    print(f"  Final PSNR:       {final['psnr']:.2f}")
    print(f"  Final SSIM:       {final['ssim']:.4f}")
    print(f"  Final LPIPS:      {final['lpips']:.4f}")
    print(f"  Final GS:         {final['gaussians']:,}")

    config_str = args.config.replace(".", "")
    save_path = repo_root / "results" / "a100" / "phase-c44" / f"{args.track}_{config_str}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {save_path}")

if __name__ == "__main__":
    main()
