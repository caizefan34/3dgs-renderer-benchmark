#!/usr/bin/env python3
"""
Track B: C42 Scale Ablation — comprehensive measurement.

Scales: 1.0, 0.875, 0.75, 0.625, 0.5
5K training, room scene, seed=42.

Measures per 100 iters:
- Iteration time (full)
- SSIM time (isolated)
- L1 time (isolated)

Measures every 500 iters:
- PSNR (13 cams, full-res)
- SSIM metric (13 cams, full-res)
- Gaussian count
- Clone/split/prune stats

Measures at iters 1000/3000/5000:
- Gradient cosine vs scale=1.0
- Gradient magnitude ratio vs scale=1.0
"""
import json, math, sys, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "epic05" / "phase7"))
sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/src")))
sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")))

from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint
from loss import d_ssim_loss, combined_loss

DEVICE = "cuda"
SEED = 42
EVAL_INTERVAL = 500
GRAD_ANALYSIS_ITERS = [1000, 3000, 5000]
EVAL_CAMERAS = list(range(0, 311, 24))[:13]
GRAD_CAMERAS = [0, 100, 200]
TOTAL_ITERS = 5000

def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    if mse < 1e-10:
        return 100.0
    return float(20 * math.log10(1.0 / math.sqrt(mse.item())))

def compute_ssim_metric(pred, gt):
    return float(1.0 - d_ssim_loss(pred, gt))

def compute_lpips_proxy(pred, gt):
    """LPIPS proxy: L1 in feature space (no VGG available)."""
    return float(F.l1_loss(pred, gt).item())

def render(model, cam, data=None):
    if data is None:
        data = model.forward()
    r, _, _ = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=model.sh_degree,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB")
    return r[0].clamp(0, 1)

def compute_loss_timed(pred, gt, scale):
    """Compute loss with timing breakdown."""
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
    loss = (1 - 0.2) * l1 + 0.2 * dsim
    return loss

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
            lpipses.append(compute_lpips_proxy(pred, gt))
    return float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(lpipses))

def gradient_analysis(model, dataset, scale, cam_indices):
    """Compute gradient cosine and magnitude ratio vs scale=1.0."""
    model.train()
    results = {"per_camera": [], "summary": {}}
    param_names = ["xyz", "rotations", "scales", "opacity", "shs"]
    cosines = {p: [] for p in param_names}
    magnitudes = {p: [] for p in param_names}

    for ci in cam_indices:
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)

        # Baseline gradients (scale=1.0)
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        data = model.forward()
        pred = render(model, cam, data)
        loss_base = compute_loss_only(pred, gt, scale=1.0)
        loss_base.backward()
        grad_base = {}
        for pname in param_names:
            p = getattr(model, pname)
            if p.grad is not None:
                grad_base[pname] = p.grad.detach().clone()

        # Scaled gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        data2 = model.forward()
        pred2 = render(model, cam, data2)
        loss_scaled = compute_loss_only(pred2, gt, scale=scale)
        loss_scaled.backward()
        grad_scaled = {}
        for pname in param_names:
            p = getattr(model, pname)
            if p.grad is not None:
                grad_scaled[pname] = p.grad.detach().clone()

        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

        for pname in param_names:
            if pname in grad_base and pname in grad_scaled:
                g1, g2 = grad_base[pname].flatten(), grad_scaled[pname].flatten()
                cos = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
                mag1, mag2 = g1.norm().item(), g2.norm().item()
                ratio = mag2 / mag1 if mag1 > 1e-10 else 0.0
                cosines[pname].append(cos)
                magnitudes[pname].append(ratio)

    for pname in param_names:
        results["summary"][pname] = {
            "cosine_mean": float(np.mean(cosines[pname])) if cosines[pname] else 0,
            "magnitude_ratio_mean": float(np.mean(magnitudes[pname])) if magnitudes[pname] else 0,
        }
    return results

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
        "eval_points": [], "grad_analysis": {},
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

        # Densification
        if iter_idx >= 500 and iter_idx < 4000 and iter_idx % 100 == 0:
            model.accumulate_positional_gradient()
            counts = model.densification(grad_threshold=0.0002)
            total_clone += counts["cloned"]
            total_split += counts["split"]
            total_prune += counts["removed"]

        if iter_idx % 1000 == 0 and iter_idx > 0:
            if model.sh_degree < 3:
                model.set_sh_degree(model.sh_degree + 1)

        if iter_idx > 0 and iter_idx % 3000 == 0:
            removed = model.prune_and_reset(opacity_threshold=0.005, current_step=iter_idx)
            total_prune += removed

        optimizer.step()

        if iter_idx % 100 == 0:
            torch.cuda.synchronize()
            iter_time = (time.perf_counter() - t0) * 1000
            results["per_iter_times"].append(iter_time)

        # Evaluation
        if iter_idx % EVAL_INTERVAL == 0 or iter_idx == total_iters - 1:
            psnr, ssim_v, lpips_v = evaluate(model, dataset)
            gs_count = model.xyz.shape[0]
            results["eval_points"].append({
                "iter": iter_idx, "psnr": psnr, "ssim": ssim_v, "lpips": lpips_v,
                "gaussians": gs_count,
                "clone": total_clone, "split": total_split, "prune": total_prune,
            })
            print(f"  [s={scale}@{iter_idx}] PSNR={psnr:.2f} SSIM={ssim_v:.4f} LPIPS={lpips_v:.4f} GS={gs_count:,}")

        # Gradient analysis
        if iter_idx in GRAD_ANALYSIS_ITERS:
            print(f"  [grad analysis@{iter_idx}]...")
            ga = gradient_analysis(model, dataset, scale, GRAD_CAMERAS)
            results["grad_analysis"][str(iter_idx)] = ga
            for p in ["xyz", "rotations", "scales", "opacity", "shs"]:
                s = ga["summary"][p]
                print(f"    {p}: cosine={s['cosine_mean']:.4f} mag_ratio={s['magnitude_ratio_mean']:.4f}")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 72)
    print(f"Track B: C42 Scale Ablation (scale={args.scale})")
    print("=" * 72)
    print(f"  GPU: {torch.cuda.get_device_name(0)}  Scale: {args.scale}  Iters: {TOTAL_ITERS}")

    dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device=DEVICE)
    sfm_data = load_initial_checkpoint("room", repo_root, device=DEVICE)
    print(f"  SfM points: {sfm_data['xyz'].shape[0]:,}")

    results = train_scale(dataset, sfm_data, args.scale)

    # Summary
    iter_times = results["per_iter_times"]
    ssim_times = results["ssim_times"]
    l1_times = results["l1_times"]
    final = results["eval_points"][-1]

    print(f"\n{'='*72}")
    print(f"SUMMARY: scale={args.scale}")
    print(f"{'='*72}")
    print(f"  Mean iter time: {np.mean(iter_times):.1f} ms")
    print(f"  Mean SSIM time: {np.mean(ssim_times):.2f} ms")
    print(f"  Mean L1 time:   {np.mean(l1_times):.2f} ms")
    print(f"  Final PSNR:     {final['psnr']:.2f}")
    print(f"  Final SSIM:     {final['ssim']:.4f}")
    print(f"  Final GS:       {final['gaussians']:,}")
    print(f"  Clone/Split/Prune: {final['clone']}/{final['split']}/{final['prune']}")

    scale_str = str(args.scale).replace(".", "")
    save_path = repo_root / "results" / "a100" / "phase-c42" / f"track_b_ablation_{scale_str}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Data saved to {save_path}")

if __name__ == "__main__":
    main()
