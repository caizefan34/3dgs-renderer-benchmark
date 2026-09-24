#!/usr/bin/env python3
"""
Track A: C42 Scale Sweep + Gradient Analysis.

Runs 5K training at a given SSIM scale, with gradient cosine/magnitude
analysis at checkpoints (1000, 3000, 5000).

Usage: python3 track_a_scale_sweep.py --scale 0.75
       python3 track_a_scale_sweep.py --scale 1.0  (baseline)
"""
import json, math, sys, time, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "epic05" / "phase7"))

from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint

DEVICE = "cuda"
TILE_SIZE = 16; PACKED = True; EPS2D = 0.1; RADIUS_CLIP = 0.0
LAMBDA_DSSIM = 0.2; SEED = 42; NUM_ITERS = 5000
EVAL_INTERVAL = 500; GRAD_ANALYSIS_ITERS = [1000, 3000, 5000]

DENSIFICATION_INTERVAL = 100; DENSIFICATION_GRAD_THRESHOLD = 0.0002
DENSIFICATION_START = 500; DENSIFICATION_END = 15000
CLONE_MAX_SCREEN_SIZE = 100.0; SPLIT_MAX_SCREEN_SIZE = 100.0
PRUNE_INTERVAL = 100; PRUNE_OPACITY_THRESHOLD = 0.005; PRUNE_START = 500
RESET_OPACITY_INTERVAL = 3000; SH_DEGREE_INTERVAL = 1000; MAX_SH_DEGREE = 3

EVAL_CAMERAS = list(range(0, 311, 25))
GRAD_CAMERAS = [0, 100, 200]  # cameras for gradient analysis


def d_ssim_loss(pred, target, scale=1.0, window_size=11, sigma=1.5):
    if pred.ndim == 3: pred = pred.unsqueeze(0).permute(0,3,1,2); target = target.unsqueeze(0).permute(0,3,1,2)
    if scale < 1.0:
        pred = F.interpolate(pred, scale_factor=scale, mode="area", recompute_scale_factor=False)
        target = F.interpolate(target, scale_factor=scale, mode="area", recompute_scale_factor=False)
    coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype) - window_size//2
    k1d = torch.exp(-(coords**2)/(2*sigma**2)); k1d = k1d/k1d.sum()
    kernel = (k1d[:,None]*k1d[None,:]).expand(pred.shape[1],1,window_size,window_size).contiguous()
    C1, C2 = 0.01**2, 0.03**2
    def blur(x): return F.conv2d(x, kernel, padding=window_size//2, groups=pred.shape[1])
    mu_p, mu_t = blur(pred), blur(target)
    ssim = ((2*mu_p*mu_t+C1)*(2*(blur(pred*target)-mu_p*mu_t)+C2))/((mu_p**2+mu_t**2+C1)*(blur(pred**2)-mu_p**2+blur(target**2)-mu_t**2+C2))
    return 1.0 - ssim.mean()


def compute_loss(pred, gt, scale):
    l1 = F.l1_loss(pred, gt)
    dsim = d_ssim_loss(pred, gt, scale=scale)
    return (1.0-LAMBDA_DSSIM)*l1 + LAMBDA_DSSIM*dsim, l1, dsim


def make_optimizer(model, sls):
    return torch.optim.Adam([
        {"params":[model.xyz],"lr":1.6e-4*sls,"eps":1e-15,"betas":(0.9,0.999)},
        {"params":[model.rotations],"lr":1e-3,"eps":1e-15,"betas":(0.9,0.999)},
        {"params":[model.scales],"lr":5e-3,"eps":1e-15,"betas":(0.9,0.999)},
        {"params":[model.opacity],"lr":5e-2,"eps":1e-15,"betas":(0.9,0.999)},
        {"params":[model.shs],"lr":2.5e-3,"eps":1e-15,"betas":(0.9,0.999)},
    ])


def render(model, cam):
    data = model.forward()
    r, _, _ = rasterization(means=data["xyz"],quats=data["rotations"],scales=data["scales"],
        opacities=data["opacity"],colors=data["shs"],viewmats=cam.viewmatrix.unsqueeze(0),
        Ks=cam.K.unsqueeze(0),width=cam.image_width,height=cam.image_height,
        tile_size=TILE_SIZE,packed=PACKED,sh_degree=model.sh_degree,
        radius_clip=RADIUS_CLIP,eps2d=EPS2D,render_mode="RGB")
    return r[0].clamp(0,1)


def evaluate(model, dataset, cam_indices):
    psnrs, ssims = [], []
    for ci in cam_indices:
        cam = dataset.get_camera(ci); gt = dataset.get_gt_image(ci)
        with torch.no_grad():
            pred = render(model, cam)
            mse = float(((pred-gt)**2).mean())
            psnrs.append(10*math.log10(1.0/max(mse,1e-10)))
            ssims.append(1.0-float(d_ssim_loss(pred,gt,scale=1.0)))
    return float(np.mean(psnrs)), float(np.mean(ssims))


def gradient_analysis(model, dataset, scale, cam_indices):
    """Compute gradient cosine and magnitude ratio vs baseline (scale=1.0)."""
    results = {"scale": scale, "per_camera": [], "summary": {}}
    cosines = {p: [] for p in ["xyz", "rotations", "scales", "opacity", "shs"]}
    magnitudes = {p: [] for p in ["xyz", "rotations", "scales", "opacity", "shs"]}

    for ci in cam_indices:
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)

        # Baseline gradients (scale=1.0)
        for p in model.parameters():
            p.grad = None
        pred1 = render(model, cam)
        loss_base, _, _ = compute_loss(pred1, gt, scale=1.0)
        loss_base.backward()
        grad_base = {p: getattr(model, p).grad.detach().clone() for p in cosines}

        # Scaled gradients
        for p in model.parameters():
            p.grad = None
        pred2 = render(model, cam)
        loss_scaled, _, _ = compute_loss(pred2, gt, scale=scale)
        loss_scaled.backward()
        grad_scaled = {p: getattr(model, p).grad.detach().clone() for p in cosines}

        # Reset grads
        for p in model.parameters():
            p.grad = None

        # Compute cosine and magnitude ratio
        cam_result = {"cam": ci}
        for p in cosines:
            g1 = grad_base[p].flatten()
            g2 = grad_scaled[p].flatten()
            cos = float(F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item())
            mag_ratio = float(g2.norm().item() / max(g1.norm().item(), 1e-10))
            cosines[p].append(cos)
            magnitudes[p].append(mag_ratio)
            cam_result[p] = {"cosine": cos, "magnitude_ratio": mag_ratio}
        results["per_camera"].append(cam_result)

    # Summary
    for p in cosines:
        results["summary"][p] = {
            "mean_cosine": float(np.mean(cosines[p])),
            "min_cosine": float(np.min(cosines[p])),
            "mean_magnitude_ratio": float(np.mean(magnitudes[p])),
            "std_magnitude_ratio": float(np.std(magnitudes[p])),
        }
    return results


def train_scale(dataset, sfm_data, scale):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = GaussianModel(num_points=sfm_data["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=DEVICE)
    model.init_from_sfm(xyz=sfm_data["xyz"],
        opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0],1),0.1,device=DEVICE)),
        scales_log=sfm_data.get("scales"),rotations_raw=sfm_data.get("rotations"),shs=sfm_data.get("shs"))
    sls = float(sfm_data["xyz"].norm(dim=-1).max().item())
    optimizer = make_optimizer(model, sls)
    num_cameras = len(dataset)

    results = {"scale": scale, "eval_points": [], "gradient_analysis": [], "topology_events": [], "timing": {}}
    cum_cloned = 0; cum_split = 0; cum_pruned = 0
    iter_times = []
    t0_total = time.perf_counter()

    for iteration in range(1, NUM_ITERS+1):
        cam_idx = (iteration-1) % num_cameras
        cam = dataset.get_camera(cam_idx); gt = dataset.get_gt_image(cam_idx)
        new_deg = min(MAX_SH_DEGREE, iteration // SH_DEGREE_INTERVAL)
        if new_deg != model.sh_degree and new_deg <= MAX_SH_DEGREE:
            model.set_sh_degree(new_deg); optimizer = make_optimizer(model, sls)

        t0 = time.perf_counter()
        pred = render(model, cam)
        loss, l1_val, dsim_val = compute_loss(pred, gt, scale)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        model.accumulate_positional_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        denf = {"cloned":0,"split":0,"removed":0}
        if iteration >= DENSIFICATION_START and iteration < DENSIFICATION_END and iteration % DENSIFICATION_INTERVAL == 0:
            denf = model.densification(grad_threshold=DENSIFICATION_GRAD_THRESHOLD,
                clone_max_screen_size=CLONE_MAX_SCREEN_SIZE, split_max_screen_size=SPLIT_MAX_SCREEN_SIZE)
        prune_count = 0
        if iteration >= PRUNE_START and iteration % PRUNE_INTERVAL == 0:
            prune_count = model.prune_and_reset(opacity_threshold=PRUNE_OPACITY_THRESHOLD,
                reset_interval=RESET_OPACITY_INTERVAL, current_step=iteration)
        if denf["cloned"]+denf["split"]+prune_count > 0:
            cum_cloned += denf["cloned"]; cum_split += denf["split"]; cum_pruned += prune_count
            optimizer = make_optimizer(model, sls)
            results["topology_events"].append({"iter": iteration, "cloned": denf["cloned"],
                "split": denf["split"], "pruned": prune_count, "n_gaussians": model.xyz.shape[0],
                "cum_cloned": cum_cloned, "cum_split": cum_split, "cum_pruned": cum_pruned})

        torch.cuda.synchronize()
        iter_ms = (time.perf_counter()-t0)*1000
        iter_times.append(iter_ms)

        if iteration % EVAL_INTERVAL == 0:
            psnr, ssim = evaluate(model, dataset, EVAL_CAMERAS)
            results["eval_points"].append({"iter": iteration, "psnr": psnr, "ssim": ssim,
                "n_gaussians": model.xyz.shape[0], "l1": float(l1_val.item()),
                "d_ssim": float(dsim_val.item()), "mean_iter_ms": float(np.mean(iter_times[-EVAL_INTERVAL:]))})
            print(f"  [eval@{iteration}] PSNR={psnr:.2f}  SSIM={ssim:.4f}  GS={model.xyz.shape[0]:,}  "
                f"L1={float(l1_val.item()):.4f}  iter={np.mean(iter_times[-EVAL_INTERVAL:]):.1f}ms")

        # Gradient analysis at checkpoints
        if iteration in GRAD_ANALYSIS_ITERS:
            print(f"  [grad analysis@{iteration}]...")
            ga = gradient_analysis(model, dataset, scale, GRAD_CAMERAS)
            ga["iter"] = iteration
            results["gradient_analysis"].append(ga)
            for p, s in ga["summary"].items():
                print(f"    {p:>12s}: cosine={s['mean_cosine']:.4f}  mag_ratio={s['mean_magnitude_ratio']:.4f}")

    total = time.perf_counter()-t0_total
    results["timing"] = {"total_wall_s": total, "mean_iter_ms": float(np.mean(iter_times))}
    results["cumulative_topology"] = {"cloned": cum_cloned, "split": cum_split, "pruned": cum_pruned}
    print(f"  Total: {total:.1f}s  Mean iter: {np.mean(iter_times):.2f}ms")
    print(f"  Cumulative: clone={cum_cloned:,} split={cum_split:,} prune={cum_pruned:,}")
    del model; torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    scale = args.scale

    print(f"{'='*72}")
    print(f"Track A: Scale Sweep (scale={scale})")
    print(f"{'='*72}")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  GPU: {gpu_name}  Scale: {scale}  Iters: {NUM_ITERS}")

    repo_root = Path(__file__).resolve().parent.parent.parent
    dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device=DEVICE)
    sfm_data = load_initial_checkpoint("room", repo_root, device=DEVICE)
    print(f"  SfM points: {sfm_data['xyz'].shape[0]:,}")

    results = train_scale(dataset, sfm_data, scale)

    output = {
        "experiment": f"Track A Scale Sweep (scale={scale})",
        "hardware": {"gpu": gpu_name},
        "config": {"seed": SEED, "num_iters": NUM_ITERS, "scene": "room", "scale": scale,
                   "lambda_dssim": LAMBDA_DSSIM, "grad_analysis_iters": GRAD_ANALYSIS_ITERS,
                   "grad_cameras": GRAD_CAMERAS},
        "results": results,
    }
    save_path = repo_root / "results" / "a100" / "phase-c42" / f"track_a_scale_{scale}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data saved to {save_path}")


if __name__ == "__main__":
    main()
