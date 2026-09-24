#!/usr/bin/env python3
"""
C44 Adaptive SH Schedule Screening (5K).

Hypothesis: SH degree should increase based on loss convergence, not fixed iteration intervals.
Baseline: SH degree increases at iters 1000, 2000, 3000 (fixed, every 1000).
Candidate: SH degree increases when L1 loss improvement over last 200 iters < threshold.

Single-module: only SH degree scheduling changes. All other training components unchanged.
"""
import json, math, sys, time
from pathlib import Path
from collections import deque

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
TILE_SIZE = 16
PACKED = True
EPS2D = 0.1
RADIUS_CLIP = 0.0
LAMBDA_DSSIM = 0.2
SEED = 42
NUM_ITERS = 5000
EVAL_INTERVAL = 500

DENSIFICATION_INTERVAL = 100
DENSIFICATION_GRAD_THRESHOLD = 0.0002
DENSIFICATION_START = 500
DENSIFICATION_END = 15000
CLONE_MAX_SCREEN_SIZE = 100.0
SPLIT_MAX_SCREEN_SIZE = 100.0
PRUNE_INTERVAL = 100
PRUNE_OPACITY_THRESHOLD = 0.005
PRUNE_START = 500
RESET_OPACITY_INTERVAL = 3000
SH_DEGREE_INTERVAL = 1000
MAX_SH_DEGREE = 3

# Adaptive SH: increase degree when L1 improvement over window < threshold
ADAPTIVE_SH_WINDOW = 200  # check every 200 iters
ADAPTIVE_SH_IMPROVEMENT_THRESHOLD = 0.005  # min L1 improvement to stay at current degree

EVAL_CAMERAS = list(range(0, 311, 25))


def d_ssim_loss(pred, target, window_size=11, sigma=1.5):
    if pred.ndim == 3:
        pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
        target = target.unsqueeze(0).permute(0, 3, 1, 2)
    coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype) - window_size // 2
    k1d = torch.exp(-(coords**2)/(2*sigma**2)); k1d = k1d/k1d.sum()
    kernel = (k1d[:,None]*k1d[None,:]).expand(pred.shape[1],1,window_size,window_size).contiguous()
    C1, C2 = 0.01**2, 0.03**2
    def blur(x): return F.conv2d(x, kernel, padding=window_size//2, groups=pred.shape[1])
    mu_p, mu_t = blur(pred), blur(target)
    ssim = ((2*mu_p*mu_t+C1)*(2*(blur(pred*target)-mu_p*mu_t)+C2))/((mu_p**2+mu_t**2+C1)*(blur(pred**2)-mu_p**2+blur(target**2)-mu_t**2+C2))
    return 1.0 - ssim.mean()


def compute_loss(pred, gt):
    l1 = F.l1_loss(pred, gt)
    dsim = d_ssim_loss(pred, gt)
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
            ssims.append(1.0-float(d_ssim_loss(pred,gt)))
    return float(np.mean(psnrs)), float(np.mean(ssims))


def train_variant(dataset, sfm_data, adaptive_sh, variant_name):
    print(f"\n{'='*60}")
    print(f"  VARIANT {variant_name}: adaptive_sh={adaptive_sh}")
    print(f"{'='*60}")

    torch.manual_seed(SEED); np.random.seed(SEED)

    model = GaussianModel(num_points=sfm_data["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=DEVICE)
    model.init_from_sfm(xyz=sfm_data["xyz"],
        opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0],1),0.1,device=DEVICE)),
        scales_log=sfm_data.get("scales"),rotations_raw=sfm_data.get("rotations"),shs=sfm_data.get("shs"))
    sls = float(sfm_data["xyz"].norm(dim=-1).max().item())
    optimizer = make_optimizer(model, sls)
    num_cameras = len(dataset)

    results = {"variant": variant_name, "adaptive_sh": adaptive_sh, "eval_points": [], "sh_events": [], "timing": {}}
    l1_history = deque(maxlen=ADAPTIVE_SH_WINDOW)
    iter_times = []
    t0_total = time.perf_counter()

    for iteration in range(1, NUM_ITERS+1):
        cam_idx = (iteration-1) % num_cameras
        cam = dataset.get_camera(cam_idx); gt = dataset.get_gt_image(cam_idx)

        # SH degree scheduling
        if adaptive_sh:
            # Check every ADAPTIVE_SH_WINDOW iters
            if iteration % ADAPTIVE_SH_WINDOW == 0 and model.sh_degree < MAX_SH_DEGREE:
                if len(l1_history) >= ADAPTIVE_SH_WINDOW // 2:
                    recent = list(l1_history)[-ADAPTIVE_SH_WINDOW//2:]
                    old = list(l1_history)[:ADAPTIVE_SH_WINDOW//2]
                    improvement = float(np.mean(old) - np.mean(recent))
                    if improvement < ADAPTIVE_SH_IMPROVEMENT_THRESHOLD:
                        model.set_sh_degree(model.sh_degree + 1)
                        optimizer = make_optimizer(model, sls)
                        results["sh_events"].append({"iter": iteration, "new_degree": model.sh_degree,
                            "l1_improvement": improvement, "trigger": "loss_plateau"})
                        print(f"  [SH@{iteration}] degree -> {model.sh_degree} (improvement={improvement:.4f})")
        else:
            new_degree = min(MAX_SH_DEGREE, iteration // SH_DEGREE_INTERVAL)
            if new_degree != model.sh_degree and new_degree <= MAX_SH_DEGREE:
                model.set_sh_degree(new_degree)
                optimizer = make_optimizer(model, sls)
                results["sh_events"].append({"iter": iteration, "new_degree": new_degree, "trigger": "fixed_interval"})
                print(f"  [SH@{iteration}] degree -> {new_degree} (fixed)")

        t0 = time.perf_counter()
        pred = render(model, cam)
        loss, l1_val, dsim_val = compute_loss(pred, gt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        model.accumulate_positional_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Densification
        if iteration >= DENSIFICATION_START and iteration < DENSIFICATION_END and iteration % DENSIFICATION_INTERVAL == 0:
            dc = model.densification(grad_threshold=DENSIFICATION_GRAD_THRESHOLD,
                clone_max_screen_size=CLONE_MAX_SCREEN_SIZE, split_max_screen_size=SPLIT_MAX_SCREEN_SIZE)
            if dc["cloned"]+dc["split"] > 0:
                optimizer = make_optimizer(model, sls)
        if iteration >= PRUNE_START and iteration % PRUNE_INTERVAL == 0:
            pc = model.prune_and_reset(opacity_threshold=PRUNE_OPACITY_THRESHOLD,
                reset_interval=RESET_OPACITY_INTERVAL, current_step=iteration)
            if pc > 0:
                optimizer = make_optimizer(model, sls)

        torch.cuda.synchronize()
        iter_ms = (time.perf_counter()-t0)*1000
        iter_times.append(iter_ms)
        l1_history.append(float(l1_val.item()))

        if iteration % EVAL_INTERVAL == 0:
            psnr, ssim = evaluate(model, dataset, EVAL_CAMERAS)
            results["eval_points"].append({"iter": iteration, "psnr": psnr, "ssim": ssim,
                "n_gaussians": model.xyz.shape[0], "sh_degree": model.sh_degree,
                "mean_iter_ms": float(np.mean(iter_times[-EVAL_INTERVAL:]))})
            print(f"  [eval@{iteration}] PSNR={psnr:.2f}  SSIM={ssim:.4f}  GS={model.xyz.shape[0]:,}  "
                f"SH={model.sh_degree}  iter={np.mean(iter_times[-EVAL_INTERVAL:]):.1f}ms")

    total = time.perf_counter()-t0_total
    results["timing"] = {"total_wall_s": total, "mean_iter_ms": float(np.mean(iter_times))}
    print(f"  Total: {total:.1f}s  Mean iter: {np.mean(iter_times):.2f}ms")

    del model; torch.cuda.empty_cache()
    return results


def main():
    print("="*72)
    print("C44 Adaptive SH Schedule Screening (5K)")
    print("="*72)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  GPU: {gpu_name}  Seed: {SEED}  Iters: {NUM_ITERS}")

    repo_root = Path(__file__).resolve().parent.parent.parent
    dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device=DEVICE)
    sfm_data = load_initial_checkpoint("room", repo_root, device=DEVICE)
    print(f"  SfM points: {sfm_data['xyz'].shape[0]:,}")

    ra = train_variant(dataset, sfm_data, adaptive_sh=False, variant_name="A_baseline_fixed_sh")
    rb = train_variant(dataset, sfm_data, adaptive_sh=True, variant_name="B_adaptive_sh")

    # Compare
    print(f"\n{'='*72}")
    print("COMPARISON")
    print(f"{'='*72}")
    for ea, eb in zip(ra["eval_points"], rb["eval_points"]):
        print(f"  iter {ea['iter']:>5d}: A PSNR={ea['psnr']:.2f} SH={ea['sh_degree']}  "
              f"B PSNR={eb['psnr']:.2f} SH={eb['sh_degree']}  dPSNR={eb['psnr']-ea['psnr']:+.2f}")

    fa, fb = ra["eval_points"][-1], rb["eval_points"][-1]
    d_psnr = fb["psnr"] - fa["psnr"]
    d_ssim = fb["ssim"] - fa["ssim"]
    speedup = (1 - rb["timing"]["mean_iter_ms"]/ra["timing"]["mean_iter_ms"]) * 100

    print(f"\n  FINAL: dPSNR={d_psnr:+.2f}  dSSIM={d_ssim:+.4f}  speedup={speedup:+.1f}%")
    print(f"  SH events A: {ra['sh_events']}")
    print(f"  SH events B: {rb['sh_events']}")

    psnr_pass = d_psnr > -0.2
    speedup_pass = speedup > 0
    decision = "KEEP" if (psnr_pass and speedup_pass) else "DROP"
    print(f"\n  DECISION: {decision}")

    output = {
        "experiment": "C44 Adaptive SH Schedule Screening",
        "hypothesis": "SH degree should increase based on loss convergence, not fixed iteration intervals",
        "implementation": "Baseline: SH degree up every 1000 iters. Adaptive: SH degree up when L1 improvement over 200-iter window < 0.005",
        "hardware": {"gpu": gpu_name},
        "config": {"seed": SEED, "num_iters": NUM_ITERS, "scene": "room",
                   "adaptive_sh_window": ADAPTIVE_SH_WINDOW,
                   "adaptive_sh_threshold": ADAPTIVE_SH_IMPROVEMENT_THRESHOLD},
        "variant_A_baseline": ra, "variant_B_adaptive": rb,
        "analysis": {"d_psnr": d_psnr, "d_ssim": d_ssim, "speedup_pct": speedup,
                     "psnr_pass": psnr_pass, "speedup_pass": speedup_pass, "decision": decision},
    }
    save_path = repo_root / "results" / "a100" / "phase-c42" / "c44_adaptive_sh_screening.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Data saved to {save_path}")


if __name__ == "__main__":
    main()
