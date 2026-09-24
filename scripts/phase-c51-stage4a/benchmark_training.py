#!/usr/bin/env python3
"""Phase C51 Stage 4A — 5K Training Validation

Tests CUDA sparse backward in actual training:
  - baseline: full backward (no mask)
  - K80-B1: K=80%, skip all gradient for masked, previous-grad densification
  - K70-B1: K=70%, skip all gradient for masked (higher speedup, unknown quality)
  - K80-B2: K=80%, compute v_means2d for masked (densification-safe)
  - K80-B3: K=80%, same kernel as B1, delayed densification ablation

Key differences from C51-R simulation:
  - C51-R: compute FULL backward, then mask gradients in Python
  - Stage 4A: CUDA kernel SKIPS gradient computation for masked Gaussians
  - This means xyz.grad is ZERO for masked Gaussians in B1/B3
  - B1: use previous-iteration stored gradient norm for densification
  - B2: CUDA computes v_means2d for masked → xyz.grad is correct for all
  - B3: same as B1 but tests whether delayed densification is acceptable

Acceptance gates:
  - Gate A: cosine >= 0.999 for selected Gaussians (validated in microbenchmark)
  - Gate B: PSNR degradation < 0.2 dB, SSIM < 0.005
  - Gate C: E2E speedup > 5%
  - Gate D: No unacceptable densification failure
"""
import json, math, sys, time, argparse, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/src")))
sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")))
from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint

DEVICE = "cuda"
SEED = 42
PRUNE_THRESHOLD = 0.01
GRAD_THRESHOLD = 0.001
EVAL_CAMERAS = list(range(0, 311, 24))[:13]
TRAJECTORY_INTERVAL = 100


class SepSSIM:
    """Separable SSIM (C44 method)."""
    def __init__(self, window_size=11, sigma=1.5, device="cuda"):
        self.C1 = (0.01) ** 2
        self.C2 = (0.03) ** 2
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        self.kernel = k1d.view(1, 1, window_size, 1).repeat(3, 1, 1, 1)
        self.window_size = window_size
        self.device = device

    def __call__(self, pred, gt):
        # pred, gt: [H, W, 3] -> [3, H, W]
        pred = pred.permute(2, 0, 1).unsqueeze(0)
        gt = gt.permute(2, 0, 1).unsqueeze(0)
        k = self.kernel
        C1 = self.C1
        C2 = self.C2
        mu_p = F.conv2d(pred, k, padding=5, groups=3)
        mu_t = F.conv2d(gt, k, padding=5, groups=3)
        mu_p2 = mu_p ** 2
        mu_t2 = mu_t ** 2
        mu_pt = mu_p * mu_t
        sp2 = F.conv2d(pred**2, k, padding=5, groups=3) - mu_p2
        st2 = F.conv2d(gt**2, k, padding=5, groups=3) - mu_t2
        spt = F.conv2d(pred*gt, k, padding=5, groups=3) - mu_pt
        ssim = (2*mu_pt+C1)*(2*spt+C2) / ((mu_p2+mu_t2+C1)*(sp2+st2+C2))
        return float(ssim.mean().item())


def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    if mse < 1e-12:
        return 100.0
    return float(20 * math.log10(1.0 / math.sqrt(mse.item())))


def render(model, cam, data=None, importance_mask=None, compute_densify_grad=False):
    if data is None:
        data = model.forward()
    r, _, _ = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=model.sh_degree,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB",
        importance_mask=importance_mask,
        compute_densify_grad=compute_densify_grad,
    )
    return r[0].clamp(0, 1)


def run_experiment(dataset, sfm_data, config):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    k_frac = config["keep_fraction"]
    design = config["design"]  # "baseline", "B1", "B2", "B3"
    iters = config["iters"]
    gpu_id = config.get("gpu_id", 0)

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    # Create model
    N = sfm_data['xyz'].shape[0]
    model = GaussianModel(num_points=N, sh_degree=3, max_sh_degree=3)
    model.init_from_sfm(
        xyz=sfm_data['xyz'].clone().float().to(device),
        opacity_logit=sfm_data['opacity'].clone().float().to(device),
        scales_log=sfm_data['scales'].clone().float().to(device),
        rotations_raw=sfm_data['rotations'].clone().float().to(device),
        shs=sfm_data['shs'].clone().float().to(device),
    )
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.get_optimizer_param_groups({
        "lr_xyz": 1.6e-4, "lr_rotation": 1e-3, "lr_scaling": 5e-3,
        "lr_opacity": 5e-2, "lr_sh": 2.5e-3,
    }), betas=(0.9, 0.999), eps=1e-8)

    sep_ssim = SepSSIM(device=device)

    # Previous gradient norm for mask prediction
    prev_grad_norm = None

    is_sparse = design in ("B1", "B2", "B3")
    compute_densify_grad = (design == "B2")
    
    # For B1/B3: densification uses previous-iteration gradient norm
    # For B2: densification uses current gradient (xyz.grad is correct for all)
    # For B3: densification is delayed by one iteration (same as B1 at kernel level)

    trajectory = []
    timing_stats = []
    densification_events = []

    def evaluate():
        model.eval()
        psnrs, ssims = [], []
        with torch.no_grad():
            for idx in EVAL_CAMERAS:
                cam = dataset.get_camera(idx)
                gt = dataset.get_gt_image(idx)
                data_eval = model.forward()
                pred = render(model, cam, data_eval)
                psnrs.append(compute_psnr(pred, gt))
                ssims.append(sep_ssim(pred, gt))
        model.train()
        return float(np.mean(psnrs)), float(np.mean(ssims))

    # Initial eval
    psnr, ssim = evaluate()
    print(f"  Iter 0: PSNR={psnr:.2f}, SSIM={ssim:.4f}, GS={model.xyz.shape[0]:,}")
    trajectory.append({
        "iter": 0, "psnr": psnr, "ssim": ssim,
        "gaussians": model.xyz.shape[0], "loss": 0.0,
    })

    # Training loop
    for iter_idx in range(1, iters + 1):
        model.train()
        cam_idx = (iter_idx - 1) % len(dataset)
        cam = dataset.get_camera(cam_idx)
        gt = dataset.get_gt_image(cam_idx)

        # Construct mask from previous gradient norm
        importance_mask = None
        if is_sparse and prev_grad_norm is not None:
            n_keep = max(1, int(N * k_frac))
            _, top_idx = torch.topk(prev_grad_norm, n_keep)
            importance_mask = torch.zeros(N, dtype=torch.uint8, device=device)
            importance_mask[top_idx] = 1

        # Forward + backward
        optimizer.zero_grad()
        
        # Time the iteration
        torch.cuda.synchronize(device)
        iter_start = time.perf_counter()
        
        data = model.forward()
        pred = render(model, cam, data, importance_mask, compute_densify_grad)
        
        # Loss (L1 only — SepSSIM in C51-R was non-differentiable)
        l1 = F.l1_loss(pred, gt)
        loss = l1  # L1-only, matching C51-R effective behavior
        
        loss.backward()
        
        torch.cuda.synchronize(device)
        backward_end = time.perf_counter()
        
        # Densification (every 100 iters, 500-15000)
        do_densify = (iter_idx % 100 == 0 and 500 <= iter_idx <= 15000)
        clone_count, split_count, prune_count = 0, 0, 0
        
        if do_densify:
            # All designs use the same accumulation path.
            # Baseline/B2: xyz.grad is correct for all Gaussians
            # B1/B3: xyz.grad is zero for masked Gaussians (densification decoupled)
            model.accumulate_positional_gradient()
            
            counts = model.densification(grad_threshold=GRAD_THRESHOLD)
            clone_count = counts["cloned"]
            split_count = counts["split"]
            
            # Prune low-opacity Gaussians (skip opacity reset — prune_and_reset has shape bug)
            prune_count = model.prune(opacity_threshold=PRUNE_THRESHOLD)
            N = model.xyz.shape[0]  # update N after densification
            
            # Re-create optimizer with new params
            optimizer = torch.optim.Adam(model.get_optimizer_param_groups({
                "lr_xyz": 1.6e-4, "lr_rotation": 1e-3, "lr_scaling": 5e-3,
                "lr_opacity": 5e-2, "lr_sh": 2.5e-3,
            }), betas=(0.9, 0.999), eps=1e-8)
            
            densification_events.append({
                "iter": iter_idx, "cloned": clone_count, "split": split_count,
                "pruned": prune_count, "gaussians": N,
            })
            
            # Reset prev_grad_norm since N changed
            prev_grad_norm = None
        
        # Store gradient norm for mask prediction (BEFORE optimizer step)
        if model.xyz.grad is not None:
            current_grad_norm = model.xyz.grad.detach().norm(dim=-1)
            
            if design in ("B1", "B3"):
                # Sparse gradient: EMA + epsilon to prevent feedback lockout
                eps = 1e-6
                current_grad_norm = current_grad_norm + eps
                if prev_grad_norm is not None and prev_grad_norm.shape[0] == N:
                    ema_decay = 0.9
                    prev_grad_norm = ema_decay * prev_grad_norm + (1 - ema_decay) * current_grad_norm
                else:
                    prev_grad_norm = current_grad_norm.clone()
            else:
                # B2 or baseline: full gradient
                prev_grad_norm = current_grad_norm.clone()
        
        # SH degree increase every 1000 iters
        if iter_idx % 1000 == 0 and model.sh_degree < model.max_sh_degree:
            model.set_sh_degree(model.sh_degree + 1)
        
        optimizer.step()
        
        torch.cuda.synchronize(device)
        iter_end = time.perf_counter()
        iter_time_ms = (iter_end - iter_start) * 1000
        backward_time_ms = (backward_end - iter_start) * 1000
        
        timing_stats.append({
            "iter": iter_idx,
            "total_ms": iter_time_ms,
            "backward_ms": backward_time_ms,
            "is_sparse": is_sparse and importance_mask is not None,
        })
        
        # Trajectory
        if iter_idx % TRAJECTORY_INTERVAL == 0 or iter_idx == iters:
            psnr, ssim = evaluate()
            print(f"  Iter {iter_idx}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, "
                  f"GS={model.xyz.shape[0]:,}, time={iter_time_ms:.1f}ms")
            trajectory.append({
                "iter": iter_idx, "psnr": psnr, "ssim": ssim,
                "gaussians": model.xyz.shape[0], "loss": float(loss.item()),
            })

    # Final stats
    timing_arr = np.array([t["total_ms"] for t in timing_stats])
    sparse_timing = [t["total_ms"] for t in timing_stats if t["is_sparse"]]
    dense_timing = [t["total_ms"] for t in timing_stats if not t["is_sparse"]]
    
    results = {
        "config": config,
        "final_psnr": trajectory[-1]["psnr"],
        "final_ssim": trajectory[-1]["ssim"],
        "final_gaussians": model.xyz.shape[0],
        "trajectory": trajectory,
        "densification_events": densification_events,
        "timing": {
            "mean_ms": float(timing_arr.mean()),
            "median_ms": float(np.median(timing_arr)),
            "p50_ms": float(np.percentile(timing_arr, 50)),
            "p90_ms": float(np.percentile(timing_arr, 90)),
            "std_ms": float(timing_arr.std()),
            "n_sparse_iters": len(sparse_timing),
            "n_dense_iters": len(dense_timing),
            "sparse_mean_ms": float(np.mean(sparse_timing)) if sparse_timing else 0,
            "dense_mean_ms": float(np.mean(dense_timing)) if dense_timing else 0,
        },
        "total_clone": sum(e["cloned"] for e in densification_events),
        "total_split": sum(e["split"] for e in densification_events),
        "total_prune": sum(e["pruned"] for e in densification_events),
    }
    
    return results


CONFIGS = {
    "baseline": {
        "keep_fraction": 1.0, "design": "baseline", "iters": 5000,
    },
    "k80_b1": {
        "keep_fraction": 0.8, "design": "B1", "iters": 5000,
    },
    "k70_b1": {
        "keep_fraction": 0.7, "design": "B1", "iters": 5000,
    },
    "k80_b2": {
        "keep_fraction": 0.8, "design": "B2", "iters": 5000,
    },
    "k80_b3": {
        "keep_fraction": 0.8, "design": "B3", "iters": 5000,
    },
    "k60_b1": {
        "keep_fraction": 0.6, "design": "B1", "iters": 5000,
    },
    "k50_b1": {
        "keep_fraction": 0.5, "design": "B1", "iters": 5000,
    },
    "baseline_30k": {
        "keep_fraction": 1.0, "design": "baseline", "iters": 30000,
    },
    "k50_b1_30k": {
        "keep_fraction": 0.5, "design": "B1", "iters": 30000,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    config = CONFIGS[args.config].copy()
    config["gpu_id"] = args.gpu

    repo_root = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Phase C51 Stage 4A — 5K Training Validation")
    print(f"  Config: {args.config}")
    print(f"  Design: {config['design']}, K={config['keep_fraction']}")
    print(f"  GPU: {args.gpu}")
    print()

    dataset = GTDataset(scene="room", repo_root=repo_root, resolution="1080p", device=f"cuda:{args.gpu}")
    sfm_data = load_initial_checkpoint("room", repo_root, device=f"cuda:{args.gpu}")
    print(f"  SfM points: {sfm_data['xyz'].shape[0]:,}")

    results = run_experiment(dataset, sfm_data, config)

    save_dir = repo_root / "results" / "a100" / "phase-c51-stage4a"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / f"training_5k_{args.config}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")
    print(f"  Final PSNR: {results['final_psnr']:.2f}")
    print(f"  Final SSIM: {results['final_ssim']:.4f}")
    print(f"  Final GS: {results['final_gaussians']:,}")
    print(f"  Mean time: {results['timing']['mean_ms']:.2f}ms")
    print(f"  Clone: {results['total_clone']}, Split: {results['total_split']}, Prune: {results['total_prune']}")


if __name__ == "__main__":
    main()
