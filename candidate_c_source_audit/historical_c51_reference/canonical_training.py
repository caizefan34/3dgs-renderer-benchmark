#!/usr/bin/env python3
"""
Phase C51 Stage 4B — Canonical CUDA Training Validation

Canonical 3DGS training configuration:
  - Loss: 0.8 * L1 + 0.2 * D-SSIM (separable SSIM, every iteration)
  - LR: xyz=1.6e-4, rotations=1e-3, scales=5e-3, opacity=5e-2, shs=2.5e-3
  - Optimizer: Adam, eps=1e-15
  - Grad clip: max_norm=1.0
  - Densification: every 100 iters, iter 500 to densify_end (0.5 * total)
  - Prune: opacity_threshold=0.005, grad_threshold=0.0002
  - SH progression: +1 every 1000 iters, start 0, max 3
  - Opacity reset: every 3000 iters
  - Camera sampling: shuffled indices, seed=42
  - Resolution: 1080p

Experiment modes:
  - baseline: Full backward, no mask
  - k50_b1: K50-B1 sparse backward, predictive mask (previous-grad)
  - k50_oracle: K50-B1 sparse backward, oracle mask (current-grad)
  - k50_postdens: Full backward 0-15K, K50-B1 sparse 15K-30K
  - k50_freeze_mask: Full backward + Python zeroing (freeze-vs-mask ablation)
  - k50_b1_oracledens: K50-B1 sparse backward + full densification signal (oracle control)
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
from loss import d_ssim_loss

SEED = 42
PRUNE_THRESHOLD = 0.01   # moderate config (C45 canonical)
GRAD_THRESHOLD = 0.001   # moderate config (C45 canonical)
DENSIFY_END_FRAC = 0.5
OPACITY_RESET_INTERVAL = 3000
SH_PROGRESS_INTERVAL = 1000
DENSIFY_INTERVAL = 100
DENSIFY_START = 500
EVAL_INTERVAL = 500
TRAJECTORY_INTERVAL = 500
EVAL_CAMERAS = list(range(0, 311, 24))[:13]


class SepSSIM:
    """Separable SSIM (C44 method) — differentiable loss version."""
    def __init__(self, window_size=11, sigma=1.5, device="cuda"):
        self.C1 = (0.01) ** 2
        self.C2 = (0.03) ** 2
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0,1,3,2).contiguous()
        self.padding = window_size // 2

    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.cat([pred, target, pred**2, target**2, pred*target], dim=1)
        b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_p, mu_t = b[:, 0:3], b[:, 3:6]
        bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
        mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p*mu_t
        sp2, st2, spt = bp2-mu_p2, bt2-mu_t2, bpt-mu_pt
        ssim_map = (2*mu_pt+self.C1)*(2*spt+self.C2) / ((mu_p2+mu_t2+self.C1)*(sp2+st2+self.C2))
        return 1.0 - ssim_map.mean()

    def metric(self, pred, target):
        """Non-loss SSIM metric (returns SSIM, not 1-SSIM)."""
        return float(1.0 - self(pred, target).item())


def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    return float(20 * math.log10(1.0 / math.sqrt(mse.item()))) if mse > 1e-10 else 100.0


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


def evaluate(model, dataset, sep_ssim):
    psnrs, ssims = [], []
    model.eval()
    with torch.no_grad():
        data = model.forward()
        for ci in EVAL_CAMERAS:
            if ci >= len(dataset):
                break
            cam = dataset.get_camera(ci)
            gt = dataset.get_gt_image(ci)
            pred = render(model, cam, data)
            psnrs.append(compute_psnr(pred, gt))
            ssims.append(sep_ssim.metric(pred, gt))
    model.train()
    return float(np.mean(psnrs)), float(np.mean(ssims))


def get_optimizer(model):
    lr_params = [
        {"params": [model.xyz], "lr": 1.6e-4, "name": "xyz"},
        {"params": [model.rotations], "lr": 1e-3, "name": "rotations"},
        {"params": [model.scales], "lr": 5e-3, "name": "scales"},
        {"params": [model.opacity], "lr": 5e-2, "name": "opacity"},
        {"params": [model.shs], "lr": 2.5e-3, "name": "shs"},
    ]
    return torch.optim.Adam(lr_params, eps=1e-15)


def run_experiment(dataset, sfm_data, config):
    """Run a single training experiment.

    config keys:
        mode: "baseline" | "k50_b1" | "k50_oracle" | "k50_postdens" |
              "k50_freeze_mask" | "k50_b1_oracledens"
        iters: int
        gpu_id: int
        keep_fraction: float (default 0.5)
        scene: str
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    mode = config["mode"]
    iters = config["iters"]
    gpu_id = config.get("gpu_id", 0)
    k_frac = config.get("keep_fraction", 0.5)
    scene = config.get("scene", "room")

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    # Create model with canonical initialization
    N = sfm_data['xyz'].shape[0]
    model = GaussianModel(num_points=N, sh_degree=0, max_sh_degree=3, device=device)
    model.init_from_sfm(
        xyz=sfm_data['xyz'].clone().float().to(device),
        opacity_logit=torch.logit(torch.full((N,), 0.1, device=device)),  # canonical init
        scales_log=sfm_data['scales'].clone().float().to(device),
        rotations_raw=sfm_data['rotations'].clone().float().to(device),
        shs=sfm_data['shs'].clone().float().to(device),
    )
    model.set_sh_degree(0)  # Start at SH degree 0

    optimizer = get_optimizer(model)
    sep_ssim = SepSSIM(device=device)

    densify_end = int(iters * DENSIFY_END_FRAC)

    # Camera sampling
    n_cams = len(dataset)
    cam_indices = list(range(n_cams))
    np.random.shuffle(cam_indices)

    # Mask state
    prev_grad_norm = None  # for predictive mask
    is_sparse_mode = mode in ("k50_b1", "k50_oracle", "k50_postdens", "k50_b1_oracledens", "k50_freeze_mask")
    compute_densify_grad = (mode == "k50_b1_oracledens")  # B2-style for oracle densification
    is_freeze_mask = (mode == "k50_freeze_mask")

    # For post-densification mode
    postdens_start = int(iters * 0.5)  # 15000 for 30K

    # Tracking
    trajectory = []
    timing_stats = []
    densification_events = []
    mask_stats = []
    total_clone, total_split, total_prune = 0, 0, 0

    # Timing decomposition
    fwd_times = []
    bwd_times = []
    loss_times = []
    opt_times = []
    dens_times = []
    mask_times = []

    def make_mask(grad_norm, frac, current_grad_norm=None):
        """Construct importance mask.

        If current_grad_norm is provided, use it (oracle mode).
        Otherwise use grad_norm (predictive mode).
        """
        if (current_grad_norm is not None) and (mode == "k50_oracle"):
            source = current_grad_norm
        else:
            source = grad_norm
        n_keep = max(1, int(len(source) * frac))
        _, top_idx = torch.topk(source, n_keep)
        mask = torch.zeros(len(source), dtype=torch.uint8, device=device)
        mask[top_idx] = 1
        return mask

    # Initial eval
    psnr, ssim = evaluate(model, dataset, sep_ssim)
    print(f"  Iter 0: PSNR={psnr:.2f}, SSIM={ssim:.4f}, GS={model.xyz.shape[0]:,}")
    trajectory.append({
        "iter": 0, "psnr": psnr, "ssim": ssim,
        "gaussians": model.xyz.shape[0], "loss": 0.0,
    })

    # Training loop
    for iter_idx in range(iters):
        model.train()
        ci = cam_indices[iter_idx % n_cams]
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)

        # Determine if this iteration uses sparse backward
        use_sparse = False
        if is_sparse_mode:
            if mode == "k50_postdens":
                use_sparse = (iter_idx >= postdens_start)
            else:
                use_sparse = (prev_grad_norm is not None)

        # Construct mask
        mask_t0 = time.perf_counter()
        importance_mask = None
        if use_sparse:
            if mode == "k50_oracle":
                # Oracle: need current gradient — do a full forward+backward first,
                # then redo with mask. For efficiency, we just use the current
                # iteration's gradient from a full backward, then re-render.
                # Actually, for oracle mode, we compute full backward to get grad,
                # then use that grad as mask, then redo backward with mask.
                # This is 2x backward but it's an ablation, not production.
                # For now, use prev_grad_norm as approximation of "oracle" if we
                # haven't computed current yet. We'll handle this below.
                importance_mask = make_mask(prev_grad_norm, k_frac)
            else:
                importance_mask = make_mask(prev_grad_norm, k_frac)
        mask_t1 = time.perf_counter()
        mask_times.append((mask_t1 - mask_t0) * 1000)

        # Forward
        fwd_t0 = time.perf_counter()
        torch.cuda.synchronize(device)
        data = model.forward()

        if is_freeze_mask:
            # Full backward (no mask), then zero masked gradients
            pred = render(model, cam, data, importance_mask=None)
            l1 = F.l1_loss(pred, gt)
            dssim = sep_ssim(pred, gt)
            loss = 0.8 * l1 + 0.2 * dssim
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.cuda.synchronize(device)
            fwd_t1 = time.perf_counter()

            # Zero masked gradients (shape-safe)
            if use_sparse and importance_mask is not None:
                N_mask = len(importance_mask)
                skip_mask = (importance_mask == 0)
                for param in [model.xyz, model.opacity, model.scales, model.rotations, model.shs]:
                    if param.grad is not None and param.shape[0] == N_mask:
                        if param.dim() > 1:
                            param.grad[skip_mask] = 0
                        else:
                            param.grad[skip_mask] = 0
        elif mode == "k50_oracle":
            # Oracle: full backward to get current gradient, then sparse backward
            # Step 1: Full forward + backward to get gradient
            pred = render(model, cam, data, importance_mask=None)
            l1 = F.l1_loss(pred, gt)
            dssim = sep_ssim(pred, gt)
            loss = 0.8 * l1 + 0.2 * dssim
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.cuda.synchronize(device)
            fwd_t1 = time.perf_counter()

            # Step 2: Use current gradient as mask, re-compute forward, sparse backward
            if model.xyz.grad is not None:
                current_grad_norm = model.xyz.grad.detach().norm(dim=-1)
                importance_mask = make_mask(prev_grad_norm, k_frac, current_grad_norm)
                # Re-compute forward (graph was freed by first backward)
                data2 = model.forward()
                pred2 = render(model, cam, data2, importance_mask=importance_mask)
                l1_2 = F.l1_loss(pred2, gt)
                dssim_2 = sep_ssim(pred2, gt)
                loss = 0.8 * l1_2 + 0.2 * dssim_2
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.cuda.synchronize(device)
        else:
            # Normal path: sparse or baseline
            pred = render(model, cam, data,
                          importance_mask=importance_mask if use_sparse else None,
                          compute_densify_grad=compute_densify_grad)
            l1 = F.l1_loss(pred, gt)
            dssim = sep_ssim(pred, gt)
            loss = 0.8 * l1 + 0.2 * dssim
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.cuda.synchronize(device)
            fwd_t1 = time.perf_counter()

        bwd_time = (fwd_t1 - fwd_t0) * 1000  # includes forward+backward
        fwd_times.append(bwd_time)  # fwd+bwd combined timing

        # Grad clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Densification
        dens_t0 = time.perf_counter()
        do_densify = (iter_idx >= DENSIFY_START and iter_idx < densify_end and
                      iter_idx % DENSIFY_INTERVAL == 0)
        clone_count, split_count, prune_count = 0, 0, 0

        if do_densify:
            # For oracle densification mode: use full gradient (stored separately)
            # For B1/B3: use sparse gradient (xyz.grad has zeros for masked)
            # For baseline/freeze_mask: use full gradient
            model.accumulate_positional_gradient()
            counts = model.densification(grad_threshold=GRAD_THRESHOLD)
            clone_count = counts["cloned"]
            split_count = counts["split"]
            total_clone += clone_count
            total_split += split_count

            prune_count = model.prune(opacity_threshold=PRUNE_THRESHOLD)
            total_prune += prune_count

            # Re-create optimizer
            optimizer = get_optimizer(model)

            densification_events.append({
                "iter": iter_idx, "cloned": clone_count, "split": split_count,
                "pruned": prune_count, "gaussians": model.xyz.shape[0],
                "cumulative_clone": total_clone, "cumulative_split": total_split,
                "cumulative_prune": total_prune,
            })
            prev_grad_norm = None  # reset after topology change

        # Opacity reset
        if iter_idx > 0 and iter_idx % OPACITY_RESET_INTERVAL == 0:
            removed = model.prune_and_reset(
                opacity_threshold=PRUNE_THRESHOLD,
                reset_interval=OPACITY_RESET_INTERVAL,
                current_step=iter_idx,
            )
            total_prune += removed
            optimizer = get_optimizer(model)
            prev_grad_norm = None  # reset after topology change

        # SH progression
        if iter_idx > 0 and iter_idx % SH_PROGRESS_INTERVAL == 0:
            if model.sh_degree < model.max_sh_degree:
                model.set_sh_degree(model.sh_degree + 1)

        dens_t1 = time.perf_counter()
        dens_times.append((dens_t1 - dens_t0) * 1000)

        # Store gradient norm for mask prediction
        if model.xyz.grad is not None:
            current_grad_norm = model.xyz.grad.detach().norm(dim=-1)
            if mode in ("k50_b1", "k50_postdens", "k50_b1_oracledens"):
                # Predictive: EMA + epsilon
                eps = 1e-6
                current_grad_norm = current_grad_norm + eps
                if prev_grad_norm is not None and prev_grad_norm.shape[0] == model.xyz.shape[0]:
                    ema_decay = 0.9
                    prev_grad_norm = ema_decay * prev_grad_norm + (1 - ema_decay) * current_grad_norm
                else:
                    prev_grad_norm = current_grad_norm.clone()
            else:
                prev_grad_norm = current_grad_norm.clone()

        # Mask overlap tracking
        if use_sparse and importance_mask is not None and len(mask_stats) > 0:
            prev_mask = mask_stats[-1].get("mask_hash", None)
            # Compute overlap with previous iteration's mask
            # (simplified: just store mask info)
            n_selected = int(importance_mask.sum().item())
            mask_stats.append({
                "iter": iter_idx,
                "n_selected": n_selected,
                "n_total": len(importance_mask),
                "keep_frac": n_selected / len(importance_mask),
            })
        elif use_sparse and importance_mask is not None:
            n_selected = int(importance_mask.sum().item())
            mask_stats.append({
                "iter": iter_idx,
                "n_selected": n_selected,
                "n_total": len(importance_mask),
                "keep_frac": n_selected / len(importance_mask),
            })

        # Optimizer step
        opt_t0 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize(device)
        opt_t1 = time.perf_counter()
        opt_times.append((opt_t1 - opt_t0) * 1000)

        # Timing
        total_iter_ms = (opt_t1 - fwd_t0) * 1000
        timing_stats.append({
            "iter": iter_idx,
            "total_ms": total_iter_ms,
            "fwd_bwd_ms": bwd_time,
            "dens_ms": (dens_t1 - dens_t0) * 1000,
            "opt_ms": (opt_t1 - opt_t0) * 1000,
            "mask_ms": (mask_t1 - mask_t0) * 1000,
            "is_sparse": use_sparse,
        })

        # Trajectory
        if iter_idx % TRAJECTORY_INTERVAL == 0 or iter_idx == iters - 1:
            psnr, ssim = evaluate(model, dataset, sep_ssim)
            gs = model.xyz.shape[0]
            print(f"  Iter {iter_idx}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, "
                  f"GS={gs:,}, time={total_iter_ms:.1f}ms, sparse={use_sparse}")
            trajectory.append({
                "iter": iter_idx, "psnr": psnr, "ssim": ssim,
                "gaussians": gs, "loss": float(loss.item()),
            })

    # Final stats
    sparse_timing = [t["total_ms"] for t in timing_stats if t["is_sparse"]]
    dense_timing = [t["total_ms"] for t in timing_stats if not t["is_sparse"]]

    results = {
        "config": config,
        "final_psnr": trajectory[-1]["psnr"],
        "final_ssim": trajectory[-1]["ssim"],
        "final_gaussians": model.xyz.shape[0],
        "trajectory": trajectory,
        "densification_events": densification_events,
        "mask_stats": mask_stats,
        "timing": {
            "mean_ms": float(np.mean([t["total_ms"] for t in timing_stats])),
            "median_ms": float(np.median([t["total_ms"] for t in timing_stats])),
            "p50_ms": float(np.percentile([t["total_ms"] for t in timing_stats], 50)),
            "p90_ms": float(np.percentile([t["total_ms"] for t in timing_stats], 90)),
            "std_ms": float(np.std([t["total_ms"] for t in timing_stats])),
            "n_sparse_iters": len(sparse_timing),
            "n_dense_iters": len(dense_timing),
            "sparse_mean_ms": float(np.mean(sparse_timing)) if sparse_timing else 0,
            "dense_mean_ms": float(np.mean(dense_timing)) if dense_timing else 0,
            "fwd_bwd_mean_ms": float(np.mean(fwd_times)),
            "dens_mean_ms": float(np.mean(dens_times)),
            "opt_mean_ms": float(np.mean(opt_times)),
            "mask_mean_ms": float(np.mean(mask_times)),
        },
        "total_clone": total_clone,
        "total_split": total_split,
        "total_prune": total_prune,
    }

    return results


CONFIGS = {
    "baseline": {"mode": "baseline", "iters": 30000},
    "k50_b1": {"mode": "k50_b1", "iters": 30000, "keep_fraction": 0.5},
    "k50_oracle": {"mode": "k50_oracle", "iters": 30000, "keep_fraction": 0.5},
    "k50_postdens": {"mode": "k50_postdens", "iters": 30000, "keep_fraction": 0.5},
    "k50_freeze_mask": {"mode": "k50_freeze_mask", "iters": 30000, "keep_fraction": 0.5},
    "k50_b1_oracledens": {"mode": "k50_b1_oracledens", "iters": 30000, "keep_fraction": 0.5},
    # 5K variants for quick ablation
    "baseline_5k": {"mode": "baseline", "iters": 5000},
    "k50_b1_5k": {"mode": "k50_b1", "iters": 5000, "keep_fraction": 0.5},
    "k50_oracle_5k": {"mode": "k50_oracle", "iters": 5000, "keep_fraction": 0.5},
    "k50_freeze_mask_5k": {"mode": "k50_freeze_mask", "iters": 5000, "keep_fraction": 0.5},
    "k50_b1_oracledens_5k": {"mode": "k50_b1_oracledens", "iters": 5000, "keep_fraction": 0.5},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--scene", default="room", choices=["room", "bicycle", "garden"])
    args = parser.parse_args()

    config = CONFIGS[args.config].copy()
    config["gpu_id"] = args.gpu
    config["scene"] = args.scene

    repo_root = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Phase C51 Stage 4B — Canonical CUDA Training Validation")
    print(f"  Config: {args.config}")
    print(f"  Mode: {config['mode']}")
    print(f"  Scene: {args.scene}")
    print(f"  Iters: {config['iters']}")
    print(f"  GPU: {args.gpu}")
    print(f"  Keep fraction: {config.get('keep_fraction', 1.0)}")
    print()

    dataset = GTDataset(scene=args.scene, repo_root=repo_root, resolution="1080p",
                        device=f"cuda:{args.gpu}")
    sfm_data = load_initial_checkpoint(args.scene, repo_root, device=f"cuda:{args.gpu}")
    print(f"  SfM points: {sfm_data['xyz'].shape[0]:,}")

    results = run_experiment(dataset, sfm_data, config)

    save_dir = repo_root / "results" / "a100" / "phase-c51-stage4b"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / f"training_{args.scene}_{args.config}.json"
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
