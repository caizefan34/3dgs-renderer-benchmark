#!/usr/bin/env python3
"""
Phase C52 Stage 0 — Predictive Density Control
Counterfactual mechanism validation: does previous-iteration gradient importance
improve density allocation at matched budget?

Four conditions:
  A (baseline):  All candidates densified (budget=100%)
  B (uniform):   Random candidate selection at budget fraction
  C (predictive): Top candidates by previous densification-window gradient norm
  D (oracle):    Top candidates by current densification-window gradient norm

Budget levels: 100%, 75%, 50%

Implementation:
  After accumulate_positional_gradient(), compute candidate set (avg_grad >= threshold).
  Rank candidates by selection signal. Select top-N. Zero non-selected candidates'
  _xyz_grad_accum so they don't pass the threshold in densification().

  prev_dens_grad_norm: stored from previous densification window, aligned to
  current Gaussian set (0 for new Gaussians, filtered after pruning).
"""
import json, math, sys, time, argparse, os
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/src")))
sys.path.insert(0, str(Path("/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7")))
from gsplat import rasterization
from gaussian_model import GaussianModel
from dataset import GTDataset, load_initial_checkpoint

SEED = 42
PRUNE_THRESHOLD = 0.01
GRAD_THRESHOLD = 0.001
DENSIFY_END_FRAC = 0.5
OPACITY_RESET_INTERVAL = 3000
SH_PROGRESS_INTERVAL = 1000
DENSIFY_INTERVAL = 100
DENSIFY_START = 500
EVAL_INTERVAL = 500
TRAJECTORY_INTERVAL = 500
EVAL_CAMERAS = list(range(0, 311, 24))[:13]


class SepSSIM:
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
        return float(1.0 - self(pred, target).item())


def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    return float(20 * math.log10(1.0 / math.sqrt(mse.item()))) if mse > 1e-10 else 100.0


def render(model, cam, data=None):
    if data is None:
        data = model.forward()
    r, _, _ = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=model.sh_degree,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB",
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


def select_candidates(avg_grad, candidate_mask, budget_fraction, strategy,
                      ema_grad_norm, rng):
    """Select which densification candidates to actually densify.

    Args:
        ema_grad_norm: [N] EMA of per-iteration gradient norm (predictive signal)
    """
    n_candidates = int(candidate_mask.sum().item())
    n_select = int(n_candidates * budget_fraction)

    if n_select >= n_candidates or n_select == 0:
        return candidate_mask, n_candidates, n_select

    candidate_indices = torch.where(candidate_mask)[0]

    if strategy == "uniform":
        # Random selection
        perm = rng.permutation(n_candidates)[:n_select]
        selected_local = torch.zeros(n_candidates, dtype=torch.bool, device=candidate_mask.device)
        selected_local[torch.from_numpy(perm).to(candidate_mask.device)] = True
    elif strategy == "predictive":
        # Rank by EMA of per-iteration gradient norm
        prev_scores = ema_grad_norm[candidate_indices]
        _, top_local = torch.topk(prev_scores, n_select)
        selected_local = torch.zeros(n_candidates, dtype=torch.bool, device=candidate_mask.device)
        selected_local[top_local] = True
    elif strategy == "oracle":
        # Rank by current gradient norm (avg_grad)
        curr_scores = avg_grad[candidate_indices]
        _, top_local = torch.topk(curr_scores, n_select)
        selected_local = torch.zeros(n_candidates, dtype=torch.bool, device=candidate_mask.device)
        selected_local[top_local] = True
    else:
        # "all" = baseline
        return candidate_mask, n_candidates, n_candidates

    selected_mask = torch.zeros_like(candidate_mask)
    selected_mask[candidate_indices[selected_local]] = True
    return selected_mask, n_candidates, n_select


def run_experiment(dataset, sfm_data, config):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    strategy = config["strategy"]  # "all", "uniform", "predictive", "oracle"
    budget_fraction = config.get("budget_fraction", 1.0)
    iters = config["iters"]
    gpu_id = config.get("gpu_id", 0)
    scene = config.get("scene", "room")
    rng = np.random.RandomState(SEED)

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)

    N = sfm_data['xyz'].shape[0]
    model = GaussianModel(num_points=N, sh_degree=0, max_sh_degree=3, device=device)
    model.init_from_sfm(
        xyz=sfm_data['xyz'].clone().float().to(device),
        opacity_logit=torch.logit(torch.full((N,), 0.1, device=device)),
        scales_log=sfm_data['scales'].clone().float().to(device),
        rotations_raw=sfm_data['rotations'].clone().float().to(device),
        shs=sfm_data['shs'].clone().float().to(device),
    )
    model.set_sh_degree(0)

    optimizer = get_optimizer(model)
    sep_ssim = SepSSIM(device=device)
    densify_end = int(iters * DENSIFY_END_FRAC)

    n_cams = len(dataset)
    cam_indices = list(range(n_cams))
    np.random.shuffle(cam_indices)

    # EMA of per-iteration gradient norm (the C50/C51 predictive signal)
    # importance_i(t) = EMA(||gradient_i(t)||) with decay=0.9
    # This tracks the trend over iterations, not just a single snapshot
    ema_decay = config.get("ema_decay", 0.9)
    ema_grad_norm = torch.zeros(N, device=device)
    # Previous densification-window gradient norm (for analysis only)
    prev_dens_grad_norm = torch.zeros(N, device=device)

    trajectory = []
    timing_stats = []
    densification_events = []
    candidate_analysis_events = []  # For correlation/transition analysis
    total_clone, total_split, total_prune = 0, 0, 0
    fwd_times = []
    opt_times = []
    dens_times = []

    # Initial eval
    psnr, ssim = evaluate(model, dataset, sep_ssim)
    print(f"  Iter 0: PSNR={psnr:.2f}, SSIM={ssim:.4f}, GS={model.xyz.shape[0]:,}")
    trajectory.append({"iter": 0, "psnr": psnr, "ssim": ssim,
                        "gaussians": model.xyz.shape[0], "loss": 0.0})

    for iter_idx in range(iters):
        model.train()
        ci = cam_indices[iter_idx % n_cams]
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)

        # Forward + backward
        fwd_t0 = time.perf_counter()
        torch.cuda.synchronize(device)
        data = model.forward()
        pred = render(model, cam, data)
        l1 = F.l1_loss(pred, gt)
        dssim = sep_ssim(pred, gt)
        loss = 0.8 * l1 + 0.2 * dssim
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize(device)
        fwd_t1 = time.perf_counter()
        fwd_times.append((fwd_t1 - fwd_t0) * 1000)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Densification
        dens_t0 = time.perf_counter()
        do_densify = (iter_idx >= DENSIFY_START and iter_idx < densify_end and
                      iter_idx % DENSIFY_INTERVAL == 0)
        clone_count, split_count, prune_count = 0, 0, 0

        if do_densify:
            # Accumulate positional gradient
            model.accumulate_positional_gradient()

            if model._xyz_grad_accum is not None and model._denf_steps > 0:
                avg_grad = model._xyz_grad_accum / model._denf_steps
                candidate_mask = avg_grad >= GRAD_THRESHOLD
                n_candidates = int(candidate_mask.sum().item())

                # Record candidate analysis (for correlation/transition)
                if n_candidates > 0:
                    # Use per-iteration gradient for correlation analysis
                    prev_scores = ema_grad_norm[candidate_mask].cpu().numpy()
                    curr_scores = avg_grad[candidate_mask].cpu().numpy()
                    # Transition matrix: prev_high → curr_high etc.
                    # "high" = above median of ALL Gaussians' prev-iter gradient
                    prev_median = float(np.median(ema_grad_norm.cpu().numpy()))
                    prev_high = ema_grad_norm >= prev_median
                    curr_high = candidate_mask
                    n_prev_high_curr_high = int((prev_high & curr_high).sum().item())
                    n_prev_low_curr_high = int((~prev_high & curr_high).sum().item())

                    # Also compute Spearman correlation (rank-based, no scipy needed)
                    spearman_val = 0.0
                    if n_candidates > 1 and np.std(prev_scores) > 0 and np.std(curr_scores) > 0:
                        prev_ranks = np.argsort(np.argsort(prev_scores)).astype(float)
                        curr_ranks = np.argsort(np.argsort(curr_scores)).astype(float)
                        spearman_val = float(np.corrcoef(prev_ranks, curr_ranks)[0, 1])

                    candidate_analysis_events.append({
                        "iter": iter_idx,
                        "n_candidates": n_candidates,
                        "n_total": model.xyz.shape[0],
                        "prev_grad_mean": float(np.mean(prev_scores)),
                        "curr_grad_mean": float(np.mean(curr_scores)),
                        "prev_curr_pearson": float(np.corrcoef(prev_scores, curr_scores)[0, 1])
                                             if n_candidates > 1 and np.std(prev_scores) > 0 and np.std(curr_scores) > 0 else 0.0,
                        "prev_curr_spearman": spearman_val,
                        "n_prev_high_curr_high": n_prev_high_curr_high,
                        "n_prev_low_curr_high": n_prev_low_curr_high,
                        "prev_median": prev_median,
                    })

                # Select candidates based on strategy
                selected_mask, total_candidates, n_select = select_candidates(
                    avg_grad, candidate_mask, budget_fraction, strategy,
                    ema_grad_norm, rng
                )

                # Zero non-selected candidates' gradient accumulation
                # so they don't pass the threshold in densification()
                if strategy != "all" and n_select < total_candidates:
                    not_selected = candidate_mask & (~selected_mask)
                    model._xyz_grad_accum[not_selected] = 0.0

                # Save current avg_grad as prev for next densification window (analysis only)
                # (before densification changes the Gaussian set)
                prev_dens_grad_norm = avg_grad.clone()

                # Densify
                counts = model.densification(grad_threshold=GRAD_THRESHOLD)
                clone_count = counts["cloned"]
                split_count = counts["split"]
                total_clone += clone_count
                total_split += split_count

                # Extend prev_dens_grad_norm AND ema_grad_norm for new Gaussians
                n_new = model.xyz.shape[0] - prev_dens_grad_norm.shape[0]
                if n_new > 0:
                    prev_dens_grad_norm = torch.cat([
                        prev_dens_grad_norm,
                        torch.zeros(n_new, device=device)
                    ])
                    ema_grad_norm = torch.cat([
                        ema_grad_norm,
                        torch.zeros(n_new, device=device)
                    ])

                # Compute prune mask BEFORE pruning to filter prev tensors
                opacities = torch.sigmoid(model.opacity).detach().squeeze(-1)
                prune_mask = opacities < PRUNE_THRESHOLD
                keep_mask = ~prune_mask

                prune_count = model.prune(opacity_threshold=PRUNE_THRESHOLD)
                total_prune += prune_count

                # Filter prev tensors to keep only surviving Gaussians
                if prune_count > 0:
                    prev_dens_grad_norm = prev_dens_grad_norm[keep_mask]
                    ema_grad_norm = ema_grad_norm[keep_mask]

                optimizer = get_optimizer(model)
                densification_events.append({
                    "iter": iter_idx, "cloned": clone_count, "split": split_count,
                    "pruned": prune_count, "gaussians": model.xyz.shape[0],
                    "n_candidates": total_candidates,
                    "n_selected": n_select,
                    "budget_fraction": budget_fraction,
                    "strategy": strategy,
                    "cumulative_clone": total_clone,
                    "cumulative_split": total_split,
                    "cumulative_prune": total_prune,
                })
            else:
                counts = model.densification(grad_threshold=GRAD_THRESHOLD)
                clone_count = counts["cloned"]
                split_count = counts["split"]
                total_clone += clone_count
                total_split += split_count

        # Opacity reset
        if iter_idx > 0 and iter_idx % OPACITY_RESET_INTERVAL == 0:
            # Compute prune mask BEFORE pruning to filter prev_dens_grad_norm
            opacities = torch.sigmoid(model.opacity).detach().squeeze(-1)
            prune_mask_reset = opacities < PRUNE_THRESHOLD
            keep_mask_reset = ~prune_mask_reset

            removed = model.prune_and_reset(
                opacity_threshold=PRUNE_THRESHOLD,
                reset_interval=OPACITY_RESET_INTERVAL,
                current_step=iter_idx,
            )
            total_prune += removed
            optimizer = get_optimizer(model)
            # Filter prev tensors to keep only surviving Gaussians
            if removed > 0 and prev_dens_grad_norm.shape[0] == keep_mask_reset.shape[0]:
                prev_dens_grad_norm = prev_dens_grad_norm[keep_mask_reset]
                ema_grad_norm = ema_grad_norm[keep_mask_reset]
            elif prev_dens_grad_norm.shape[0] != model.xyz.shape[0]:
                prev_dens_grad_norm = torch.zeros(model.xyz.shape[0], device=device)
                ema_grad_norm = torch.zeros(model.xyz.shape[0], device=device)

        # SH progression
        if iter_idx > 0 and iter_idx % SH_PROGRESS_INTERVAL == 0:
            if model.sh_degree < model.max_sh_degree:
                model.set_sh_degree(model.sh_degree + 1)

        dens_t1 = time.perf_counter()
        dens_times.append((dens_t1 - dens_t0) * 1000)

        # Update EMA of per-iteration gradient norm (the C50/C51 predictive signal)
        # importance_i(t) = EMA(||gradient_i(t)||) with decay=0.9
        if model.xyz.grad is not None:
            current_grad_norm = model.xyz.grad.detach().norm(dim=-1)
            if ema_grad_norm.shape[0] == current_grad_norm.shape[0]:
                ema_grad_norm = ema_decay * ema_grad_norm + (1 - ema_decay) * current_grad_norm
            else:
                # Shape mismatch (e.g. after SH resize) — reset
                ema_grad_norm = torch.zeros(model.xyz.shape[0], device=device)

        # Optimizer step
        opt_t0 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize(device)
        opt_t1 = time.perf_counter()
        opt_times.append((opt_t1 - opt_t0) * 1000)

        total_iter_ms = (opt_t1 - fwd_t0) * 1000
        timing_stats.append({
            "iter": iter_idx,
            "total_ms": total_iter_ms,
            "fwd_bwd_ms": (fwd_t1 - fwd_t0) * 1000,
            "dens_ms": (dens_t1 - dens_t0) * 1000,
            "opt_ms": (opt_t1 - opt_t0) * 1000,
        })

        if iter_idx % TRAJECTORY_INTERVAL == 0 or iter_idx == iters - 1:
            psnr, ssim = evaluate(model, dataset, sep_ssim)
            gs = model.xyz.shape[0]
            print(f"  Iter {iter_idx}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, "
                  f"GS={gs:,}, time={total_iter_ms:.1f}ms")
            trajectory.append({"iter": iter_idx, "psnr": psnr, "ssim": ssim,
                                "gaussians": gs, "loss": float(loss.item())})

    results = {
        "config": config,
        "final_psnr": trajectory[-1]["psnr"],
        "final_ssim": trajectory[-1]["ssim"],
        "final_gaussians": model.xyz.shape[0],
        "trajectory": trajectory,
        "densification_events": densification_events,
        "candidate_analysis_events": candidate_analysis_events,
        "timing": {
            "mean_ms": float(np.mean([t["total_ms"] for t in timing_stats])),
            "fwd_bwd_mean_ms": float(np.mean(fwd_times)),
            "dens_mean_ms": float(np.mean(dens_times)),
            "opt_mean_ms": float(np.mean(opt_times)),
        },
        "total_clone": total_clone,
        "total_split": total_split,
        "total_prune": total_prune,
    }
    return results


CONFIGS = {
    "baseline": {"strategy": "all", "budget_fraction": 1.0, "iters": 5000},
    "uniform_75": {"strategy": "uniform", "budget_fraction": 0.75, "iters": 5000},
    "uniform_50": {"strategy": "uniform", "budget_fraction": 0.50, "iters": 5000},
    "predictive_75": {"strategy": "predictive", "budget_fraction": 0.75, "iters": 5000},
    "predictive_50": {"strategy": "predictive", "budget_fraction": 0.50, "iters": 5000},
    "oracle_75": {"strategy": "oracle", "budget_fraction": 0.75, "iters": 5000},
    "oracle_50": {"strategy": "oracle", "budget_fraction": 0.50, "iters": 5000},
    # 30K versions
    "baseline_30k": {"strategy": "all", "budget_fraction": 1.0, "iters": 30000},
    "uniform_75_30k": {"strategy": "uniform", "budget_fraction": 0.75, "iters": 30000},
    "predictive_75_30k": {"strategy": "predictive", "budget_fraction": 0.75, "iters": 30000},
    "oracle_75_30k": {"strategy": "oracle", "budget_fraction": 0.75, "iters": 30000},
    "uniform_50_30k": {"strategy": "uniform", "budget_fraction": 0.50, "iters": 30000},
    "predictive_50_30k": {"strategy": "predictive", "budget_fraction": 0.50, "iters": 30000},
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

    print(f"Phase C52 Stage 0 — Predictive Density Control")
    print(f"  Config: {args.config}")
    print(f"  Strategy: {config['strategy']}")
    print(f"  Budget: {config['budget_fraction']:.0%}")
    print(f"  Scene: {args.scene}")
    print(f"  Iters: {config['iters']}")
    print(f"  GPU: {args.gpu}")
    print()

    dataset = GTDataset(scene=args.scene, repo_root=repo_root, resolution="1080p",
                        device=f"cuda:{args.gpu}")
    sfm_data = load_initial_checkpoint(args.scene, repo_root, device=f"cuda:{args.gpu}")
    print(f"  SfM points: {sfm_data['xyz'].shape[0]:,}")

    results = run_experiment(dataset, sfm_data, config)

    save_dir = repo_root / "results" / "a100" / "phase-c52-stage0"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / f"{args.config}_{args.scene}.json"
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
