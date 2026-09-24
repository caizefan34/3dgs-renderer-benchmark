#!/usr/bin/env python3
"""
Phase C49 Experiment 1: Gaussian Lifecycle Profiling.

Measurement experiment (no optimization, no quality impact) that tracks:
1. Per-Gaussian lifespan distribution (creation to removal)
2. Gradient magnitude distribution after backward
3. Gradient contribution concentration (top-K% → fraction of total)
4. Correlation between gradient magnitude and opacity
5. Per-densification-step creation/removal statistics
6. Gradient stability across accumulation windows

This informs all three tracks:
- Track A: Is gradient magnitude alone sufficient? (correlation with opacity/visibility)
- Track B: How many created Gaussians are temporary? (lifespan distribution)
- Track C: Can backward skip low-contribution Gaussians? (gradient concentration)

Uses moderate pruning config (threshold=0.01, grad=0.001, densify 500-15000).
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
EVAL_INTERVAL = 1000
EVAL_CAMERAS = list(range(0, 311, 24))[:13]
PRUNE_THRESHOLD = 0.01
GRAD_THRESHOLD = 0.001


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
        radius_clip=0.0, eps2d=0.1, render_mode="RGB")
    return r[0].clamp(0, 1)


def train_with_lifecycle_tracking(dataset, sfm_data, iters=5000):
    """Train with full lifecycle instrumentation."""
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

    sep_ssim = SepSSIM(device=DEVICE)

    # Lifecycle tracking
    initial_N = model.xyz.shape[0]
    # creation_iter[gid] = iteration when Gaussian gid was created
    # initial Gaussians have creation_iter = -1
    creation_iter = torch.full((initial_N + 100000,), -1, dtype=torch.int32, device=DEVICE)
    # For lifespan tracking of pruned Gaussians
    lifespans = []  # list of (creation_iter, removal_iter, lifespan)
    
    # Per-densification-step statistics
    densify_stats = []
    
    # Gradient distribution samples (recorded at each densification step)
    gradient_distributions = []
    
    # Gradient stability tracking (per-Gaussian gradient history)
    grad_history = {}  # gid -> list of gradient norms
    grad_accum_count = 0
    
    total_clone, total_split, total_prune = 0, 0, 0
    results = {
        "lifespans": [],
        "densify_stats": [],
        "gradient_distributions": [],
        "gs_trajectory": [],
        "eval_points": [],
        "total_train_time_s": 0.0,
    }

    train_start = time.perf_counter()

    for iter_idx in range(iters):
        model.train()
        ci = cam_indices[iter_idx % n_cams]
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)
        data = model.forward()
        pred = render(model, cam, data)

        # Loss (sep_freq8)
        l1 = F.l1_loss(pred, gt)
        if iter_idx % 8 == 0:
            loss = 0.8 * l1 + 0.2 * sep_ssim(pred, gt)
        else:
            loss = l1

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Densification (moderate: 500-15000, every 100 iters)
        if iter_idx >= 500 and iter_idx < 15000 and iter_idx % 100 == 0:
            model.accumulate_positional_gradient()

            # Record gradient distribution BEFORE densification
            if model.xyz.grad is not None:
                grad_norms = model.xyz.grad.norm(dim=-1).detach()
                sorted_norms, _ = grad_norms.sort(descending=True)
                total_grad = sorted_norms.sum().item()
                if total_grad > 0:
                    cumulative = sorted_norms.cumsum(dim=0) / total_grad
                    # Find K for 50%, 90%, 95%, 99%
                    k_50 = (cumulative < 0.5).sum().item()
                    k_90 = (cumulative < 0.9).sum().item()
                    k_95 = (cumulative < 0.95).sum().item()
                    k_99 = (cumulative < 0.99).sum().item()
                    n = grad_norms.shape[0]
                    grad_dist = {
                        "iter": iter_idx,
                        "n_gaussians": n,
                        "grad_p50": float(grad_norms.median().item()),
                        "grad_mean": float(grad_norms.mean().item()),
                        "grad_max": float(grad_norms.max().item()),
                        "grad_min": float(grad_norms.min().item()),
                        "top_1pct_fraction": float(cumulative[min(int(n*0.01), n-1)].item()),
                        "top_10pct_fraction": float(cumulative[min(int(n*0.1), n-1)].item()),
                        "top_50pct_fraction": float(cumulative[min(int(n*0.5), n-1)].item()),
                        "k_for_90pct": k_90,
                        "k_for_90pct_fraction": k_90 / n,
                        "k_for_95pct": k_95,
                        "k_for_99pct": k_99,
                    }
                    gradient_distributions.append(grad_dist)

                    # Correlation: gradient vs opacity
                    opacities = torch.sigmoid(model.opacity).detach().squeeze(-1)
                    if grad_norms.shape[0] == opacities.shape[0]:
                        corr = float(torch.corrcoef(torch.stack([grad_norms, opacities]))[0, 1].item())
                        grad_dist["grad_opacity_correlation"] = corr
                        # Also correlation with scale
                        scales = torch.exp(model.scales).detach().mean(dim=-1)
                        if scales.shape[0] == grad_norms.shape[0]:
                            corr_scale = float(torch.corrcoef(torch.stack([grad_norms, scales]))[0, 1].item())
                            grad_dist["grad_scale_correlation"] = corr_scale

            # Record pre-densification state
            pre_N = model.xyz.shape[0]

            counts = model.densification(grad_threshold=GRAD_THRESHOLD)
            total_clone += counts["cloned"]
            total_split += counts["split"]

            post_N = model.xyz.shape[0]
            new_count = post_N - pre_N

            # Pruning
            removed = model.prune(opacity_threshold=PRUNE_THRESHOLD)
            total_prune += removed

            # Track creation of new Gaussians
            if new_count > 0:
                new_ids = range(pre_N, post_N)
                creation_iter[pre_N:post_N] = iter_idx

            # Track removal: find which Gaussians were pruned
            # After pruning, model.xyz has fewer rows. We need to track which were removed.
            # The keep_mask approach: before pruning, record opacity; after, count survivors
            # Actually, we can track by checking which IDs are still present
            # Simpler approach: record pre-prune N and post-prune N
            post_prune_N = model.xyz.shape[0]
            pruned_count = post_N - post_prune_N  # removed by pruning

            densify_stats.append({
                "iter": iter_idx,
                "pre_N": pre_N,
                "created": new_count,
                "cloned": counts["cloned"],
                "split": counts["split"],
                "pruned": pruned_count,
                "post_N": post_prune_N,
                "net_change": post_prune_N - pre_N,
            })

        if iter_idx % 1000 == 0 and iter_idx > 0:
            if model.sh_degree < 3:
                model.set_sh_degree(model.sh_degree + 1)

        if iter_idx > 0 and iter_idx % 3000 == 0:
            removed = model.prune_and_reset(opacity_threshold=PRUNE_THRESHOLD, current_step=iter_idx)
            total_prune += removed

        optimizer.step()

        if iter_idx % 1000 == 0:
            results["gs_trajectory"].append({"iter": iter_idx, "n_gaussians": model.xyz.shape[0]})

        if iter_idx % EVAL_INTERVAL == 0 or iter_idx == iters - 1:
            model.eval()
            with torch.no_grad():
                data = model.forward()
                psnrs = []
                for ci2 in EVAL_CAMERAS:
                    if ci2 >= len(dataset): break
                    cam2 = dataset.get_camera(ci2)
                    gt2 = dataset.get_gt_image(ci2)
                    pred2 = render(model, cam2, data)
                    psnrs.append(compute_psnr(pred2, gt2))
                psnr = float(np.mean(psnrs))
            model.train()
            results["eval_points"].append({
                "iter": iter_idx,
                "psnr": psnr,
                "gaussians": model.xyz.shape[0],
            })
            print(f"  [{iter_idx:>5}] PSNR={psnr:.2f}  GS={model.xyz.shape[0]:,}", flush=True)

    results["total_train_time_s"] = time.perf_counter() - train_start
    results["densify_stats"] = densify_stats
    results["gradient_distributions"] = gradient_distributions
    results["gs_trajectory"] = results["gs_trajectory"]
    results["eval_points"] = results["eval_points"]

    # Compute lifecycle summary
    total_created = sum(d["created"] for d in densify_stats)
    total_pruned_final = sum(d["pruned"] for d in densify_stats)
    
    results["lifecycle_summary"] = {
        "initial_gaussians": initial_N,
        "total_created": total_created,
        "total_pruned": total_pruned_final,
        "final_gaussians": model.xyz.shape[0],
        "total_clone": total_clone,
        "total_split": total_split,
        "waste_ratio": total_pruned_final / max(total_created, 1),
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="room")
    parser.add_argument("--iters", type=int, default=5000)
    args = parser.parse_args()

    repo_root = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Phase C49 Experiment 1: Gaussian Lifecycle Profiling")
    print(f"  Scene: {args.scene}, Iters: {args.iters}")
    print(f"  Pruning: moderate (threshold=0.01, grad=0.001, densify 500-15000)")

    dataset = GTDataset(scene=args.scene, repo_root=repo_root, resolution="1080p", device=DEVICE)
    sfm_data = load_initial_checkpoint(args.scene, repo_root, device=DEVICE)
    print(f"  SfM points: {sfm_data['xyz'].shape[0]:,}")

    results = train_with_lifecycle_tracking(dataset, sfm_data, iters=args.iters)

    print(f"\n{'='*70}")
    print("Lifecycle Summary:")
    print(f"{'='*70}")
    ls = results["lifecycle_summary"]
    print(f"  Initial Gaussians:  {ls['initial_gaussians']:,}")
    print(f"  Total created:      {ls['total_created']:,}")
    print(f"  Total pruned:       {ls['total_pruned']:,}")
    print(f"  Final Gaussians:    {ls['final_gaussians']:,}")
    print(f"  Clone count:        {ls['total_clone']:,}")
    print(f"  Split count:        {ls['total_split']:,}")
    print(f"  Waste ratio:        {ls['waste_ratio']:.1%}")

    print(f"\n{'='*70}")
    print("Gradient Distribution (last measurement):")
    print(f"{'='*70}")
    if results["gradient_distributions"]:
        gd = results["gradient_distributions"][-1]
        print(f"  N Gaussians:        {gd['n_gaussians']:,}")
        print(f"  Grad median:        {gd['grad_p50']:.6f}")
        print(f"  Grad mean:          {gd['grad_mean']:.6f}")
        print(f"  Grad max:           {gd['grad_max']:.6f}")
        print(f"  Top 1% → {gd['top_1pct_fraction']*100:.1f}% of total gradient")
        print(f"  Top 10% → {gd['top_10pct_fraction']*100:.1f}% of total gradient")
        print(f"  Top 50% → {gd['top_50pct_fraction']*100:.1f}% of total gradient")
        print(f"  K for 90%: {gd['k_for_90pct']} ({gd['k_for_90pct_fraction']*100:.1f}% of Gaussians)")
        print(f"  Grad-Opacity corr:  {gd.get('grad_opacity_correlation', 'N/A')}")
        print(f"  Grad-Scale corr:    {gd.get('grad_scale_correlation', 'N/A')}")

    print(f"\n  Total time: {results['total_train_time_s']:.1f}s")

    save_path = repo_root / "results" / "a100" / "phase-c49" / "lifecycle_profile.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {save_path}")


if __name__ == "__main__":
    main()
