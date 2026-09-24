#!/usr/bin/env python3
"""
Phase C51-R: Error-Controlled Predictive Sparse Backward.

Unified experiment script supporting all 8 configurations:
  - baseline: full backward, no mask
  - K=90%: keep top 90% by previous gradient
  - K=80%: keep top 80%
  - refresh50/100/200/500: K=50% with periodic full backward every N iters
  - post_densification: full until iter 15000, then K=50% (requires 30K iters)

Key design (from C51 V2):
  1. Full gradient is computed every iteration (standard backward)
  2. accumulate_positional_gradient() uses FULL gradient (before masking)
  3. Mask is applied to gradients AFTER densification, BEFORE optimizer step
  4. Previous-iteration full gradient norm is the predictor
  5. Refresh iterations: skip mask entirely (full backward + full optimizer update)

Metrics recorded every 100 iterations:
  - PSNR, SSIM, Gaussian count, loss
  - Gradient cosine + rel_l2 (vs dense, at measurement iters)
  - PSNR_gap vs baseline (computed in post-processing)
  - Clone/split/prune counts
  - Iteration timing
"""
import json, math, sys, time, argparse, numpy as np
from pathlib import Path
import torch, torch.nn.functional as F

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
GRAD_MEASURE_ITERS = [500, 1000, 1500, 2000, 3000, 4000, 5000, 7000, 10000, 15000, 20000, 25000, 29999]
TRAJECTORY_INTERVAL = 100  # record trajectory every 100 iters


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


def compute_ssim(pred, gt):
    C1, C2 = (0.01)**2, (0.03)**2
    if pred.ndim == 3:
        pred = pred.unsqueeze(0).permute(0,3,1,2)
        gt = gt.unsqueeze(0).permute(0,3,1,2)
    kernel = torch.ones(3, 1, 11, 11, device=pred.device) / 121.0
    mu_p = F.conv2d(pred, kernel, padding=5, groups=3)
    mu_t = F.conv2d(gt, kernel, padding=5, groups=3)
    mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p*mu_t
    sp2 = F.conv2d(pred**2, kernel, padding=5, groups=3) - mu_p2
    st2 = F.conv2d(gt**2, kernel, padding=5, groups=3) - mu_t2
    spt = F.conv2d(pred*gt, kernel, padding=5, groups=3) - mu_pt
    ssim = (2*mu_pt+C1)*(2*spt+C2) / ((mu_p2+mu_t2+C1)*(sp2+st2+C2))
    return float(ssim.mean().item())


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


def compute_grad_correctness(dense_grads, masked_grads, param_names):
    results = {}
    for name in param_names:
        if name not in dense_grads or name not in masked_grads:
            continue
        d = dense_grads[name].flatten()
        m = masked_grads[name].flatten()
        d_norm = torch.norm(d).item()
        m_norm = torch.norm(m).item()
        if d_norm < 1e-12:
            rel_l2, cosine = 0.0, 1.0
        else:
            rel_l2 = torch.norm(m - d).item() / d_norm
            cosine = torch.dot(m, d).item() / (m_norm * d_norm) if m_norm > 1e-12 else 0.0
        results[name] = {"rel_l2": rel_l2, "cosine": cosine}
    return results


def run_experiment(dataset, sfm_data, config):
    """Run a single experiment with the given configuration."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    k_frac = config["keep_fraction"]          # fraction to KEEP (1.0 = no mask)
    refresh_period = config.get("refresh_period", 0)  # 0 = no refresh
    sparse_start_iter = config.get("sparse_start_iter", 0)  # 0 = sparse from start
    iters = config["iters"]
    eval_interval = config.get("eval_interval", 1000)

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

    prev_grad_norm = None
    total_clone, total_split, total_prune = 0, 0, 0
    eval_points = []
    trajectory = []  # every TRAJECTORY_INTERVAL iters
    grad_measurements = []
    iter_times = []

    train_start = time.perf_counter()

    for iter_idx in range(iters):
        iter_start = time.perf_counter()
        model.train()
        ci = cam_indices[iter_idx % n_cams]
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)

        N = model.xyz.shape[0]

        # === Determine if this iteration is sparse or full ===
        is_sparse_iter = (
            k_frac < 1.0
            and iter_idx >= sparse_start_iter
            and prev_grad_norm is not None
            and prev_grad_norm.shape[0] == N
        )
        is_refresh_iter = (
            refresh_period > 0
            and is_sparse_iter
            and iter_idx > sparse_start_iter
            and (iter_idx - sparse_start_iter) % refresh_period == 0
        )
        if is_refresh_iter:
            is_sparse_iter = False  # full backward on refresh iters

        # === Build mask ===
        if is_sparse_iter:
            K = max(int(N * k_frac), 1)
            _, top_idx = torch.topk(prev_grad_norm, K)
            mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
            mask[top_idx] = True
        else:
            mask = None

        # === Forward + Loss ===
        data = model.forward()
        pred = render(model, cam, data)
        l1 = F.l1_loss(pred, gt)
        if iter_idx % 8 == 0:
            loss = 0.8 * l1 + 0.2 * sep_ssim(pred, gt)
        else:
            loss = l1

        # === Backward (always full) ===
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # === Save dense gradient norm (predictor for next iter) ===
        if model.xyz.grad is not None:
            curr_grad_norm = model.xyz.grad.detach().norm(dim=-1)
            prev_grad_norm = curr_grad_norm.clone()

        # === Gradient correctness measurement ===
        if iter_idx in GRAD_MEASURE_ITERS and mask is not None:
            dense_grads = {}
            for name, p in [("xyz", model.xyz), ("scales", model.scales),
                            ("rotations", model.rotations), ("opacity", model.opacity),
                            ("shs", model.shs)]:
                if p.grad is not None:
                    dense_grads[name] = p.grad.detach().clone()

            masked_grads = {}
            for name, p in [("xyz", model.xyz), ("scales", model.scales),
                            ("rotations", model.rotations), ("opacity", model.opacity),
                            ("shs", model.shs)]:
                if p.grad is not None and name in dense_grads:
                    m = mask.view(N, *([1] * (p.grad.dim() - 1)))
                    masked = p.grad.clone()
                    masked.mul_(m.float())
                    masked_grads[name] = masked

            correctness = compute_grad_correctness(dense_grads, masked_grads,
                ["xyz", "scales", "rotations", "opacity", "shs"])
            grad_measurements.append({
                "iter": iter_idx, "N": N, "is_refresh": is_refresh_iter,
                "correctness": correctness
            })
            print(f"  [GRAD iter={iter_idx}] xyz_cos={correctness['xyz']['cosine']:.4f} "
                  f"refresh={is_refresh_iter}", flush=True)

        # === Densification (uses FULL gradient, BEFORE masking) ===
        if iter_idx >= 500 and iter_idx < 15000 and iter_idx % 100 == 0:
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

        # === Apply mask to gradients (AFTER densification, BEFORE optimizer) ===
        if mask is not None:
            for p in [model.xyz, model.scales, model.rotations, model.opacity, model.shs]:
                if p.grad is not None:
                    m = mask.view(N, *([1] * (p.grad.dim() - 1)))
                    p.grad.mul_(m.float())

        optimizer.step()

        iter_time = time.perf_counter() - iter_start
        iter_times.append(iter_time)

        # === Trajectory recording (every 100 iters) ===
        if iter_idx % TRAJECTORY_INTERVAL == 0 or iter_idx == iters - 1:
            model.eval()
            with torch.no_grad():
                data_eval = model.forward()
                psnrs = []
                for ci2 in EVAL_CAMERAS:
                    if ci2 >= len(dataset): break
                    cam2 = dataset.get_camera(ci2)
                    gt2 = dataset.get_gt_image(ci2)
                    pred2 = render(model, cam2, data_eval)
                    psnrs.append(compute_psnr(pred2, gt2))
                psnr = float(np.mean(psnrs))
            model.train()
            gs = model.xyz.shape[0]
            trajectory.append({
                "iter": iter_idx, "psnr": psnr, "gaussians": gs,
                "loss": float(loss.item()), "is_sparse": is_sparse_iter,
                "is_refresh": is_refresh_iter,
            })

        # === Full evaluation (every eval_interval) ===
        if iter_idx % eval_interval == 0 or iter_idx == iters - 1:
            last_traj = trajectory[-1]
            eval_points.append({
                "iter": iter_idx,
                "psnr": last_traj["psnr"],
                "gaussians": last_traj["gaussians"],
                "loss": last_traj["loss"],
            })
            print(f"  [{iter_idx:>5}] PSNR={last_traj['psnr']:.2f}  GS={last_traj['gaussians']:,}  "
                  f"sparse={is_sparse_iter}  refresh={is_refresh_iter}", flush=True)

    total_time = time.perf_counter() - train_start

    return {
        "config": config,
        "eval_points": eval_points,
        "trajectory": trajectory,
        "grad_measurements": grad_measurements,
        "total_train_time_s": total_time,
        "total_clone": total_clone,
        "total_split": total_split,
        "total_prune": total_prune,
        "iter_time_stats": {
            "mean": float(np.mean(iter_times)),
            "median": float(np.median(iter_times)),
            "p50": float(np.percentile(iter_times, 50)),
            "p90": float(np.percentile(iter_times, 90)),
            "std": float(np.std(iter_times)),
        },
        "final_psnr": eval_points[-1]["psnr"] if eval_points else 0,
        "final_gaussians": eval_points[-1]["gaussians"] if eval_points else 0,
    }


CONFIGS = {
    "baseline": {
        "keep_fraction": 1.0, "refresh_period": 0, "sparse_start_iter": 0,
        "iters": 5000, "eval_interval": 1000,
    },
    "k90": {
        "keep_fraction": 0.9, "refresh_period": 0, "sparse_start_iter": 0,
        "iters": 5000, "eval_interval": 1000,
    },
    "k80": {
        "keep_fraction": 0.8, "refresh_period": 0, "sparse_start_iter": 0,
        "iters": 5000, "eval_interval": 1000,
    },
    "refresh50": {
        "keep_fraction": 0.5, "refresh_period": 50, "sparse_start_iter": 0,
        "iters": 5000, "eval_interval": 1000,
    },
    "refresh100": {
        "keep_fraction": 0.5, "refresh_period": 100, "sparse_start_iter": 0,
        "iters": 5000, "eval_interval": 1000,
    },
    "refresh200": {
        "keep_fraction": 0.5, "refresh_period": 200, "sparse_start_iter": 0,
        "iters": 5000, "eval_interval": 1000,
    },
    "refresh500": {
        "keep_fraction": 0.5, "refresh_period": 500, "sparse_start_iter": 0,
        "iters": 5000, "eval_interval": 1000,
    },
    "post_densification": {
        "keep_fraction": 0.5, "refresh_period": 0, "sparse_start_iter": 15000,
        "iters": 30000, "eval_interval": 5000,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--scene", default="room")
    args = parser.parse_args()

    config = CONFIGS[args.config].copy()
    config["scene"] = args.scene
    config["seed"] = SEED

    repo_root = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Phase C51-R: Error-Controlled Predictive Sparse Backward")
    print(f"  Config: {args.config}")
    print(f"  Scene: {args.scene}, Iters: {config['iters']}")
    print(f"  Keep fraction: {config['keep_fraction']}")
    print(f"  Refresh period: {config.get('refresh_period', 0)}")
    print(f"  Sparse start: {config.get('sparse_start_iter', 0)}")
    print()

    dataset = GTDataset(scene=args.scene, repo_root=repo_root, resolution="1080p", device=DEVICE)
    sfm_data = load_initial_checkpoint(args.scene, repo_root, device=DEVICE)
    print(f"  SfM points: {sfm_data['xyz'].shape[0]:,}")

    results = run_experiment(dataset, sfm_data, config)

    save_dir = repo_root / "results" / "a100" / "phase-c51r"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / f"{args.config}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {out_file}")
    print(f"  Total time: {results['total_train_time_s']:.1f}s")
    print(f"  Final PSNR: {results['final_psnr']:.2f}")
    print(f"  Final GS: {results['final_gaussians']:,}")
    print(f"  Clone/Split/Prune: {results['total_clone']}/{results['total_split']}/{results['total_prune']}")
    print(f"  Iter time: mean={results['iter_time_stats']['mean']*1000:.1f}ms "
          f"median={results['iter_time_stats']['median']*1000:.1f}ms")


if __name__ == "__main__":
    main()
