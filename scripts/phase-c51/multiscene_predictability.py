#!/usr/bin/env python3
"""
Phase C51 Stage 3: Multi-Scene Predictor Validation.

Validates previous-gradient prediction on room, bicycle, and garden scenes.

For each scene, measures:
  - Recall@32% (previous gradient as predictor)
  - Coverage@32% (gradient signal preserved)
  - Coverage@50% (gradient signal preserved at 50% selection)

Success criterion:
  Recall@32% >= 0.90 AND Coverage@32% >= 0.85 AND Coverage@50% >= 0.95

If one scene fails, analyze why before implementing CUDA.

This is a lightweight version of C50's gradient_predictability.py — only measures
the key decision metrics, not the full correlation analysis.
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
PRUNE_THRESHOLD = 0.01
GRAD_THRESHOLD = 0.001
EVAL_INTERVAL = 1000
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


def compute_recall_and_coverage(prev_scores, curr_scores, N, K):
    """Compute Recall@K and gradient coverage."""
    if K >= N:
        return 1.0, 1.0
    _, prev_top_idx = torch.topk(prev_scores, K)
    _, curr_top_idx = torch.topk(curr_scores, K)
    mask_curr = torch.zeros(N, dtype=torch.bool, device=curr_scores.device)
    mask_curr[curr_top_idx] = True
    recall = mask_curr[prev_top_idx].sum().item() / K
    total_grad = curr_scores.sum().item()
    if total_grad == 0:
        return recall, 0.0
    selected_grad = curr_scores[prev_top_idx].sum().item()
    coverage = selected_grad / total_grad
    return recall, coverage


def train_and_measure_multiscene(dataset, sfm_data, scene_name, iters=5000):
    """Train and measure predictor quality for a single scene."""
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
    prev_grad_norm = None

    # Measurements per phase
    measurements = {"early": [], "middle": [], "late": []}
    eval_points = []

    def get_phase(iter_idx):
        if iter_idx < 500: return "init"
        elif iter_idx < 2000: return "early"
        elif iter_idx < 4000: return "middle"
        else: return "late"

    train_start = time.perf_counter()

    for iter_idx in range(iters):
        model.train()
        ci = cam_indices[iter_idx % n_cams]
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)
        data = model.forward()
        pred = render(model, cam, data)

        l1 = F.l1_loss(pred, gt)
        if iter_idx % 8 == 0:
            loss = 0.8 * l1 + 0.2 * sep_ssim(pred, gt)
        else:
            loss = l1

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Measure predictor quality
        if model.xyz.grad is not None:
            curr_grad_norm = model.xyz.grad.detach().norm(dim=-1)
            N = curr_grad_norm.shape[0]
            phase = get_phase(iter_idx)

            if phase != "init" and prev_grad_norm is not None and prev_grad_norm.shape[0] == N:
                K_32 = max(int(N * 0.32), 1)
                K_50 = max(int(N * 0.50), 1)
                recall_32, coverage_32 = compute_recall_and_coverage(
                    prev_grad_norm, curr_grad_norm, N, K_32)
                _, coverage_50 = compute_gradient_coverage_only(
                    prev_grad_norm, curr_grad_norm, N, K_50)
                measurements[phase].append({
                    "iter": iter_idx,
                    "recall_32": recall_32,
                    "coverage_32": coverage_32,
                    "coverage_50": coverage_50,
                })

            if prev_grad_norm is None or prev_grad_norm.shape[0] != N:
                prev_grad_norm = curr_grad_norm.clone()
            else:
                prev_grad_norm = curr_grad_norm.clone()

        # Densification
        if iter_idx >= 500 and iter_idx < 15000 and iter_idx % 100 == 0:
            model.accumulate_positional_gradient()
            model.densification(grad_threshold=GRAD_THRESHOLD)
            model.prune(opacity_threshold=PRUNE_THRESHOLD)

        if iter_idx % 1000 == 0 and iter_idx > 0:
            if model.sh_degree < 3:
                model.set_sh_degree(model.sh_degree + 1)

        if iter_idx > 0 and iter_idx % 3000 == 0:
            model.prune_and_reset(opacity_threshold=PRUNE_THRESHOLD, current_step=iter_idx)

        optimizer.step()

        if iter_idx % EVAL_INTERVAL == 0 or iter_idx == iters - 1:
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
            eval_points.append({"iter": iter_idx, "psnr": psnr, "gaussians": gs})
            print(f"  [{scene_name}] [{iter_idx:>5}] PSNR={psnr:.2f}  GS={gs:,}", flush=True)

    total_time = time.perf_counter() - train_start

    # Compute phase statistics
    phase_stats = {}
    for phase in ("early", "middle", "late"):
        data = measurements[phase]
        if not data:
            phase_stats[phase] = {"count": 0}
            continue
        phase_stats[phase] = {"count": len(data)}
        for key in ("recall_32", "coverage_32", "coverage_50"):
            vals = [d[key] for d in data]
            arr = np.array(vals)
            phase_stats[phase][key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }

    return {
        "scene": scene_name,
        "eval_points": eval_points,
        "phase_statistics": phase_stats,
        "total_train_time_s": total_time,
        "final_psnr": eval_points[-1]["psnr"] if eval_points else 0,
        "final_gaussians": eval_points[-1]["gaussians"] if eval_points else 0,
    }


def compute_gradient_coverage_only(predictor_scores, curr_grad_norm, N, K):
    """Compute only coverage (not recall) for K=50%."""
    if K >= N:
        return None, 1.0
    _, selected_idx = torch.topk(predictor_scores, K)
    total_grad = curr_grad_norm.sum().item()
    if total_grad == 0:
        return None, 0.0
    selected_grad = curr_grad_norm[selected_idx].sum().item()
    return None, selected_grad / total_grad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, choices=["room", "bicycle", "garden"])
    parser.add_argument("--iters", type=int, default=5000)
    args = parser.parse_args()

    repo_root = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Phase C51 Stage 3: Multi-Scene Predictor Validation")
    print(f"  Scene: {args.scene}, Iters: {args.iters}")
    print()

    dataset = GTDataset(scene=args.scene, repo_root=repo_root, resolution="1080p", device=DEVICE)
    sfm_data = load_initial_checkpoint(args.scene, repo_root, device=DEVICE)
    print(f"  SfM points: {sfm_data['xyz'].shape[0]:,}")

    results = train_and_measure_multiscene(
        dataset, sfm_data, args.scene, iters=args.iters)

    save_dir = repo_root / "results" / "a100" / "phase-c51"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / f"multiscene_{args.scene}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results for {args.scene}:")
    print(f"  Total time: {results['total_train_time_s']:.1f}s")
    print(f"  Final PSNR: {results['final_psnr']:.2f}")
    print(f"  Final GS: {results['final_gaussians']:,}")

    for phase in ("early", "middle", "late"):
        ps = results["phase_statistics"].get(phase, {})
        if "recall_32" in ps:
            print(f"  {phase:>6}: R@32%={ps['recall_32']['mean']:.3f}  "
                  f"Cov@32%={ps['coverage_32']['mean']:.3f}  "
                  f"Cov@50%={ps['coverage_50']['mean']:.3f}  "
                  f"(n={ps['count']})")

    # Check success criteria (using late phase)
    late = results["phase_statistics"].get("late", {})
    if "recall_32" in late:
        r32 = late["recall_32"]["mean"]
        c32 = late["coverage_32"]["mean"]
        c50 = late["coverage_50"]["mean"]
        print(f"\n  Success criteria (late phase):")
        print(f"    Recall@32% >= 0.90: {r32:.3f} → {'PASS' if r32 >= 0.90 else 'FAIL'}")
        print(f"    Coverage@32% >= 0.85: {c32:.3f} → {'PASS' if c32 >= 0.85 else 'FAIL'}")
        print(f"    Coverage@50% >= 0.95: {c50:.3f} → {'PASS' if c50 >= 0.95 else 'FAIL'}")

    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
