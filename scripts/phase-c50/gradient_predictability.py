#!/usr/bin/env python3
"""
Phase C50: Gradient Predictability Analysis.

Three experiments in a SINGLE training run:

Experiment 1: Temporal Gradient Correlation
  - Pearson correlation between |g(t-1)| and |g(t)|
  - Spearman correlation (rank-based)
  - Split by phase: early (500-2000), middle (2000-4000), late (4000-5000)

Experiment 2: Top-K Importance Persistence
  - Recall@K = |TopK_prev ∩ TopK_curr| / K
  - K = 1%, 5%, 10%, 32%, 50%

Experiment 3: Predictor Comparison
  - Oracle (current gradient — upper bound)
  - Previous gradient (g(t-1))
  - EMA beta=0.9
  - EMA beta=0.99
  - Opacity (sigmoid(opacity))
  - Metrics: Recall@10%, @32%, @50% and Gradient Coverage

Measurement ONLY — no CUDA modification, no sparse backward implementation.
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

# Phases
def get_phase(iter_idx):
    if iter_idx < 500:
        return "init"
    elif iter_idx < 2000:
        return "early"
    elif iter_idx < 4000:
        return "middle"
    else:
        return "late"

# K fractions for Recall@K
K_FRACTIONS = [
    (0.01, "1pct"),
    (0.05, "5pct"),
    (0.10, "10pct"),
    (0.32, "32pct"),
    (0.50, "50pct"),
]

# Predictor K fractions (subset for Experiment 3)
PRED_K_FRACTIONS = [
    (0.10, "10pct"),
    (0.32, "32pct"),
    (0.50, "50pct"),
]


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


def compute_recall_at_k(prev_scores, curr_scores, N, K):
    """Compute Recall@K: fraction of prev top-K that are also in curr top-K."""
    if K >= N:
        return 1.0
    _, prev_top_idx = torch.topk(prev_scores, K)
    _, curr_top_idx = torch.topk(curr_scores, K)
    mask_curr = torch.zeros(N, dtype=torch.bool, device=curr_scores.device)
    mask_curr[curr_top_idx] = True
    recall = mask_curr[prev_top_idx].sum().item() / K
    return recall


def compute_gradient_coverage(predictor_scores, curr_grad_norm, N, K):
    """Compute gradient coverage: fraction of total gradient captured by top-K selected by predictor."""
    if K >= N:
        return 1.0
    _, selected_idx = torch.topk(predictor_scores, K)
    total_grad = curr_grad_norm.sum().item()
    if total_grad == 0:
        return 0.0
    selected_grad = curr_grad_norm[selected_idx].sum().item()
    return selected_grad / total_grad


def train_and_measure(dataset, sfm_data, iters=5000):
    """Train with full gradient predictability measurement."""
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

    # Temporal tracking state
    prev_grad_norm = None      # [N] gradient norm from previous iteration
    ema_09 = None              # [N] EMA with beta=0.9
    ema_099 = None             # [N] EMA with beta=0.99

    # Per-iteration measurements (CPU lists of dicts)
    temporal_data = []         # Experiment 1: Pearson + Spearman
    topk_data = []             # Experiment 2: Recall@K
    predictor_data = []        # Experiment 3: Predictor comparison

    total_clone, total_split, total_prune = 0, 0, 0
    eval_points = []

    train_start = time.perf_counter()

    for iter_idx in range(iters):
        model.train()
        ci = cam_indices[iter_idx % n_cams]
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)
        data = model.forward()
        pred = render(model, cam, data)

        # Loss (sep_freq8 — same as C49 baseline)
        l1 = F.l1_loss(pred, gt)
        if iter_idx % 8 == 0:
            loss = 0.8 * l1 + 0.2 * sep_ssim(pred, gt)
        else:
            loss = l1

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # === MEASUREMENT SECTION ===
        # All measurements use the gradient AFTER backward, BEFORE optimizer step
        if model.xyz.grad is not None:
            curr_grad_norm = model.xyz.grad.detach().norm(dim=-1)  # [N]
            N = curr_grad_norm.shape[0]
            curr_opacity = torch.sigmoid(model.opacity).detach().squeeze(-1)  # [N]
            phase = get_phase(iter_idx)

            # Only measure when phase is not "init" (need stable training)
            if phase != "init":
                # --- Experiment 1: Temporal Gradient Correlation ---
                if prev_grad_norm is not None and prev_grad_norm.shape[0] == N:
                    # Pearson correlation
                    stacked = torch.stack([prev_grad_norm, curr_grad_norm])
                    pearson = float(torch.corrcoef(stacked)[0, 1].item())

                    # Spearman correlation (rank-based)
                    rank_prev = prev_grad_norm.argsort().argsort().float()
                    rank_curr = curr_grad_norm.argsort().argsort().float()
                    spearman = float(torch.corrcoef(torch.stack([rank_prev, rank_curr]))[0, 1].item())

                    temporal_data.append({
                        "iter": iter_idx, "phase": phase, "N": N,
                        "pearson": pearson, "spearman": spearman,
                    })

                    # --- Experiment 2: Top-K Importance Persistence ---
                    topk_record = {"iter": iter_idx, "phase": phase, "N": N}
                    for frac, label in K_FRACTIONS:
                        K = max(int(N * frac), 1)
                        recall = compute_recall_at_k(prev_grad_norm, curr_grad_norm, N, K)
                        topk_record[f"recall_{label}"] = recall
                    topk_data.append(topk_record)

                    # --- Experiment 3: Predictor Comparison ---
                    pred_record = {"iter": iter_idx, "phase": phase, "N": N}
                    predictors = {
                        "oracle": curr_grad_norm,
                        "previous": prev_grad_norm,
                        "ema_09": ema_09,
                        "ema_099": ema_099,
                        "opacity": curr_opacity,
                    }
                    for pred_name, pred_scores in predictors.items():
                        if pred_scores is None or pred_scores.shape[0] != N:
                            continue
                        for frac, label in PRED_K_FRACTIONS:
                            K = max(int(N * frac), 1)
                            recall = compute_recall_at_k(pred_scores, curr_grad_norm, N, K)
                            coverage = compute_gradient_coverage(pred_scores, curr_grad_norm, N, K)
                            pred_record[f"{pred_name}_recall_{label}"] = recall
                            pred_record[f"{pred_name}_coverage_{label}"] = coverage
                    predictor_data.append(pred_record)

                # Update EMA and prev_grad_norm
                if ema_09 is None or ema_09.shape[0] != N:
                    # Initialize or reset (N changed due to densification)
                    ema_09 = curr_grad_norm.clone()
                    ema_099 = curr_grad_norm.clone()
                    prev_grad_norm = None  # Reset — can't correlate across N change
                else:
                    # Normal EMA update (BEFORE setting prev, so EMA used as predictor is from t-1)
                    ema_09 = 0.9 * ema_09 + 0.1 * curr_grad_norm
                    ema_099 = 0.99 * ema_099 + 0.01 * curr_grad_norm
                    prev_grad_norm = curr_grad_norm.clone()
        # === END MEASUREMENT SECTION ===

        # Densification (moderate: 500-15000, every 100 iters)
        if iter_idx >= 500 and iter_idx < 15000 and iter_idx % 100 == 0:
            model.accumulate_positional_gradient()
            counts = model.densification(grad_threshold=GRAD_THRESHOLD)
            total_clone += counts["cloned"]
            total_split += counts["split"]
            removed = model.prune(opacity_threshold=PRUNE_THRESHOLD)
            total_prune += removed
            # After densification, N changes. prev_grad_norm and EMA will be
            # reset on next iteration when shape mismatch is detected.

        if iter_idx % 1000 == 0 and iter_idx > 0:
            if model.sh_degree < 3:
                model.set_sh_degree(model.sh_degree + 1)

        if iter_idx > 0 and iter_idx % 3000 == 0:
            removed = model.prune_and_reset(opacity_threshold=PRUNE_THRESHOLD, current_step=iter_idx)
            total_prune += removed

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
            print(f"  [{iter_idx:>5}] PSNR={psnr:.2f}  GS={gs:,}  "
                  f"measurements: temporal={len(temporal_data)} topk={len(topk_data)} pred={len(predictor_data)}",
                  flush=True)

    total_time = time.perf_counter() - train_start

    return {
        "temporal_data": temporal_data,
        "topk_data": topk_data,
        "predictor_data": predictor_data,
        "eval_points": eval_points,
        "total_train_time_s": total_time,
        "total_clone": total_clone,
        "total_split": total_split,
        "total_prune": total_prune,
    }


def compute_phase_stats(data_list, metric_keys, phases=("early", "middle", "late")):
    """Compute mean, median, std for each metric per phase."""
    results = {}
    for phase in phases:
        phase_data = [d for d in data_list if d.get("phase") == phase]
        if not phase_data:
            results[phase] = {"count": 0}
            continue
        results[phase] = {"count": len(phase_data)}
        for key in metric_keys:
            values = [d[key] for d in phase_data if key in d]
            if values:
                arr = np.array(values)
                results[phase][key] = {
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
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

    print(f"Phase C50: Gradient Predictability Analysis")
    print(f"  Scene: {args.scene}, Iters: {args.iters}")
    print(f"  Pruning: moderate (threshold=0.01, grad=0.001, densify 500-15000)")
    print()

    dataset = GTDataset(scene=args.scene, repo_root=repo_root, resolution="1080p", device=DEVICE)
    sfm_data = load_initial_checkpoint(args.scene, repo_root, device=DEVICE)
    print(f"  SfM points: {sfm_data['xyz'].shape[0]:,}")

    results = train_and_measure(dataset, sfm_data, iters=args.iters)

    total_time = results["total_train_time_s"]
    print(f"\n{'='*70}")
    print(f"Training complete: {total_time:.1f}s")
    print(f"  Measurements: temporal={len(results['temporal_data'])}, "
          f"topk={len(results['topk_data'])}, "
          f"predictor={len(results['predictor_data'])}")
    print(f"  Clone: {results['total_clone']:,}, Split: {results['total_split']:,}, "
          f"Prune: {results['total_prune']:,}")

    # === Compute phase statistics ===
    save_dir = repo_root / "results" / "a100" / "phase-c50"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Experiment 1: Temporal correlation statistics
    temporal_stats = compute_phase_stats(
        results["temporal_data"], ["pearson", "spearman"])
    temporal_output = {
        "experiment": "temporal_gradient_correlation",
        "description": "Pearson and Spearman correlation between consecutive iteration gradient norms",
        "raw_data_count": len(results["temporal_data"]),
        "phase_statistics": temporal_stats,
        "raw_data": results["temporal_data"],  # Include raw data for distribution analysis
    }
    with open(save_dir / "gradient_temporal.json", "w") as f:
        json.dump(temporal_output, f, indent=2)
    print(f"\nExperiment 1 (Temporal Correlation) saved to gradient_temporal.json")
    for phase in ("early", "middle", "late"):
        ps = temporal_stats.get(phase, {})
        if "pearson" in ps:
            print(f"  {phase:>6}: Pearson={ps['pearson']['mean']:.4f}±{ps['pearson']['std']:.4f}  "
                  f"Spearman={ps['spearman']['mean']:.4f}±{ps['spearman']['std']:.4f}  "
                  f"(n={ps['count']})")

    # Experiment 2: Top-K persistence statistics
    topk_metric_keys = [f"recall_{label}" for _, label in K_FRACTIONS]
    topk_stats = compute_phase_stats(results["topk_data"], topk_metric_keys)
    topk_output = {
        "experiment": "topk_importance_persistence",
        "description": "Recall@K: fraction of previous top-K Gaussians still in current top-K",
        "k_fractions": {label: frac for frac, label in K_FRACTIONS},
        "raw_data_count": len(results["topk_data"]),
        "phase_statistics": topk_stats,
    }
    with open(save_dir / "topk_persistence.json", "w") as f:
        json.dump(topk_output, f, indent=2)
    print(f"\nExperiment 2 (Top-K Persistence) saved to topk_persistence.json")
    for phase in ("early", "middle", "late"):
        ps = topk_stats.get(phase, {})
        if "recall_32pct" in ps:
            print(f"  {phase:>6}: R@1%={ps['recall_1pct']['mean']:.3f}  "
                  f"R@10%={ps['recall_10pct']['mean']:.3f}  "
                  f"R@32%={ps['recall_32pct']['mean']:.3f}  "
                  f"R@50%={ps['recall_50pct']['mean']:.3f}  (n={ps['count']})")

    # Experiment 3: Predictor comparison statistics
    pred_metric_keys = []
    for pred_name in ["oracle", "previous", "ema_09", "ema_099", "opacity"]:
        for _, label in PRED_K_FRACTIONS:
            pred_metric_keys.append(f"{pred_name}_recall_{label}")
            pred_metric_keys.append(f"{pred_name}_coverage_{label}")
    pred_stats = compute_phase_stats(results["predictor_data"], pred_metric_keys)
    pred_output = {
        "experiment": "predictor_comparison",
        "description": "Compare pre-backward importance predictors: Oracle, Previous, EMA, Opacity",
        "predictors": ["oracle", "previous", "ema_09", "ema_099", "opacity"],
        "k_fractions": {label: frac for frac, label in PRED_K_FRACTIONS},
        "raw_data_count": len(results["predictor_data"]),
        "phase_statistics": pred_stats,
    }
    with open(save_dir / "predictor_comparison.json", "w") as f:
        json.dump(pred_output, f, indent=2)
    print(f"\nExperiment 3 (Predictor Comparison) saved to predictor_comparison.json")
    for phase in ("early", "middle", "late"):
        ps = pred_stats.get(phase, {})
        if "previous_recall_32pct" in ps:
            print(f"  {phase:>6}:")
            for pred_name in ["oracle", "previous", "ema_09", "ema_099", "opacity"]:
                r32_key = f"{pred_name}_recall_32pct"
                c32_key = f"{pred_name}_coverage_32pct"
                c50_key = f"{pred_name}_coverage_50pct"
                if r32_key in ps:
                    print(f"    {pred_name:>10}: R@32%={ps[r32_key]['mean']:.3f}  "
                          f"Cov@32%={ps[c32_key]['mean']:.3f}  "
                          f"Cov@50%={ps[c50_key]['mean']:.3f}")

    # === Decision Summary ===
    print(f"\n{'='*70}")
    print("DECISION CRITERIA CHECK")
    print(f"{'='*70}")
    # Use "late" phase as the most representative (stable training)
    late_pred = pred_stats.get("late", {})
    late_topk = topk_stats.get("late", {})

    if "previous_recall_32pct" in late_pred:
        r32 = late_pred["previous_recall_32pct"]["mean"]
        c32 = late_pred["previous_coverage_32pct"]["mean"]
        c50 = late_pred["previous_coverage_50pct"]["mean"]

        print(f"\n  Previous gradient predictor (late phase):")
        print(f"    Recall@32% = {r32:.3f}  (threshold: >= 0.70)  → {'PASS' if r32 >= 0.70 else 'FAIL'}")
        print(f"    Coverage@32% = {c32:.3f}  (threshold: >= 0.85)  → {'PASS' if c32 >= 0.85 else 'FAIL'}")
        print(f"    Coverage@50% = {c50:.3f}  (threshold: >= 0.95)  → {'PASS' if c50 >= 0.95 else 'FAIL'}")

        if r32 >= 0.70 and c32 >= 0.85 and c50 >= 0.95:
            decision = "KEEP"
        elif c50 >= 0.90:
            decision = "MODIFY"
        else:
            decision = "DROP"
        print(f"\n  DECISION: {decision}")

    # Save summary
    summary = {
        "total_train_time_s": total_time,
        "total_clone": results["total_clone"],
        "total_split": results["total_split"],
        "total_prune": results["total_prune"],
        "eval_points": results["eval_points"],
        "measurement_counts": {
            "temporal": len(results["temporal_data"]),
            "topk": len(results["topk_data"]),
            "predictor": len(results["predictor_data"]),
        },
    }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {save_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
