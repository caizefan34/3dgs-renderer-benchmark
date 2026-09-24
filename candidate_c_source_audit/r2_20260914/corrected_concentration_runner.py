"""
R2 Part 1: Corrected U_loss Concentration + Parts 10-12: Oracle Attribute Masks.

Loads checkpoints, runs ~50 iterations each, computes:
  - Corrected U_loss concentration (positive/absolute)
  - Family-specific oracle masks and their utility retention
  - Combined oracle combination

Does NOT save per-Gaussian arrays. Only aggregate stats.
"""

import sys, os, json, math, argparse
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
BASELINE_DIR = os.path.join(REPO_ROOT, "baseline", "reference_v1")
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, BASELINE_DIR)
sys.path.insert(0, SRC_DIR)

from gaussian_model import GaussianModel
from config import ReferenceV1Config
from gsplat import rasterization

OLD_SCRIPTS = os.path.join(REPO_ROOT, "scripts", "epic05", "phase7")
sys.path.insert(0, OLD_SCRIPTS)
from dataset import GTDataset

BETA1, BETA2, EPS = 0.9, 0.999, 1e-15


class SepSSIM:
    def __init__(self, window_size=11, sigma=1.5, device="cuda"):
        self.C1 = (0.01) ** 2; self.C2 = (0.03) ** 2
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0, 1, 3, 2).contiguous()
        self.padding = window_size // 2

    def __call__(self, pred, target):
        if pred.ndim == 3: pred = pred.unsqueeze(0).permute(0, 3, 1, 2); target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.cat([pred, target, pred ** 2, target ** 2, pred * target], dim=1)
        b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_p, mu_t = b[:, 0:3], b[:, 3:6]
        bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
        mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
        sp2, st2, spt = bp2 - mu_p2, bt2 - mu_t2, bpt - mu_pt
        ssim_map = (2 * mu_pt + self.C1) * (2 * spt + self.C2) / ((mu_p2 + mu_t2 + self.C1) * (sp2 + st2 + self.C2))
        return 1.0 - ssim_map.mean()


def capture_adam_state(model):
    NAME_MAP = {"xyz": "xyz", "shs": "shs", "opacity": "opacity", "scaling": "scale", "rotation": "rot"}
    state = {}
    for group in model.optimizer.param_groups:
        name = NAME_MAP.get(group["name"], group["name"])
        p = group["params"][0]
        stored = model.optimizer.state.get(p, None)
        if stored and "exp_avg" in stored:
            exp_avg = stored["exp_avg"].detach().clone()
            exp_avg_sq = stored["exp_avg_sq"].detach().clone()
            step = stored.get("step", 0)
            if isinstance(step, torch.Tensor): step = step.item()
        else:
            exp_avg = torch.zeros_like(p.data); exp_avg_sq = torch.zeros_like(p.data); step = 0
        state[name] = {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq, "step": step, "lr": group["lr"]}
    return state


def reconstruct_adam_update(state, grad, name):
    exp_avg, exp_avg_sq, step, lr = state["exp_avg"], state["exp_avg_sq"], state["step"], state["lr"]
    t = step + 1
    g = grad.squeeze(-1) if (name == "opacity" and grad.dim() > 1 and grad.shape[-1] == 1) else grad
    m_t = BETA1 * exp_avg + (1 - BETA1) * g
    v_t = BETA2 * exp_avg_sq + (1 - BETA2) * g * g
    m_hat = m_t / (1.0 - BETA1 ** t)
    v_hat = v_t / (1.0 - BETA2 ** t)
    return -lr * m_hat / (torch.sqrt(v_hat) + EPS)


def compute_update_utility(delta, grad):
    if delta.dim() > 1: delta_flat, grad_flat = delta.flatten(start_dim=1), grad.flatten(start_dim=1)
    else: delta_flat, grad_flat = delta.unsqueeze(-1), grad.unsqueeze(-1)
    return delta_flat.norm(dim=-1), -(grad_flat * delta_flat).sum(dim=-1)


def compute_state_score(state, name):
    exp_avg, exp_avg_sq, step, lr = state["exp_avg"], state["exp_avg_sq"], state["step"], state["lr"]
    if step == 0: return torch.zeros(exp_avg.shape[0], device=exp_avg.device)
    t = step
    m_hat = exp_avg / (1.0 - BETA1 ** t)
    v_hat = exp_avg_sq / (1.0 - BETA2 ** t)
    score = lr * m_hat / (torch.sqrt(v_hat) + EPS)
    return score.abs().squeeze(-1) if name == "opacity" else score.flatten(start_dim=1).norm(dim=-1)


def run_corrected_analysis(checkpoint_path, start_iter, n_iters, output_dir, config, camera_sequence_path):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda"
    dataset = GTDataset(config.scene, config.repo_root)
    camera_sequence = np.load(camera_sequence_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = GaussianModel(max_sh_degree=config.sh_degree)
    model.restore(ckpt, {
        "position_lr_init": config.position_lr_init, "position_lr_final": config.position_lr_final,
        "position_lr_delay_mult": config.position_lr_delay_mult, "position_lr_max_steps": config.position_lr_max_steps,
        "feature_lr": config.feature_lr, "opacity_lr": config.opacity_lr,
        "scaling_lr": config.scaling_lr, "rotation_lr": config.rotation_lr,
        "percent_dense": config.percent_dense,
    })
    N = model._xyz.shape[0]
    print(f"Loaded: N={N}")
    ssim_fn = SepSSIM(device=device)

    # Storage for corrected concentration
    corrected_conc = {}  # iter -> {total, geometry, sh, opacity: {pos_total, neg_total, neg_frac, C_K+, C_K_abs}}
    # Storage for oracle masks
    oracle_masks = {}     # iter -> {family: {k: {pos_utility_retained, abs_utility_retained}}}
    # SH state vs oracle
    sh_state_vs_oracle = {}  # iter -> {precision, recall, pos_utility_coverage}

    print(f"Running {n_iters} iterations...")

    for i in range(n_iters):
        iteration = start_iter + 1 + i
        cam_idx = int(camera_sequence[iteration - 1])
        cam, gt_image = dataset.get_item(cam_idx)
        model.update_learning_rate(iteration)

        # Forward
        image, _, meta = rasterization(
            means=model.get_xyz, quats=model.get_rotation, scales=model.get_scaling,
            opacities=model.get_opacity, colors=model.get_features,
            viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            tile_size=16, packed=False, sh_degree=model.active_sh_degree,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
        )
        if image.ndim == 4: image = image.squeeze(0)
        radii = meta["radii"]
        if radii.ndim > 2: radii = radii[0]
        if radii.ndim > 2: radii = radii.squeeze(-1)
        vis_filter = (radii > 0).any(dim=-1)
        vis_np = vis_filter.cpu().numpy().astype(bool)
        vis_count = int(vis_np.sum())

        pre_state = capture_adam_state(model)

        L1 = F.l1_loss(image, gt_image)
        dssim = ssim_fn(image, gt_image)
        loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim
        loss.backward()

        current_N = model._xyz.shape[0]

        # Compute per-family U_loss and U_step
        u_loss = {}
        u_step = {}
        s_state = {}
        for name, param in [("xyz", model._xyz), ("shs", model._shs),
                            ("opacity", model._opacity), ("scale", model._scaling),
                            ("rot", model._rotation)]:
            grad = param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param.data)
            delta = reconstruct_adam_update(pre_state[name], grad, name)
            step_u, loss_u = compute_update_utility(delta, grad)
            u_step[name] = step_u.detach().cpu().numpy().astype(np.float32)
            u_loss[name] = loss_u.detach().cpu().numpy().astype(np.float32)
            s_state[name] = compute_state_score(pre_state[name], name).detach().cpu().numpy().astype(np.float32)

        # Group U_loss
        u_loss["geometry"] = u_loss["xyz"] + u_loss["scale"] + u_loss["rot"]
        u_loss["sh"] = u_loss["shs"]
        u_loss["opacity_g"] = u_loss["opacity"]
        u_loss["total"] = u_loss["geometry"] + u_loss["sh"] + u_loss["opacity_g"]

        # U_step for groups
        u_step["geometry"] = u_step["xyz"] + u_step["scale"] + u_step["rot"]
        u_step["sh"] = u_step["shs"]
        u_step["opacity_g"] = u_step["opacity"]

        # === Part 1: Corrected concentration ===
        iter_conc = {}
        for fam in ["total", "geometry", "sh", "opacity_g"]:
            vals = u_loss[fam][vis_np]
            pos_vals = np.maximum(vals, 0)
            abs_vals = np.abs(vals)
            pos_total = float(pos_vals.sum())
            neg_total = float(np.minimum(vals, 0).sum())  # negative
            neg_frac = float((vals < 0).sum() / max(len(vals), 1))
            neg_pos_ratio = abs(neg_total) / max(pos_total, 1e-20)

            sorted_pos = np.sort(pos_vals)[::-1]
            sorted_abs = np.sort(abs_vals)[::-1]
            n = len(sorted_pos)

            entry = {
                "positive_total": pos_total,
                "negative_total": neg_total,
                "neg_pos_ratio": neg_pos_ratio,
                "neg_fraction": neg_frac,
            }
            for k_pct in [10, 20, 32, 50]:
                k = max(1, int(n * k_pct / 100))
                # C_K+ : TopK by positive utility / total positive
                entry[f"C{k_pct}_pos"] = float(sorted_pos[:k].sum() / max(pos_total, 1e-20))
                # C_K_abs : TopK by absolute utility / total absolute
                entry[f"C{k_pct}_abs"] = float(sorted_abs[:k].sum() / max(abs_vals.sum(), 1e-20))
            iter_conc[fam] = entry
        corrected_conc[iteration] = iter_conc

        # === Parts 10-11: Oracle attribute masks ===
        iter_oracle = {}
        for fam, ukey in [("geometry", "geometry"), ("sh", "sh"), ("opacity", "opacity_g")]:
            vals = u_loss[fam][vis_np]
            pos_vals = np.maximum(vals, 0)
            abs_vals = np.abs(vals)
            pos_total = max(float(pos_vals.sum()), 1e-20)
            abs_total = max(float(abs_vals.sum()), 1e-20)

            # Rank by positive utility (oracle)
            sorted_idx = np.argsort(pos_vals)[::-1]
            fam_result = {}
            for k_pct in [20, 32, 50, 75, 100]:
                k = max(1, int(vis_count * k_pct / 100))
                top_k = sorted_idx[:k]
                fam_result[f"k{k_pct}"] = {
                    "pos_utility_retained": float(pos_vals[top_k].sum() / pos_total),
                    "abs_utility_retained": float(abs_vals[top_k].sum() / abs_total),
                }
            iter_oracle[fam] = fam_result
        oracle_masks[iteration] = iter_oracle

        # === Part 12: Combined oracle ===
        # Find smallest K that retains >=95% positive utility per family
        min_k_per_fam = {}
        for fam in ["geometry", "sh", "opacity"]:
            for k_pct in [20, 32, 50, 75, 100]:
                cov = iter_oracle[fam][f"k{k_pct}"]["pos_utility_retained"]
                if cov >= 0.95:
                    min_k_per_fam[fam] = k_pct
                    break
            if fam not in min_k_per_fam:
                min_k_per_fam[fam] = 100

        # === Part 13: SH state vs oracle ===
        sh_pos = np.maximum(u_loss["sh"][vis_np], 0)
        sh_pos_total = max(float(sh_pos.sum()), 1e-20)
        sh_state_vis = s_state["shs"][vis_np]
        # Oracle top-50% by positive utility
        oracle_idx = set(np.argsort(sh_pos)[::-1][:max(1, int(vis_count * 0.5))].tolist())
        # State predictor top-50%
        state_idx = set(np.argsort(sh_state_vis)[::-1][:max(1, int(vis_count * 0.5))].tolist())
        intersection = oracle_idx & state_idx
        union = oracle_idx | state_idx
        precision = len(intersection) / max(len(state_idx), 1)
        recall = len(intersection) / max(len(oracle_idx), 1)
        # Positive utility coverage of state-selected
        state_arr = np.array(list(state_idx))
        pos_cov = float(sh_pos[state_arr].sum() / sh_pos_total)
        sh_state_vs_oracle[iteration] = {
            "precision": precision, "recall": recall,
            "pos_utility_coverage": pos_cov,
            "k_retain": 50,
            "min_k_for_95pct_sh": min_k_per_fam.get("sh", 100),
        }

        # Densification
        if iteration < config.densify_until_iter:
            model.max_radii2D[vis_filter] = torch.max(
                model.max_radii2D[vis_filter], radii[vis_filter].float().max(dim=-1).values)
            means2d = meta["means2d"]
            model.add_densification_stats(means2d, vis_filter, width=cam.image_width, height=cam.image_height)

        if (iteration > config.densify_from_iter and iteration < config.densify_until_iter and
            iteration % config.densification_interval == 0):
            size_threshold = config.max_screen_size if iteration > config.opacity_reset_interval else None
            current_radii = radii.float().max(dim=-1).values
            model.densify_and_prune(max_grad=config.densify_grad_threshold, min_opacity=config.min_opacity,
                                     extent=1.0, max_screen_size=size_threshold, radii=current_radii)

        if iteration < config.densify_until_iter and iteration % config.opacity_reset_interval == 0:
            model.reset_opacity()

        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [iter {iteration}] N={model._xyz.shape[0]} vis={vis_count} "
                  f"neg_frac={iter_conc['total']['neg_fraction']:.4f} "
                  f"C50_pos={iter_conc['total']['C50_pos']:.4f}")

    # Save
    with open(os.path.join(output_dir, "corrected_concentration.json"), "w") as f:
        json.dump({str(k): v for k, v in corrected_conc.items()}, f, indent=2)
    with open(os.path.join(output_dir, "oracle_masks.json"), "w") as f:
        json.dump({str(k): v for k, v in oracle_masks.items()}, f, indent=2)
    with open(os.path.join(output_dir, "sh_state_vs_oracle.json"), "w") as f:
        json.dump({str(k): v for k, v in sh_state_vs_oracle.items()}, f, indent=2)

    # Aggregate min K for 95% positive utility
    min_k_summary = {"geometry": [], "sh": [], "opacity": []}
    for it, m in oracle_masks.items():
        for fam in min_k_summary:
            for k_pct in [20, 32, 50, 75, 100]:
                if m[fam][f"k{k_pct}"]["pos_utility_retained"] >= 0.95:
                    min_k_summary[fam].append(k_pct); break
            else: min_k_summary[fam].append(100)

    summary = {fam: {"min_k_for_95pct_median": float(np.median(vals)) if vals else 100,
                     "min_k_for_95pct_mean": float(np.mean(vals)) if vals else 100}
               for fam, vals in min_k_summary.items()}
    with open(os.path.join(output_dir, "oracle_min_k_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved to {output_dir}")
    print(f"Min K for 95% pos utility: {summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--start-iter", type=int, required=True)
    parser.add_argument("--n-iters", type=int, default=50)
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera-sequence", required=True)
    args = parser.parse_args()
    config = ReferenceV1Config(scene="room", iterations=30000)
    run_corrected_analysis(args.checkpoint, args.start_iter, args.n_iters, args.output, config, args.camera_sequence)
