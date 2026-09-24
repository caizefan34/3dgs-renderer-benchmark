"""
R1 Continuation Runner — Optimizer-Space Utility Audit.

At every iteration (no skipping), records:
  1. Pre-backward Adam state (exp_avg, exp_avg_sq, step, lr) for visible Gaussians
  2. Full backward to get gradients
  3. Exact Adam update reconstruction (validated against actual optimizer.step())
  4. U_step (per-Gaussian per-param update norm) and U_loss (first-order loss decrease)
  5. G_opt (raw combined gradient norm) for C49 audit
  6. State-only pre-backward predictor scores S_state
  7. Per-parameter Top-K overlap

Does NOT:
  - Modify training, skip gradients, or alter the optimizer.
"""

import sys, os, json, math, random, argparse
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


class SepSSIM:
    def __init__(self, window_size=11, sigma=1.5, device="cuda"):
        self.C1 = (0.01) ** 2
        self.C2 = (0.03) ** 2
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(15, 1, 1, 1).permute(0, 1, 3, 2).contiguous()
        self.padding = window_size // 2

    def __call__(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.cat([pred, target, pred ** 2, target ** 2, pred * target], dim=1)
        b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=15)
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=15)
        mu_p, mu_t = b[:, 0:3], b[:, 3:6]
        bp2, bt2, bpt = b[:, 6:9], b[:, 9:12], b[:, 12:15]
        mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
        sp2, st2, spt = bp2 - mu_p2, bt2 - mu_t2, bpt - mu_pt
        ssim_map = (2 * mu_pt + self.C1) * (2 * spt + self.C2) / \
                   ((mu_p2 + mu_t2 + self.C1) * (sp2 + st2 + self.C2))
        return 1.0 - ssim_map.mean()


# === Adam hyperparameters (PyTorch defaults, matching baseline) ===
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-15


def extract_gopt(model):
    """Extract per-Gaussian gradient norms for all parameter families."""
    result = {}
    for name, param, ndim in [
        ("xyz", model._xyz, 1),
        ("shs", model._shs, 1),
        ("opacity", model._opacity, None),
        ("scale", model._scaling, 1),
        ("rot", model._rotation, 1),
    ]:
        if param.grad is not None:
            if name == "opacity":
                result[name] = param.grad.abs().detach().cpu().numpy().astype(np.float32)
            else:
                result[name] = param.grad.flatten(start_dim=1).norm(dim=-1).detach().cpu().numpy().astype(np.float32)
        else:
            result[name] = np.zeros(param.shape[0], dtype=np.float32)
    # Total G_opt
    total_sq = (result["xyz"]**2 + result["opacity"]**2 +
                result["scale"]**2 + result["rot"]**2 + result["shs"]**2)
    result["total"] = np.sqrt(total_sq + 1e-20)
    return result


def capture_adam_state(model):
    """Capture pre-backward Adam state (exp_avg, exp_avg_sq, step, lr) for all groups.
    
    Maps optimizer group names to short names:
      xyz -> xyz, shs -> shs, opacity -> opacity,
      scaling -> scale, rotation -> rot
    """
    NAME_MAP = {"xyz": "xyz", "shs": "shs", "opacity": "opacity",
                "scaling": "scale", "rotation": "rot"}
    state = {}
    for group in model.optimizer.param_groups:
        raw_name = group["name"]
        name = NAME_MAP.get(raw_name, raw_name)
        p = group["params"][0]
        lr = group["lr"]
        stored = model.optimizer.state.get(p, None)
        if stored is not None and "exp_avg" in stored:
            exp_avg = stored["exp_avg"].detach().clone()
            exp_avg_sq = stored["exp_avg_sq"].detach().clone()
            step = stored.get("step", 0)
            if isinstance(step, torch.Tensor):
                step = step.item()
            # pre-update step count
        else:
            # No state for this parameter yet (e.g., just created, step 0)
            exp_avg = torch.zeros_like(p.data)
            exp_avg_sq = torch.zeros_like(p.data)
            step = 0
        state[name] = {
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
            "step": step,
            "lr": lr,
        }
    return state


def reconstruct_adam_update(state, grad, name):
    """
    Reconstruct the exact Adam update delta = -lr * m_hat / (sqrt(v_hat) + eps).
    
    PyTorch Adam semantics (from step()):
      m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
      v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
      bias_correction1 = 1 - beta1^step
      bias_correction2 = 1 - beta2^step
      step_size = lr / bias_correction1
      delta = -step_size * (m_t / bias_correction1) / (sqrt(v_t / bias_correction2) + eps)
    
    Which simplifies to:
      m_hat = (beta1 * m_{t-1} + (1-beta1)*g) / (1 - beta1^t)
      v_hat = (beta2 * v_{t-1} + (1-beta2)*g^2) / (1 - beta2^t)
      delta = -lr * m_hat / (sqrt(v_hat) + eps)
    
    Where t = step + 1 (this is the step we're about to take).
    
    Handles 1D parameters (opacity) vs multi-dimensional ones.
    """
    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]
    step = state["step"]
    lr = state["lr"]
    
    t = step + 1  # current step number
    
    if name == "opacity":
        g = grad.squeeze(-1) if grad.dim() > 1 and grad.shape[-1] == 1 else grad
    else:
        g = grad
    
    m_t = BETA1 * exp_avg + (1 - BETA1) * g
    v_t = BETA2 * exp_avg_sq + (1 - BETA2) * g * g
    
    bc1 = 1.0 - BETA1 ** t
    bc2 = 1.0 - BETA2 ** t
    
    m_hat = m_t / bc1
    v_hat = v_t / bc2
    
    delta = -lr * m_hat / (torch.sqrt(v_hat) + EPS)
    
    return delta


def compute_update_utility(delta, grad):
    """
    Compute:
      U_step = ||delta||_2            (per-Gaussian L2 norm)
      U_loss = -sum(g_j * delta_j)    (first-order loss decrease, per-Gaussian)
    
    For multi-dimensional parameters, delta and grad are flattened per-Gaussian.
    """
    # Flatten per-Gaussian
    if delta.dim() > 1:
        delta_flat = delta.flatten(start_dim=1)
        grad_flat = grad.flatten(start_dim=1)
    else:
        delta_flat = delta.unsqueeze(-1)
        grad_flat = grad.unsqueeze(-1)
    
    u_step = delta_flat.norm(dim=-1)  # [N]
    u_loss = -(grad_flat * delta_flat).sum(dim=-1)  # [N]
    
    return u_step, u_loss


def compute_state_score(state, name):
    """
    Pre-backward state-only score: S_state = lr * ||m_hat / (sqrt(v_hat) + eps)||_2
    
    Uses previous-step moments (exp_avg = m_{t-1}, exp_avg_sq = v_{t-1}) with their
    EXISTING bias correction factors (based on step count BEFORE current step).
    
    This score is available before current backward — it uses only optimizer state.
    """
    exp_avg = state["exp_avg"]
    exp_avg_sq = state["exp_avg_sq"]
    step = state["step"]
    lr = state["lr"]
    
    if step == 0:
        # No steps taken yet — score is zero (no optimizer signal)
        if name == "opacity":
            return torch.zeros(exp_avg.shape[0], device=exp_avg.device)
        else:
            return torch.zeros(exp_avg.shape[0], device=exp_avg.device)
    
    t = step  # bias correction uses CURRENT step number (the last completed step)
    
    bc1 = 1.0 - BETA1 ** t
    bc2 = 1.0 - BETA2 ** t
    
    m_hat = exp_avg / bc1
    v_hat = exp_avg_sq / bc2
    
    score = lr * m_hat / (torch.sqrt(v_hat) + EPS)
    
    # Per-Gaussian norm
    if name == "opacity":
        return score.abs().squeeze(-1)
    else:
        return score.flatten(start_dim=1).norm(dim=-1)


def parametrized_jaccard(set1, set2):
    """Jaccard similarity between two index sets (as torch tensors of indices)."""
    s1 = set(set1.tolist())
    s2 = set(set2.tolist())
    if len(s1 | s2) == 0:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def compute_topk_metrics(scores, targets, k_pcts, n_visible):
    """
    For a ranking signal (scores) and target utility, compute:
      - Top-K coverage: mass of targets in top K(scores) / total target mass
      - Oracle Top-K coverage (by target itself)
      - Random Top-K coverage

    scores and targets are already masked to visible Gaussians.
    n_visible = len(scores) = len(targets) = visible count.
    """
    result = {}
    n = len(scores)
    if n == 0 or n_visible == 0:
        return {f"k{k}": {"coverage": 0.0, "oracle": 0.0, "random": 0.0} for k in k_pcts}
    total_target = max(float(targets.sum().item()), 1e-20)

    for k_pct in k_pcts:
        k = max(1, int(n_visible * k_pct / 100))
        if k > n:
            k = n
        # Top-K by scores
        top_idx = torch.argsort(scores, descending=True)[:k]
        cov_k = float(targets[top_idx].sum().item() / total_target)
        
        # Oracle
        oracle_idx = torch.argsort(targets, descending=True)[:k]
        cov_oracle = float(targets[oracle_idx].sum().item() / total_target)
        
        # Random
        rng = np.random.RandomState(42)
        rand_covs = []
        for _ in range(5):
            rand_idx = torch.from_numpy(rng.choice(n, k, replace=False)).to(targets.device)
            rand_covs.append(float(targets[rand_idx].sum().item() / total_target))
        cov_random = float(np.mean(rand_covs))
        
        result[f"k{k_pct}"] = {
            "coverage": cov_k,
            "oracle": cov_oracle,
            "random": cov_random,
        }
    
    return result


def compute_overlap_matrix(u_step_dict, vis_np, k_pcts=(20, 32, 50)):
    """
    Compute pairwise Jaccard/Recall between parameter groups' top-K indices.
    """
    param_names = ["xyz", "scale", "rot", "opacity", "shs"]
    result = {}
    
    for k_pct in k_pcts:
        topk_indices = {}
        for name in param_names:
            if name in u_step_dict:
                vals = torch.from_numpy(u_step_dict[name]).to("cpu")
                if isinstance(vis_np, np.ndarray):
                    mask = torch.from_numpy(vis_np)
                else:
                    mask = vis_np
                masked_vals = vals.clone()
                masked_vals[~mask] = -1.0  # invisible get lowest rank
                k = max(1, int(mask.sum().item() * k_pct / 100))
                idx = torch.argsort(masked_vals, descending=True)[:k]
                topk_indices[name] = idx
        
        pairwise = {}
        for i, n1 in enumerate(param_names):
            for j, n2 in enumerate(param_names):
                if n1 in topk_indices and n2 in topk_indices:
                    jac = parametrized_jaccard(topk_indices[n1], topk_indices[n2])
                    recall_1in2 = len(set(topk_indices[n1].tolist()) & set(topk_indices[n2].tolist())) / max(len(topk_indices[n1]), 1)
                else:
                    jac = 0.0
                    recall_1in2 = 0.0
                pairwise[f"{n1}_vs_{n2}"] = {"jaccard": jac, "recall_1in2": recall_1in2}
        
        result[f"k{k_pct}"] = pairwise
    
    return result


def run_continuation(checkpoint_path, start_iter, n_iters, output_dir,
                     config, camera_sequence_path, gpu_id=0):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda"

    # Dataset
    print(f"Loading dataset: {config.scene}")
    dataset = GTDataset(config.scene, config.repo_root)
    n_cameras = len(dataset)
    centers = []
    for i in range(min(50, n_cameras)):
        cam = dataset.get_camera(i)
        centers.append(cam.camera_center.cpu().numpy())
    centers = np.array(centers)
    scene_extent = 0.0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = np.linalg.norm(centers[i] - centers[j])
            scene_extent = max(scene_extent, d)
    scene_extent = max(scene_extent, 0.1)
    print(f"  {n_cameras} cameras, extent={scene_extent:.4f}")

    camera_sequence = np.load(camera_sequence_path)
    print(f"  Camera sequence: {len(camera_sequence)} entries")

    # Checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"  Checkpoint N: {ckpt['num_points']}, SH degree: {ckpt['active_sh_degree']}")

    model = GaussianModel(max_sh_degree=config.sh_degree)
    training_config = {
        "position_lr_init": config.position_lr_init,
        "position_lr_final": config.position_lr_final,
        "position_lr_delay_mult": config.position_lr_delay_mult,
        "position_lr_max_steps": config.position_lr_max_steps,
        "feature_lr": config.feature_lr,
        "opacity_lr": config.opacity_lr,
        "scaling_lr": config.scaling_lr,
        "rotation_lr": config.rotation_lr,
        "percent_dense": config.percent_dense,
    }
    model.restore(ckpt, training_config)
    N = model._xyz.shape[0]
    print(f"  Model restored: N={N}, SH degree={model.active_sh_degree}")

    ssim_fn = SepSSIM(device=device)

    # --- Storage ---
    gopt_vs_utility = {}      # iter -> {Pearson, Spearman, coverage, etc.}
    utility_concentration = {}  # iter -> {Gini, TopK masses}
    state_prediction = {}     # iter -> {per-group state score coverage}
    param_overlap = {}        # iter -> {K: pairwise Jaccard}
    geo_app_overlap = {}      # iter -> geometry vs appearance

    # Validation accumulator
    validation_rows = {}      # iter -> {name: {delta_pred, delta_actual, max_err}}

    print(f"\nStarting R1 continuation: {n_iters} iterations")

    for i in range(n_iters):
        iteration = start_iter + 1 + i
        cam_idx = int(camera_sequence[iteration - 1])
        cam, gt_image = dataset.get_item(cam_idx)
        model.update_learning_rate(iteration)

        # --- Forward pass ---
        # gsplat returns (render_colors, render_alphas, meta)
        image, _, meta = rasterization(
            means=model.get_xyz, quats=model.get_rotation,
            scales=model.get_scaling, opacities=model.get_opacity,
            colors=model.get_features,
            viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            tile_size=16, packed=False, sh_degree=model.active_sh_degree,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
        )
        if image.ndim == 4:
            image = image.squeeze(0)
        means2d_full = meta["means2d"]
        means2d_full.retain_grad()
        radii = meta["radii"]
        if radii.ndim > 2:
            radii = radii[0]
        if radii.ndim > 2:
            radii = radii.squeeze(-1)
        visibility_filter = (radii > 0).any(dim=-1)
        vis_np = visibility_filter.cpu().numpy().astype(bool)
        vis_count = int(vis_np.sum())

        # --- CAPTURE PRE-BACKWARD ADAM STATE (Part 2) ---
        pre_state = capture_adam_state(model)

        # --- Loss + backward ---
        L1 = F.l1_loss(image, gt_image)
        dssim = ssim_fn(image, gt_image)
        loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim
        loss.backward()

        # --- Extract gradients ---
        gopt_np = extract_gopt(model)

        # --- Collect gradients for reconstruction ---
        grads = {}
        for name, param in [
            ("xyz", model._xyz), ("shs", model._shs),
            ("opacity", model._opacity), ("scale", model._scaling),
            ("rot", model._rotation),
        ]:
            if param.grad is not None:
                grads[name] = param.grad.detach().clone()
            else:
                grads[name] = torch.zeros_like(param.data)

        # --- PART 3: Reconstruct Adam update ---
        deltas = {}  # predicted deltas
        u_step = {}  # predicted U_step per-Gaussian
        u_loss = {}  # predicted U_loss per-Gaussian
        s_state = {} # pre-backward state score
        total_u_loss = None
        total_s_state = None
        current_N = model._xyz.shape[0]

        # Validation: sample some rows before optimizer.step()
        validation_samples = {}

        for name in ["xyz", "shs", "scale", "rot", "opacity"]:
            st = pre_state[name]
            delta = reconstruct_adam_update(st, grads[name], name)
            deltas[name] = delta

            step_u, loss_u = compute_update_utility(delta, grads[name])
            u_step[name] = step_u.detach().cpu().numpy().astype(np.float32)
            u_loss[name] = loss_u.detach().cpu().numpy().astype(np.float32)

            # State score
            s_state_score = compute_state_score(st, name)
            s_state[name] = s_state_score.detach().cpu().numpy().astype(np.float32)

            # Validation: save prediction for a few rows
            n_val = min(10, current_N)
            if name not in validation_samples:
                validation_samples[name] = {
                    "delta_pred": delta[:n_val].detach().clone().cpu(),
                }

        # --- Compute total U_loss ---
        total_u_loss_np = np.zeros(current_N, dtype=np.float32)
        total_s_state_np = np.zeros(current_N, dtype=np.float32)
        for name in ["xyz", "shs", "scale", "rot", "opacity"]:
            total_u_loss_np += u_loss[name]
            total_s_state_np += s_state[name]

        # --- PART 5: Validate reconstruction against actual optimizer.step() ---
        # Save parameter snapshots
        params_before = {}
        for name, param in [
            ("xyz", model._xyz), ("shs", model._shs),
            ("opacity", model._opacity), ("scale", model._scaling),
            ("rot", model._rotation),
        ]:
            params_before[name] = param.data.detach().clone().cpu()

        # Run optimizer.step()
        model.optimizer.step()

        # Check actual delta
        max_errs = {}
        for name, param in [
            ("xyz", model._xyz), ("shs", model._shs),
            ("opacity", model._opacity), ("scale", model._scaling),
            ("rot", model._rotation),
        ]:
            actual_delta = (param.data.detach().clone().cpu() - params_before[name])
            pred_delta = validation_samples[name]["delta_pred"].cpu()
            n_val = min(len(pred_delta), len(actual_delta))
            err = (pred_delta[:n_val] - actual_delta[:n_val]).abs()
            max_err = float(err.max().item())
            max_errs[name] = max_err

        validation_rows[iteration] = max_errs

        # --- G_opt vs U_loss correlation (Part 5) ---
        gopt = gopt_np["total"]
        gopt_vis = gopt[vis_np]
        uloss_vis = total_u_loss_np[vis_np]

        iter_gopt_utility = {}
        if vis_count >= 10 and np.std(gopt_vis) > 1e-10 and np.std(uloss_vis) > 1e-10:
            pearson = float(np.corrcoef(gopt_vis, uloss_vis)[0, 1])
            rank_gopt = np.argsort(np.argsort(gopt_vis)).astype(float)
            rank_uloss = np.argsort(np.argsort(uloss_vis)).astype(float)
            spearman = float(np.corrcoef(rank_gopt, rank_uloss)[0, 1])
        else:
            pearson = 0.0
            spearman = 0.0
        iter_gopt_utility["pearson"] = pearson
        iter_gopt_utility["spearman"] = spearman

        # TopK overlap between G_opt and U_loss
        for k_pct in [10, 20, 32, 50]:
            k = max(1, int(vis_count * k_pct / 100))
            top_gopt = np.argsort(gopt_vis)[::-1][:k]
            top_uloss = np.argsort(uloss_vis)[::-1][:k]
            jac = len(set(top_gopt) & set(top_uloss)) / max(len(set(top_gopt) | set(top_uloss)), 1)
            recall = len(set(top_gopt) & set(top_uloss)) / max(k, 1)
            iter_gopt_utility[f"top{k_pct}_jaccard"] = jac
            iter_gopt_utility[f"top{k_pct}_recall"] = recall

            # TopK(G_opt) -> U_loss mass coverage
            total_uloss_vis = max(uloss_vis.sum(), 1e-20)
            cov = gopt_vis[top_gopt].sum() / max(gopt_vis.sum(), 1e-20)
            uloss_cov = uloss_vis[top_gopt].sum() / total_uloss_vis
            iter_gopt_utility[f"top{k_pct}_gopt_mass_cov"] = float(cov)
            iter_gopt_utility[f"top{k_pct}_uloss_mass_cov"] = float(uloss_cov)

        gopt_vs_utility[iteration] = iter_gopt_utility

        # --- Update utility concentration (Part 6) ---
        conc = {}
        for name in ["xyz", "scale", "rot", "opacity", "shs"]:
            vals = u_step[name][vis_np]
            sorted_vals = np.sort(vals)[::-1]
            total = max(sorted_vals.sum(), 1e-20)
            nv = len(sorted_vals)
            # Gini
            cumsum = np.cumsum(sorted_vals)
            gini = (2 * np.sum((np.arange(1, nv+1) * sorted_vals)) / (nv * total) - (nv + 1) / nv) if nv > 0 and total > 0 else 0.0
            conc[name] = {
                "gini": float(gini),
                "top1_mass": float(sorted_vals[0] / total) if nv > 0 else 0.0,
                "top10_mass": float(sorted_vals[:max(1, nv//10)].sum() / total) if nv >= 10 else float(sorted_vals.sum() / total),
            }
            for k_pct in [10, 20, 32, 50]:
                k = max(1, int(nv * k_pct / 100))
                conc[name][f"top{k_pct}_mass"] = float(sorted_vals[:k].sum() / total)

        # Also for total U_loss
        uloss_vis_sorted = np.sort(uloss_vis)[::-1]
        total_ul = max(uloss_vis.sum(), 1e-20)
        nv = len(uloss_vis)
        cumsum = np.cumsum(uloss_vis_sorted)
        gini_u = (2 * np.sum((np.arange(1, nv+1) * uloss_vis_sorted)) / (nv * total_ul) - (nv + 1) / nv) if nv > 0 and total_ul > 0 else 0.0
        conc["total_uloss"] = {
            "gini": float(gini_u),
            "top1_mass": float(uloss_vis_sorted[0] / total_ul) if nv > 0 else 0.0,
            "top10_mass": float(uloss_vis_sorted[:max(1, nv//10)].sum() / total_ul) if nv >= 10 else float(uloss_vis_sorted.sum() / total_ul),
        }
        for k_pct in [10, 20, 32, 50]:
            k = max(1, int(nv * k_pct / 100))
            conc["total_uloss"][f"top{k_pct}_mass"] = float(uloss_vis_sorted[:k].sum() / total_ul)

        utility_concentration[iteration] = conc

        # --- State predictor evaluation (Part 7) ---
        iter_state_pred = {}
        for name in ["xyz", "shs", "scale", "rot", "opacity"]:
            s_vis = s_state[name][vis_np]
            u_vis = u_step[name][vis_np]
            
            # Correlation
            if vis_count >= 10 and np.std(s_vis) > 1e-10 and np.std(u_vis) > 1e-10:
                sp = float(np.corrcoef(s_vis, u_vis)[0, 1])
                rs = np.argsort(np.argsort(s_vis)).astype(float)
                ru = np.argsort(np.argsort(u_vis)).astype(float)
                sr = float(np.corrcoef(rs, ru)[0, 1]) if np.std(rs) > 0 and np.std(ru) > 0 else 0.0
            else:
                sp = 0.0; sr = 0.0

            # Coverage
            cov = compute_topk_metrics(
                torch.from_numpy(s_vis), torch.from_numpy(u_vis),
                [20, 32, 50], vis_count
            )

            # Also compare vs opacity ranking and previous-gradient ranking
            iter_state_pred[name] = {
                "pearson": sp,
                "spearman": sr,
                "coverage": cov,
            }

        # Aggregate state predictor (Part 8)
        s_total_vis = total_s_state_np[vis_np]
        if vis_count >= 10 and np.std(s_total_vis) > 1e-10 and np.std(uloss_vis) > 1e-10:
            sp_agg = float(np.corrcoef(s_total_vis, uloss_vis)[0, 1])
        else:
            sp_agg = 0.0
        agg_cov = compute_topk_metrics(
            torch.from_numpy(s_total_vis), torch.from_numpy(uloss_vis),
            [20, 32, 50], vis_count
        )
        iter_state_pred["aggregate"] = {
            "pearson": sp_agg,
            "coverage": agg_cov,
        }

        state_prediction[iteration] = iter_state_pred

        # --- Parameter group overlap (Part 9) ---
        overlap = compute_overlap_matrix(u_step, vis_np, [20, 32, 50])
        param_overlap[iteration] = overlap

        # --- Geometry vs appearance overlap (Part 10) ---
        geo_names = ["xyz", "scale", "rot"]
        app_names = ["shs"]
        geo_u = np.zeros(current_N, dtype=np.float32)
        app_u = np.zeros(current_N, dtype=np.float32)
        opa_u = u_step["opacity"].copy()
        for gn in geo_names:
            geo_u += u_step[gn]
        for an in app_names:
            app_u += u_step[an]
        
        geo_app_result = {}
        for k_pct in [20, 32, 50]:
            k = max(1, int(vis_count * k_pct / 100))
            # Among visible
            geo_vis = geo_u[vis_np]
            app_vis = app_u[vis_np]
            opa_vis = opa_u[vis_np]

            top_geo = set(np.argsort(geo_vis)[::-1][:k].tolist())
            top_app = set(np.argsort(app_vis)[::-1][:k].tolist())
            top_opa = set(np.argsort(opa_vis)[::-1][:k].tolist())

            geo_app_jac = len(top_geo & top_app) / max(len(top_geo | top_app), 1)
            geo_opa_jac = len(top_geo & top_opa) / max(len(top_geo | top_opa), 1)

            geo_app_result[f"k{k_pct}"] = {
                "geometry_appearance_jaccard": geo_app_jac,
                "geometry_opacity_jaccard": geo_opa_jac,
                "geo_app_recall_geo_in_app": len(top_geo & top_app) / max(len(top_geo), 1),
                "geo_app_recall_app_in_geo": len(top_geo & top_app) / max(len(top_app), 1),
            }

        geo_app_overlap[iteration] = geo_app_result

        # --- Part 4: Report negative U_loss fraction ---
        neg_frac = float((total_u_loss_np[vis_np] < 0).sum() / max(vis_count, 1))
        gopt_vs_utility[iteration]["neg_uloss_fraction"] = neg_frac

        # Densification + step
        model.optimizer.zero_grad(set_to_none=True)

        # Densification (matching baseline protocol)
        if iteration < config.densify_until_iter:
            model.max_radii2D[visibility_filter] = torch.max(
                model.max_radii2D[visibility_filter],
                radii[visibility_filter].float().max(dim=-1).values
            )
            model.add_densification_stats(means2d_full, visibility_filter,
                                           width=cam.image_width, height=cam.image_height)

        if (iteration > config.densify_from_iter and
            iteration < config.densify_until_iter and
            iteration % config.densification_interval == 0):
            size_threshold = config.max_screen_size if iteration > config.opacity_reset_interval else None
            current_radii = radii.float().max(dim=-1).values
            model.densify_and_prune(
                max_grad=config.densify_grad_threshold,
                min_opacity=config.min_opacity,
                extent=scene_extent,
                max_screen_size=size_threshold,
                radii=current_radii,
            )

        if iteration < config.densify_until_iter and iteration % config.opacity_reset_interval == 0:
            model.reset_opacity()

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [iter {iteration}] N={model._xyz.shape[0]} vis={vis_count} "
                  f"neg_uloss={neg_frac:.4f} val_err={max(max_errs.values()):.2e} "
                  f"G_vs_U_pear={pearson:.4f} G_vs_U_spear={spearman:.4f}")

    # --- Save ---
    print("\n=== Saving outputs ===")

    # Save per-iteration data
    with open(os.path.join(output_dir, "gopt_vs_update_utility.json"), "w") as f:
        json.dump({str(k): v for k, v in gopt_vs_utility.items()}, f, indent=2)
    with open(os.path.join(output_dir, "update_utility_concentration.json"), "w") as f:
        json.dump({str(k): v for k, v in utility_concentration.items()}, f, indent=2)
    with open(os.path.join(output_dir, "optimizer_state_prediction.json"), "w") as f:
        json.dump({str(k): v for k, v in state_prediction.items()}, f, indent=2)
    with open(os.path.join(output_dir, "parameter_group_overlap.json"), "w") as f:
        json.dump({str(k): v for k, v in param_overlap.items()}, f, indent=2)
    with open(os.path.join(output_dir, "geometry_appearance_overlap.json"), "w") as f:
        json.dump({str(k): v for k, v in geo_app_overlap.items()}, f, indent=2)
    with open(os.path.join(output_dir, "validation_errors.json"), "w") as f:
        json.dump({str(k): v for k, v in validation_rows.items()}, f, indent=2)

    # Summary provenance
    provenance = {
        "experiment": "r1_optimizer_utility",
        "semantic_label": "REFERENCE_V1_ABSGRAD",
        "checkpoint_path": checkpoint_path,
        "start_iter": start_iter,
        "n_iters": n_iters,
        "gpu_id": gpu_id,
        "scene": config.scene,
        "N_at_checkpoint": ckpt["num_points"],
        "N_at_end": model._xyz.shape[0],
        "adam_beta1": BETA1,
        "adam_beta2": BETA2,
        "adam_eps": EPS,
    }
    with open(os.path.join(output_dir, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)

    print(f"  Outputs saved to: {output_dir}")
    print(f"  Final N: {model._xyz.shape[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R1 Continuation Runner")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--start-iter", type=int, required=True)
    parser.add_argument("--n-iters", type=int, default=200)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--camera-sequence", required=True)
    args = parser.parse_args()

    config = ReferenceV1Config(scene="room", iterations=30000)
    run_continuation(
        checkpoint_path=args.checkpoint,
        start_iter=args.start_iter,
        n_iters=args.n_iters,
        output_dir=args.output,
        config=config,
        camera_sequence_path=args.camera_sequence,
        gpu_id=args.gpu,
    )
