"""
R2.1 Benchmark: Shared-Raster Attribute Gating Oracle Test.

Tests the modified rasterize_to_pixels_3dgs_bwd kernel with three independent
per-branch masks (geo_mask, app_mask, opacity_mask).

Measures:
  1. Gradient correctness: FULL vs masked (all-1s) vs masked (all-0s)
  2. FULL backward timing
  3. Synthetic mask scaling (geo/app/opacity suppression series)
  4. ORACLE_DECOUPLED_95 (geo=50%, app=20%, opacity=32% oracle masks)
  5. ALL_BRANCH_K50 (single mask at 50% for all branches)

Uses real training code (GaussianModel, GTDataset) with checkpoint iter_10000.
20 warmup, 100 measured. CUDA events for timing.
"""

import sys, os, json, math, argparse, time
import numpy as np

# Set RTLD_GLOBAL before importing torch/gsplat so the .so can resolve PyTorch symbols
old_dlopen_flags = sys.getdlopenflags()
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)

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


def cuda_time(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), result


def forward_pass(model, cam, gt_image, config, ssim_fn, masks=None):
    """Forward pass with optional r2 masks. Returns (image, meta, loss, vis_filter, radii)."""
    kwargs = dict(
        means=model.get_xyz, quats=model.get_rotation, scales=model.get_scaling,
        opacities=model.get_opacity, colors=model.get_features,
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=model.active_sh_degree,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
    )
    if masks is not None:
        kwargs.update(masks)
    r, _, meta = rasterization(**kwargs)
    image = r.squeeze(0) if r.ndim == 4 else r
    radii = meta["radii"]
    if radii.ndim > 2: radii = radii[0]
    if radii.ndim > 2: radii = radii.squeeze(-1)
    vis_filter = (radii > 0).any(dim=-1)
    L1 = F.l1_loss(image, gt_image)
    dssim = ssim_fn(image, gt_image)
    loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim
    return image, meta, loss, vis_filter, radii


def stats(arr):
    a = np.array(arr, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)), "n": len(a)}


def run_benchmark(checkpoint_path, output_dir, config, camera_sequence_path):
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

    benchmark_iters = [10001, 10002, 10003, 10004, 10005]
    n_warmup = 20
    n_measure = 100

    # First, compute FULL gradients for correctness comparison
    print("\n=== Computing FULL reference gradients ===")
    ref_grads = {}
    for bench_iter in benchmark_iters[:1]:  # just 1 camera for correctness
        cam_idx = int(camera_sequence[bench_iter - 1])
        cam, gt_image = dataset.get_item(cam_idx)
        model.update_learning_rate(bench_iter)
        model.optimizer.zero_grad(set_to_none=True)
        _, _, loss, vis_filter, radii = forward_pass(model, cam, gt_image, config, ssim_fn)
        loss.backward()
        ref_grads["v_colors"] = model._shs.grad.detach().clone()
        ref_grads["v_opacities"] = model._opacity.grad.detach().clone()
        ref_grads["v_means2d"] = getattr(model.get_xyz, 'absgrad', None)
        if ref_grads["v_means2d"] is not None:
            ref_grads["v_means2d"] = ref_grads["v_means2d"].detach().clone()
        ref_grads["vis_filter"] = vis_filter
        ref_grads["radii"] = radii
        model.optimizer.zero_grad(set_to_none=True)
    print("  Reference gradients captured")

    # === Gradient correctness test ===
    print("\n=== Gradient correctness: masked (all 1s) vs FULL ===")
    cam_idx = int(camera_sequence[benchmark_iters[0] - 1])
    cam, gt_image = dataset.get_item(cam_idx)
    model.update_learning_rate(benchmark_iters[0])

    all_ones_masks = {
        "r2_geo_mask": torch.ones(N, dtype=torch.uint8, device=device),
        "r2_app_mask": torch.ones(N, dtype=torch.uint8, device=device),
        "r2_opacity_mask": torch.ones(N, dtype=torch.uint8, device=device),
    }
    model.optimizer.zero_grad(set_to_none=True)
    _, _, loss, _, _ = forward_pass(model, cam, gt_image, config, ssim_fn, masks=all_ones_masks)
    loss.backward()

    # Compare
    correctness = {}
    for name, ref in ref_grads.items():
        if name in ["vis_filter", "radii"]: continue
        curr = None
        if name == "v_colors": curr = model._shs.grad.detach()
        elif name == "v_opacities": curr = model._opacity.grad.detach()
        elif name == "v_means2d": curr = getattr(model.get_xyz, 'absgrad', None)
        if curr is not None and ref is not None:
            diff = (curr - ref).abs()
            max_abs = float(diff.max())
            rel_l2 = float(diff.norm() / max(ref.norm(), 1e-20))
            cosine = float(F.cosine_similarity(curr.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)))
            correctness[name] = {"max_abs_error": max_abs, "relative_l2": rel_l2, "cosine_similarity": cosine}
            print(f"  {name}: max_abs={max_abs:.2e}, rel_l2={rel_l2:.2e}, cosine={cosine:.8f}")
    model.optimizer.zero_grad(set_to_none=True)

    # === Gradient correctness: all-0s masks should produce zero gradients ===
    print("\n=== Gradient correctness: masked (all 0s) should be zero ===")
    all_zeros_masks = {
        "r2_geo_mask": torch.zeros(N, dtype=torch.uint8, device=device),
        "r2_app_mask": torch.zeros(N, dtype=torch.uint8, device=device),
        "r2_opacity_mask": torch.zeros(N, dtype=torch.uint8, device=device),
    }
    model.optimizer.zero_grad(set_to_none=True)
    _, _, loss, _, _ = forward_pass(model, cam, gt_image, config, ssim_fn, masks=all_zeros_masks)
    loss.backward()
    zero_check = {}
    for name, param in [("v_colors", model._shs), ("v_opacities", model._opacity)]:
        if param.grad is not None:
            max_val = float(param.grad.abs().max())
            zero_check[name] = {"max_abs": max_val, "is_zero": max_val < 1e-10}
            print(f"  {name}: max_abs={max_val:.2e}, is_zero={max_val < 1e-10}")
    model.optimizer.zero_grad(set_to_none=True)

    # === Compute per-family oracle utility for mask construction ===
    print("\n=== Computing oracle utility for mask construction ===")
    pre_state = capture_adam_state(model)
    model.update_learning_rate(benchmark_iters[0])
    model.optimizer.zero_grad(set_to_none=True)
    _, _, loss, vis_filter, _ = forward_pass(model, cam, gt_image, config, ssim_fn)
    loss.backward()

    vis_np = vis_filter.cpu().numpy().astype(bool)
    vis_count = int(vis_np.sum())

    u_loss = {}
    for name, param in [("xyz", model._xyz), ("shs", model._shs),
                        ("opacity", model._opacity), ("scale", model._scaling),
                        ("rot", model._rotation)]:
        grad = param.grad.detach().clone() if param.grad is not None else torch.zeros_like(param.data)
        delta = reconstruct_adam_update(pre_state[name], grad, name)
        _, loss_u = compute_update_utility(delta, grad)
        u_loss[name] = loss_u.cpu().numpy().astype(np.float32)

    u_loss["geometry"] = u_loss["xyz"] + u_loss["scale"] + u_loss["rot"]
    u_loss["sh"] = u_loss["shs"]
    u_loss["opacity_g"] = u_loss["opacity"]
    model.optimizer.zero_grad(set_to_none=True)

    # Build oracle masks
    def make_oracle_mask(utility, vis_np, keep_frac):
        pos_u = np.maximum(utility[vis_np], 0)
        vis_indices = np.where(vis_np)[0]
        k = max(1, int(vis_count * keep_frac))
        top_k_local = np.argsort(pos_u)[::-1][:k]
        mask = np.zeros(len(utility), dtype=np.uint8)
        mask[vis_indices[top_k_local]] = 1
        return torch.from_numpy(mask).to(device)

    # Oracle masks for 95% positive utility
    oracle_masks_95 = {
        "r2_geo_mask": make_oracle_mask(u_loss["geometry"], vis_np, 0.50),
        "r2_app_mask": make_oracle_mask(u_loss["sh"], vis_np, 0.20),
        "r2_opacity_mask": make_oracle_mask(u_loss["opacity_g"], vis_np, 0.32),
    }
    print(f"  Oracle masks: geo={int(oracle_masks_95['r2_geo_mask'].sum())}, "
          f"app={int(oracle_masks_95['r2_app_mask'].sum())}, "
          f"opacity={int(oracle_masks_95['r2_opacity_mask'].sum())}")

    # Compute actual utility retained by oracle masks
    oracle_utility = {}
    for fam, mask_key, ukey in [("geometry", "r2_geo_mask", "geometry"),
                                 ("appearance", "r2_app_mask", "sh"),
                                 ("opacity", "r2_opacity_mask", "opacity_g")]:
        mask_np = oracle_masks_95[mask_key].cpu().numpy().astype(bool)
        vals = u_loss[ukey][vis_np]
        pos_vals = np.maximum(vals, 0)
        abs_vals = np.abs(vals)
        pos_total = max(float(pos_vals.sum()), 1e-20)
        abs_total = max(float(abs_vals.sum()), 1e-20)
        mask_vis = mask_np[vis_np]
        oracle_utility[fam] = {
            "pos_utility_retained": float(pos_vals[mask_vis].sum() / pos_total),
            "abs_utility_retained": float(abs_vals[mask_vis].sum() / abs_total),
            "mask_count": int(mask_vis.sum()),
            "vis_count": vis_count,
        }
        print(f"  {fam}: pos_retained={oracle_utility[fam]['pos_utility_retained']:.4f}, "
              f"abs_retained={oracle_utility[fam]['abs_utility_retained']:.4f}")

    # === Timing function ===
    def time_config(masks, label, n_warmup=20, n_measure=100):
        """Time backward with given mask configuration."""
        raster_times, total_bwd_times, e2e_times = [], [], []
        for bench_iter in benchmark_iters:
            cam_idx = int(camera_sequence[bench_iter - 1])
            cam, gt_image = dataset.get_item(cam_idx)
            model.update_learning_rate(bench_iter)

            # Warmup
            for _ in range(n_warmup):
                model.optimizer.zero_grad(set_to_none=True)
                _, _, loss, _, _ = forward_pass(model, cam, gt_image, config, ssim_fn, masks=masks)
                loss.backward()
                model.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

            # Measure
            for _ in range(n_measure):
                model.optimizer.zero_grad(set_to_none=True)
                t_fwd, (_, _, loss, _, _) = cuda_time(lambda: forward_pass(model, cam, gt_image, config, ssim_fn, masks=masks))
                t_bwd, _ = cuda_time(lambda: loss.backward())
                total_bwd_times.append(t_bwd)
                e2e_times.append(t_fwd + t_bwd)
                model.optimizer.zero_grad(set_to_none=True)

        result = {
            "label": label,
            "raster_bwd_ms": stats(total_bwd_times),  # Note: this is total backward, not just raster kernel
            "total_backward_ms": stats(total_bwd_times),
            "e2e_ms": stats(e2e_times),
        }
        return result

    # === 1. FULL baseline ===
    print("\n=== Timing FULL (no masks) ===")
    full_result = time_config(None, "FULL")
    print(f"  backward: {full_result['total_backward_ms']['median']:.2f}ms, "
          f"e2e: {full_result['e2e_ms']['median']:.2f}ms")

    # === 2. FULL with all-1s masks (should match FULL) ===
    print("\n=== Timing FULL_MASKED_ALL_ONES (masks all 1s) ===")
    all_ones_result = time_config(all_ones_masks, "FULL_MASKED_ALL_ONES")
    print(f"  backward: {all_ones_result['total_backward_ms']['median']:.2f}ms, "
          f"e2e: {all_ones_result['e2e_ms']['median']:.2f}ms")

    # === 3. Synthetic mask scaling series ===
    print("\n=== Synthetic mask scaling series ===")
    synthetic_results = {}
    vis_indices = np.where(vis_np)[0]

    def make_random_mask(keep_frac):
        n_keep = max(1, int(vis_count * keep_frac))
        chosen = np.random.choice(vis_indices, size=n_keep, replace=False)
        mask = np.zeros(N, dtype=np.uint8)
        mask[chosen] = 1
        return torch.from_numpy(mask).to(device)

    np.random.seed(42)  # deterministic

    # Geometry suppression series
    for keep_frac in [0.75, 0.50, 0.25]:
        label = f"GEO_K{int(keep_frac*100)}"
        masks = {
            "r2_geo_mask": make_random_mask(keep_frac),
            "r2_app_mask": torch.ones(N, dtype=torch.uint8, device=device),
            "r2_opacity_mask": torch.ones(N, dtype=torch.uint8, device=device),
        }
        print(f"  Timing {label}...")
        synthetic_results[label] = time_config(masks, label)

    # Appearance suppression series
    for keep_frac in [0.75, 0.50, 0.20]:
        label = f"APP_K{int(keep_frac*100)}"
        masks = {
            "r2_geo_mask": torch.ones(N, dtype=torch.uint8, device=device),
            "r2_app_mask": make_random_mask(keep_frac),
            "r2_opacity_mask": torch.ones(N, dtype=torch.uint8, device=device),
        }
        print(f"  Timing {label}...")
        synthetic_results[label] = time_config(masks, label)

    # Opacity suppression series
    for keep_frac in [0.75, 0.50, 0.32]:
        label = f"OPA_K{int(keep_frac*100)}"
        masks = {
            "r2_geo_mask": torch.ones(N, dtype=torch.uint8, device=device),
            "r2_app_mask": torch.ones(N, dtype=torch.uint8, device=device),
            "r2_opacity_mask": make_random_mask(keep_frac),
        }
        print(f"  Timing {label}...")
        synthetic_results[label] = time_config(masks, label)

    # === 4. ORACLE_DECOUPLED_95 ===
    print("\n=== Timing ORACLE_DECOUPLED_95 ===")
    oracle_result = time_config(oracle_masks_95, "ORACLE_DECOUPLED_95")
    print(f"  backward: {oracle_result['total_backward_ms']['median']:.2f}ms, "
          f"e2e: {oracle_result['e2e_ms']['median']:.2f}ms")

    # === 5. ALL_BRANCH_K50 (single mask, all branches at 50%) ===
    print("\n=== Timing ALL_BRANCH_K50 ===")
    k50_mask = make_random_mask(0.50)
    all_branch_k50_masks = {
        "r2_geo_mask": k50_mask,
        "r2_app_mask": k50_mask,
        "r2_opacity_mask": k50_mask,
    }
    all_branch_result = time_config(all_branch_k50_masks, "ALL_BRANCH_K50")
    print(f"  backward: {all_branch_result['total_backward_ms']['median']:.2f}ms, "
          f"e2e: {all_branch_result['e2e_ms']['median']:.2f}ms")

    # === Save results ===
    results = {
        "checkpoint": checkpoint_path,
        "N_gaussians": N,
        "n_warmup": n_warmup,
        "n_measure": n_measure,
        "benchmark_iters": benchmark_iters,
        "vis_count": vis_count,
        "gradient_correctness": correctness,
        "zero_mask_check": zero_check,
        "full": full_result,
        "full_masked_all_ones": all_ones_result,
        "synthetic_masks": synthetic_results,
        "oracle_decoupled_95": oracle_result,
        "oracle_utility": oracle_utility,
        "all_branch_k50": all_branch_result,
    }

    # Compute speedups
    full_bwd = full_result["total_backward_ms"]["median"]
    full_e2e = full_result["e2e_ms"]["median"]

    oracle_bwd = oracle_result["total_backward_ms"]["median"]
    oracle_e2e = oracle_result["e2e_ms"]["median"]

    all_branch_bwd = all_branch_result["total_backward_ms"]["median"]
    all_branch_e2e = all_branch_result["e2e_ms"]["median"]

    results["speedups"] = {
        "oracle_decoupled_95": {
            "raster_bwd_speedup_pct": float((full_bwd - oracle_bwd) / full_bwd * 100),
            "e2e_gain_pct": float((full_e2e - oracle_e2e) / full_e2e * 100),
            "full_bwd_ms": full_bwd,
            "oracle_bwd_ms": oracle_bwd,
            "full_e2e_ms": full_e2e,
            "oracle_e2e_ms": oracle_e2e,
        },
        "all_branch_k50": {
            "raster_bwd_speedup_pct": float((full_bwd - all_branch_bwd) / full_bwd * 100),
            "e2e_gain_pct": float((full_e2e - all_branch_e2e) / full_e2e * 100),
            "full_bwd_ms": full_bwd,
            "all_branch_bwd_ms": all_branch_bwd,
            "full_e2e_ms": full_e2e,
            "all_branch_e2e_ms": all_branch_e2e,
        },
    }

    # Print summary
    print("\n" + "="*60)
    print("=== R2.1 SUMMARY ===")
    print("="*60)
    print(f"FULL backward: {full_bwd:.2f}ms, E2E: {full_e2e:.2f}ms")
    print(f"ORACLE_DECOUPLED_95 backward: {oracle_bwd:.2f}ms, E2E: {oracle_e2e:.2f}ms")
    print(f"  Raster bwd speedup: {(full_bwd - oracle_bwd) / full_bwd * 100:.1f}%")
    print(f"  E2E gain: {(full_e2e - oracle_e2e) / full_e2e * 100:.1f}%")
    print(f"ALL_BRANCH_K50 backward: {all_branch_bwd:.2f}ms, E2E: {all_branch_e2e:.2f}ms")
    print(f"  Raster bwd speedup: {(full_bwd - all_branch_bwd) / full_bwd * 100:.1f}%")
    print(f"  E2E gain: {(full_e2e - all_branch_e2e) / full_e2e * 100:.1f}%")

    print(f"\nUtility retained:")
    for fam in ["geometry", "appearance", "opacity"]:
        u = oracle_utility[fam]
        print(f"  {fam}: pos={u['pos_utility_retained']:.4f}, abs={u['abs_utility_retained']:.4f}")

    # Save
    output_path = os.path.join(output_dir, "r21_benchmark.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/reference_v1/ckpt_14k/checkpoints/iter_10000.pt")
    parser.add_argument("--output", default="results/reference_v1/r2.1")
    parser.add_argument("--camera-sequence", default="results/reference_v1/room_30k/camera_sequence.npy")
    args = parser.parse_args()
    config = ReferenceV1Config(scene="room", iterations=30000)
    run_benchmark(args.checkpoint, args.output, config, args.camera_sequence)
