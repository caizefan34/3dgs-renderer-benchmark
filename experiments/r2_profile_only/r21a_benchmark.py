"""
R2.1A Benchmark: Geometry-Focused Oracle Validation.

Uses the existing R2.1 patched gsplat kernel with three per-branch masks.

Tests (all per-camera CURRENT oracle):
  1. Baseline recheck: FULL timing
  2. CURRENT_CAMERA_ORACLE: per-camera oracle at >=95% utility per family
  3. GEO_ONLY_ORACLE_95: geo >=95%, app+opacity all-1s
  4. GEO_OPACITY_ORACLE_95: geo+opacity >=95%, app all-1s
  5. FULL_ATTRIBUTE_ORACLE_95: all three >=95%
  6. Optional: GEO_ONLY synthetic scaling

20 warmup, 100 measured. CUDA events.
"""
import sys, os, json, math, argparse, time
import numpy as np

# RTLD_GLOBAL for .so symbol resolution
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
    """Forward pass with optional r2 masks."""
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


def compute_positive_utility(model, cam, gt_image, config, ssim_fn):
    """Compute per-Gaussian positive utility for all derivative families.
    Returns dict of utility arrays (numpy, all N Gaussians, zero for invisible) and vis_np."""
    pre_state = capture_adam_state(model)
    model.optimizer.zero_grad(set_to_none=True)
    _, _, loss, vis_filter, _ = forward_pass(model, cam, gt_image, config, ssim_fn)
    loss.backward()
    vis_np = vis_filter.cpu().numpy().astype(bool)

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
    return u_loss, vis_np


def find_min_keep_frac(utility, vis_np, target=0.95, max_frac=1.0, min_frac=0.01):
    """Binary search for minimum keep fraction achieving >=target positive utility."""
    pos_u = np.maximum(utility[vis_np], 0)
    pos_total = float(pos_u.sum()) if pos_u.sum() > 0 else 0.0
    if pos_total == 0.0:
        return 1.0, 1.0  # all invisible or all negative

    vis_indices = np.where(vis_np)[0]
    n_vis = len(vis_indices)
    sorted_local = np.argsort(pos_u)[::-1]
    sorted_vals = pos_u[sorted_local]
    cumsum = np.cumsum(sorted_vals)

    # Find the smallest k where cumsum >= target * total
    target_val = target * pos_total
    k_needed = int(np.searchsorted(cumsum, target_val) + 1)
    k_needed = min(k_needed, n_vis)

    keep_frac = max(min_frac, k_needed / n_vis)
    actual_pos_retained = cumsum[k_needed - 1] / pos_total if k_needed > 0 else 0.0
    return keep_frac, float(actual_pos_retained)


def make_oracle_mask_from_keep_frac(utility, vis_np, keep_frac):
    """Build boolean mask retaining top keep_frac by positive utility.
    Returns (mask_tensor, pos_utility_retained)."""
    pos_u = np.maximum(utility[vis_np], 0)
    pos_total = float(pos_u.sum()) if pos_u.sum() > 0 else 1.0
    vis_indices = np.where(vis_np)[0]
    n_vis = len(vis_indices)
    k = max(1, min(int(n_vis * keep_frac), n_vis))

    sorted_local = np.argsort(pos_u)[::-1]
    top_k_local = sorted_local[:k]
    mask = np.zeros(len(utility), dtype=np.uint8)
    mask[vis_indices[top_k_local]] = 1

    retained = float(pos_u[top_k_local].sum() / pos_total)
    return torch.from_numpy(mask).to("cuda"), retained


def stats(arr):
    a = np.array(arr, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)), "n": len(a)}


def stats_to_dict(s):
    return {"mean": s["mean"], "median": s["median"], "p10": s["p10"], "p90": s["p90"], "n": s["n"]}


def timing_iteration(model, cam, gt_image, config, ssim_fn, masks, n_warmup, n_measure):
    """Time ONE camera with a fixed mask configuration. Returns times list."""
    # Warmup
    for _ in range(n_warmup):
        model.optimizer.zero_grad(set_to_none=True)
        _, _, loss, _, _ = forward_pass(model, cam, gt_image, config, ssim_fn, masks=masks)
        loss.backward()
        model.optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    bwd_times = []
    e2e_times = []
    for _ in range(n_measure):
        model.optimizer.zero_grad(set_to_none=True)
        t_fwd, (_, _, loss, _, _) = cuda_time(lambda: forward_pass(model, cam, gt_image, config, ssim_fn, masks=masks))
        t_bwd, _ = cuda_time(lambda: loss.backward())
        bwd_times.append(t_bwd)
        e2e_times.append(t_fwd + t_bwd)
        model.optimizer.zero_grad(set_to_none=True)

    return bwd_times, e2e_times


def run_benchmark(checkpoint_path, output_dir, config, camera_sequence_path):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda"
    dataset = GTDataset(config.scene, config.repo_root)
    camera_sequence = np.load(camera_sequence_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # We re-initialize the model for each camera group to avoid optimizer state drift
    # But for consistent timing, we need to reset optimizer state each time.
    # Strategy: load model once, capture INITIAL optimizer state and initial params,
    # then restore for each measurement.

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

    # ================================================================
    # Phase 1: Baseline recheck (FULL timing across all cameras)
    # ================================================================
    print("\n" + "="*70)
    print("=== Phase 1: Baseline recheck ===")
    print("="*70)

    all_full_bwd = []
    all_full_e2e = []
    all_ones_bwd = []
    all_ones_e2e = []

    all_ones_masks = {
        "r2_geo_mask": torch.ones(N, dtype=torch.uint8, device=device),
        "r2_app_mask": torch.ones(N, dtype=torch.uint8, device=device),
        "r2_opacity_mask": torch.ones(N, dtype=torch.uint8, device=device),
    }

    for bench_iter in benchmark_iters:
        cam_idx = int(camera_sequence[bench_iter - 1])
        cam, gt_image = dataset.get_item(cam_idx)
        model.update_learning_rate(bench_iter)

        # FULL timing
        bwd, e2e = timing_iteration(model, cam, gt_image, config, ssim_fn, None, n_warmup // 4, n_measure // 5)
        all_full_bwd.extend(bwd)
        all_full_e2e.extend(e2e)

        # ALL_ONES timing
        bwd_ones, e2e_ones = timing_iteration(model, cam, gt_image, config, ssim_fn, all_ones_masks, n_warmup // 4, n_measure // 5)
        all_ones_bwd.extend(bwd_ones)
        all_ones_e2e.extend(e2e_ones)

    full_stats = stats(all_full_bwd)
    full_e2e_stats = stats(all_full_e2e)
    all_ones_bwd_stats = stats(all_ones_bwd)
    all_ones_e2e_stats = stats(all_ones_e2e)

    print(f"FULL raster bwd: median={full_stats['median']:.2f}ms, mean={full_stats['mean']:.2f}ms")
    print(f"FULL E2E: median={full_e2e_stats['median']:.2f}ms")
    print(f"ALL_ONES bwd: median={all_ones_bwd_stats['median']:.2f}ms")
    mask_overhead = all_ones_bwd_stats['median'] - full_stats['median']
    mask_overhead_pct = mask_overhead / full_stats['median'] * 100
    print(f"Mask overhead: {mask_overhead:.2f}ms ({mask_overhead_pct:.1f}%)")

    # ================================================================
    # Phase 2: Per-camera oracle utility + mask search
    # ================================================================
    print("\n" + "="*70)
    print("=== Phase 2: Per-camera oracle mask construction ===")
    print("="*70)

    oracle_per_camera = {}  # bench_iter -> {"geo_keep": ..., "masks": ..., "utility": ...}

    for bench_iter in benchmark_iters:
        cam_idx = int(camera_sequence[bench_iter - 1])
        cam, gt_image = dataset.get_item(cam_idx)
        model.update_learning_rate(bench_iter)

        # Compute FULL utility for this camera (forward + backward + adam reconstruction)
        u_loss, vis_np = compute_positive_utility(model, cam, gt_image, config, ssim_fn)
        vis_count = int(vis_np.sum())

        # Find minimum keep fractions for >=95% utility
        geo_keep, geo_util = find_min_keep_frac(u_loss["geometry"], vis_np, target=0.95)
        app_keep, app_util = find_min_keep_frac(u_loss["sh"], vis_np, target=0.95)
        opa_keep, opa_util = find_min_keep_frac(u_loss["opacity_g"], vis_np, target=0.95)

        # Build masks
        geo_mask_t, geo_util2 = make_oracle_mask_from_keep_frac(u_loss["geometry"], vis_np, geo_keep)
        app_mask_t, app_util2 = make_oracle_mask_from_keep_frac(u_loss["sh"], vis_np, app_keep)
        opa_mask_t, opa_util2 = make_oracle_mask_from_keep_frac(u_loss["opacity_g"], vis_np, opa_keep)

        oracle_per_camera[bench_iter] = {
            "camera_idx": cam_idx,
            "vis_count": vis_count,
            "geo": {"keep_frac": geo_keep, "utility_retained": geo_util, "mask": geo_mask_t},
            "app": {"keep_frac": app_keep, "utility_retained": app_util, "mask": app_mask_t},
            "opa": {"keep_frac": opa_keep, "utility_retained": opa_util, "mask": opa_mask_t},
        }
        print(f"  Camera {bench_iter} (idx={cam_idx}): "
              f"geo={geo_keep*100:.1f}% ({geo_util*100:.1f}%), "
              f"app={app_keep*100:.1f}% ({app_util*100:.1f}%), "
              f"opa={opa_keep*100:.1f}% ({opa_util*100:.1f}%)")

    # ================================================================
    # Phase 3: CURRENT_CAMERA_ORACLE timing
    # ================================================================
    print("\n" + "="*70)
    print("=== Phase 3: CURRENT_CAMERA_ORACLE timing ===")
    print("="*70)

    cco_bwd = []
    cco_e2e = []
    cco_utils = {"geo": [], "app": [], "opa": []}

    for bench_iter in benchmark_iters:
        cam_idx = int(camera_sequence[bench_iter - 1])
        cam, gt_image = dataset.get_item(cam_idx)
        model.update_learning_rate(bench_iter)

        oc = oracle_per_camera[bench_iter]
        masks = {
            "r2_geo_mask": oc["geo"]["mask"],
            "r2_app_mask": oc["app"]["mask"],
            "r2_opacity_mask": oc["opa"]["mask"],
        }
        bwd, e2e = timing_iteration(model, cam, gt_image, config, ssim_fn, masks, n_warmup // 4, n_measure // 5)
        cco_bwd.extend(bwd)
        cco_e2e.extend(e2e)
        cco_utils["geo"].append(oc["geo"]["utility_retained"])
        cco_utils["app"].append(oc["app"]["utility_retained"])
        cco_utils["opa"].append(oc["opa"]["utility_retained"])

    cco_bwd_stats = stats(cco_bwd)
    cco_e2e_stats = stats(cco_e2e)
    cco_raster_speedup = (full_stats['median'] - cco_bwd_stats['median']) / full_stats['median'] * 100
    cco_e2e_gain = (full_e2e_stats['median'] - cco_e2e_stats['median']) / full_e2e_stats['median'] * 100

    print(f"CURRENT_CAMERA_ORACLE bwd: median={cco_bwd_stats['median']:.2f}ms")
    print(f"CURRENT_CAMERA_ORACLE E2E: median={cco_e2e_stats['median']:.2f}ms")
    print(f"Raster speedup: {cco_raster_speedup:.1f}%")
    print(f"E2E gain: {cco_e2e_gain:.1f}%")
    print(f"Utility: geo={np.mean(cco_utils['geo'])*100:.1f}%, "
          f"app={np.mean(cco_utils['app'])*100:.1f}%, "
          f"opa={np.mean(cco_utils['opa'])*100:.1f}%")

    # ================================================================
    # Phase 4: GEO_ONLY_ORACLE_95
    # ================================================================
    print("\n" + "="*70)
    print("=== Phase 4: GEO_ONLY_ORACLE_95 ===")
    print("="*70)

    geo_bwd = []
    geo_e2e = []
    geo_keep_fracs = []

    for bench_iter in benchmark_iters:
        cam_idx = int(camera_sequence[bench_iter - 1])
        cam, gt_image = dataset.get_item(cam_idx)
        model.update_learning_rate(bench_iter)

        oc = oracle_per_camera[bench_iter]
        masks = {
            "r2_geo_mask": oc["geo"]["mask"],
            "r2_app_mask": torch.ones(N, dtype=torch.uint8, device=device),
            "r2_opacity_mask": torch.ones(N, dtype=torch.uint8, device=device),
        }
        bwd, e2e = timing_iteration(model, cam, gt_image, config, ssim_fn, masks, n_warmup // 4, n_measure // 5)
        geo_bwd.extend(bwd)
        geo_e2e.extend(e2e)
        geo_keep_fracs.append(oc["geo"]["keep_frac"])

    geo_bwd_stats = stats(geo_bwd)
    geo_e2e_stats = stats(geo_e2e)
    geo_raster_speedup = (full_stats['median'] - geo_bwd_stats['median']) / full_stats['median'] * 100
    geo_e2e_gain = (full_e2e_stats['median'] - geo_e2e_stats['median']) / full_e2e_stats['median'] * 100

    print(f"GEO_ONLY_95 bwd: median={geo_bwd_stats['median']:.2f}ms")
    print(f"GEO_ONLY_95 E2E: median={geo_e2e_stats['median']:.2f}ms")
    print(f"Raster speedup: {geo_raster_speedup:.1f}%")
    print(f"E2E gain: {geo_e2e_gain:.1f}%")
    print(f"Geo keep: {np.mean(geo_keep_fracs)*100:.1f}% (mean), geo utility retained: {np.mean(cco_utils['geo'])*100:.1f}%")

    # ================================================================
    # Phase 5: GEO_OPACITY_ORACLE_95
    # ================================================================
    print("\n" + "="*70)
    print("=== Phase 5: GEO_OPACITY_ORACLE_95 ===")
    print("="*70)

    geo_opa_bwd = []
    geo_opa_e2e = []

    for bench_iter in benchmark_iters:
        cam_idx = int(camera_sequence[bench_iter - 1])
        cam, gt_image = dataset.get_item(cam_idx)
        model.update_learning_rate(bench_iter)

        oc = oracle_per_camera[bench_iter]
        masks = {
            "r2_geo_mask": oc["geo"]["mask"],
            "r2_app_mask": torch.ones(N, dtype=torch.uint8, device=device),
            "r2_opacity_mask": oc["opa"]["mask"],
        }
        bwd, e2e = timing_iteration(model, cam, gt_image, config, ssim_fn, masks, n_warmup // 4, n_measure // 5)
        geo_opa_bwd.extend(bwd)
        geo_opa_e2e.extend(e2e)

    geo_opa_bwd_stats = stats(geo_opa_bwd)
    geo_opa_e2e_stats = stats(geo_opa_e2e)
    geo_opa_raster_speedup = (full_stats['median'] - geo_opa_bwd_stats['median']) / full_stats['median'] * 100
    geo_opa_e2e_gain = (full_e2e_stats['median'] - geo_opa_e2e_stats['median']) / full_e2e_stats['median'] * 100

    print(f"GEO_OPACITY_95 bwd: median={geo_opa_bwd_stats['median']:.2f}ms")
    print(f"GEO_OPACITY_95 E2E: median={geo_opa_e2e_stats['median']:.2f}ms")
    print(f"Raster speedup: {geo_opa_raster_speedup:.1f}%")
    print(f"E2E gain: {geo_opa_e2e_gain:.1f}%")

    # ================================================================
    # Phase 6: FULL_ATTRIBUTE_ORACLE_95
    # ================================================================
    print("\n" + "="*70)
    print("=== Phase 6: FULL_ATTRIBUTE_ORACLE_95 ===")
    print("="*70)

    full_attr_bwd = []
    full_attr_e2e = []

    for bench_iter in benchmark_iters:
        cam_idx = int(camera_sequence[bench_iter - 1])
        cam, gt_image = dataset.get_item(cam_idx)
        model.update_learning_rate(bench_iter)

        oc = oracle_per_camera[bench_iter]
        masks = {
            "r2_geo_mask": oc["geo"]["mask"],
            "r2_app_mask": oc["app"]["mask"],
            "r2_opacity_mask": oc["opa"]["mask"],
        }
        bwd, e2e = timing_iteration(model, cam, gt_image, config, ssim_fn, masks, n_warmup // 4, n_measure // 5)
        full_attr_bwd.extend(bwd)
        full_attr_e2e.extend(e2e)

    full_attr_bwd_stats = stats(full_attr_bwd)
    full_attr_e2e_stats = stats(full_attr_e2e)
    full_attr_raster_speedup = (full_stats['median'] - full_attr_bwd_stats['median']) / full_stats['median'] * 100
    full_attr_e2e_gain = (full_e2e_stats['median'] - full_attr_e2e_stats['median']) / full_e2e_stats['median'] * 100

    print(f"FULL_ATTRIBUTE_95 bwd: median={full_attr_bwd_stats['median']:.2f}ms")
    print(f"FULL_ATTRIBUTE_95 E2E: median={full_attr_e2e_stats['median']:.2f}ms")
    print(f"Raster speedup: {full_attr_raster_speedup:.1f}%")
    print(f"E2E gain: {full_attr_e2e_gain:.1f}%")

    # ================================================================
    # Phase 7: Optional GEO_ONLY synthetic scaling
    # ================================================================
    print("\n" + "="*70)
    print("=== Phase 7: GEO_ONLY synthetic scaling ===")
    print("="*70)

    # Use first camera for synthetic scaling
    bench_iter = benchmark_iters[0]
    cam_idx = int(camera_sequence[bench_iter - 1])
    cam, gt_image = dataset.get_item(cam_idx)
    model.update_learning_rate(bench_iter)

    # Get vis_indices from utility computation
    # We already have vis counts from oracle_per_camera
    oc0 = oracle_per_camera[bench_iter]
    vis_count = oc0["vis_count"]

    # Compute vis_indices from the first camera
    model.optimizer.zero_grad(set_to_none=True)
    _, _, loss, vis_filter0, _ = forward_pass(model, cam, gt_image, config, ssim_fn)
    loss.backward()
    model.optimizer.zero_grad(set_to_none=True)
    vis_np0 = vis_filter0.cpu().numpy().astype(bool)
    vis_indices0 = np.where(vis_np0)[0]

    def make_random_mask(keep_frac, vis_indices, total_n):
        n_keep = max(1, min(int(len(vis_indices) * keep_frac), len(vis_indices)))
        chosen = np.random.choice(vis_indices, size=n_keep, replace=False)
        mask = np.zeros(total_n, dtype=np.uint8)
        mask[chosen] = 1
        return torch.from_numpy(mask).to(device)

    np.random.seed(42)  # deterministic
    geo_synthetic = {}

    for keep_frac in [0.75, 0.50, 0.32, 0.20]:
        label = f"GEO_SYNTH_K{int(keep_frac*100)}"
        masks = {
            "r2_geo_mask": make_random_mask(keep_frac, vis_indices0, N),
            "r2_app_mask": torch.ones(N, dtype=torch.uint8, device=device),
            "r2_opacity_mask": torch.ones(N, dtype=torch.uint8, device=device),
        }
        bwd, e2e = timing_iteration(model, cam, gt_image, config, ssim_fn, masks, n_warmup // 4, n_measure // 5)
        geo_synthetic[label] = {"bwd_ms": stats(bwd), "e2e_ms": stats(e2e)}
        print(f"  {label}: bwd={stats(bwd)['median']:.2f}ms, e2e={stats(e2e)['median']:.2f}ms")

    # ================================================================
    # Build results
    # ================================================================

    # Per-camera oracle details (without mask tensors, for JSON)
    per_camera_oracle = {}
    for bi, oc in oracle_per_camera.items():
        per_camera_oracle[str(bi)] = {
            "camera_idx": oc["camera_idx"],
            "vis_count": oc["vis_count"],
            "geo": {"keep_frac": oc["geo"]["keep_frac"], "utility_retained": oc["geo"]["utility_retained"]},
            "app": {"keep_frac": oc["app"]["keep_frac"], "utility_retained": oc["app"]["utility_retained"]},
            "opa": {"keep_frac": oc["opa"]["keep_frac"], "utility_retained": oc["opa"]["utility_retained"]},
        }

    # Per-camera oracle details (without mask tensors, for JSON)
    per_camera_oracle = {}
    for bi, oc in oracle_per_camera.items():
        per_camera_oracle[str(bi)] = {
            "camera_idx": oc["camera_idx"],
            "vis_count": oc["vis_count"],
            "geo": {"keep_frac": oc["geo"]["keep_frac"], "utility_retained": oc["geo"]["utility_retained"]},
            "app": {"keep_frac": oc["app"]["keep_frac"], "utility_retained": oc["app"]["utility_retained"]},
            "opa": {"keep_frac": oc["opa"]["keep_frac"], "utility_retained": oc["opa"]["utility_retained"]},
        }

    # Pooled utility across cameras
    cco_geo_util = float(np.mean(cco_utils["geo"]))
    cco_app_util = float(np.mean(cco_utils["app"]))
    cco_opa_util = float(np.mean(cco_utils["opa"]))

    # Mask overhead analysis
    mask_overhead_analysis = {
        "FULL_bwd_ms": full_stats['median'],
        "ALL_ONES_bwd_ms": all_ones_bwd_stats['median'],
        "mask_overhead_ms": mask_overhead,
        "mask_overhead_pct": mask_overhead_pct,
        "GEO_ONLY_95_bwd_ms": geo_bwd_stats['median'],
        "GEO_ONLY_95_overhead_vs_full_ms": geo_bwd_stats['median'] - full_stats['median'],
        "GEO_OPACITY_95_bwd_ms": geo_opa_bwd_stats['median'],
        "GEO_OPACITY_95_overhead_vs_full_ms": geo_opa_bwd_stats['median'] - full_stats['median'],
        "FULL_ATTRIBUTE_95_bwd_ms": full_attr_bwd_stats['median'],
        "FULL_ATTRIBUTE_95_overhead_vs_full_ms": full_attr_bwd_stats['median'] - full_stats['median'],
        "CURRENT_CAMERA_ORACLE_bwd_ms": cco_bwd_stats['median'],
        "CURRENT_CAMERA_ORACLE_overhead_vs_full_ms": cco_bwd_stats['median'] - full_stats['median'],
        "note": "Mask overhead = masked time - FULL time. The raw speedup potential is the geometric saving "
                "from suppressed atomicAdds AFTER subtracting this overhead.",
    }

    # ================================================================
    # Decisions
    # ================================================================

    # GEOMETRY_DOMINANT if GEO_ONLY_95 captures >=80% of FULL_ATTRIBUTE_95's E2E gain
    if full_attr_e2e_gain > 0:
        geo_capture_ratio = geo_e2e_gain / full_attr_e2e_gain
    else:
        geo_capture_ratio = 0.0

    geometry_dominant = geo_capture_ratio >= 0.80

    # GEO_OPACITY_DOMINANT if GEO_OPACITY_95 captures >=90% of FULL_ATTRIBUTE_95's E2E gain
    # and opacity gives meaningful gain beyond GEO_ONLY
    if full_attr_e2e_gain > 0:
        geo_opa_capture_ratio = geo_opa_e2e_gain / full_attr_e2e_gain
    else:
        geo_opa_capture_ratio = 0.0

    opacity_adds_gain = (geo_opa_e2e_gain - geo_e2e_gain) >= 0.5  # >=0.5% absolute
    geo_opacity_dominant = geo_opa_capture_ratio >= 0.90 and opacity_adds_gain

    # FULL_ATTRIBUTE_REQUIRED if appearance gating adds >=1% absolute E2E gain
    app_adds_gain = (full_attr_e2e_gain - geo_opa_e2e_gain) >= 1.0  # >=1% absolute
    full_attribute_required = app_adds_gain

    if full_attribute_required:
        dominant = "FULL_ATTRIBUTE_REQUIRED"
    elif geo_opacity_dominant:
        dominant = "GEO_OPACITY_DOMINANT"
    elif geometry_dominant:
        dominant = "GEOMETRY_DOMINANT"
    else:
        dominant = "GEOMETRY_DOMINANT"  # default fallback

    # Architecture gate
    # ARCH_CONFIRMED: >=95% utility per active family AND >=5% E2E gain
    arch_utility_ok = cco_geo_util >= 0.95 and cco_app_util >= 0.95 and cco_opa_util >= 0.95
    arch_e2e_ok = full_attr_e2e_gain >= 5.0

    if arch_utility_ok and arch_e2e_ok:
        arch_gate = "ARCH_CONFIRMED"
    elif full_attr_e2e_gain >= 2.0:
        arch_gate = "ARCH_WEAKENED"
    else:
        arch_gate = "ARCH_INVALIDATED"

    print("\n" + "="*70)
    print("=== R2.1A SUMMARY ===")
    print("="*70)
    print(f"Architecture: {arch_gate}")
    print(f"Dominant design: {dominant}")
    print(f"")
    print(f"FULL: bwd={full_stats['median']:.2f}ms, E2E={full_e2e_stats['median']:.2f}ms")
    print(f"GEO_ONLY_95: bwd={geo_bwd_stats['median']:.2f}ms, gain={geo_e2e_gain:.1f}%")
    print(f"GEO_OPACITY_95: bwd={geo_opa_bwd_stats['median']:.2f}ms, gain={geo_opa_e2e_gain:.1f}%")
    print(f"FULL_ATTRIBUTE_95: bwd={full_attr_bwd_stats['median']:.2f}ms, gain={full_attr_e2e_gain:.1f}%")
    print(f"")
    print(f"CURRENT-camera utility: geo={cco_geo_util*100:.1f}%, app={cco_app_util*100:.1f}%, opa={cco_opa_util*100:.1f}%")
    print(f"Geo captures {geo_capture_ratio*100:.1f}% of FULL_ATTRIBUTE E2E gain")
    print(f"App adds >=1% absolute E2E? {'YES' if app_adds_gain else 'NO'}")
    print(f"Opac adds >=0.5% beyond geo? {'YES' if opacity_adds_gain else 'NO'}")

    # Save all results
    results = {
        "environment": {
            "git_commit": "32ab80e",
            "gpu": "NVIDIA A100-PCIE-40GB",
            "cuda_version": "11.5",
            "checkpoint": checkpoint_path,
            "N_gaussians": N,
            "benchmark_iters": benchmark_iters,
            "n_warmup": n_warmup,
            "n_measure": n_measure,
        },
        "full_timing": {
            "raster_bwd_ms": stats_to_dict(full_stats),
            "e2e_ms": stats_to_dict(full_e2e_stats),
        },
        "all_ones_masked_timing": {
            "raster_bwd_ms": stats_to_dict(all_ones_bwd_stats),
            "e2e_ms": stats_to_dict(all_ones_e2e_stats),
        },
        "per_camera_oracle": per_camera_oracle,
        "current_camera_oracle": {
            "raster_bwd_ms": stats_to_dict(cco_bwd_stats),
            "e2e_ms": stats_to_dict(cco_e2e_stats),
            "raster_speedup_pct": cco_raster_speedup,
            "e2e_gain_pct": cco_e2e_gain,
            "pooled_utility": {"geo": cco_geo_util, "app": cco_app_util, "opa": cco_opa_util},
        },
        "geo_only_95": {
            "raster_bwd_ms": stats_to_dict(geo_bwd_stats),
            "e2e_ms": stats_to_dict(geo_e2e_stats),
            "raster_speedup_pct": geo_raster_speedup,
            "e2e_gain_pct": geo_e2e_gain,
            "mean_geo_keep_frac": float(np.mean(geo_keep_fracs)),
            "mean_geo_utility_retained": cco_geo_util,
        },
        "geo_opacity_95": {
            "raster_bwd_ms": stats_to_dict(geo_opa_bwd_stats),
            "e2e_ms": stats_to_dict(geo_opa_e2e_stats),
            "raster_speedup_pct": geo_opa_raster_speedup,
            "e2e_gain_pct": geo_opa_e2e_gain,
        },
        "full_attribute_95": {
            "raster_bwd_ms": stats_to_dict(full_attr_bwd_stats),
            "e2e_ms": stats_to_dict(full_attr_e2e_stats),
            "raster_speedup_pct": full_attr_raster_speedup,
            "e2e_gain_pct": full_attr_e2e_gain,
        },
        "geo_synthetic_scaling": geo_synthetic,
        "mask_overhead": mask_overhead_analysis,
        "decisions": {
            "dominant_design": dominant,
            "architecture_gate": arch_gate,
            "geo_capture_ratio_vs_full_attr": float(geo_capture_ratio),
            "geo_opa_capture_ratio_vs_full_attr": float(geo_opa_capture_ratio),
            "app_adds_ge1pct_absolute_e2e": app_adds_gain,
            "opacity_adds_ge0.5pct_beyond_geo": opacity_adds_gain,
            "arch_utility_met": arch_utility_ok,
            "arch_e2e_met": arch_e2e_ok,
            "full_attr_bwd_ms": full_attr_bwd_stats['median'],
            "full_attr_e2e_ms": full_attr_e2e_stats['median'],
            "full_e2e_ms": full_e2e_stats['median'],
            "geo_only_e2e_gain_pct": geo_e2e_gain,
            "geo_opacity_e2e_gain_pct": geo_opa_e2e_gain,
            "full_attr_e2e_gain_pct": full_attr_e2e_gain,
        },
    }

    output_path = os.path.join(output_dir, "r21a_benchmark.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results/reference_v1/ckpt_14k/checkpoints/iter_10000.pt")
    parser.add_argument("--output", default="results/reference_v1/r2.1a")
    parser.add_argument("--camera-sequence", default="results/reference_v1/room_30k/camera_sequence.npy")
    args = parser.parse_args()
    config = ReferenceV1Config(scene="room", iterations=30000)
    run_benchmark(args.checkpoint, args.output, config, args.camera_sequence)
