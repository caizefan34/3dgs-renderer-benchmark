#!/usr/bin/env python3
"""
Phase C53-Validation — Actual Work Collection

Collects ACTUAL computational work targets (not future visibility):
  T1 future_tile_work    = Σ tiles_per_gauss(τ)        — rasterization work
  T2 future_backward_work = Σ tiles_per_gauss(τ) × grad_norm(τ) — backward work proxy
  T3 future_pixel_work   = Σ projected_area(τ)          — pixel rendering work
  T4 future_update       = Σ ‖Δθ(τ)‖                    — parameter update (secondary)

Also collects per-iteration timing for instrumentation validation.

Signals (same as C53-Discovery):
  S1-S7: prev_grad, ema_grad, visibility, screen_radius, opacity, scale, age
"""
import sys, math, json, time, argparse, os
from pathlib import Path
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
EMA_DECAY = 0.9

CHECKPOINTS = [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]
DELTAS = [10, 50, 100]
MAX_DELTA = 100
N_SAMPLE = 100_000
SAMPLING_SEED = 42


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


def render_with_meta(model, cam, data=None):
    if data is None:
        data = model.forward()
    r, _, meta = rasterization(
        means=data["xyz"], quats=data["rotations"], scales=data["scales"],
        opacities=data["opacity"], colors=data["shs"],
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size=16, packed=False, sh_degree=model.sh_degree,
        radius_clip=0.0, eps2d=0.1, render_mode="RGB",
    )
    return r[0].clamp(0, 1), meta


def get_optimizer(model):
    return torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4},
        {"params": [model.rotations], "lr": 1e-3},
        {"params": [model.scales], "lr": 5e-3},
        {"params": [model.opacity], "lr": 5e-2},
        {"params": [model.shs], "lr": 2.5e-3},
    ], eps=1e-15)


def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    return float(20 * math.log10(1.0 / math.sqrt(mse.item()))) if mse > 1e-10 else 100.0


class FutureWorkWindow:
    """Tracks ACTUAL computational work for sampled Gaussians."""

    def __init__(self, cp_iter, sampled_ids_np, device):
        self.cp_iter = cp_iter
        self.sampled_ids_np = sampled_ids_np
        self.n = len(sampled_ids_np)
        self.device = device
        self.end_iter = cp_iter + MAX_DELTA
        self.ids_tensor = torch.from_numpy(sampled_ids_np).long().to(device)

        # Actual work accumulators
        self.tile_work_sum = torch.zeros(self.n, device=device)    # T1: Σ tiles_per_gauss
        self.pixel_work_sum = torch.zeros(self.n, device=device)   # T3: Σ projected_area
        self.backward_work_sum = torch.zeros(self.n, device=device) # T2: Σ tiles × grad_norm
        self.vis_count = torch.zeros(self.n, device=device)         # For comparison
        self.update_xyz = torch.zeros(self.n, device=device)        # T4: Σ ‖Δxyz‖
        self.grad_sum = torch.zeros(self.n, device=device)          # Future gradient sum

        self.alive = torch.ones(self.n, dtype=torch.bool, device=device)
        self.outcome = torch.zeros(self.n, dtype=torch.int8, device=device)
        self.snapshots = {}

        # Parameter tracking for update
        self.prev_xyz = None
        self.prev_valid = None

    def get_indices(self, id_to_index):
        return id_to_index[self.ids_tensor]

    def accumulate_work(self, radii, tiles_per_gauss, grad_norm, id_to_index):
        """Accumulate actual work: tiles, area, backward work."""
        idx = self.get_indices(id_to_index)
        valid = idx >= 0
        if valid.any():
            i = idx[valid]
            r = radii[i].max(dim=-1).values.float()  # screen radius
            vis = (radii[i] > 0).any(dim=-1).float()  # 1.0 if visible, 0.0
            tpg = tiles_per_gauss[i].float()          # tile intersection count
            area = 3.14159 * r * r * vis              # projected area (0 if invisible)
            gn = grad_norm[i]                          # gradient norm

            # T1: rasterization work = tiles per gauss
            self.tile_work_sum[valid] += tpg
            # T3: pixel work = projected area
            self.pixel_work_sum[valid] += area
            # T2: backward work = tiles × gradient (computational cost weighted by gradient flow)
            self.backward_work_sum[valid] += tpg * gn
            # Visibility count (for comparison)
            self.vis_count[valid] += vis
            # Future gradient
            self.grad_sum[valid] += gn

    def save_params(self, model, id_to_index):
        idx = self.get_indices(id_to_index)
        valid = idx >= 0
        self.prev_valid = valid
        self.prev_xyz = torch.zeros(self.n, 3, device=self.device)
        if valid.any():
            self.prev_xyz[valid] = model.xyz.data[idx[valid]]

    def accumulate_update(self, model, id_to_index):
        if self.prev_valid is None:
            return
        idx = self.get_indices(id_to_index)
        valid = idx >= 0
        both = valid & self.prev_valid
        if both.any():
            self.update_xyz[both] += (model.xyz.data[idx[both]] - self.prev_xyz[both]).norm(dim=-1)

    def record_cloned_split(self, clone_mask, split_mask, id_to_index):
        idx = self.get_indices(id_to_index)
        valid = idx >= 0
        if not valid.any():
            return
        i = idx[valid]
        if clone_mask is not None:
            is_cloned = clone_mask[i]
            to_set = valid.clone()
            to_set[valid] = is_cloned
            not_pruned = self.outcome != 3
            self.outcome[to_set & not_pruned] = 1
        if split_mask is not None:
            is_split = split_mask[i]
            to_set = valid.clone()
            to_set[valid] = is_split
            not_pruned = self.outcome != 3
            self.outcome[to_set & not_pruned] = 2

    def record_pruned(self, prune_mask, id_to_index):
        idx = self.get_indices(id_to_index)
        valid = idx >= 0
        if valid.any():
            i = idx[valid]
            is_pruned = prune_mask[i]
            to_set = valid.clone()
            to_set[valid] = is_pruned
            self.outcome[to_set] = 3
            self.alive[to_set] = False
        newly_dead = (~valid) & self.alive
        self.outcome[newly_dead] = 3
        self.alive[newly_dead] = False

    def snapshot(self, delta, n_iters):
        self.snapshots[delta] = {
            "tile_work_sum": self.tile_work_sum.clone(),
            "tile_work_mean": self.tile_work_sum / max(n_iters, 1),
            "pixel_work_sum": self.pixel_work_sum.clone(),
            "pixel_work_mean": self.pixel_work_sum / max(n_iters, 1),
            "backward_work_sum": self.backward_work_sum.clone(),
            "backward_work_mean": self.backward_work_sum / max(n_iters, 1),
            "vis_count": self.vis_count.clone(),
            "update_xyz": self.update_xyz.clone(),
            "grad_sum": self.grad_sum.clone(),
            "grad_mean": self.grad_sum / max(n_iters, 1),
            "alive": self.alive.clone(),
        }


def run_collection(dataset, sfm_data, gpu_id, scene, seed, iters=30000):
    torch.manual_seed(seed)
    np.random.seed(seed)
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

    # Tracking
    gaussian_ids = torch.arange(N, dtype=torch.long, device=device)
    creation_iter = torch.zeros(N, dtype=torch.long, device=device)
    next_id = N
    ema_grad_norm = torch.zeros(N, device=device)
    prev_grad_norm = torch.zeros(N, device=device)
    vis_count_window = torch.zeros(N, device=device)
    radius_sum_window = torch.zeros(N, device=device)

    id_to_index = torch.full((N + 1,), -1, dtype=torch.long, device=device)
    id_to_index[gaussian_ids] = torch.arange(N, device=device)

    active_window = None
    checkpoint_data = {}

    # Instrumentation timing
    iter_times = []
    instrumented_iters = 0

    def update_id_to_index():
        nonlocal id_to_index
        current_max = int(gaussian_ids.max().item())
        if current_max >= id_to_index.shape[0]:
            new_size = current_max + 1
            new_map = torch.full((new_size,), -1, dtype=torch.long, device=device)
            new_map[:id_to_index.shape[0]] = id_to_index
            id_to_index = new_map
        id_to_index[:] = -1
        id_to_index[gaussian_ids] = torch.arange(len(gaussian_ids), device=device)

    def extend_tracking(n_new, current_iter):
        nonlocal next_id, gaussian_ids, creation_iter, ema_grad_norm, prev_grad_norm
        nonlocal vis_count_window, radius_sum_window
        new_ids = torch.arange(next_id, next_id + n_new, device=device, dtype=torch.long)
        gaussian_ids = torch.cat([gaussian_ids, new_ids])
        creation_iter = torch.cat([creation_iter, torch.full((n_new,), current_iter, dtype=torch.long, device=device)])
        ema_grad_norm = torch.cat([ema_grad_norm, torch.zeros(n_new, device=device)])
        prev_grad_norm = torch.cat([prev_grad_norm, torch.zeros(n_new, device=device)])
        vis_count_window = torch.cat([vis_count_window, torch.zeros(n_new, device=device)])
        radius_sum_window = torch.cat([radius_sum_window, torch.zeros(n_new, device=device)])
        next_id += n_new

    def filter_tracking(keep_mask):
        nonlocal gaussian_ids, creation_iter, ema_grad_norm, prev_grad_norm
        nonlocal vis_count_window, radius_sum_window
        gaussian_ids = gaussian_ids[keep_mask]
        creation_iter = creation_iter[keep_mask]
        ema_grad_norm = ema_grad_norm[keep_mask]
        prev_grad_norm = prev_grad_norm[keep_mask]
        vis_count_window = vis_count_window[keep_mask]
        radius_sum_window = radius_sum_window[keep_mask]

    print(f"  Starting: {iters} iters, scene={scene}, seed={seed}, N={N:,}")

    for iter_idx in range(iters):
        model.train()
        ci = cam_indices[iter_idx % n_cams]
        cam = dataset.get_camera(ci)
        gt = dataset.get_gt_image(ci)

        iter_t0 = time.perf_counter()

        # Forward + backward
        data = model.forward()
        pred, meta = render_with_meta(model, cam, data)
        l1 = F.l1_loss(pred, gt)
        dssim = sep_ssim(pred, gt)
        loss = 0.8 * l1 + 0.2 * dssim
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Work metrics from current render
        radii_meta = meta["radii"][0]
        tiles_meta = meta["tiles_per_gauss"][0]
        current_visible = (radii_meta > 0).any(dim=-1)
        current_radius = radii_meta.max(dim=-1).values.float()
        vis_count_window[current_visible] += 1
        radius_sum_window[current_visible] += current_radius[current_visible]

        # Gradient norm
        current_grad_norm = torch.zeros(model.xyz.shape[0], device=device)
        if model.xyz.grad is not None:
            current_grad_norm = model.xyz.grad.detach().norm(dim=-1)

        # Accumulate future work
        if active_window is not None and iter_idx > active_window.cp_iter:
            active_window.accumulate_work(radii_meta, tiles_meta, current_grad_norm, id_to_index)

        # Densification
        do_densify = (iter_idx >= DENSIFY_START and iter_idx < densify_end and
                      iter_idx % DENSIFY_INTERVAL == 0)
        if do_densify:
            model.accumulate_positional_gradient()
            if model._xyz_grad_accum is not None and model._denf_steps > 0:
                avg_grad = model._xyz_grad_accum / model._denf_steps
                high_grad_mask = avg_grad >= GRAD_THRESHOLD
                scales_act = torch.exp(model.scales).detach()
                median_scale = scales_act.median(dim=0).values
                is_small = (scales_act <= median_scale).all(dim=-1)
                clone_mask = high_grad_mask & is_small
                split_mask = high_grad_mask & (~is_small)

                if active_window is not None:
                    active_window.record_cloned_split(clone_mask, split_mask, id_to_index)

                old_N = model.xyz.shape[0]
                model.densification(grad_threshold=GRAD_THRESHOLD)
                n_new = model.xyz.shape[0] - old_N
                if n_new > 0:
                    extend_tracking(n_new, iter_idx)

                opacities = torch.sigmoid(model.opacity).detach().squeeze(-1)
                prune_mask = opacities < PRUNE_THRESHOLD
                if active_window is not None:
                    active_window.record_pruned(prune_mask, id_to_index)
                prune_count = model.prune(opacity_threshold=PRUNE_THRESHOLD)
                if prune_count > 0:
                    filter_tracking(~prune_mask)

                optimizer = get_optimizer(model)
                update_id_to_index()

        # Opacity reset
        if iter_idx > 0 and iter_idx % OPACITY_RESET_INTERVAL == 0:
            opacities = torch.sigmoid(model.opacity).detach().squeeze(-1)
            prune_mask_r = opacities < PRUNE_THRESHOLD
            if active_window is not None:
                active_window.record_pruned(prune_mask_r, id_to_index)
            removed = model.prune_and_reset(
                opacity_threshold=PRUNE_THRESHOLD,
                reset_interval=OPACITY_RESET_INTERVAL,
                current_step=iter_idx,
            )
            if removed > 0:
                filter_tracking(~prune_mask_r)
            optimizer = get_optimizer(model)
            update_id_to_index()

        # Visibility window reset at offset-50
        if iter_idx > 0 and (iter_idx - 50) % 100 == 0:
            vis_count_window = torch.zeros(model.xyz.shape[0], device=device)
            radius_sum_window = torch.zeros(model.xyz.shape[0], device=device)

        # SH progression
        if iter_idx > 0 and iter_idx % SH_PROGRESS_INTERVAL == 0:
            if model.sh_degree < model.max_sh_degree:
                model.set_sh_degree(model.sh_degree + 1)
                if ema_grad_norm.shape[0] != model.xyz.shape[0]:
                    ema_grad_norm = torch.zeros(model.xyz.shape[0], device=device)
                    prev_grad_norm = torch.zeros(model.xyz.shape[0], device=device)

        # Update EMA
        if current_grad_norm.shape[0] == ema_grad_norm.shape[0]:
            ema_grad_norm = EMA_DECAY * ema_grad_norm + (1 - EMA_DECAY) * current_grad_norm
            prev_grad_norm = current_grad_norm.clone()

        # Save params before step
        if active_window is not None and iter_idx > active_window.cp_iter:
            active_window.save_params(model, id_to_index)

        optimizer.step()

        # Accumulate update
        if active_window is not None and iter_idx > active_window.cp_iter:
            active_window.accumulate_update(model, id_to_index)

        # Snapshot at deltas
        if active_window is not None and iter_idx > active_window.cp_iter:
            n_in_window = iter_idx - active_window.cp_iter
            for d in DELTAS:
                if n_in_window == d:
                    active_window.snapshot(d, n_in_window)

        # Checkpoint
        if iter_idx in CHECKPOINTS:
            n_total = model.xyz.shape[0]
            n_sample = min(N_SAMPLE, n_total)
            rng = np.random.RandomState(SAMPLING_SEED + iter_idx)
            sample_pos = rng.choice(n_total, n_sample, replace=False)
            sampled_ids = gaussian_ids[sample_pos].cpu().numpy().astype(np.int64)

            opacity_vals = torch.sigmoid(model.opacity).detach().squeeze(-1)
            scale_norm_vals = torch.exp(model.scales).detach().norm(dim=-1)
            age_vals = (iter_idx - creation_iter).float()
            vis_signal = vis_count_window
            rad_signal = radius_sum_window / vis_count_window.clamp(min=1)

            cp_data = {
                "ids": sampled_ids, "n_total": n_total, "cp_iter": iter_idx,
                "signals": {
                    "prev_grad_norm": prev_grad_norm[sample_pos].cpu().numpy(),
                    "ema_grad_norm": ema_grad_norm[sample_pos].cpu().numpy(),
                    "visibility_count": vis_signal[sample_pos].cpu().numpy(),
                    "screen_radius_mean": rad_signal[sample_pos].cpu().numpy(),
                    "projected_area": (3.14159 * rad_signal[sample_pos] ** 2).cpu().numpy(),
                    "opacity": opacity_vals[sample_pos].cpu().numpy(),
                    "scale_norm": scale_norm_vals[sample_pos].cpu().numpy(),
                    "age": age_vals[sample_pos].cpu().numpy(),
                },
            }
            checkpoint_data[iter_idx] = cp_data
            active_window = FutureWorkWindow(iter_idx, sampled_ids, device)
            print(f"  [CP {iter_idx}] Sampled {n_sample}/{n_total}, "
                  f"vis_mean={vis_signal[sample_pos].float().mean().item():.1f}")

        # End future window
        if active_window is not None and iter_idx >= active_window.end_iter:
            cp = active_window.cp_iter
            for d in DELTAS:
                if d in active_window.snapshots:
                    snap = active_window.snapshots[d]
                    checkpoint_data[cp][f"delta_{d}"] = {
                        k: v.cpu().numpy() for k, v in snap.items()
                    }
            checkpoint_data[cp]["outcome"] = active_window.outcome.cpu().numpy()
            print(f"  [CP {cp}] Window done. Alive: {active_window.alive.sum().item()}/{active_window.n}")
            active_window = None

        iter_t1 = time.perf_counter()
        iter_times.append((iter_t1 - iter_t0) * 1000)
        instrumented_iters += 1

        if iter_idx % 1000 == 0 or iter_idx == iters - 1:
            mean_ms = np.mean(iter_times[-100:]) if len(iter_times) >= 100 else np.mean(iter_times)
            print(f"  Iter {iter_idx}: PSNR={compute_psnr(pred, gt):.2f}, "
                  f"GS={model.xyz.shape[0]:,}, loss={loss.item():.4f}, "
                  f"mean_iter={mean_ms:.1f}ms")

    # Handle remaining window
    if active_window is not None:
        cp = active_window.cp_iter
        for d in DELTAS:
            if d in active_window.snapshots:
                snap = active_window.snapshots[d]
                checkpoint_data[cp][f"delta_{d}"] = {
                    k: v.cpu().numpy() for k, v in snap.items()
                }
        checkpoint_data[cp]["outcome"] = active_window.outcome.cpu().numpy()

    # Instrumentation timing summary
    timing_summary = {
        "mean_iter_ms": float(np.mean(iter_times)),
        "median_iter_ms": float(np.median(iter_times)),
        "p95_iter_ms": float(np.percentile(iter_times, 95)),
        "n_iters": len(iter_times),
    }

    return checkpoint_data, timing_summary


def save_raw_data(checkpoint_data, timing_summary, scene, seed, save_dir):
    suffix = f"_seed{seed}" if seed != SEED else ""
    filepath = save_dir / f"{scene}{suffix}_actual_work.npz"
    save_dict = {
        "checkpoints": np.array(CHECKPOINTS),
        "timing_mean_ms": timing_summary["mean_iter_ms"],
        "timing_median_ms": timing_summary["median_iter_ms"],
    }
    for cp_iter in sorted(checkpoint_data.keys()):
        cp = checkpoint_data[cp_iter]
        p = f"cp{cp_iter}"
        save_dict[f"{p}_n_total"] = cp["n_total"]
        save_dict[f"{p}_ids"] = cp["ids"]
        for sig_name, sig_val in cp["signals"].items():
            save_dict[f"{p}_s_{sig_name}"] = sig_val
        for d in DELTAS:
            if f"delta_{d}" in cp:
                for key, val in cp[f"delta_{d}"].items():
                    save_dict[f"{p}_d{d}_{key}"] = val
        if "outcome" in cp:
            save_dict[f"{p}_outcome"] = cp["outcome"]
    np.savez_compressed(filepath, **save_dict)
    print(f"\nRaw data saved to {filepath} ({filepath.stat().st_size / 1e6:.1f} MB)")
    return filepath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--scene", default="room", choices=["room", "bicycle", "garden"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--iters", type=int, default=30000)
    args = parser.parse_args()

    repo_root = Path("/home/liaoyuanjun/3dgs-renderer-benchmark")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Phase C53-Validation — Actual Work Collection")
    print(f"  Scene={args.scene}, Seed={args.seed}, GPU={args.gpu}, Iters={args.iters}\n")

    dataset = GTDataset(scene=args.scene, repo_root=repo_root, resolution="1080p",
                        device=f"cuda:{args.gpu}")
    sfm_data = load_initial_checkpoint(args.scene, repo_root, device=f"cuda:{args.gpu}")
    print(f"  SfM points: {sfm_data['xyz'].shape[0]:,}")

    checkpoint_data, timing_summary = run_collection(
        dataset, sfm_data, args.gpu, args.scene, args.seed, args.iters)

    save_dir = repo_root / "results" / "a100" / "phase-c53-validation"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_raw_data(checkpoint_data, timing_summary, args.scene, args.seed, save_dir)

    # Save timing summary
    suffix = f"_seed{args.seed}" if args.seed != SEED else ""
    timing_file = save_dir / f"{args.scene}{suffix}_timing.json"
    with open(timing_file, 'w') as f:
        json.dump(timing_summary, f, indent=2)
    print(f"Timing saved to {timing_file}")


if __name__ == "__main__":
    main()
