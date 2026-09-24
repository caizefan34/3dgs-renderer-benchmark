"""
R0.3 Part 10: Masked Backward Scaling Benchmark.

Uses the existing C51 patched CUDA kernel (importance_mask) to measure
actual rasterization-backward runtime at different retention fractions.

Masks are DETERMINISTIC RANDOM (not importance-based) — this measures
the kernel's actual scaling behavior, not prediction quality.

Retention fractions: 100%, 75%, 50%, 32% of visible Gaussians.

No training. No optimization. Pure runtime microbenchmark.
"""

import sys
import os
import json
import time
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


def cuda_time(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), result


def benchmark_masked_backward(checkpoint_path, output_path, config, camera_sequence_path):
    device = "cuda"

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

    camera_sequence = np.load(camera_sequence_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    print(f"Loaded checkpoint: N={N}, SH degree={model.active_sh_degree}")

    ssim_fn = SepSSIM(device=device)

    # Pick 3 camera viewpoints for benchmarking
    benchmark_iters = [10001, 10002, 10003]

    # Retention fractions
    retention_fractions = [1.0, 0.75, 0.50, 0.32]
    n_warmup = 3
    n_measure = 10

    results = {}

    for ret_frac in retention_fractions:
        print(f"\n=== Retention fraction: {ret_frac*100:.0f}% ===")
        times_raster_bwd = []
        times_total_bwd = []
        times_forward = []

        for bench_iter in benchmark_iters:
            cam_idx = int(camera_sequence[bench_iter - 1])
            cam, gt_image = dataset.get_item(cam_idx)
            model.update_learning_rate(bench_iter)

            # Forward to get visibility
            r, _, meta = rasterization(
                means=model.get_xyz, quats=model.get_rotation,
                scales=model.get_scaling, opacities=model.get_opacity,
                colors=model.get_features,
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=16, packed=False, sh_degree=model.active_sh_degree,
                radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
            )
            image = r.squeeze(0)
            radii = meta["radii"][0]
            visibility_filter = (radii > 0).any(dim=-1)
            vis_count = int(visibility_filter.sum().item())

            # Build deterministic random mask
            n_retain = int(vis_count * ret_frac) if ret_frac < 1.0 else N
            mask = torch.zeros(N, dtype=torch.uint8, device=device)

            if ret_frac < 1.0:
                # Deterministic random: select ret_frac of visible Gaussians
                vis_indices = torch.where(visibility_filter)[0]
                # Use deterministic seed for reproducibility
                gen = torch.Generator(device=device)
                gen.manual_seed(42)
                perm = torch.randperm(len(vis_indices), generator=gen, device=device)
                retain_indices = vis_indices[perm[:n_retain]]
                mask[retain_indices] = 1
            else:
                # 100%: keep all visible
                mask[visibility_filter] = 1

            n_masked = int(mask.sum().item())
            print(f"  iter {bench_iter}: N={N}, vis={vis_count}, masked={n_masked} ({100*n_masked/N:.1f}%)")

            # Warmup
            for _ in range(n_warmup):
                model.optimizer.zero_grad(set_to_none=True)
                r, _, meta = rasterization(
                    means=model.get_xyz, quats=model.get_rotation,
                    scales=model.get_scaling, opacities=model.get_opacity,
                    colors=model.get_features,
                    viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                    width=cam.image_width, height=cam.image_height,
                    tile_size=16, packed=False, sh_degree=model.active_sh_degree,
                    radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
                    importance_mask=mask, compute_densify_grad=False,
                )
                image = r.squeeze(0)
                L1 = F.l1_loss(image, gt_image)
                dssim = ssim_fn(image, gt_image)
                loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim
                loss.backward()
                model.optimizer.step()
                model.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

            # Measure
            for _ in range(n_measure):
                model.optimizer.zero_grad(set_to_none=True)

                # Time forward
                t_fwd, (r, _, meta) = cuda_time(lambda: rasterization(
                    means=model.get_xyz, quats=model.get_rotation,
                    scales=model.get_scaling, opacities=model.get_opacity,
                    colors=model.get_features,
                    viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                    width=cam.image_width, height=cam.image_height,
                    tile_size=16, packed=False, sh_degree=model.active_sh_degree,
                    radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
                    importance_mask=mask, compute_densify_grad=False,
                ))
                times_forward.append(t_fwd)

                image = r.squeeze(0)
                L1 = F.l1_loss(image, gt_image)
                dssim = ssim_fn(image, gt_image)
                loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim

                # Time backward (includes raster bwd + projection bwd + SH bwd)
                t_bwd, _ = cuda_time(lambda: loss.backward())
                times_total_bwd.append(t_bwd)

                model.optimizer.step()
                model.optimizer.zero_grad(set_to_none=True)

        # Aggregate
        results[f"retention_{int(ret_frac*100)}"] = {
            "retention_fraction": ret_frac,
            "n_masked_mean": n_masked,
            "forward_ms": {
                "mean": float(np.mean(times_forward)),
                "median": float(np.median(times_forward)),
                "std": float(np.std(times_forward)),
            },
            "backward_total_ms": {
                "mean": float(np.mean(times_total_bwd)),
                "median": float(np.median(times_total_bwd)),
                "std": float(np.std(times_total_bwd)),
            },
        }

        print(f"  Forward: {np.mean(times_forward):.2f}ms")
        print(f"  Backward: {np.mean(times_total_bwd):.2f}ms")

    # Compute scaling function f(K) = backward(K) / backward(100%)
    bwd_100 = results["retention_100"]["backward_total_ms"]["median"]
    print(f"\n=== Backward Scaling Function f(K) ===")
    for key, val in results.items():
        ret = val["retention_fraction"]
        bwd = val["backward_total_ms"]["median"]
        f_k = bwd / bwd_100 if bwd_100 > 0 else 0
        val["f_k"] = f_k
        val["backward_saving_pct"] = float((1 - f_k) * 100)
        print(f"  K={ret*100:.0f}%: backward={bwd:.2f}ms, f(K)={f_k:.3f}, saving={100*(1-f_k):.1f}%")

    # Also measure forward-only (no mask, for reference)
    print(f"\n=== Reference: No-mask backward ===")
    ref_bwd_times = []
    for bench_iter in benchmark_iters:
        cam_idx = int(camera_sequence[bench_iter - 1])
        cam, gt_image = dataset.get_item(cam_idx)
        model.update_learning_rate(bench_iter)

        # Warmup
        for _ in range(3):
            model.optimizer.zero_grad(set_to_none=True)
            r, _, meta = rasterization(
                means=model.get_xyz, quats=model.get_rotation,
                scales=model.get_scaling, opacities=model.get_opacity,
                colors=model.get_features,
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=16, packed=False, sh_degree=model.active_sh_degree,
                radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
            )
            image = r.squeeze(0)
            L1 = F.l1_loss(image, gt_image)
            dssim = ssim_fn(image, gt_image)
            loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        for _ in range(10):
            model.optimizer.zero_grad(set_to_none=True)
            r, _, meta = rasterization(
                means=model.get_xyz, quats=model.get_rotation,
                scales=model.get_scaling, opacities=model.get_opacity,
                colors=model.get_features,
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=16, packed=False, sh_degree=model.active_sh_degree,
                radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
            )
            image = r.squeeze(0)
            L1 = F.l1_loss(image, gt_image)
            dssim = ssim_fn(image, gt_image)
            loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim
            t_bwd, _ = cuda_time(lambda: loss.backward())
            ref_bwd_times.append(t_bwd)
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)

    results["no_mask_reference"] = {
        "backward_total_ms": {
            "mean": float(np.mean(ref_bwd_times)),
            "median": float(np.median(ref_bwd_times)),
        }
    }
    print(f"  No-mask backward: {np.mean(ref_bwd_times):.2f}ms")

    # Save
    result = {
        "checkpoint": checkpoint_path,
        "N_gaussians": N,
        "sh_degree": model.active_sh_degree,
        "retention_results": results,
        "methodology": "Deterministic random masks (seed=42). 3 camera viewpoints, 3 warmup + 10 measured iterations each. importance_mask parameter gates gradient computation in rasterize_to_pixels_3dgs_bwd kernel.",
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    config = ReferenceV1Config(scene="room", iterations=30000)
    ckpt = "results/reference_v1/ckpt_14k/checkpoints/iter_10000.pt"
    cam_seq = "results/reference_v1/room_30k/camera_sequence.npy"
    output = "results/reference_v1/r0.3/masked_backward_scaling.json"
    benchmark_masked_backward(ckpt, output, config, cam_seq)
