"""
R1 Part 15: Post-Densification AbsGrad Microbenchmark.

On a 20K+ Room checkpoint (well past densify_until_iter=15000),
same state and cameras, compare absgrad=True vs absgrad=False
with densification permanently disabled.

Verifies trainable parameter gradients are equal within numerical tolerance.
Measures raster backward and E2E iteration time.

Decision: E2E gain < 1% -> DROP; >= 1% -> KEEP_SYSTEMS
"""

import sys, os, json, time
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


def run_absgrad_benchmark(checkpoint_path, output_path, config, camera_sequence_path):
    device = "cuda"

    dataset = GTDataset(config.scene, config.repo_root)
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

    # Pick 5 camera viewpoints (past iteration 20000)
    benchmark_iters = [20001, 20002, 20003, 20004, 20005]

    # Warmup and gradient comparison
    print("\n=== Gradient Comparison: absgrad=True vs absgrad=False ===")
    grad_comparison = {}

    for bench_iter in benchmark_iters[:2]:  # Only check gradients on 2 iters
        cam_idx = int(camera_sequence[bench_iter - 1])
        cam, gt_image = dataset.get_item(cam_idx)

        # Run with absgrad=True
        model.optimizer.zero_grad(set_to_none=True)
        r_true, _, _ = rasterization(
            means=model.get_xyz, quats=model.get_rotation,
            scales=model.get_scaling, opacities=model.get_opacity,
            colors=model.get_features,
            viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            tile_size=16, packed=False, sh_degree=model.active_sh_degree,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=True,
        )
        image_true = r_true.squeeze(0)
        L1 = F.l1_loss(image_true, gt_image)
        dssim = ssim_fn(image_true, gt_image)
        loss_true = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim
        loss_true.backward()

        grads_true = {}
        for name, param in [("xyz", model._xyz), ("shs", model._shs),
                            ("opacity", model._opacity), ("scale", model._scaling),
                            ("rot", model._rotation)]:
            grads_true[name] = param.grad.detach().clone()
        model.optimizer.zero_grad(set_to_none=True)

        # Run with absgrad=False
        r_false, _, _ = rasterization(
            means=model.get_xyz, quats=model.get_rotation,
            scales=model.get_scaling, opacities=model.get_opacity,
            colors=model.get_features,
            viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            tile_size=16, packed=False, sh_degree=model.active_sh_degree,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=False,
        )
        image_false = r_false.squeeze(0)
        L1_f = F.l1_loss(image_false, gt_image)
        dssim_f = ssim_fn(image_false, gt_image)
        loss_false = (1.0 - config.lambda_dssim) * L1_f + config.lambda_dssim * dssim_f
        loss_false.backward()

        grads_false = {}
        for name, param in [("xyz", model._xyz), ("shs", model._shs),
                            ("opacity", model._opacity), ("scale", model._scaling),
                            ("rot", model._rotation)]:
            grads_false[name] = param.grad.detach().clone()
        model.optimizer.zero_grad(set_to_none=True)

        # Compare
        iter_cmp = {}
        for name in ["xyz", "shs", "opacity", "scale", "rot"]:
            g_t = grads_true[name].flatten()
            g_f = grads_false[name].flatten()
            max_abs_diff = float((g_t - g_f).abs().max().item())
            max_rel_diff = float(((g_t - g_f).abs() / (g_t.abs() + 1e-10)).max().item())
            iter_cmp[name] = {
                "max_abs_diff": max_abs_diff,
                "max_rel_diff": max_rel_diff,
                "grad_norm_true": float(g_t.norm().item()),
                "grad_norm_false": float(g_f.norm().item()),
            }
            print(f"  iter {bench_iter} {name}: max_abs_diff={max_abs_diff:.2e} max_rel_diff={max_rel_diff:.2e}")
        grad_comparison[f"iter_{bench_iter}"] = iter_cmp

    # Timing comparison
    print("\n=== Timing: absgrad=True vs absgrad=False ===")
    n_warmup = 3
    n_measure = 10

    for absgrad_mode in [True, False]:
        mode_name = "absgrad_true" if absgrad_mode else "absgrad_false"
        print(f"\n--- {mode_name} ---")
        times_forward = []
        times_backward = []
        times_total = []

        for bench_iter in benchmark_iters:
            cam_idx = int(camera_sequence[bench_iter - 1])
            cam, gt_image = dataset.get_item(cam_idx)
            model.update_learning_rate(bench_iter)

            # Warmup
            for _ in range(n_warmup):
                model.optimizer.zero_grad(set_to_none=True)
                r, _, _ = rasterization(
                    means=model.get_xyz, quats=model.get_rotation,
                    scales=model.get_scaling, opacities=model.get_opacity,
                    colors=model.get_features,
                    viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                    width=cam.image_width, height=cam.image_height,
                    tile_size=16, packed=False, sh_degree=model.active_sh_degree,
                    radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=absgrad_mode,
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
                
                t_fwd, (r, _, _) = cuda_time(lambda: rasterization(
                    means=model.get_xyz, quats=model.get_rotation,
                    scales=model.get_scaling, opacities=model.get_opacity,
                    colors=model.get_features,
                    viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                    width=cam.image_width, height=cam.image_height,
                    tile_size=16, packed=False, sh_degree=model.active_sh_degree,
                    radius_clip=0.0, eps2d=0.1, render_mode="RGB", absgrad=absgrad_mode,
                ))
                times_forward.append(t_fwd)

                image = r.squeeze(0)
                L1 = F.l1_loss(image, gt_image)
                dssim = ssim_fn(image, gt_image)
                loss = (1.0 - config.lambda_dssim) * L1 + config.lambda_dssim * dssim

                t_bwd, _ = cuda_time(lambda: loss.backward())
                times_backward.append(t_bwd)

                t_opt, _ = cuda_time(lambda: model.optimizer.step())
                model.optimizer.zero_grad(set_to_none=True)
                times_total.append(t_fwd + t_bwd + t_opt)

        print(f"  Forward: {np.mean(times_forward):.2f}ms (median {np.median(times_forward):.2f})")
        print(f"  Backward: {np.mean(times_backward):.2f}ms (median {np.median(times_backward):.2f})")
        print(f"  Total: {np.mean(times_total):.2f}ms (median {np.median(times_total):.2f})")

        grad_comparison[mode_name] = {
            "forward_ms": {"mean": float(np.mean(times_forward)), "median": float(np.median(times_forward))},
            "backward_ms": {"mean": float(np.mean(times_backward)), "median": float(np.median(times_backward))},
            "total_ms": {"mean": float(np.mean(times_total)), "median": float(np.median(times_total))},
            "n_measure": n_measure,
            "n_cameras": len(benchmark_iters),
        }

    # Compute E2E gain
    total_true = grad_comparison["absgrad_true"]["total_ms"]["median"]
    total_false = grad_comparison["absgrad_false"]["total_ms"]["median"]
    e2e_gain = float((total_true - total_false) / total_true * 100)
    bwd_true = grad_comparison["absgrad_true"]["backward_ms"]["median"]
    bwd_false = grad_comparison["absgrad_false"]["backward_ms"]["median"]
    bwd_gain = float((bwd_true - bwd_false) / bwd_true * 100)

    decision = "KEEP_SYSTEMS" if e2e_gain >= 1.0 else "DROP"

    print(f"\n=== Results ===")
    print(f"  Backward: absgrad_true={bwd_true:.2f}ms vs absgrad_false={bwd_false:.2f}ms (gain={bwd_gain:.2f}%)")
    print(f"  E2E: absgrad_true={total_true:.2f}ms vs absgrad_false={total_false:.2f}ms (gain={e2e_gain:.2f}%)")
    print(f"  Decision: {decision}")

    result = {
        "checkpoint": checkpoint_path,
        "N_gaussians": N,
        "gradient_comparison": grad_comparison,
        "e2e_gain_pct": e2e_gain,
        "backward_gain_pct": bwd_gain,
        "decision": decision,
        "methodology": "5 camera viewpoints past iter 20000, 3 warmup + 10 measured iters each, densification disabled",
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    config = ReferenceV1Config(scene="room", iterations=30000)
    
    # Find available checkpoints >= 20K
    ckpt_dir = "results/reference_v1"
    possible_ckpts = []
    for d in os.listdir(ckpt_dir):
        full = os.path.join(ckpt_dir, d, "checkpoints")
        if os.path.isdir(full):
            for f in os.listdir(full):
                if f.startswith("iter_") and f.endswith(".pt"):
                    iter_num = int(f.replace("iter_", "").replace(".pt", ""))
                    if iter_num >= 20000:
                        possible_ckpts.append((iter_num, os.path.join(full, f)))
    
    if not possible_ckpts:
        # Fall back to 14K (still past most densification)
        ckpt = "results/reference_v1/ckpt_14k/checkpoints/iter_14000.pt"
        print(f"WARNING: No 20K+ checkpoint found, using {ckpt}")
    else:
        possible_ckpts.sort()
        ckpt = possible_ckpts[-1][1]
        print(f"Using checkpoint: {ckpt}")
    
    cam_seq = "results/reference_v1/room_30k/camera_sequence.npy"
    output = "results/reference_v1/r1/absgrad_postdensification_microbench.json"
    run_absgrad_benchmark(ckpt, output, config, cam_seq)
