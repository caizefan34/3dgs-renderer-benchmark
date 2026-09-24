#!/usr/bin/env python3
"""C31-A/B: Multi-GPU communication granularity + scaling envelope.

Measures gradient sizes, per-parameter communication cost, and estimates
DDP scaling from actual per-GPU timing data. Runs independently on each GPU.
"""
from __future__ import annotations
import argparse, gc, json, math, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from gsplat import rasterization
from scripts.epic05.phase7.gaussian_model import GaussianModel
from scripts.epic05.phase7.loss import combined_loss, d_ssim_loss
from scripts.epic05.phase7.dataset import GTDataset, load_initial_checkpoint
import torch.nn.functional as F

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--scene", default="room"); p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--measured", type=int, default=150); p.add_argument("--tile-size", type=int, default=16)
    args = p.parse_args()
    device = f"cuda:{args.gpu}"; torch.cuda.set_device(device)
    torch.manual_seed(42 + args.gpu); np.random.seed(42 + args.gpu)

    dataset = GTDataset(scene=args.scene, repo_root=ROOT, resolution="1080p", device=device)
    sfm_data = load_initial_checkpoint(args.scene, ROOT, device=device)
    model = GaussianModel(num_points=sfm_data["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=device)
    model.init_from_sfm(xyz=sfm_data["xyz"],
        opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0], 1), 0.1, device=device)),
        scales_log=sfm_data.get("scales"), rotations_raw=sfm_data.get("rotations"), shs=sfm_data.get("shs"))
    spatial_lr_scale = sfm_data["xyz"].norm(dim=-1).max().item()
    print(f"[GPU{args.gpu}] Gs: {model.xyz.shape[0]:,}", flush=True)

    optimizer = torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
        {"params": [model.rotations], "lr": 1e-3},
        {"params": [model.scales], "lr": 5e-3},
        {"params": [model.opacity], "lr": 5e-2},
        {"params": [model.shs], "lr": 2.5e-3},
    ], eps=1e-15, betas=(0.9, 0.999))

    def make_opt():
        return torch.optim.Adam([
            {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
            {"params": [model.rotations], "lr": 1e-3},
            {"params": [model.scales], "lr": 5e-3},
            {"params": [model.opacity], "lr": 5e-2},
            {"params": [model.shs], "lr": 2.5e-3},
        ], eps=1e-15, betas=(0.9, 0.999))

    total = args.warmup + args.measured
    measured = []
    grad_sizes = {}  # sample once

    for step in range(total):
        cam_idx = step % len(dataset)
        camera = dataset.get_camera(cam_idx); gt = dataset.get_gt_image(cam_idx)
        new_deg = min(3, step // 500)
        if new_deg != model.sh_degree:
            model.set_sh_degree(new_deg); optimizer = make_opt()
        data = model.forward()
        t0 = time.perf_counter()
        r, _, _ = rasterization(means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=camera.viewmatrix.unsqueeze(0), Ks=camera.K.unsqueeze(0),
            width=camera.image_width, height=camera.image_height,
            tile_size=args.tile_size, packed=True, sh_degree=model.sh_degree,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB")
        torch.cuda.synchronize(); fwd_ms = (time.perf_counter() - t0) * 1000
        rendered = r[0].clamp(0, 1)
        loss = combined_loss(rendered, gt, lambda_dssim=0.2)["loss"]
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(); bwd_ms = (time.perf_counter() - t0) * 1000

        # Gradient sizes (measure once)
        if step == 0:
            for name in ["xyz","rotations","scales","opacity","shs"]:
                p = getattr(model, name)
                grad_sizes[name] = {"numel": p.numel(), "bytes": p.numel() * p.element_size()}
            grad_sizes["total_bytes"] = sum(v["bytes"] for v in grad_sizes.values())
            print(f"  Grad sizes: {grad_sizes['total_bytes']/1e6:.1f}MB total", flush=True)

        model.accumulate_positional_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        t0 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize(); opt_ms = (time.perf_counter() - t0) * 1000

        if step >= 200 and step % 100 == 0:
            model.densification(grad_threshold=2e-4); optimizer = make_opt()
        if step >= 200 and step % 100 == 0:
            model.prune(opacity_threshold=0.005); optimizer = make_opt()

        with torch.no_grad():
            mse = torch.mean((rendered - gt) ** 2).item()
            psnr = 10 * math.log10(1.0 / max(mse, 1e-10))

        if step >= args.warmup:
            measured.append({"fwd_ms": fwd_ms, "bwd_ms": bwd_ms, "opt_ms": opt_ms,
                "t_iter_ms": fwd_ms + bwd_ms + opt_ms, "psnr": psnr, "n_g": model.xyz.shape[0]})

        if step % 50 == 0 or step == total - 1:
            w = "WARM" if step < args.warmup else "MEAS"
            print(f"  [{w}] Step {step}: PSNR={psnr:.2f} Gs={model.xyz.shape[0]:,} "
                  f"fwd={fwd_ms:.2f} bwd={bwd_ms:.2f} opt={opt_ms:.2f}", flush=True)

    fwd_a = np.array([m["fwd_ms"] for m in measured])
    bwd_a = np.array([m["bwd_ms"] for m in measured])
    opt_a = np.array([m["opt_ms"] for m in measured])
    ti_a = np.array([m["t_iter_ms"] for m in measured])

    # Estimate DDP scaling
    n_gpu = torch.cuda.device_count()
    grad_mb = grad_sizes["total_bytes"] / 1e6
    nvlink_bw = 300.0  # GB/s empirical for A100 NVLink
    nvlink_comm_ms = (grad_sizes["total_bytes"] * 2) / (nvlink_bw * 1e9) * 1000
    mean_t = float(np.mean(ti_a))

    scaling = {}
    for ng in [1, 2, 4, 8]:
        if ng > n_gpu: break
        if ng == 1: t_est = mean_t
        else:
            parallel_ms = float(np.mean(fwd_a) + np.mean(bwd_a))
            serial_ms = float(np.mean(opt_a))
            t_est = serial_ms + parallel_ms / ng + nvlink_comm_ms
        scaling[f"{ng}gpu"] = {"t_iter_est_ms": round(t_est, 2),
            "speedup": round(mean_t / max(t_est, 1e-9), 3), "efficiency_pct": round(mean_t / max(t_est * ng, 1e-9), 1),
            "note": "actual DDP needed for validation" if ng > 1 else "single GPU measured"}

    results = {"schema_version": 2, "phase": "C31-AB",
        "gpu": args.gpu, "measured_steps": args.measured, "warmup_steps": args.warmup,
        "gradient_sizes": grad_sizes, "gradient_mb": round(grad_mb, 2),
        "nvlink_comm_ms_estimate": round(nvlink_comm_ms, 3),
        "single_gpu_t_iter_ms": {"mean": mean_t, "median": float(np.median(ti_a)),
            "std": float(np.std(ti_a)), "min": float(np.min(ti_a)), "max": float(np.max(ti_a))},
        "single_gpu_fwd_ms": {"mean": float(np.mean(fwd_a)), "median": float(np.median(fwd_a))},
        "single_gpu_bwd_ms": {"mean": float(np.mean(bwd_a)), "median": float(np.median(bwd_a))},
        "single_gpu_opt_ms": {"mean": float(np.mean(opt_a)), "median": float(np.median(opt_a))},
        "scaling_estimate": scaling,
        "final_psnr": measured[-1]["psnr"], "final_gaussians": measured[-1]["n_g"],
    }

    print(f"\n[GPU{args.gpu}] T_iter: {mean_t:.2f}ms ± {float(np.std(ti_a)):.2f}")
    print(f"  Fwd: {float(np.mean(fwd_a)):.3f} Bwd: {float(np.mean(bwd_a)):.3f} Opt: {float(np.mean(opt_a)):.3f}")
    print(f"  Grad: {grad_mb:.1f}MB, NVLink comm est: {nvlink_comm_ms:.3f}ms")
    for k, v in scaling.items():
        print(f"  {k}: {v['t_iter_est_ms']}ms ({v['speedup']}x, {v['efficiency_pct']}%)")
    print(f"  C30-G INVALID FOR COMPARISON: baseline now = {mean_t:.1f}ms (was 98.7ms in C30)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
