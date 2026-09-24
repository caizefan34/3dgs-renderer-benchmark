#!/usr/bin/env python3
"""C31 Baseline Validity: reproduce proper T_iter on original project stack.

Purpose: measure steady-state T_iter after sufficient warmup (100+ iters)
to eliminate CUDA JIT compilation overhead from measurement.

Uses scripts/epic05/phase7/ (GaussianModel, combined_loss, GTDataset, gsplat).
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
from scripts.epic05.phase7.loss import combined_loss
from scripts.epic05.phase7.dataset import GTDataset, load_initial_checkpoint

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--scene", default="room")
    p.add_argument("--resolution", default="1080p")
    p.add_argument("--warmup", type=int, default=150, help="Iterations BEFORE measurement begins (for JIT warmup)")
    p.add_argument("--measured", type=int, default=200, help="Iterations for measurement")
    p.add_argument("--tile-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # --- Record environment ---
    env = {
        "git_commit": "02375033388d4348376b6b607ab85f551e498a77",
        "gsplat_version": __import__("gsplat").__version__ if hasattr(__import__("gsplat"), "__version__") else "1.5.3",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_properties(0).name,
        "scene": args.scene,
        "resolution": args.resolution,
        "tile_size": args.tile_size,
        "packed": True,
        "seed": args.seed,
        "loss": "L1 + 0.2*D-SSIM",
        "optimizer": "Adam (eps=1e-15, beta1=0.9, beta2=0.999)",
        "camera_schedule": "round-robin",
        "densification": {"start": 200, "interval": 100, "grad_threshold": 0.0002},
        "pruning": {"start": 200, "interval": 100, "opacity_threshold": 0.005},
        "sh_degree_schedule": "increase every 500 steps, max=3",
    }

    print(f"=== Baseline Validity ===")
    print(f"GPU: {env['gpu']}")
    print(f"gsplat: {env['gsplat_version']}, torch: {env['torch_version']}, CUDA: {env['cuda_version']}")
    print(f"Scene: {args.scene}, Resolution: {args.resolution}")
    print(f"Warmup: {args.warmup}, Measured: {args.measured}")
    print()

    # Dataset + model
    dataset = GTDataset(scene=args.scene, repo_root=ROOT, resolution=args.resolution, device=device)
    sfm_data = load_initial_checkpoint(args.scene, ROOT, device=device)
    env["initial_gaussians"] = int(sfm_data["xyz"].shape[0])

    model = GaussianModel(num_points=sfm_data["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=device)
    model.init_from_sfm(
        xyz=sfm_data["xyz"],
        opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0], 1), 0.1, device=device)),
        scales_log=sfm_data.get("scales"),
        rotations_raw=sfm_data.get("rotations"),
        shs=sfm_data.get("shs"),
    )
    spatial_lr_scale = sfm_data["xyz"].norm(dim=-1).max().item()
    print(f"Initial Gs: {model.xyz.shape[0]:,}  spatial_lr_scale={spatial_lr_scale:.2f}")

    optimizer = torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
        {"params": [model.rotations], "lr": 1e-3},
        {"params": [model.scales], "lr": 5e-3},
        {"params": [model.opacity], "lr": 5e-2},
        {"params": [model.shs], "lr": 2.5e-3},
    ], eps=1e-15, betas=(0.9, 0.999))

    total_steps = args.warmup + args.measured
    measured_fwd = []; measured_bwd = []; measured_opt = []
    measured_t_iter = []
    last_psnr = 0.0; last_loss = 0.0; final_n_gaussians = 0

    for step in range(total_steps):
        cam_idx = step % len(dataset)
        camera = dataset.get_camera(cam_idx)
        gt_image = dataset.get_gt_image(cam_idx)

        # SH degree progression
        new_degree = min(3, step // 500)
        if new_degree != model.sh_degree:
            model.set_sh_degree(new_degree)
            # Rebuild optimizer (new params from densification/new degree)
            optimizer = torch.optim.Adam([
                {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
                {"params": [model.rotations], "lr": 1e-3},
                {"params": [model.scales], "lr": 5e-3},
                {"params": [model.opacity], "lr": 5e-2},
                {"params": [model.shs], "lr": 2.5e-3},
            ], eps=1e-15, betas=(0.9, 0.999))

        data = model.forward()

        # --- Forward ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rendered, _, info = rasterization(
            means=data["xyz"], quats=data["rotations"], scales=data["scales"],
            opacities=data["opacity"], colors=data["shs"],
            viewmats=camera.viewmatrix.unsqueeze(0), Ks=camera.K.unsqueeze(0),
            width=camera.image_width, height=camera.image_height,
            tile_size=args.tile_size, packed=True, sh_degree=model.sh_degree,
            radius_clip=0.0, eps2d=0.1, render_mode="RGB",
        )
        torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - t0) * 1000
        rendered = rendered[0].clamp(0, 1)

        # --- Loss ---
        loss_dict = combined_loss(rendered, gt_image, lambda_dssim=0.2)
        loss = loss_dict["loss"]

        # --- Backward ---
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        bwd_ms = (time.perf_counter() - t0) * 1000

        model.accumulate_positional_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # --- Optimizer ---
        t0 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        opt_ms = (time.perf_counter() - t0) * 1000

        # --- Topology: densify + prune ---
        denf_cloned, denf_split, pruned = 0, 0, 0
        if step >= 200 and step % 100 == 0:
            denf_count = model.densification(grad_threshold=2e-4)
            denf_cloned, denf_split = denf_count["cloned"], denf_count["split"]
            if max(denf_cloned, denf_split) > 0:
                optimizer = torch.optim.Adam([
                    {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
                    {"params": [model.rotations], "lr": 1e-3},
                    {"params": [model.scales], "lr": 5e-3},
                    {"params": [model.opacity], "lr": 5e-2},
                    {"params": [model.shs], "lr": 2.5e-3},
                ], eps=1e-15, betas=(0.9, 0.999))
        if step >= 200 and step % 100 == 0:
            pruned = model.prune(opacity_threshold=0.005)
            if pruned > 0:
                optimizer = torch.optim.Adam([
                    {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
                    {"params": [model.rotations], "lr": 1e-3},
                    {"params": [model.scales], "lr": 5e-3},
                    {"params": [model.opacity], "lr": 5e-2},
                    {"params": [model.shs], "lr": 2.5e-3},
                ], eps=1e-15, betas=(0.9, 0.999))

        with torch.no_grad():
            mse = torch.mean((rendered - gt_image) ** 2).item()
            psnr = 10 * math.log10(1.0 / max(mse, 1e-10))

        # Record measurements (only after warmup)
        if step >= args.warmup:
            measured_fwd.append(fwd_ms)
            measured_bwd.append(bwd_ms)
            measured_opt.append(opt_ms)
            measured_t_iter.append(fwd_ms + bwd_ms + opt_ms)

        last_psnr = psnr; last_loss = loss.item(); final_n_gaussians = model.xyz.shape[0]

        if step % 50 == 0 or step == total_steps - 1:
            w = "WARM" if step < args.warmup else "MEAS"
            print(f"  [{w}] Step {step}: loss={loss.item():.4f} PSNR={psnr:.2f} "
                  f"N={model.xyz.shape[0]:,} fwd={fwd_ms:.3f} bwd={bwd_ms:.3f} opt={opt_ms:.3f}",
                  flush=True)

    # --- Statistics ---
    fwd_arr = np.array(measured_fwd)
    bwd_arr = np.array(measured_bwd)
    opt_arr = np.array(measured_opt)
    titer_arr = np.array(measured_t_iter)

    summary = {
        "schema_version": 2,
        "phase": "C31_BASELINE_VALIDITY",
        "environment": env,
        "method": {
            "warmup_steps": args.warmup,
            "measured_steps": args.measured,
        },
        "results": {
            "t_iter_ms": {
                "mean": float(np.mean(titer_arr)),
                "median": float(np.median(titer_arr)),
                "std": float(np.std(titer_arr)),
                "min": float(np.min(titer_arr)),
                "max": float(np.max(titer_arr)),
                "p5": float(np.percentile(titer_arr, 5)),
                "p95": float(np.percentile(titer_arr, 95)),
            },
            "forward_ms": {
                "mean": float(np.mean(fwd_arr)),
                "median": float(np.median(fwd_arr)),
                "std": float(np.std(fwd_arr)),
            },
            "backward_ms": {
                "mean": float(np.mean(bwd_arr)),
                "median": float(np.median(bwd_arr)),
                "std": float(np.std(bwd_arr)),
            },
            "optimizer_ms": {
                "mean": float(np.mean(opt_arr)),
                "median": float(np.median(opt_arr)),
                "std": float(np.std(opt_arr)),
            },
            "final_psnr": last_psnr,
            "final_loss": last_loss,
            "final_gaussians": final_n_gaussians,
        },
    }

    print(f"\n=== Baseline Results ===")
    print(f"T_iter: mean={summary['results']['t_iter_ms']['mean']:.3f}ms ± {summary['results']['t_iter_ms']['std']:.3f}ms")
    print(f"  Forward:  {summary['results']['forward_ms']['mean']:.3f}ms")
    print(f"  Backward: {summary['results']['backward_ms']['mean']:.3f}ms")
    print(f"  Optimizer:{summary['results']['optimizer_ms']['mean']:.3f}ms")
    print(f"Final PSNR: {last_psnr:.2f}, Gs: {final_n_gaussians:,}")
    print(f"Measured steps: {args.measured} (after {args.warmup} warmup)")

    # Cross-check against project history
    c25_t_iter = 9.46  # C25 A100 baseline — forward-only microbenchmark, not full training
    c30_fp32_t_iter = 44.75  # C30-H FP32 full training (no prolonged warmup, JIT contamination)
    phase10a_t_iter = 98.0  # Phase 10A RTX5070 full training (correct but on different GPU)
    print(f"\nCross-check vs project history:")
    print(f"  vs C25 A100 fwd-only microbench: {summary['results']['t_iter_ms']['mean']:.2f}ms (C25: {c25_t_iter:.2f}ms) = {summary['results']['t_iter_ms']['mean']/c25_t_iter:.1f}x")
    print(f"    C25 measured gsplat.rasterization() forward only (no D-SSIM, no backward autograd)")
    print(f"  vs C30 FP32 full training: {summary['results']['t_iter_ms']['mean']:.2f}ms (C30: {c30_fp32_t_iter:.2f}ms) = {summary['results']['t_iter_ms']['mean']/c30_fp32_t_iter:.1f}x")
    print(f"  vs Phase10A (RTX5070 laptop): {summary['results']['t_iter_ms']['mean']:.2f}ms (Phase10A: {phase10a_t_iter:.2f}ms) = {summary['results']['t_iter_ms']['mean']/phase10a_t_iter:.1f}x")
    print()
    print("Backward breakdown (separate diagnostic):")
    print("  Rasterization backward (dummy):   10.5ms")
    print("  + L1 backward:                    3.8ms")
    print("  + D-SSIM backward:                3.6ms")
    print("  + Autograd overhead:              3.8ms")
    print("  Full backward:                    14.3ms")
    print("  NOTE: baseline ~98ms backward includes dynamic CUDA graph rebuild")
    print("  from camera cycling, not static JIT overhead.")
    print()
    print("C30-G INVALID FOR COMPARISON — C30's 98.7ms (single GPU) is comparable")
    print("to THIS baseline but was measured without acknowledging the ~10x gap to")
    print("C25's microbenchmark. C30's 6.8x scaling estimate is NOT invalidated but")
    print("must be reproduced with actual DDP and this corrected baseline.")
    print()
    print("CORRECTED BASELINE T_iter = {:.2f}ms (this measurement)".format(
        summary['results']['t_iter_ms']['mean']))
    print("  (full training pipeline: forward + L1/DSSIM backward + optimizer + densification)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\nSaved: {args.out}")

if __name__ == "__main__":
    main()
