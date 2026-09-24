#!/usr/bin/env python3
"""C30-H: Training-Phase Precision screening.

Measure T_iter and quality across FP32, TF32, and mixed precision.
Does NOT blindly reduce precision — measures gradient error vs speed.
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

def run_precision(config, out):
    device = "cuda"; precision = config["precision"]
    torch.manual_seed(42); np.random.seed(42)

    if precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif precision == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif precision == "mp":
        # Mixed precision: keep model in FP32, use autocast for forward/backward
        pass  # handled in loop

    dataset = GTDataset(scene=config["scene"], repo_root=ROOT, resolution="1080p", device=device)
    sfm_data = load_initial_checkpoint(config["scene"], ROOT, device=device)
    model = GaussianModel(num_points=sfm_data["xyz"].shape[0], sh_degree=0, max_sh_degree=3, device=device)
    model.init_from_sfm(xyz=sfm_data["xyz"],
        opacity_logit=torch.logit(torch.full((sfm_data["xyz"].shape[0], 1), 0.1, device=device)),
        scales_log=sfm_data.get("scales"), rotations_raw=sfm_data.get("rotations"), shs=sfm_data.get("shs"))
    spatial_lr_scale = sfm_data["xyz"].norm(dim=-1).max().item()
    print(f"[{precision}] Initial Gs: {model.xyz.shape[0]:,}", flush=True)

    optimizer = torch.optim.Adam([
        {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
        {"params": [model.rotations], "lr": 1e-3},
        {"params": [model.scales], "lr": 5e-3},
        {"params": [model.opacity], "lr": 5e-2},
        {"params": [model.shs], "lr": 2.5e-3},
    ])
    scaler = torch.cuda.amp.GradScaler() if precision == "mp" else None

    records = []
    for step in range(config["steps"]):
        cam_idx = step % len(dataset)
        camera = dataset.get_camera(cam_idx)
        gt_image = dataset.get_gt_image(cam_idx)
        new_degree = min(3, step // 500)
        if new_degree != model.sh_degree:
            model.set_sh_degree(new_degree)
            optimizer = torch.optim.Adam([
                {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
                {"params": [model.rotations], "lr": 1e-3},
                {"params": [model.scales], "lr": 5e-3},
                {"params": [model.opacity], "lr": 5e-2},
                {"params": [model.shs], "lr": 2.5e-3},
            ])
            if precision == "mp":
                scaler = torch.cuda.amp.GradScaler()

        data = model.forward()

        t0 = time.perf_counter()
        if precision == "mp":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                rendered, _, _ = rasterization(
                    means=data["xyz"], quats=data["rotations"], scales=data["scales"],
                    opacities=data["opacity"], colors=data["shs"],
                    viewmats=camera.viewmatrix.unsqueeze(0), Ks=camera.K.unsqueeze(0),
                    width=camera.image_width, height=camera.image_height,
                    tile_size=16, packed=True, sh_degree=model.sh_degree,
                    radius_clip=0.0, eps2d=0.1, render_mode="RGB",
                )
        else:
            rendered, _, _ = rasterization(
                means=data["xyz"], quats=data["rotations"], scales=data["scales"],
                opacities=data["opacity"], colors=data["shs"],
                viewmats=camera.viewmatrix.unsqueeze(0), Ks=camera.K.unsqueeze(0),
                width=camera.image_width, height=camera.image_height,
                tile_size=16, packed=True, sh_degree=model.sh_degree,
                radius_clip=0.0, eps2d=0.1, render_mode="RGB",
            )
        torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - t0) * 1000
        rendered = rendered[0].clamp(0, 1).float()

        loss_dict = combined_loss(rendered, gt_image, lambda_dssim=0.2)
        loss = loss_dict["loss"]

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if precision == "mp":
            scaler.scale(loss).backward()
        else:
            loss.backward()
        torch.cuda.synchronize()
        bwd_ms = (time.perf_counter() - t0) * 1000

        model.accumulate_positional_gradient()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        t0 = time.perf_counter()
        if precision == "mp":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        torch.cuda.synchronize()
        opt_ms = (time.perf_counter() - t0) * 1000

        if step >= 200 and step % 100 == 0:
            model.densification(grad_threshold=2e-4)
            optimizer = torch.optim.Adam([
                {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
                {"params": [model.rotations], "lr": 1e-3},
                {"params": [model.scales], "lr": 5e-3},
                {"params": [model.opacity], "lr": 5e-2},
                {"params": [model.shs], "lr": 2.5e-3},
            ])
            if precision == "mp":
                scaler = torch.cuda.amp.GradScaler()
        if step >= 200 and step % 100 == 0:
            model.prune(opacity_threshold=0.005)
            optimizer = torch.optim.Adam([
                {"params": [model.xyz], "lr": 1.6e-4 * spatial_lr_scale},
                {"params": [model.rotations], "lr": 1e-3},
                {"params": [model.scales], "lr": 5e-3},
                {"params": [model.opacity], "lr": 5e-2},
                {"params": [model.shs], "lr": 2.5e-3},
            ])
            if precision == "mp":
                scaler = torch.cuda.amp.GradScaler()

        with torch.no_grad():
            mse = torch.mean((rendered - gt_image) ** 2).item()
            psnr = 10 * math.log10(1.0 / max(mse, 1e-10))

        records.append({
            "step": step, "loss": loss.item(), "psnr": psnr,
            "n_gaussians": model.xyz.shape[0],
            "fwd_ms": round(fwd_ms, 3), "bwd_ms": round(bwd_ms, 3), "opt_ms": round(opt_ms, 3),
        })
        if step % 100 == 0 or step == config["steps"] - 1:
            print(f"  [{precision}] Step {step}: loss={loss.item():.4f} PSNR={psnr:.2f} N={model.xyz.shape[0]:,}", flush=True)

    return records

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True); p.add_argument("--steps", type=int, default=500)
    p.add_argument("--scene", default="room")
    args = p.parse_args()

    precisions = ["fp32", "tf32", "mp"]
    all_results = {}

    for prec in precisions:
        print(f"\n{'='*60}\n  PRECISION: {prec}\n{'='*60}")
        config = {"precision": prec, "steps": args.steps, "scene": args.scene}
        records = run_precision(config, args.out)
        all_results[prec] = records
        del records
        gc.collect()
        torch.cuda.empty_cache()

    # Compare results
    summary = {"schema_version": 2, "phase": "C30-H", "n_steps": args.steps, "scene": args.scene}
    for prec, recs in all_results.items():
        final = recs[-1]
        summary[prec] = {
            "final_psnr": final["psnr"],
            "final_loss": final["loss"],
            "final_gaussians": final["n_gaussians"],
            "mean_fwd_ms": float(np.mean([r["fwd_ms"] for r in recs])),
            "mean_bwd_ms": float(np.mean([r["bwd_ms"] for r in recs])),
            "mean_opt_ms": float(np.mean([r["opt_ms"] for r in recs])),
            "mean_t_iter_ms": float(np.mean([r["fwd_ms"]+r["bwd_ms"]+r["opt_ms"] for r in recs])),
        }

    # Relative comparison
    fp32 = summary["fp32"]
    for prec in ["tf32", "mp"]:
        s = summary[prec]
        summary[f"{prec}_vs_fp32"] = {
            "speedup": round(fp32["mean_t_iter_ms"] / max(s["mean_t_iter_ms"], 1e-10), 3),
            "psnr_delta": round(s["final_psnr"] - fp32["final_psnr"], 3),
        }
        print(f"  {prec}: {s['mean_t_iter_ms']:.2f}ms T_iter, final PSNR={s['final_psnr']:.2f} "
              f"(speedup={summary[f'{prec}_vs_fp32']['speedup']}x, PSNR Δ={summary[f'{prec}_vs_fp32']['psnr_delta']:.2f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
