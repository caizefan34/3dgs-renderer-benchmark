#!/usr/bin/env python3
"""
Phase 9A — M2 packed/dense 500-step training sanity (standalone).
Runs packed=True and packed=False sequentially on room scene with real GT.
"""

import json, math, sys, gc, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
from gsplat import rasterization

device = "cuda"
scene = "room"
steps = 500
tile_size = 16

# Load cameras
cameras = load_cameras_from_json(str(REPO_ROOT / "data" / "official" / "mipnerf360" / scene / "cameras.json"))
cameras = resize_cameras(cameras, 1920, 1080)

# Load GT image list
gt_dir = REPO_ROOT / "data" / "datasets" / "mipnerf360" / scene / "images"
gt_files = sorted(gt_dir.glob("*.jpg")) + sorted(gt_dir.glob("*.png"))
print(f"  {len(gt_files)} GT images, {len(cameras)} cameras")

# Load initial SfM data
data = load_ply(str(REPO_ROOT / "data" / "official" / "mipnerf360" / scene / "point_cloud.ply"), device=device)

def get_gt_image(idx):
    from PIL import Image
    img = Image.open(gt_files[idx]).convert("RGB")
    img = img.resize((1920, 1080), Image.LANCZOS)
    return torch.tensor(np.array(img), device=device, dtype=torch.float32) / 255.0

results = {
    "experiment_id": f"phase9a_m2_sanity_{scene}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    "date": datetime.now(timezone.utc).isoformat(),
    "config": {"scene": scene, "steps": steps, "tile_size": tile_size, "resolution": "1080p"},
    "modes": {},
}

for packed in [True, False]:
    label = "packed" if packed else "dense"
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"\n{'='*60}")
    print(f"  M2 Sanity: {label.upper()} mode")
    print(f"{'='*60}")

    # Model
    xyz = data["xyz"].clone().detach().requires_grad_(True)
    quats = torch.nn.functional.normalize(data["rotations"].clone().detach(), dim=-1).requires_grad_(True)
    scales_log = data["scales"].clone().detach().requires_grad_(True)
    opacity_logit = torch.logit(torch.full((xyz.shape[0],), 0.1, device=device)).requires_grad_(True)
    shs = data["shs"].clone().detach().requires_grad_(True)

    def get_params():
        return {
            "xyz": xyz, "rotations": quats, "scales_log": scales_log,
            "opacity": torch.sigmoid(opacity_logit), "shs": shs,
            "opacity_logit": opacity_logit,
        }

    spatial_lr_scale = xyz.norm(dim=-1).max().item()
    optimizer = torch.optim.Adam([
        {"params": [xyz], "lr": 1.6e-4 * spatial_lr_scale},
        {"params": [quats], "lr": 1e-3},
        {"params": [scales_log], "lr": 5e-3},
        {"params": [opacity_logit], "lr": 5e-2},
        {"params": [shs], "lr": 2.5e-3},
    ])

    nan_detected = False
    inf_detected = False
    fwd_times = []
    bwd_times = []
    metrics = []
    sh_degree = 0

    t0 = time.perf_counter()
    for step in range(steps):
        cam_idx = step % len(cameras)
        cam = cameras[cam_idx]
        vm = cam.viewmatrix.unsqueeze(0).to(device)
        K = cam.K.unsqueeze(0).to(device)
        gt = get_gt_image(cam_idx)

        # SH scheduling
        new_deg = min(3, step // 500)
        if new_deg != sh_degree:
            sh_degree = new_deg
            optimizer = torch.optim.Adam([
                {"params": [xyz], "lr": 1.6e-4 * spatial_lr_scale},
                {"params": [quats], "lr": 1e-3},
                {"params": [scales_log], "lr": 5e-3},
                {"params": [opacity_logit], "lr": 5e-2},
                {"params": [shs], "lr": 2.5e-3},
            ])
            print(f"    [Step {step}] SH degree -> {sh_degree}")

        scales_a = torch.exp(scales_log)
        opacities_a = torch.sigmoid(opacity_logit).squeeze(-1)

        # Forward
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        ev_s.record()
        rendered, _, _ = rasterization(
            means=xyz, quats=quats, scales=scales_a, opacities=opacities_a,
            colors=shs, viewmats=vm, Ks=K,
            width=cam.image_width, height=cam.image_height,
            tile_size=tile_size, packed=packed, sh_degree=sh_degree,
            render_mode="RGB",
        )
        ev_e.record()
        rendered = rendered[0].clamp(0, 1)

        # Loss
        l1 = torch.abs(rendered - gt).mean()
        ssim_val = 1 - ((2 * rendered.mean() * gt.mean() + 0.01) /
                        (rendered.mean()**2 + gt.mean()**2 + 0.01))
        loss = (1 - 0.2) * l1 + 0.2 * ssim_val

        # Backward
        ev_bwd_s = torch.cuda.Event(enable_timing=True)
        ev_bwd_e = torch.cuda.Event(enable_timing=True)
        ev_bwd_s.record()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize()
        ev_bwd_e.record()

        # Check NaN/Inf
        for name, p in [("xyz", xyz), ("quats", quats), ("scales", scales_log),
                        ("opacity", opacity_logit), ("shs", shs)]:
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    nan_detected = True; print(f"  WARNING: NaN grad in {name} at step {step}")
                if torch.isinf(p.grad).any():
                    inf_detected = True; print(f"  WARNING: Inf grad in {name} at step {step}")
            if torch.isnan(p).any() or torch.isinf(p).any():
                nan_detected = True; print(f"  WARNING: NaN param in {name} at step {step}")

        # Gradient accumulation (for densification)
        if xyz.grad is not None:
            if not hasattr(xyz, 'absgrad'):
                xyz.absgrad = torch.zeros_like(xyz)
            xyz.absgrad += xyz.grad.abs()

        grad_norm = torch.nn.utils.clip_grad_norm_([xyz, quats, scales_log, opacity_logit, shs], max_norm=1.0)
        ev_opt_s = torch.cuda.Event(enable_timing=True)
        ev_opt_e = torch.cuda.Event(enable_timing=True)
        ev_opt_s.record()
        optimizer.step()
        torch.cuda.synchronize()
        ev_opt_e.record()

        # Sync timings
        ev_e.synchronize()
        ev_bwd_e.synchronize()
        ev_opt_e.synchronize()
        fwd_ms = ev_s.elapsed_time(ev_e)
        bwd_ms = ev_bwd_s.elapsed_time(ev_bwd_e)

        # Densification
        if step >= 200 and step % 100 == 0:
            with torch.no_grad():
                grad_abs = xyz.absgrad
                grad_thresh = 2e-4
                mask_clone = (grad_abs.mean(dim=-1) >= grad_thresh) & (torch.sigmoid(opacity_logit).squeeze(-1) >= 0.005)
                n_clone = mask_clone.sum().item()
                if n_clone > 0:
                    xyz_clone = xyz[mask_clone].clone()
                    opacity_logit_clone = opacity_logit[mask_clone].clone()
                    scales_log_clone = scales_log[mask_clone].clone()
                    quats_clone = quats[mask_clone].clone()
                    shs_clone = shs[mask_clone].clone()

                    xyz = torch.cat([xyz, xyz_clone], dim=0).requires_grad_(True)
                    opacity_logit = torch.cat([opacity_logit, opacity_logit_clone], dim=0).requires_grad_(True)
                    scales_log = torch.cat([scales_log, scales_log_clone], dim=0).requires_grad_(True)
                    quats = torch.cat([quats, quats_clone], dim=0).requires_grad_(True)
                    shs = torch.cat([shs, shs_clone], dim=0).requires_grad_(True)
                    xyz.absgrad = torch.zeros_like(xyz)

            # Pruning
            with torch.no_grad():
                opacity_prune = torch.sigmoid(opacity_logit).squeeze(-1)
                prune_mask = opacity_prune < 0.005
                n_prune = prune_mask.sum().item()
                if n_prune > 0:
                    keep = ~prune_mask
                    xyz = xyz[keep].detach().requires_grad_(True)
                    opacity_logit = opacity_logit[keep].detach().requires_grad_(True)
                    scales_log = scales_log[keep].detach().requires_grad_(True)
                    quats = quats[keep].detach().requires_grad_(True)
                    shs = shs[keep].detach().requires_grad_(True)
                    xyz.absgrad = torch.zeros_like(xyz)

            optimizer = torch.optim.Adam([
                {"params": [xyz], "lr": 1.6e-4 * spatial_lr_scale},
                {"params": [quats], "lr": 1e-3},
                {"params": [scales_log], "lr": 5e-3},
                {"params": [opacity_logit], "lr": 5e-2},
                {"params": [shs], "lr": 2.5e-3},
            ])

        fwd_times.append(fwd_ms)
        bwd_times.append(bwd_ms)
        mse = torch.mean((rendered - gt) ** 2).item()
        psnr = 10 * math.log10(1.0 / max(mse, 1e-10)) if mse > 0 else 100.0

        metrics.append({
            "step": step, "loss": loss.item(), "l1": l1.item(),
            "psnr": psnr, "grad_norm": grad_norm,
            "gaussians": xyz.shape[0],
            "fwd_ms": fwd_ms, "bwd_ms": bwd_ms,
        })

        if step % 100 == 0 or step == steps - 1:
            print(f"    Step {step:4d}: loss={loss.item():.4f} PSNR={psnr:.2f}dB N={xyz.shape[0]:,} fwd={fwd_ms:.1f}ms bwd={bwd_ms:.1f}ms")

    total_time = time.perf_counter() - t0
    print(f"\n  [{label.upper()}] Complete: {total_time:.1f}s, {total_time*1000/steps:.1f}ms/step")
    print(f"    Initial: {metrics[0]['gaussians']:,} → Final: {xyz.shape[0]:,}")
    print(f"    Best PSNR: {max(m['psnr'] for m in metrics):.2f}dB")
    print(f"    NaN: {nan_detected}, Inf: {inf_detected}")
    print(f"    Avg fwd: {np.mean(fwd_times):.2f}ms, Avg bwd: {np.mean(bwd_times):.2f}ms")

    results["modes"][label] = {
        "total_time_s": total_time,
        "avg_step_ms": total_time * 1000 / steps,
        "avg_fwd_ms": float(np.mean(fwd_times)),
        "avg_bwd_ms": float(np.mean(bwd_times)),
        "initial_gaussians": metrics[0]["gaussians"],
        "final_gaussians": xyz.shape[0],
        "best_psnr_db": float(max(m["psnr"] for m in metrics)),
        "final_psnr_db": float(metrics[-1]["psnr"]),
        "nan_detected": nan_detected,
        "inf_detected": inf_detected,
        "trajectory": metrics,
    }

    del xyz, quats, scales_log, opacity_logit, shs, optimizer
    gc.collect()
    torch.cuda.empty_cache()

# Comparison
print(f"\n{'='*60}")
print(f"  M2 COMPARISON SUMMARY")
print(f"{'='*60}")
pr = results["modes"]["packed"]
dr = results["modes"]["dense"]
checks = 0
passed = 0

if not pr["nan_detected"] and not dr["nan_detected"]:
    print(f"  ✅ [1/5] No NaN"); passed += 1
else: print(f"  ❌ [1/5] NaN detected")
checks += 1

if not pr["inf_detected"] and not dr["inf_detected"]:
    print(f"  ✅ [2/5] No Inf"); passed += 1
else: print(f"  ❌ [2/5] Inf detected")
checks += 1

psnr_diff = abs(pr["best_psnr_db"] - dr["best_psnr_db"])
if psnr_diff < 1.0:
    print(f"  ✅ [3/5] PSNR consistent (Δ={psnr_diff:.2f} dB)"); passed += 1
else: print(f"  ❌ [3/5] PSNR differs (Δ={psnr_diff:.2f} dB)")
checks += 1

gauss_diff = abs(pr["final_gaussians"] - dr["final_gaussians"]) / max(pr["final_gaussians"], 1)
if gauss_diff < 0.05:
    print(f"  ✅ [4/5] Gaussian count similar (Δ={gauss_diff*100:.2f}%)"); passed += 1
else: print(f"  ❌ [4/5] Gaussian count diverges (Δ={gauss_diff*100:.2f}%)")
checks += 1

print(f"  {'✅' if passed >= 4 else '❌'} [{passed}/{checks}] Verdict: {'PASS' if passed >= 4 else 'FAIL'}")
results["comparison"] = {
    "checks_passed": passed, "total_checks": checks,
    "verdict": "PASS" if passed >= 4 else "FAIL",
    "psnr_delta_db": float(f"{psnr_diff:.3f}"),
    "gaussian_delta_pct": float(f"{gauss_diff*100:.2f}"),
    "fwd_speedup": float(f"{dr['avg_fwd_ms']/max(pr['avg_fwd_ms'],0.01):.3f}"),
}
results["eligible_for_full_training"] = passed >= 4

out_dir = REPO_ROOT / "results" / "epic05" / "phase9a"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"m2_sanity_{scene}_{steps}steps.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n  Saved: {out_path}")
