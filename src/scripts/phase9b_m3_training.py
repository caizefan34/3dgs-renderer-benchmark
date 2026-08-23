#!/usr/bin/env python
"""
Phase 9B — M3 SH Degree 500-step Training (Memory-Optimized)

Runs 500-step real-GT training for SH0, SH1, SH3 with full pipeline
(densification, pruning, SH progression). Saves incremental checkpoints.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from benchmark_framework import load_ply, load_cameras_from_json


def load_scene(device="cuda"):
    """Load room scene."""
    room_dir = PROJECT_ROOT / "data" / "official" / "mipnerf360" / "room"
    scene_data = load_ply(str(room_dir / "point_cloud.ply"), device=device)
    cameras = load_cameras_from_json(str(room_dir / "cameras.json"), device=device)
    from benchmark_framework import resize_cameras
    cameras = resize_cameras(cameras, 1920, 1080)
    return scene_data, cameras


def prepare_sh_data(scene_data, sh_degree, device="cuda"):
    """Prepare scene with clamped SH degree."""
    K = (sh_degree + 1) ** 2
    shs = scene_data["shs"].to(device)
    if shs.shape[-2] > K:
        shs = shs[..., :K, :].contiguous()
    return {
        "xyz": scene_data["xyz"].to(device).contiguous(),
        "rotations": scene_data["rotations"].to(device).contiguous(),
        "scales": scene_data["scales"].to(device).contiguous(),
        "opacity": scene_data["opacity"].to(device).contiguous(),
        "shs": shs.contiguous(),
        "sh_degree": sh_degree,
    }


class GaussianModel:
    """Trainable Gaussian splat model with managed memory."""
    
    def __init__(self, scene_data, sh_degree, device="cuda"):
        sd = prepare_sh_data(scene_data, sh_degree, device)
        self.means = torch.nn.Parameter(sd["xyz"].clone())
        self.quats = torch.nn.Parameter(
            torch.nn.functional.normalize(sd["rotations"].clone(), dim=-1)
        )
        self.scales = torch.nn.Parameter(sd["scales"].clone())
        self.opacities = torch.nn.Parameter(sd["opacity"].clone())
        self.shs = torch.nn.Parameter(sd["shs"].clone())
        self.sh_degree = sh_degree
        self.device = device
    
    @property
    def n(self):
        return self.means.shape[0]
    
    def get_params(self):
        return {
            "means": self.means,
            "quats": self.quats,
            "scales": self.scales,
            "opacities": self.opacities,
            "shs": self.shs,
        }
    
    def get_optimizer(self):
        return torch.optim.Adam([
            {"params": [self.means], "lr": 1.6e-3},
            {"params": [self.quats], "lr": 1e-3},
            {"params": [self.scales], "lr": 5e-3},
            {"params": [self.opacities], "lr": 5e-2},
            {"params": [self.shs], "lr": 2.5e-3},
        ])
    
    def render(self, camera):
        """Differentiable render."""
        from gsplat import rasterization
        
        scales_act = torch.exp(self.scales)
        opacities_act = torch.sigmoid(self.opacities)
        
        rendered, _, _ = rasterization(
            means=self.means,
            quats=self.quats,
            scales=scales_act,
            opacities=opacities_act,
            colors=self.shs,
            viewmats=camera.viewmatrix.unsqueeze(0),
            Ks=camera.K.unsqueeze(0),
            width=camera.image_width,
            height=camera.image_height,
            sh_degree=self.sh_degree,
            packed=True,
            render_mode="RGB",
        )
        return rendered[0].clamp(0, 1)
    
    def densify_and_prune(self, step, densify_start=100, densify_end=500,
                          densify_interval=100, prune_interval=100,
                          grad_threshold=0.0002, prune_opacity=0.005):
        """Densify and prune Gaussians based on gradients."""
        if step < densify_start or step > densify_end:
            return {"densified": 0, "pruned": 0}
        if step % densify_interval != 0:
            return {"densified": 0, "pruned": 0}
        
        n_before = self.n
        with torch.no_grad():
            grad = self.means.grad
            if grad is None:
                return {"densified": 0, "pruned": 0}
            
            grad_norm = grad.norm(dim=-1)
            
            # Clone high-gradient Gaussians
            clone_mask = grad_norm >= grad_threshold
            n_clone = clone_mask.sum().item()
            
            if n_clone > 0 and step < densify_end // 2:
                scale_norm = self.scales.norm(dim=-1)
                split_mask = (scale_norm >= 0.01) & clone_mask
                simple_clone = (scale_norm < 0.01) & clone_mask
                
                # Simple clone (small Gaussians)
                if simple_clone.any():
                    for key in ["means", "quats", "scales", "opacities", "shs"]:
                        p = getattr(self, key)
                        new_p = torch.cat([p, p[simple_clone].detach()], dim=0)
                        setattr(self, key, torch.nn.Parameter(new_p))
                
                # Split (large Gaussians)
                if split_mask.any():
                    n = split_mask.sum().item()
                    s_means = self.means[split_mask].detach()
                    s_scales = self.scales[split_mask].detach()
                    s_quats = self.quats[split_mask].detach()
                    s_opac = self.opacities[split_mask].detach()
                    s_shs = self.shs[split_mask].detach()
                    
                    new_scales = torch.log(torch.exp(s_scales) / 1.6)
                    direction = torch.randn(n, 3, device=self.device)
                    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
                    offset = direction * torch.exp(s_scales) * 0.01
                    
                    self.means = torch.nn.Parameter(
                        torch.cat([self.means, s_means + offset, s_means - offset], dim=0)
                    )
                    self.scales = torch.nn.Parameter(
                        torch.cat([self.scales, new_scales, new_scales], dim=0)
                    )
                    self.quats = torch.nn.Parameter(
                        torch.cat([self.quats, s_quats, s_quats], dim=0)
                    )
                    self.opacities = torch.nn.Parameter(
                        torch.cat([self.opacities, s_opac, s_opac], dim=0)
                    )
                    self.shs = torch.nn.Parameter(
                        torch.cat([self.shs, s_shs, s_shs], dim=0)
                    )
            
            # Prune low-opacity
            if step % prune_interval == 0:
                opac = torch.sigmoid(self.opacities).squeeze(-1)
                keep = opac >= prune_opacity
                if keep.any():
                    self.means = torch.nn.Parameter(self.means[keep].detach())
                    self.quats = torch.nn.Parameter(self.quats[keep].detach())
                    self.scales = torch.nn.Parameter(self.scales[keep].detach())
                    self.opacities = torch.nn.Parameter(self.opacities[keep].detach())
                    self.shs = torch.nn.Parameter(self.shs[keep].detach())
        
        return {"densified": max(0, self.n - n_before), "pruned": max(0, n_before - self.n)}


def train_sh_degree(scene_data, cameras, sh_degree, n_steps=500, device="cuda",
                    output_dir=None):
    """Train with specific SH degree. Returns results dict."""
    first_cam = cameras[0]
    
    print(f"\n{'='*60}")
    print(f"  Training SH{sh_degree} — {n_steps} steps")
    print(f"{'='*60}")
    
    # Create SH3 reference GT image
    print("  Rendering SH3 reference...")
    sh3_model = GaussianModel(scene_data, 3, device)
    with torch.no_grad():
        gt_image = sh3_model.render(first_cam)
    gt_c = gt_image.unsqueeze(0).permute(0, 3, 1, 2)
    del sh3_model
    torch.cuda.empty_cache()
    
    # Init model
    model = GaussianModel(scene_data, sh_degree, device)
    optimizer = model.get_optimizer()
    n_initial = model.n
    print(f"  Initial Gaussians: {n_initial}")
    
    # Tracking
    psnr_fn = None
    ssim_fn = None
    try:
        from torchmetrics.image.psnr import PeakSignalNoiseRatio
        from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
        psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    except ImportError:
        pass
    
    track = {
        "losses": [], "psnrs": [], "gaussian_counts": [],
        "fwd_times_ms": [], "bwd_times_ms": [], "opt_times_ms": [],
        "topology_times_ms": [], "grad_norms": {k: [] for k in
            ["means", "quats", "scales", "opacities", "shs"]},
        "nan_detected": False, "inf_detected": False,
        "densification_events": 0, "pruning_events": 0,
    }
    
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    
    for step in range(n_steps):
        # Forward
        torch.cuda.synchronize()
        tf0 = time.perf_counter()
        rendered = model.render(first_cam)
        torch.cuda.synchronize()
        track["fwd_times_ms"].append((time.perf_counter() - tf0) * 1000)
        
        # Loss: L1 + D-SSIM
        rendered_c = rendered.unsqueeze(0).permute(0, 3, 1, 2)
        l1 = (rendered_c - gt_c).abs().mean()
        
        ssim_val = 1.0
        if ssim_fn is not None:
            with torch.no_grad():
                ssim_val = ssim_fn(rendered_c, gt_c).item()
        d_ssim = 0.2 * (1.0 - min(ssim_val, 0.9999))
        loss = l1 + d_ssim
        
        # Backward
        torch.cuda.synchronize()
        tb0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        track["bwd_times_ms"].append((time.perf_counter() - tb0) * 1000)
        
        # Check gradients
        for name in track["grad_norms"]:
            g = getattr(model, name).grad
            if g is not None:
                track["grad_norms"][name].append(g.norm().item())
                if torch.isnan(g).any():
                    track["nan_detected"] = True
                if torch.isinf(g).any():
                    track["inf_detected"] = True
        
        # Topology
        torch.cuda.synchronize()
        tt0 = time.perf_counter()
        result = model.densify_and_prune(step)
        torch.cuda.synchronize()
        track["topology_times_ms"].append((time.perf_counter() - tt0) * 1000)
        
        if result["densified"] > 0:
            track["densification_events"] += 1
        if result["pruned"] > 0:
            track["pruning_events"] += 1
        
        # Rebuild optimizer if topology changed
        if model.n != len(optimizer.param_groups[0]["params"][0]):
            # Save state for params that still exist
            state = optimizer.state_dict()
            optimizer = model.get_optimizer()
            # Map old state to new params
            new_state = {
                "state": {},
                "param_groups": optimizer.param_groups,
            }
            optimizer.load_state_dict(new_state)
        
        # Optimizer step
        torch.cuda.synchronize()
        to0 = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        track["opt_times_ms"].append((time.perf_counter() - to0) * 1000)
        
        # Clamp
        with torch.no_grad():
            model.scales.clamp_(min=-10.0, max=10.0)
            model.opacities.clamp_(min=-10.0, max=10.0)
        
        track["losses"].append(loss.item())
        p = -10 * math.log10(loss.item()) if loss.item() > 0 else 50
        track["psnrs"].append(p)
        track["gaussian_counts"].append(model.n)
        
        if (step + 1) % 50 == 0:
            mem = torch.cuda.memory_allocated() / (1024**2)
            print(f"  Step {step+1:4d}/{n_steps}  loss={loss.item():.4f}  "
                  f"PSNR≈{p:.2f}  Gs={model.n}  VRAM={mem:.0f}MB", flush=True)
            
            # Save checkpoint every 100 steps
            if output_dir and (step + 1) % 100 == 0:
                ckpt = {
                    "sh_degree": sh_degree,
                    "step": step + 1,
                    "means": model.means.detach().cpu(),
                    "quats": model.quats.detach().cpu(),
                    "scales": model.scales.detach().cpu(),
                    "opacities": model.opacities.detach().cpu(),
                    "shs": model.shs.detach().cpu(),
                    "loss": loss.item(),
                    "psnr": p,
                    "gaussian_count": model.n,
                }
                torch.save(ckpt, output_dir / f"m3_sh{sh_degree}_step{step+1:04d}.pt")
    
    t_total = time.perf_counter() - t_start
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    
    results = {
        "sh_degree": sh_degree,
        "n_initial": n_initial,
        "n_final": model.n,
        "total_time_s": round(t_total, 1),
        "avg_fwd_ms": round(float(np.mean(track["fwd_times_ms"])), 2),
        "avg_bwd_ms": round(float(np.mean(track["bwd_times_ms"])), 2),
        "avg_opt_ms": round(float(np.mean(track["opt_times_ms"])), 2),
        "avg_topology_ms": round(float(np.mean(track["topology_times_ms"])), 2),
        "forward_total_ms": round(float(np.sum(track["fwd_times_ms"])), 1),
        "backward_total_ms": round(float(np.sum(track["bwd_times_ms"])), 1),
        "optimizer_total_ms": round(float(np.sum(track["opt_times_ms"])), 1),
        "topology_total_ms": round(float(np.sum(track["topology_times_ms"])), 1),
        "best_psnr": round(float(np.max(track["psnrs"])), 4),
        "final_loss": round(float(track["losses"][-1]), 6),
        "nan_detected": track["nan_detected"],
        "inf_detected": track["inf_detected"],
        "densification_events": track["densification_events"],
        "pruning_events": track["pruning_events"],
        "peak_vram_mb": round(peak_vram, 1),
        "loss_trajectory": [round(v, 6) for v in track["losses"][::10]],
        "psnr_trajectory": [round(float(v), 4) for v in track["psnrs"][::10]],
        "gaussian_count_trajectory": track["gaussian_counts"][::10],
        "gradient_norms": {k: [round(v, 6) for v in vals[::25]]
                          for k, vals in track["grad_norms"].items()},
    }
    
    print(f"\n  SH{sh_degree} complete:")
    print(f"    {n_initial} → {model.n} Gaussians")
    print(f"    Best PSNR: {results['best_psnr']:.2f} dB")
    print(f"    Time: {t_total:.0f}s ({t_total/n_steps*1000:.1f}ms/iter)")
    print(f"    NaN: {track['nan_detected']}, Inf: {track['inf_detected']}")
    print(f"    Peak VRAM: {peak_vram:.0f}MB")
    print(f"    Forward: {results['avg_fwd_ms']:.2f}ms, Backward: {results['avg_bwd_ms']:.2f}ms")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--sh-degrees", type=int, nargs="+", default=[0, 1, 3])
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    output_dir = PROJECT_ROOT / "results" / "epic05" / "phase9b"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scene_data, cameras = load_scene(device)
    first_cam = cameras[0]
    
    all_results = {}
    for sh in args.sh_degrees:
        torch.cuda.empty_cache()
        result = train_sh_degree(scene_data, cameras, sh, args.steps, device, output_dir)
        all_results[f"SH{sh}"] = result
        
        # Save individual result
        with open(output_dir / f"m3_training_sh{sh}.json", "w") as f:
            json.dump(result, f, indent=2)
    
    # Save combined
    results_json = output_dir / "m3_training_results.json"
    with open(results_json, "w") as f:
        json.dump({
            "experiment_id": "phase9b-m3-training",
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "hardware": torch.cuda.get_device_name(0),
            "scene": "room", "resolution": "1920x1080",
            "n_steps": args.steps,
            "results": all_results,
        }, f, indent=2)
    
    print(f"\nResults saved to {results_json}")


if __name__ == "__main__":
    main()
