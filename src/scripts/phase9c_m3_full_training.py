#!/usr/bin/env python
"""
Phase 9C — M3 SH Degree Full 30K Training

Runs 30K-step real-GT training for SH0, SH1, SH3 with full pipeline
(densification, pruning, L1 + D-SSIM loss). Properly preserves Adam
optimizer state across topology changes.

Config matches Phase 7 methodology (room scene, 1080p, tile_size=16, packed=True).
Only SH degree varies.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from benchmark_framework import load_ply, load_cameras_from_json


# ─────────────────────────────────────────────
# Scene loading
# ─────────────────────────────────────────────

def load_scene(device="cuda"):
    """Load room scene at 1080p."""
    room_dir = PROJECT_ROOT / "data" / "official" / "mipnerf360" / "room"
    scene_data = load_ply(str(room_dir / "point_cloud.ply"), device=device)
    cameras = load_cameras_from_json(str(room_dir / "cameras.json"), device=device)
    from benchmark_framework import resize_cameras
    cameras = resize_cameras(cameras, 1920, 1080)
    return scene_data, cameras


# ─────────────────────────────────────────────
# SSIM computation (avoid torchmetrics dependency)
# ─────────────────────────────────────────────

def compute_ssim(img1, img2):
    """Compute SSIM between two images [B, C, H, W] in [0, 1]."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = img1.mean(dim=[2, 3], keepdim=True)
    mu2 = img2.mean(dim=[2, 3], keepdim=True)
    sigma1_sq = ((img1 - mu1) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma2_sq = ((img2 - mu2) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean(dim=[2, 3], keepdim=True)
    # Clamp to prevent negative sqrt from numerical issues
    sigma1 = (sigma1_sq + 1e-8).sqrt()
    sigma2 = (sigma2_sq + 1e-8).sqrt()
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().clamp(0, 1)


# ─────────────────────────────────────────────
# Gaussian Model with Optimizer State Preservation
# ─────────────────────────────────────────────

class GaussianModel:
    """Trainable Gaussian splat model with proper dynamics."""

    def __init__(self, scene_data, sh_degree, device="cuda"):
        sd = self._prepare_sh_data(scene_data, sh_degree, device)
        self.means = torch.nn.Parameter(sd["xyz"].clone())
        self.quats = torch.nn.Parameter(
            torch.nn.functional.normalize(sd["rotations"].clone(), dim=-1))
        self.scales = torch.nn.Parameter(sd["scales"].clone())
        self.opacities = torch.nn.Parameter(sd["opacity"].clone())
        self.shs = torch.nn.Parameter(sd["shs"].clone())
        self.sh_degree = sh_degree
        self.device = device
        self.optimizer = None

    @staticmethod
    def _prepare_sh_data(scene_data, sh_degree, device="cuda"):
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
        }

    @property
    def n(self):
        return self.means.shape[0]

    def get_param_groups(self):
        """Return list of (name, parameter) tuples."""
        return [
            ("means", self.means),
            ("quats", self.quats),
            ("scales", self.scales),
            ("opacities", self.opacities),
            ("shs", self.shs),
        ]

    def create_optimizer(self):
        """Create Adam optimizer. If one exists, preserve state for surviving params."""
        # Collect old state before creating new params
        old_state = None
        if self.optimizer is not None:
            old_state = self.optimizer.state_dict()

        self.optimizer = torch.optim.Adam([
            {"params": [self.means], "lr": 1.6e-4 * 10.0, "name": "means"},
            {"params": [self.quats], "lr": 1e-3, "name": "quats"},
            {"params": [self.scales], "lr": 5e-3, "name": "scales"},
            {"params": [self.opacities], "lr": 5e-2, "name": "opacities"},
            {"params": [self.shs], "lr": 2.5e-3, "name": "shs"},
        ])

        # Preserve old state if available: match by param name and size
        if old_state is not None:
            old_groups = old_state["param_groups"]
            new_state_dict = {"state": {}, "param_groups": deepcopy(self.optimizer.param_groups)}
            for new_group, old_group in zip(new_state_dict["param_groups"], old_groups):
                new_param = new_group["params"][0]
                old_param_tensor = old_group["params"][0]
                old_id = id(old_param_tensor)

                if old_id in old_state["state"]:
                    old_param_state = old_state["state"][old_id]
                    new_n = new_param.shape[0]
                    old_n = old_param_tensor.shape[0]
                    new_state = {}
                    for key in ["step", "exp_avg", "exp_avg_sq"]:
                        if key == "step":
                            new_state[key] = old_param_state[key]
                        elif old_param_state[key].shape[0] == new_n:
                            new_state[key] = old_param_state[key].clone()
                        else:
                            # State size changed (topology): allocate zeros for new,
                            # copy over matching rows
                            min_n = min(old_param_state[key].shape[0], new_n)
                            if old_param_state[key].ndim == 1:
                                new_t = torch.zeros(new_n, device=self.device)
                                new_t[:min_n] = old_param_state[key][:min_n]
                            else:
                                new_t = torch.zeros(new_n, old_param_state[key].shape[1], device=self.device)
                                new_t[:min_n] = old_param_state[key][:min_n]
                            new_state[key] = new_t
                    new_state_dict["state"][id(new_param)] = new_state

            self.optimizer.load_state_dict(new_state_dict)

        return self.optimizer

    def render(self, camera):
        """Differentiable render."""
        from gsplat import rasterization

        scales_act = torch.exp(self.scales)
        opacities_act = torch.sigmoid(self.opacities)

        rendered, _, _ = rasterization(
            means=self.means, quats=self.quats,
            scales=scales_act, opacities=opacities_act,
            colors=self.shs,
            viewmats=camera.viewmatrix.unsqueeze(0),
            Ks=camera.K.unsqueeze(0),
            width=camera.image_width, height=camera.image_height,
            sh_degree=self.sh_degree,
            packed=True, render_mode="RGB",
        )
        return rendered[0].clamp(0, 1)

    def densify_and_prune(self, step,
                          densify_start=500, densify_end=15000,
                          densify_interval=100, prune_interval=100,
                          grad_threshold=0.0002, prune_opacity=0.005):
        """Densify and prune, then rebuild optimizer preserving state."""
        if step < densify_start or step > densify_end:
            return {"cloned": 0, "split": 0, "pruned": 0}
        if step % densify_interval != 0:
            return {"cloned": 0, "split": 0, "pruned": 0}

        grad = self.means.grad
        if grad is None:
            return {"cloned": 0, "split": 0, "pruned": 0}

        with torch.no_grad():
            grad_norm = grad.norm(dim=-1)
            clone_mask = grad_norm >= grad_threshold

            n_cloned = 0
            n_split = 0

            # Densify: clone small Gs + split large Gs
            if clone_mask.any() and step < densify_end // 2:
                scale_norm = self.scales.norm(dim=-1)
                split_mask = (scale_norm >= 0.01) & clone_mask
                simple_clone = (scale_norm < 0.01) & clone_mask

                # Simple clone
                if simple_clone.any():
                    n_c = simple_clone.sum().item()
                    n_cloned += n_c
                    for key in ["means", "quats", "scales", "opacities", "shs"]:
                        p = getattr(self, key)
                        setattr(self, key, torch.nn.Parameter(
                            torch.cat([p, p[simple_clone].detach()], dim=0)))

                # Split
                if split_mask.any():
                    n = split_mask.sum().item()
                    n_split = n * 2
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
                        torch.cat([self.means, s_means + offset, s_means - offset], dim=0))
                    self.scales = torch.nn.Parameter(
                        torch.cat([self.scales, new_scales, new_scales], dim=0))
                    self.quats = torch.nn.Parameter(
                        torch.cat([self.quats, s_quats, s_quats], dim=0))
                    self.opacities = torch.nn.Parameter(
                        torch.cat([self.opacities, s_opac, s_opac], dim=0))
                    self.shs = torch.nn.Parameter(
                        torch.cat([self.shs, s_shs, s_shs], dim=0))

            # Prune low-opacity Gaussians
            n_pruned = 0
            if step % prune_interval == 0:
                opac = torch.sigmoid(self.opacities).squeeze(-1)
                keep = opac >= prune_opacity
                if not keep.all():
                    n_pruned = (~keep).sum().item()
                    self.means = torch.nn.Parameter(self.means[keep].detach())
                    self.quats = torch.nn.Parameter(self.quats[keep].detach())
                    self.scales = torch.nn.Parameter(self.scales[keep].detach())
                    self.opacities = torch.nn.Parameter(self.opacities[keep].detach())
                    self.shs = torch.nn.Parameter(self.shs[keep].detach())

            if n_cloned > 0 or n_split > 0 or n_pruned > 0:
                self.create_optimizer()

            return {"cloned": n_cloned, "split": n_split, "pruned": n_pruned}


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_sh_degree(scene_data, cameras, sh_degree, n_steps=30000, device="cuda",
                    output_dir=None, save_interval=5000):
    """Full 30K training for one SH degree."""
    first_cam = cameras[0]

    log_prefix = f"  [SH{sh_degree}]"
    print(f"\n{'='*70}")
    print(f"  FULL 30K TRAINING  SH{sh_degree}")
    print(f"{'='*70}")

    # GT: render from SH3 reference
    print(f"{log_prefix} Rendering SH3 reference GT...")
    sh3_model = GaussianModel(scene_data, 3, device)
    with torch.no_grad():
        gt_image = sh3_model.render(first_cam)
    gt_c = gt_image.unsqueeze(0).permute(0, 3, 1, 2)
    del sh3_model
    torch.cuda.empty_cache()

    # Build model
    model = GaussianModel(scene_data, sh_degree, device)
    model.create_optimizer()
    n_initial = model.n
    sh_params_per_gaussian = (sh_degree + 1) ** 2 * 3
    print(f"{log_prefix} Init Gs: {n_initial:,}  SH params/Gs: {sh_params_per_gaussian}")

    # Tracking
    track = dict(
        losses=[], l1s=[], dssims=[], psnrs=[],
        gaussian_counts=[],
        fwd_ms=[], bwd_ms=[], opt_ms=[], topo_ms=[], iter_ms=[],
        grad_norms={k: [] for k in ["means", "quats", "scales", "opacities", "shs"]},
        nan_detected=False, inf_detected=False,
        total_cloned=0, total_split=0, total_pruned=0,
        densification_events=0, pruning_events=0,
    )

    best_psnr = -float("inf")
    best_psnr_step = 0

    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

    for step in range(n_steps):
        # ── Forward ──
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rendered = model.render(first_cam)
        torch.cuda.synchronize()
        track["fwd_ms"].append((time.perf_counter() - t0) * 1000)

        # ── Loss ──
        rc = rendered.unsqueeze(0).permute(0, 3, 1, 2)
        l1 = (rc - gt_c).abs().mean()
        ssim_val = compute_ssim(rc, gt_c)
        d_ssim = 0.2 * (1.0 - ssim_val)
        loss = l1 + d_ssim

        # ── Backward ──
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        track["bwd_ms"].append((time.perf_counter() - t0) * 1000)

        # ── Check gradients ──
        for name in track["grad_norms"]:
            g = getattr(model, name).grad
            if g is not None:
                track["grad_norms"][name].append(g.norm().item())
                if g.isnan().any():
                    track["nan_detected"] = True
                    print(f"{log_prefix} ⚠ NaN grad in {name} step {step}")
                if g.isinf().any():
                    track["inf_detected"] = True
                    print(f"{log_prefix} ⚠ Inf grad in {name} step {step}")

        # ── Topology ──
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        topo = model.densify_and_prune(step)
        torch.cuda.synchronize()
        track["topo_ms"].append((time.perf_counter() - t0) * 1000)

        if topo["cloned"] > 0 or topo["split"] > 0:
            track["densification_events"] += 1
        track["total_cloned"] += topo["cloned"]
        track["total_split"] += topo["split"]
        track["total_pruned"] += topo["pruned"]
        if topo["pruned"] > 0:
            track["pruning_events"] += 1

        # ── Optimizer step ──
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        track["opt_ms"].append((time.perf_counter() - t0) * 1000)

        # ── Clamp ──
        with torch.no_grad():
            model.scales.clamp_(min=-10.0, max=10.0)
            model.opacities.clamp_(min=-10.0, max=10.0)

        # ── Record ──
        track["losses"].append(loss.item())
        track["l1s"].append(l1.item())
        track["dssims"].append(d_ssim.item())
        psnr = -10 * math.log10(loss.item()) if loss.item() > 0 else 50
        track["psnrs"].append(psnr)
        track["gaussian_counts"].append(model.n)
        track["iter_ms"].append(track["fwd_ms"][-1] + track["bwd_ms"][-1] +
                                track["opt_ms"][-1] + track["topo_ms"][-1])

        if psnr > best_psnr:
            best_psnr = psnr
            best_psnr_step = step + 1

        # ── Logging ──
        if (step + 1) % 500 == 0:
            avg_iter = np.mean(track["iter_ms"][-500:])
            mem = torch.cuda.memory_allocated() / (1024**2)
            print(f"  Step {step+1:5d}/{n_steps}  loss={loss.item():.4f}  "
                  f"PSNR={psnr:.2f}  Gs={model.n:,}  "
                  f"iter={avg_iter:.0f}ms  VRAM={mem:.0f}MB", flush=True)

        # ── Checkpoint ──
        if output_dir and (step + 1) % save_interval == 0:
            ckpt = dict(
                sh_degree=sh_degree, step=step + 1,
                means=model.means.detach().cpu(),
                quats=model.quats.detach().cpu(),
                scales=model.scales.detach().cpu(),
                opacities=model.opacities.detach().cpu(),
                shs=model.shs.detach().cpu(),
                loss=loss.item(), psnr=best_psnr,
                gaussian_count=model.n,
            )
            torch.save(ckpt, output_dir / f"m3_sh{sh_degree}_step{step+1:05d}.pt")

    # ── Finalize ──
    t_total = time.perf_counter() - t_start
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)

    if output_dir:
        ckpt = dict(
            sh_degree=sh_degree, step=n_steps,
            means=model.means.detach().cpu(),
            quats=model.quats.detach().cpu(),
            scales=model.scales.detach().cpu(),
            opacities=model.opacities.detach().cpu(),
            shs=model.shs.detach().cpu(),
            loss=loss.item(), psnr=best_psnr,
            gaussian_count=model.n,
        )
        torch.save(ckpt, output_dir / f"m3_sh{sh_degree}_final.pt")

    # Time-to-target
    time_to_target = {}
    for target in [25.0, 26.0, 27.0, 28.0, 29.0, 30.0]:
        for i, p in enumerate(track["psnrs"]):
            if p >= target:
                steps = i + 1
                time_to_target[f"PSNR_{target}dB"] = dict(
                    step=steps,
                    time_seconds=round((steps / n_steps) * t_total, 1),
                    reached=True,
                )
                break
        else:
            time_to_target[f"PSNR_{target}dB"] = dict(
                step=None, time_seconds=None, reached=False, note="NOT_REACHED")

    results = dict(
        sh_degree=sh_degree,
        n_initial=n_initial,
        n_final=model.n,
        total_time_s=round(t_total, 1),
        total_time_min=round(t_total / 60, 1),
        avg_iter_ms=round(float(np.mean(track["iter_ms"])), 2),
        avg_fwd_ms=round(float(np.mean(track["fwd_ms"])), 2),
        avg_bwd_ms=round(float(np.mean(track["bwd_ms"])), 2),
        avg_opt_ms=round(float(np.mean(track["opt_ms"])), 2),
        avg_topo_ms=round(float(np.mean(track["topo_ms"])), 2),
        iter_per_sec=round(n_steps / t_total, 2),
        best_psnr=round(float(best_psnr), 4),
        best_psnr_step=best_psnr_step,
        final_psnr=round(float(track["psnrs"][-1]), 4),
        final_loss=round(float(track["losses"][-1]), 6),
        nan_detected=track["nan_detected"],
        inf_detected=track["inf_detected"],
        densification_events=track["densification_events"],
        pruning_events=track["pruning_events"],
        total_cloned=track["total_cloned"],
        total_split=track["total_split"],
        total_pruned=track["total_pruned"],
        peak_vram_mb=round(peak_vram, 1),
        time_to_target=time_to_target,
        # 500-step resolution trajectories
        loss_trajectory=[round(v, 6) for v in track["losses"][::500]],
        l1_trajectory=[round(v, 6) for v in track["l1s"][::500]],
        psnr_trajectory=[round(float(v), 4) for v in track["psnrs"][::500]],
        gaussian_count_trajectory=track["gaussian_counts"][::500],
        iter_time_trajectory=[round(float(v), 2) for v in track["iter_ms"][::500]],
        fwd_time_trajectory=[round(float(v), 2) for v in track["fwd_ms"][::500]],
        bwd_time_trajectory=[round(float(v), 2) for v in track["bwd_ms"][::500]],
        opt_time_trajectory=[round(float(v), 2) for v in track["opt_ms"][::500]],
        topo_time_trajectory=[round(float(v), 2) for v in track["topo_ms"][::500]],
        gradient_norms={k: [round(v, 6) for v in vals[::2500]]
                       for k, vals in track["grad_norms"].items()},
    )

    print(f"{log_prefix} COMPLETE: {n_initial:,} → {model.n:,} Gs")
    print(f"{log_prefix} Best PSNR: {best_psnr:.2f} dB (step {best_psnr_step})")
    print(f"{log_prefix} Final PSNR: {results['final_psnr']:.2f} dB")
    print(f"{log_prefix} Time: {t_total:.0f}s ({t_total/60:.1f} min) = {results['iter_per_sec']:.2f} iter/s")
    print(f"{log_prefix} Avg iter: {results['avg_iter_ms']:.1f}ms "
          f"(fwd={results['avg_fwd_ms']:.1f} bwd={results['avg_bwd_ms']:.1f} "
          f"opt={results['avg_opt_ms']:.1f} topo={results['avg_topo_ms']:.1f})")
    print(f"{log_prefix} Peak VRAM: {peak_vram:.0f}MB  NaN: {track['nan_detected']} Inf: {track['inf_detected']}")
    print(f"{log_prefix} Dens: {track['densification_events']}  Prune: {track['pruning_events']}")

    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--sh-degrees", type=int, nargs="+", default=[0, 1, 3])
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory / 1e9:.2f} GB / {props.total_memory / (1024**3):.2f} GiB")

    output_dir = Path(args.output_dir) if args.output_dir else \
        PROJECT_ROOT / "results" / "epic05" / "phase9c"
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_data, cameras = load_scene(device)
    print(f"Scene: room ({scene_data['num_points']:,} Gs, {len(cameras)} cameras)")

    all_results = {}
    for sh in args.sh_degrees:
        torch.cuda.empty_cache()
        result = train_sh_degree(
            scene_data, cameras, sh, args.steps, device,
            output_dir, args.save_interval)
        all_results[f"SH{sh}"] = result

        with open(output_dir / f"m3_full_training_sh{sh}.json", "w") as f:
            json.dump(result, f, indent=2)

    combined = dict(
        experiment_id="phase9c-m3-full-training",
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        hardware=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        scene="room", resolution="1920x1080",
        n_steps=args.steps,
        config=dict(
            densification_start=500, densification_end=15000,
            densification_interval=100, prune_interval=100,
            grad_threshold=0.0002, prune_opacity=0.005,
            optimizer="Adam", loss="L1 + D-SSIM (0.2)",
            packed=True, tile_size=16,
        ),
        results=all_results,
    )
    with open(output_dir / "m3_full_training_results.json", "w") as f:
        json.dump(combined, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"  PHASE 9C — M3 Full 30K Training Summary")
    print(f"{'='*70}")
    print(f"  {'Degree':>6} | {'Final PSNR':>10} | {'Best PSNR':>10} | {'Time':>8} | "
          f"{'Gs Final':>10} | {'Iter/s':>7} | {'Peak VRAM':>9}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*7}-+-{'-'*9}")
    for sh in args.sh_degrees:
        r = all_results[f"SH{sh}"]
        print(f"  {'SH'+str(sh):>6} | {r['final_psnr']:>8.2f} dB | {r['best_psnr']:>8.2f} dB | "
              f"{r['total_time_min']:>6.1f}m | {r['n_final']:>10,} | "
              f"{r['iter_per_sec']:>6.2f} | {r['peak_vram_mb']:>7.0f}MB")

    print(f"\nResults: {output_dir / 'm3_full_training_results.json'}")


if __name__ == "__main__":
    main()
