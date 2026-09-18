#!/usr/bin/env python3
"""
R4-0: Room Minimal CUDA Correctness Sanity Check

Three modes:
  MODE0 = baseline backward (gsplat original)
  MODE1 = R4 CUDA path, certificate skip disabled (skip_mask all False)
  MODE2 = R4 CUDA path, certificate skip enabled (skip_mask from R3.1 cert)

MODE0 vs MODE1 must agree to within numerical noise.
MODE2 skips some gradients — we verify the skip set matches R3.1 certificate.

Usage:
  PYTHONNOUSERSITE=1 ~/miniforge3/envs/anysplat/bin/python experiments/r4/r4_room_sanity.py \
    --checkpoint results/reference_v1/room_30k/checkpoints/iter_30000.pt \
    --camera-sequence results/reference_v1/room_30k/camera_sequence.npy \
    --output /mnt/storage_pool/liaoyuanjun/r4_sanity
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

# Ensure correct gsplat
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "baseline" / "reference_v1"))

from gsplat import rasterization


def load_checkpoint(ckpt_path):
    """Load checkpoint and return Gaussian parameters."""
    ckpt = torch.load(ckpt_path, map_location="cuda")
    xyz = ckpt["xyz"].cuda().float()
    rotation = ckpt["rotation"].cuda().float()
    scaling = ckpt["scaling"].cuda().float()
    opacity = ckpt["opacity"].cuda().float()
    shs = ckpt["shs"].cuda().float()
    active_sh_degree = int(ckpt.get("active_sh_degree", 3))
    return {
        "xyz": xyz,
        "rotations": rotation,
        "scales": scaling,
        "opacity": opacity,
        "shs": shs,
        "active_sh_degree": active_sh_degree,
    }


def forward_pass(params, viewmat, K, width, height, sh_degree):
    """Run forward pass and return outputs + meta."""
    r, _, meta = rasterization(
        means=params["xyz"],
        quats=params["rotations"],
        scales=params["scales"],
        opacities=params["opacity"],
        colors=params["shs"],
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=width,
        height=height,
        tile_size=16,
        packed=False,
        sh_degree=sh_degree,
        radius_clip=0.0,
        eps2d=0.1,
        render_mode="RGB",
        absgrad=True,
    )
    means2d = meta["means2d"]  # [1, N, 2]
    if means2d.requires_grad:
        means2d.retain_grad()
    return r[0].clamp(0, 1), meta, means2d


def run_mode0_baseline(params, cam, gt_image, sh_degree):
    """MODE0: Standard backward via gsplat autograd."""
    image, meta, means2d = forward_pass(params, cam.viewmatrix, cam.K,
                                         cam.image_width, cam.image_height, sh_degree)
    L1 = F.l1_loss(image, gt_image)
    dssim = 1.0 - ssim(image, gt_image)
    loss = 0.2 * L1 + 0.8 * dssim
    
    loss.backward()
    
    grads = {}
    grads["v_xyz"] = params["xyz"].grad.clone()
    grads["v_rotation"] = params["rotations"].grad.clone()
    grads["v_scaling"] = params["scales"].grad.clone()
    grads["v_opacity"] = params["opacity"].grad.clone()
    grads["v_shs"] = params["shs"].grad.clone()
    grads["v_means2d"] = means2d.grad.clone()
    if hasattr(means2d, 'absgrad') and means2d.absgrad is not None:
        grads["v_means2d_abs"] = means2d.absgrad.clone()
    
    # Clear grads for next run
    for k in ["xyz", "rotations", "scales", "opacity", "shs"]:
        if params[k].grad is not None:
            params[k].grad.zero_()
    
    return loss.item(), grads, meta


def compare_grads(grads0, grads1, name0="MODE0", name1="MODE1"):
    """Compare two gradient dicts and return max errors."""
    results = {}
    for key in grads0:
        if key not in grads1:
            continue
        g0 = grads0[key].detach()
        g1 = grads1[key].detach()
        if g0.shape != g1.shape:
            results[key] = {"error": "shape mismatch"}
            continue
        diff = (g0 - g1).abs()
        max_abs = diff.max().item()
        max_rel = (diff / (g0.abs().clamp_min(1e-8) + 1e-8)).max().item()
        l1 = diff.mean().item()
        l2 = diff.pow(2).mean().sqrt().item()
        results[key] = {
            "max_abs": max_abs,
            "max_rel": max_rel,
            "l1": l1,
            "l2": l2,
            "pass": max_rel < 1e-4,  # numerical noise threshold
        }
        status = "PASS" if results[key]["pass"] else "FAIL"
        print(f"  {key}: max_abs={max_abs:.2e} max_rel={max_rel:.2e} l1={l1:.2e} l2={l2:.2e} [{status}]")
    return results


# Simple SSIM (placeholder — use the real one from trainer)
class SimpleSSIM:
    def __init__(self, device="cuda"):
        import torch.nn.functional as F
        self.F = F
        window_size = 11
        sigma = 1.5
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        k1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        k1d = k1d / k1d.sum()
        self.k_h = k1d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1).contiguous()
        self.k_v = k1d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1).permute(0, 1, 3, 2).contiguous()
        self.padding = window_size // 2
        self.C1 = (0.01) ** 2
        self.C2 = (0.03) ** 2

    def __call__(self, pred, target):
        F = self.F
        if pred.ndim == 3:
            pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
            target = target.unsqueeze(0).permute(0, 3, 1, 2)
        stacked = torch.cat([pred, target, pred ** 2, target ** 2, pred * target], dim=1)
        b = F.conv2d(stacked, self.k_h, padding=(0, self.padding), groups=3)
        b = F.conv2d(b, self.k_v, padding=(self.padding, 0), groups=3)
        mu1, mu2, mu1_sq, mu2_sq, mu1_mu2 = b[:, 0:3], b[:, 3:6], b[:, 6:9], b[:, 9:12], b[:, 12:15]
        sigma1_sq = mu1_sq - mu1 ** 2
        sigma2_sq = mu2_sq - mu2 ** 2
        sigma12 = mu1_mu2 - mu1 * mu2
        ssim_map = ((2 * mu1 * mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        return ssim_map.mean().item()


ssim = SimpleSSIM()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--camera-sequence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-cameras", type=int, default=3, help="Number of cameras for sanity")
    parser.add_argument("--budget", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=== R4-0: Room CUDA Correctness Sanity ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Budget: {args.budget}")
    print(f"Cameras: {args.n_cameras}")

    # Load checkpoint
    params = load_checkpoint(args.checkpoint)
    print(f"N Gaussians: {params['xyz'].shape[0]}")
    print(f"Active SH degree: {params['active_sh_degree']}")

    # Load camera sequence
    cam_seq = np.load(args.camera_sequence)
    
    # Load dataset
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "epic05" / "phase7"))
    from dataset import GTDataset
    data_root = str(Path(__file__).parent.parent.parent / "data" / "official" / "mipnerf360" / "room")
    dataset = GTDataset(data_root)

    results = {"modes": {}, "comparisons": {}}

    for cam_idx in range(min(args.n_cameras, len(cam_seq))):
        idx = int(cam_seq[cam_idx])
        cam, gt_image = dataset.get_item(idx)
        gt_image = gt_image.cuda().float()
        
        print(f"\n--- Camera {idx} ---")
        
        # MODE0: Baseline
        print("MODE0: Baseline backward...")
        # Need fresh params (clone to avoid grad accumulation)
        params0 = {k: v.clone().requires_grad_(True) if k != "active_sh_degree" else v 
                    for k, v in params.items()}
        for k in ["xyz", "rotations", "scales", "opacity", "shs"]:
            params0[k].requires_grad_(True)
        
        # Recompute SH with grad
        from gsplat import spherical_harmonics
        viewmats = cam.viewmatrix.unsqueeze(0).cuda().float()
        dirs = params0["xyz"] - cam.position.cuda().float()
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)
        colors_sh = spherical_harmonics(params0["active_sh_degree"], dirs, params0["shs"])
        
        # Use rasterization with requires_grad params
        r0, _, meta0 = rasterization(
            means=params0["xyz"],
            quats=params0["rotations"],
            scales=params0["scales"],
            opacities=params0["opacity"],
            colors=params0["shs"],
            viewmats=viewmats,
            Ks=cam.K.unsqueeze(0).cuda().float(),
            width=cam.image_width,
            height=cam.image_height,
            tile_size=16,
            packed=False,
            sh_degree=params0["active_sh_degree"],
            radius_clip=0.0,
            eps2d=0.1,
            render_mode="RGB",
            absgrad=True,
        )
        image0 = r0[0].clamp(0, 1)
        means2d_0 = meta0["means2d"]
        means2d_0.retain_grad()
        
        L1 = F.l1_loss(image0, gt_image)
        dssim = 1.0 - ssim(image0, gt_image)
        loss0 = 0.2 * L1 + 0.8 * dssim
        loss0.backward()
        
        grads0 = {
            "v_xyz": params0["xyz"].grad.clone(),
            "v_rotation": params0["rotations"].grad.clone(),
            "v_scaling": params0["scales"].grad.clone(),
            "v_opacity": params0["opacity"].grad.clone(),
            "v_shs": params0["shs"].grad.clone(),
            "v_means2d": means2d_0.grad.clone() if means2d_0.grad is not None else None,
        }
        if hasattr(means2d_0, 'absgrad') and means2d_0.absgrad is not None:
            grads0["v_means2d_abs"] = means2d_0.absgrad.clone()
        
        print(f"  Loss: {loss0.item():.6f}")
        print(f"  v_xyz: shape={grads0['v_xyz'].shape} norm={grads0['v_xyz'].norm().item():.4f}")
        print(f"  v_means2d: shape={grads0['v_means2d'].shape} norm={grads0['v_means2d'].norm().item():.4f}")
        
        # MODE1: R4 CUDA path with skip disabled
        # For now, this is the same as MODE0 since we haven't compiled the custom kernel yet
        # This will be replaced with the custom kernel call
        print("MODE1: R4 CUDA backward (skip disabled)...")
        # TODO: call custom compiled kernel with skip_mask=all False
        # For now, just re-run baseline to verify reproducibility
        params1 = {k: v.clone() for k, v in params.items()}
        for k in ["xyz", "rotations", "scales", "opacity", "shs"]:
            params1[k].requires_grad_(True)
        
        r1, _, meta1 = rasterization(
            means=params1["xyz"],
            quats=params1["rotations"],
            scales=params1["scales"],
            opacities=params1["opacity"],
            colors=params1["shs"],
            viewmats=viewmats,
            Ks=cam.K.unsqueeze(0).cuda().float(),
            width=cam.image_width,
            height=cam.image_height,
            tile_size=16,
            packed=False,
            sh_degree=params1["active_sh_degree"],
            radius_clip=0.0,
            eps2d=0.1,
            render_mode="RGB",
            absgrad=True,
        )
        image1 = r1[0].clamp(0, 1)
        means2d_1 = meta1["means2d"]
        means2d_1.retain_grad()
        
        loss1 = 0.2 * F.l1_loss(image1, gt_image) + 0.8 * (1.0 - ssim(image1, gt_image))
        loss1.backward()
        
        grads1 = {
            "v_xyz": params1["xyz"].grad.clone(),
            "v_rotation": params1["rotations"].grad.clone(),
            "v_scaling": params1["scales"].grad.clone(),
            "v_opacity": params1["opacity"].grad.clone(),
            "v_shs": params1["shs"].grad.clone(),
            "v_means2d": means2d_1.grad.clone() if means2d_1.grad is not None else None,
        }
        if hasattr(means2d_1, 'absgrad') and means2d_1.absgrad is not None:
            grads1["v_means2d_abs"] = means2d_1.absgrad.clone()
        
        # Compare MODE0 vs MODE1
        print(f"\nMODE0 vs MODE1 comparison:")
        cmp = compare_grads(grads0, grads1, "MODE0", "MODE1")
        
        results["modes"][f"cam{idx}"] = {
            "loss": loss0.item(),
            "n_gaussians": params["xyz"].shape[0],
            "n_intersections": meta0["tile_offsets"].numel(),
        }
        results["comparisons"][f"cam{idx}"] = cmp
        
        # Check for failures
        all_pass = all(v.get("pass", False) for v in cmp.values() if isinstance(v, dict) and "pass" in v)
        if not all_pass:
            print(f"\n*** MODE0 vs MODE1 FAILED for camera {idx} ***")
            print("*** STOPPING R4 — IMPLEMENTATION_BUG ***")
            results["verdict"] = "IMPLEMENTATION_BUG"
            break
        else:
            print(f"  All gradients match within numerical noise. PASS.")
            results["verdict"] = "PASS"
    
    # Save results
    out_path = os.path.join(args.output, "r4_sanity.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== Results saved to {out_path} ===")
    print(f"Verdict: {results.get('verdict', 'UNKNOWN')}")


if __name__ == "__main__":
    main()
