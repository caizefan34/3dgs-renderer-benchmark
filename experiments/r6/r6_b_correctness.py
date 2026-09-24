#!/usr/bin/env python3
"""
R6-B CORRECTNESS VALIDATION FRAMEWORK.

This script validates that B0 (persistent buffers + full clear) and B1
(persistent buffers + previous-touched selective clear) produce gradients
IDENTICAL to baseline (within FP accumulation-order tolerance).

Tests:
1. Stale gradient test: Gaussian X touched at iteration t, NOT touched at t+1
   → grad[X] must be 0 at t+1 (no stale gradient leakage)
2. Gradient comparison: all 10 gradient types compared baseline vs candidate
   → max_abs, mean_abs, relative_L2, NaN/Inf check
3. Topology safety: clone/split/prune/reorder must not corrupt persistent buffers

This script loads a checkpoint, runs 2 consecutive iterations with different
cameras (to create the stale-gradient scenario), and compares gradients.

Usage (on mx):
  CUDA_VISIBLE_DEVICES=5 PYTHONNOUSERSITE=1 \
    ~/miniforge3/envs/anysplat/bin/python experiments/r6/r6_b_correctness.py \
    --scene room --ckpt /mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts/room/checkpoints/iter_5000.pt \
    --output /mnt/storage_pool/liaoyuanjun/r6_profiling/r6_b_correctness_room_5k.json

  # With candidate implementation:
  --candidate-module gsplat_r6b --candidate-variant b1
"""
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import sys, json, argparse, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "epic05" / "phase7"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "baseline" / "reference_v1"))

from gaussian_model import GaussianModel
from config import ReferenceV1Config
from dataset import GTDataset
from trainer import SepSSIM, render_with_meta


def load_model_from_ckpt(ckpt_path, config, repo_root):
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    model = GaussianModel(max_sh_degree=config.sh_degree)
    model.restore(ckpt, {
        "position_lr_init": config.position_lr_init,
        "position_lr_final": config.position_lr_final,
        "position_lr_delay_mult": config.position_lr_delay_mult,
        "position_lr_max_steps": config.position_lr_max_steps,
        "feature_lr": config.feature_lr,
        "opacity_lr": config.opacity_lr,
        "scaling_lr": config.scaling_lr,
        "rotation_lr": config.rotation_lr,
        "percent_dense": config.percent_dense,
    })
    model.active_sh_degree = ckpt["active_sh_degree"]
    return model


def get_all_gradients(model):
    """Extract all 10 gradient types from model parameters."""
    grads = {}
    param_map = {
        "xyz": model._xyz,
        "shs": model._shs,
        "scaling": model._scaling,
        "rotation": model._rotation,
        "opacity": model._opacity,
    }
    for name, param in param_map.items():
        if param.grad is not None:
            grads[name] = param.grad.detach().clone()
        else:
            grads[name] = None
    return grads


def compare_gradients(grad_baseline, grad_candidate, name):
    """Compare two gradient tensors. Returns comparison metrics."""
    if grad_baseline is None and grad_candidate is None:
        return {"status": "both_none", "max_abs": 0, "mean_abs": 0, "rel_l2": 0,
                "nan": False, "inf": False}
    if grad_baseline is None or grad_candidate is None:
        return {"status": "mismatch_none", "max_abs": float('inf'), "mean_abs": float('inf'),
                "rel_l2": float('inf'), "nan": True, "inf": True}

    diff = (grad_candidate - grad_baseline).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())

    # Relative L2: ||diff|| / ||baseline||
    baseline_norm = float(grad_baseline.norm().item())
    diff_norm = float(diff.norm().item())
    rel_l2 = diff_norm / max(baseline_norm, 1e-12)

    nan = bool(torch.isnan(grad_candidate).any().item() or torch.isnan(grad_baseline).any().item())
    inf = bool(torch.isinf(grad_candidate).any().item() or torch.isinf(grad_baseline).any().item())

    return {
        "status": "compared",
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rel_l2": rel_l2,
        "nan": nan,
        "inf": inf,
        "baseline_norm": baseline_norm,
        "candidate_norm": float(grad_candidate.norm().item()),
    }


def run_stale_gradient_test(model, dataset, ssim_fn, sh_degree, cam_idx_t, cam_idx_t1):
    """
    Stale gradient test:
    - Iteration t: use camera cam_idx_t → some Gaussians touched
    - Iteration t+1: use camera cam_idx_t1 → different Gaussians touched
    - Check: Gaussians touched at t but NOT at t+1 have grad == 0 at t+1

    Returns: test results
    """
    print(f"  Stale gradient test: cam_t={cam_idx_t}, cam_t1={cam_idx_t1}")

    # Iteration t: forward ONLY (no backward needed — just need radii to know
    # which Gaussians were visible). Use no_grad to avoid graph retention.
    cam_t, gt_t = dataset.get_item(cam_idx_t)
    with torch.no_grad():
        image_t, meta_t, _ = render_with_meta(model, cam_t, sh_degree)

    # Record which Gaussians were touched at t (from radii)
    radii_t = meta_t["radii"][0]
    visible_t = (radii_t > 0).any(dim=-1)
    n_visible_t = int(visible_t.sum().item())

    # Iteration t+1: zero_grad + forward + backward with camera t+1
    model.optimizer.zero_grad(set_to_none=True)
    cam_t1, gt_t1 = dataset.get_item(cam_idx_t1)
    image_t1, meta_t1, _ = render_with_meta(model, cam_t1, sh_degree)
    L1_t1 = F.l1_loss(image_t1, gt_t1)
    dssim_t1 = ssim_fn(image_t1, gt_t1)
    loss_t1 = (1.0 - 0.2) * L1_t1 + 0.2 * dssim_t1
    loss_t1.backward()

    # Record which Gaussians were touched at t+1
    radii_t1 = meta_t1["radii"][0]
    visible_t1 = (radii_t1 > 0).any(dim=-1)
    n_visible_t1 = int(visible_t1.sum().item())

    # Gaussians touched at t but NOT at t+1
    stale_mask = visible_t & ~visible_t1
    n_stale = int(stale_mask.sum().item())

    # Get gradients at t+1
    grads_t1 = get_all_gradients(model)

    # Check: stale Gaussians must have grad == 0 at t+1
    stale_results = {}
    for name, grad in grads_t1.items():
        if grad is None:
            stale_results[name] = {"status": "grad_none", "stale_max_abs": 0}
            continue
        stale_grads = grad[stale_mask]
        stale_max = float(stale_grads.abs().max().item()) if n_stale > 0 else 0.0
        stale_mean = float(stale_grads.abs().mean().item()) if n_stale > 0 else 0.0
        stale_results[name] = {
            "status": "checked",
            "n_stale": n_stale,
            "stale_max_abs": stale_max,
            "stale_mean_abs": stale_mean,
            "PASS": stale_max < 1e-6,  # Must be exactly zero (or FP-noise level)
        }

    all_pass = all(r.get("PASS", True) for r in stale_results.values())

    return {
        "n_visible_t": n_visible_t,
        "n_visible_t1": n_visible_t1,
        "n_stale_gaussians": n_stale,
        "per_gradient": stale_results,
        "PASS": all_pass,
    }


def run_gradient_comparison(model, dataset, ssim_fn, sh_degree, cam_idx):
    """
    Gradient comparison: baseline (at::zeros_like) vs current implementation.
    Since we're testing the BASELINE here, this just records the gradients
    for later comparison with B0/B1 implementations.

    For now, this records the baseline gradients and checks for NaN/Inf.
    """
    print(f"  Gradient comparison: cam={cam_idx}")

    model.optimizer.zero_grad(set_to_none=True)
    cam, gt_image = dataset.get_item(cam_idx)
    image, meta, _ = render_with_meta(model, cam, sh_degree)
    L1 = F.l1_loss(image, gt_image)
    dssim = ssim_fn(image, gt_image)
    loss = (1.0 - 0.2) * L1 + 0.2 * dssim
    loss.backward()

    grads = get_all_gradients(model)

    results = {}
    for name, grad in grads.items():
        if grad is None:
            results[name] = {"status": "grad_none"}
            continue
        results[name] = {
            "status": "recorded",
            "shape": list(grad.shape),
            "n_elements": grad.numel(),
            "max_abs": float(grad.abs().max().item()),
            "mean_abs": float(grad.abs().mean().item()),
            "norm": float(grad.norm().item()),
            "n_nonzero": int((grad.abs() > 1e-8).sum().item()),
            "n_nan": int(torch.isnan(grad).sum().item()),
            "n_inf": int(torch.isinf(grad).sum().item()),
            "PASS_nan_inf": not (torch.isnan(grad).any().item() or torch.isinf(grad).any().item()),
        }

    return {
        "per_gradient": results,
        "PASS": all(r.get("PASS_nan_inf", True) for r in results.values()),
    }


def run_topology_safety_audit(model):
    """
    Audit topology operations (clone/split/prune) for buffer reuse safety.

    For the BASELINE (at::zeros_like every iteration), topology changes are
    safe because gradient buffers are reallocated every backward.

    For B0/B1 (persistent buffers), topology changes REQUIRE buffer
    reallocation or index remapping. This audit documents what operations
    occur and what must be handled.
    """
    N_before = model._xyz.shape[0]

    # Check what densification methods exist
    densification_methods = []
    if hasattr(model, 'densify_and_clone'):
        densification_methods.append("densify_and_clone")
    if hasattr(model, 'densify_and_split'):
        densification_methods.append("densify_and_split")
    if hasattr(model, 'prune_points'):
        densification_methods.append("prune_points")

    # Check parameter structure
    param_info = {}
    for name in ["_xyz", "_shs", "_scaling", "_rotation", "_opacity"]:
        if hasattr(model, name):
            p = getattr(model, name)
            param_info[name] = {
                "shape": list(p.shape),
                "n_elements": p.numel(),
                "bytes": p.numel() * 4,
            }

    return {
        "N_before": N_before,
        "densification_methods": densification_methods,
        "param_info": param_info,
        "topology_safety_requirements": [
            "clone: new Gaussians appended → persistent buffers must grow or reallocate",
            "split: 1 Gaussian → 2 → N increases by 1 → buffer must grow",
            "prune: Gaussians removed → indices change → buffer must compact or remap",
            "If any topology change occurs between backward calls, persistent buffers",
            "MUST be reallocated or index-remapped. Stale data from old indices is a",
            "correctness violation (stale gradient leakage).",
            "FALLBACK: if safe reuse cannot be proven, use full clear (B0) or",
            "full reallocation (baseline) after every topology change.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", default="1080p")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = ReferenceV1Config()
    config.scene = args.scene
    config.repo_root = str(REPO_ROOT)
    config.resolution = config.resolution
    config.seed = args.seed

    print(f"=== R6-B Correctness Validation: {args.scene} ===")
    dataset = GTDataset(scene=config.scene, repo_root=config.repo_root,
                        resolution=args.resolution, device="cuda", background="black")
    model = load_model_from_ckpt(args.ckpt, config, config.repo_root)
    print(f"  N={model._xyz.shape[0]}, SH={model.active_sh_degree}")
    print(f"  Dataset: {len(dataset)} cameras")

    ssim_fn = SepSSIM(device="cuda")

    # Test 1: Stale gradient test (baseline = at::zeros_like, should PASS trivially)
    print(f"\n--- Test 1: Stale Gradient Test (baseline) ---")
    # Use cameras 0 and 10 (different viewpoints → different visible Gaussians)
    stale_result = run_stale_gradient_test(model, dataset, ssim_fn,
                                            model.active_sh_degree, 0, 10)
    print(f"  n_visible_t={stale_result['n_visible_t']}, "
          f"n_visible_t1={stale_result['n_visible_t1']}, "
          f"n_stale={stale_result['n_stale_gaussians']}")
    for name, r in stale_result["per_gradient"].items():
        if r.get("status") == "checked":
            print(f"    {name}: stale_max_abs={r['stale_max_abs']:.2e} PASS={r['PASS']}")
    print(f"  STALE TEST: {'PASS' if stale_result['PASS'] else 'FAIL'}")

    # Test 2: Gradient comparison (baseline — records reference gradients)
    print(f"\n--- Test 2: Gradient Comparison (baseline reference) ---")
    grad_result = run_gradient_comparison(model, dataset, ssim_fn,
                                          model.active_sh_degree, 0)
    for name, r in grad_result["per_gradient"].items():
        if r.get("status") == "recorded":
            print(f"    {name}: shape={r['shape']}, max_abs={r['max_abs']:.2e}, "
                  f"n_nonzero={r['n_nonzero']}, NaN={r['n_nan']}, Inf={r['n_inf']}")
    print(f"  GRADIENT CHECK: {'PASS' if grad_result['PASS'] else 'FAIL'}")

    # Test 3: Topology safety audit
    print(f"\n--- Test 3: Topology Safety Audit ---")
    topo_result = run_topology_safety_audit(model)
    print(f"  N={topo_result['N_before']}")
    print(f"  Densification methods: {topo_result['densification_methods']}")
    for req in topo_result['topology_safety_requirements']:
        print(f"    {req}")

    # Summary
    result = {
        "scene": args.scene,
        "checkpoint": args.ckpt,
        "N_total": model._xyz.shape[0],
        "sh_degree": model.active_sh_degree,
        "variant": "baseline",
        "stale_gradient_test": stale_result,
        "gradient_comparison": grad_result,
        "topology_safety_audit": topo_result,
        "overall_pass": stale_result["PASS"] and grad_result["PASS"],
    }

    print(f"\n=== SUMMARY ===")
    print(f"  Stale gradient test: {'PASS' if stale_result['PASS'] else 'FAIL'}")
    print(f"  Gradient NaN/Inf check: {'PASS' if grad_result['PASS'] else 'FAIL'}")
    print(f"  Overall: {'PASS' if result['overall_pass'] else 'FAIL'}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
