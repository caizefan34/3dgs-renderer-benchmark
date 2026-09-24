#!/usr/bin/env python3
"""Real Room checkpoint, identical-mask, zero-step topology comparison."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
from common import OFFICIAL_SHA, environment, write_json
from experiment import (
    EVAL_CAMERAS, GRAD_THRESHOLD, PARAMETER_NAMES, PRUNE_THRESHOLD, SEED,
    SepSSIM, load_room, make_model, make_optimizer, masks, psnr, render,
)
from variants.reference_semantics import apply_prune_transaction, apply_topology_transaction


def prepare_checkpoint(iterations: int, device: str, path: Path):
    torch.manual_seed(SEED); np.random.seed(SEED)
    dataset, sfm = load_room(device)
    model, ssim = make_model(sfm, device), SepSSIM(device)
    optimizer = make_optimizer(model)
    order = np.arange(len(dataset)); np.random.shuffle(order)
    for iteration in range(iterations):
        camera, gt = dataset.get_camera(int(order[iteration % len(order)])), dataset.get_gt_image(int(order[iteration % len(order)]))
        optimizer.zero_grad(set_to_none=True)
        image, _, _ = render(model, camera); pred = image[0].clamp(0, 1)
        loss = 0.8 * F.l1_loss(pred, gt) + 0.2 * (1 - ssim(pred, gt))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    next_index = int(order[iterations % len(order)])
    camera, gt = dataset.get_camera(next_index), dataset.get_gt_image(next_index)
    optimizer.zero_grad(set_to_none=True)
    image, _, _ = render(model, camera); pred = image[0].clamp(0, 1)
    loss = 0.8 * F.l1_loss(pred, gt) + 0.2 * (1 - ssim(pred, gt))
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    model.accumulate_positional_gradient(); clone_mask, split_mask = masks(model)
    checkpoint = {
        "model": {name: getattr(model, name).detach().cpu() for name in PARAMETER_NAMES},
        "sh_degree": model.sh_degree, "optimizer": optimizer.state_dict(),
        "clone_mask": clone_mask.cpu(), "split_mask": split_mask.cpu(),
        "next_camera_index": next_index, "iterations": iterations,
    }
    # Move optimizer state to CPU before persisting the audit checkpoint.
    for state in checkpoint["optimizer"]["state"].values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor): state[key] = value.cpu()
    path.parent.mkdir(parents=True, exist_ok=True); torch.save(checkpoint, path)
    return dataset, checkpoint


def restore(checkpoint: dict, device: str):
    tensors = checkpoint["model"]
    sfm = {"xyz": tensors["xyz"], "scales": tensors["scales"], "rotations": tensors["rotations"], "shs": tensors["shs"]}
    model = make_model(sfm, device)
    model.opacity = torch.nn.Parameter(tensors["opacity"].to(device))
    model.sh_degree = checkpoint["sh_degree"]
    optimizer = make_optimizer(model); optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer


def info_intersections(info: dict) -> int | None:
    value = info.get("n_isects")
    if value is None and "isect_ids" in info: value = info["isect_ids"].numel()
    if isinstance(value, torch.Tensor): value = value.item()
    return None if value is None else int(value)


def apply_and_render(dataset, checkpoint, device, split_semantics):
    model, optimizer = restore(checkpoint, device)
    clone = checkpoint["clone_mask"].to(device); split = checkpoint["split_mask"].to(device)
    before = model.xyz.shape[0]
    tx = apply_topology_transaction(
        model, optimizer, clone, split, split_semantics=split_semantics,
        optimizer_semantics="preserve",
        generator=torch.Generator(device=device).manual_seed(SEED + checkpoint["iterations"]),
    )
    prune_mask = torch.sigmoid(model.opacity).squeeze(-1) < PRUNE_THRESHOLD
    pruned = int(prune_mask.sum())
    if pruned: tx.optimizer = apply_prune_transaction(model, tx.optimizer, prune_mask, optimizer_semantics="preserve")
    images, alphas, cameras = [], [], []
    for index in EVAL_CAMERAS:
        camera = dataset.get_camera(index)
        for _ in range(5): render(model, camera)
        timings = []
        for _ in range(15):
            torch.cuda.synchronize(); start = time.perf_counter()
            image, alpha, info = render(model, camera)
            torch.cuda.synchronize(); timings.append((time.perf_counter() - start) * 1000)
        elapsed = float(np.median(timings))
        images.append(image[0].detach().cpu()); alphas.append(alpha[0].detach().cpu())
        cameras.append({"camera": index, "forward_ms_median_15": elapsed, "forward_ms_samples": timings, "tile_intersections": info_intersections(info)})
    return {
        "n_before": before, "n_after": model.xyz.shape[0], "clone": tx.clone_count,
        "split": tx.split_count, "removed_parent": tx.removed_parent_count, "prune": pruned,
        "cameras": cameras,
    }, images, alphas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-iterations", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    checkpoint_path = ROOT / "results" / "a100" / "phase-c0" / "room_pre_densification_500.pt"
    dataset, checkpoint = prepare_checkpoint(args.prepare_iterations, args.device, checkpoint_path)
    current, current_images, current_alphas = apply_and_render(dataset, checkpoint, args.device, "current")
    torch.cuda.empty_cache(); gc.collect()
    reference, reference_images, reference_alphas = apply_and_render(dataset, checkpoint, args.device, "reference")
    ssim = SepSSIM("cpu"); comparisons = []
    for index, a, b, aa, ab in zip(EVAL_CAMERAS, current_images, reference_images, current_alphas, reference_alphas):
        comparisons.append({
            "camera": index, "psnr_ab": psnr(a, b), "l1_ab": float(F.l1_loss(a, b)),
            "ssim_ab": float(ssim(a, b)), "alpha_l1_ab": float(F.l1_loss(aa, ab)),
        })
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    payload = {
        "status": "COMPLETE", "official_reference_commit": OFFICIAL_SHA,
        "environment": environment(), "checkpoint": {"path": str(checkpoint_path), "sha256": digest, "training_iterations": args.prepare_iterations},
        "selection": {"same_masks": True, "same_seed": SEED + args.prepare_iterations, "grad_threshold": GRAD_THRESHOLD, "clone_selected": int(checkpoint["clone_mask"].sum()), "split_selected": int(checkpoint["split_mask"].sum())},
        "current": current, "reference": reference, "per_camera_comparison": comparisons,
        "aggregate": {
            "mean_psnr_ab": float(np.mean([x["psnr_ab"] for x in comparisons])),
            "mean_l1_ab": float(np.mean([x["l1_ab"] for x in comparisons])),
            "mean_ssim_ab": float(np.mean([x["ssim_ab"] for x in comparisons])),
            "mean_alpha_l1_ab": float(np.mean([x["alpha_l1_ab"] for x in comparisons])),
            "current_total_tile_intersections": sum(x["tile_intersections"] or 0 for x in current["cameras"]),
            "reference_total_tile_intersections": sum(x["tile_intersections"] or 0 for x in reference["cameras"]),
            "current_mean_camera_median_forward_ms": float(np.mean([x["forward_ms_median_15"] for x in current["cameras"]])),
            "reference_mean_camera_median_forward_ms": float(np.mean([x["forward_ms_median_15"] for x in reference["cameras"]])),
        },
        "warning": "Zero-step discontinuity is not final training quality.",
    }
    print(write_json("zero_step_topology_comparison.json", payload))


if __name__ == "__main__":
    main()
