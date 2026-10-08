#!/usr/bin/env python3
"""Read-only fixed-state SAD hybrid SH last-mask incidence diagnostic."""
import argparse
import json
import random

import torch
from arguments import ModelParams, PipelineParams, get_combined_args
from fused_ssim import fused_ssim
from gaussian_renderer import render_structgs
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, optimizer_type="hybrid")
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cameras = sorted(scene.getTrainCameras(), key=lambda c: c.image_name)
    assert len(cameras) == 272
    assert {(c.image_width, c.image_height) for c in cameras} == {(1600, 1066)}
    rng = random.Random(20260928)
    selected = rng.sample(range(272), 8)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    n = gaussians._features_rest.shape[0]
    results = []
    for pair in range(4):
        indices = selected[2 * pair:2 * pair + 2]
        vis = []
        for index in indices:
            camera = cameras[index]
            package = render_structgs(camera, gaussians, pipe, background, 0.25)
            render = package["render"]
            gt = camera.original_image[:3].cuda()
            loss = 0.8 * l1_loss(render, gt) + 0.2 * (1.0 - fused_ssim(render.unsqueeze(0), gt.unsqueeze(0)))
            loss.backward()
            vis.append(package["radii"].reshape(-1) > 0)
        grad = gaussians._features_rest.grad
        if grad is None:
            raise RuntimeError("higher-order SH gradient is None")
        row_sq = grad.detach().float().reshape(n, -1).square().sum(dim=1)
        lost = vis[0] & ~vis[1] & (row_sq > 0)
        skipped = ~vis[1] & (row_sq > 0)
        total_sq = row_sq.sum().item()
        results.append({
            "pair": pair,
            "camera_indices": indices,
            "camera_names": [cameras[i].image_name for i in indices],
            "visible_first": int(vis[0].sum().item()),
            "visible_last": int(vis[1].sum().item()),
            "first_only_visible": int((vis[0] & ~vis[1]).sum().item()),
            "first_only_nonzero_sh_grad": int(lost.sum().item()),
            "all_nonzero_sh_grad": int((row_sq > 0).sum().item()),
            "skipped_nonzero_sh_grad": int(skipped.sum().item()),
            "skipped_sh_grad_squared_norm_fraction": float(row_sq[skipped].sum().item() / total_sq) if total_sq else None,
            "total_sh_grad_squared_norm": float(total_sq),
        })
        gaussians._features_rest.grad = None
        print(json.dumps(results[-1]), flush=True)
    output = {
        "source": "frozen SAD-GS author checkpoint; no optimizer step or parameter edit",
        "model_path": dataset.model_path,
        "iteration": args.iteration,
        "selection": "four pairs from 8 sorted training indices sampled without replacement using Python Random(20260928)",
        "loss": "per-camera 0.8 L1 + 0.2 (1-fusedSSIM), summed over pair, lambda_l2=0",
        "mult": 0.25,
        "gaussians": n,
        "pairs": results,
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
