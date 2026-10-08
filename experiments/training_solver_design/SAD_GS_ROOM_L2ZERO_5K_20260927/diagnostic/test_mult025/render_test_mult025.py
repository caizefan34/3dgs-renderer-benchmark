#!/usr/bin/env python3
"""Render all held-out cameras at the training rasterizer mult, with pair counts."""
import argparse
import json
from pathlib import Path

import torch
import torchvision
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render_structgs
from scene import Scene
from utils.general_utils import safe_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mult", type=float, default=0.25)
    parser.add_argument("--counts_output", required=True)
    args = get_combined_args(parser)
    if args.iteration != 5000 or abs(args.mult - 0.25) > 1e-9:
        raise RuntimeError("this frozen diagnostic accepts only iteration=5000 and mult=0.25")
    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default")
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    views = scene.getTestCameras()
    if len(views) != 39:
        raise RuntimeError(f"expected 39 held-out cameras; found {len(views)}")
    sizes = {(int(v.image_width), int(v.image_height)) for v in views}
    if sizes != {(1600, 1066)}:
        raise RuntimeError(f"held-out camera dimensions differ from 1600x1066: {sorted(sizes)}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    run_name = f"ours_{scene.loaded_iter}"
    render_dir = Path(dataset.model_path) / "test" / run_name / "renders"
    gt_dir = Path(dataset.model_path) / "test" / run_name / "gt"
    render_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total_pairs = 0
    with torch.no_grad():
        for index, camera in enumerate(views):
            height, width = int(camera.image_height), int(camera.image_width)
            metric_map = torch.ones((height, width), dtype=torch.int32, device="cuda")
            package = render_structgs(
                camera,
                gaussians,
                pipe,
                background,
                args.mult,
                get_flag=True,
                metric_map=metric_map,
                compute_extra=False,
            )
            pairs = int(package["accum_metric_counts"].to(torch.int64).sum().item())
            total_pairs += pairs
            image_name = f"{index:05d}.png"
            torchvision.utils.save_image(package["render"], render_dir / image_name)
            torchvision.utils.save_image(camera.original_image[:3].to(package["render"].device), gt_dir / image_name)
            rows.append({
                "render_index": index,
                "camera_name": camera.image_name,
                "numeric_png": image_name,
                "width": width,
                "height": height,
                "accepted_gaussian_pixel_pairs": pairs,
            })
            print(f"{index + 1}/39 {camera.image_name} pairs={pairs}", flush=True)

    result = {
        "status": "passed",
        "model_path": dataset.model_path,
        "iteration": args.iteration,
        "mult": args.mult,
        "resolution": [1600, 1066],
        "camera_order": "exact Scene.getTestCameras() order with shuffle=False, matching author render.py",
        "accepted_work_definition": "sum of rasterizer accum_metric_counts over every pixel; one count per Gaussian-pixel pair passing power, alpha>=1/255, and test_T>=1e-4",
        "total_accepted_gaussian_pixel_pairs": total_pairs,
        "per_view": rows,
    }
    output = Path(args.counts_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "passed", "total_accepted_gaussian_pixel_pairs": total_pairs, "counts_output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
