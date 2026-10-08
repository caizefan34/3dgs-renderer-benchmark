#!/usr/bin/env python3
"""Read-only train-view rendering and accepted-contribution diagnostic."""
import argparse
import json
import math
import os
from pathlib import Path

import torch
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render_structgs
from scene import Scene
from utils.general_utils import safe_state


def qsummary(values):
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return {"mean": None, "p50": None, "p90": None}
    q = torch.quantile(values, torch.tensor([0.5, 0.9], device=values.device))
    return {"mean": float(values.mean().item()), "p50": float(q[0].item()), "p90": float(q[1].item())}


def accepted_counts(camera, gaussians, pipeline, background, mult, mask):
    package = render_structgs(
        camera,
        gaussians,
        pipeline,
        background,
        mult,
        get_flag=True,
        metric_map=mask.to(device="cuda", dtype=torch.int32).reshape(-1).contiguous(),
        compute_extra=False,
    )
    return package["accum_metric_counts"].reshape(-1).to(torch.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mult", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, optimizer_type="default")
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    views = sorted(scene.getTrainCameras(), key=lambda camera: camera.image_name)
    if len(views) != 272:
        raise RuntimeError(f"expected 272 training cameras; found {len(views)}")
    selected_indices = [round(i * (len(views) - 1) / 7) for i in range(8)]
    selected = [(index, views[index]) for index in selected_indices]
    sizes = {(int(camera.image_width), int(camera.image_height)) for _, camera in selected}
    if sizes != {(1600, 1066)}:
        raise RuntimeError(f"selected camera dimensions differ from 1600x1066: {sorted(sizes)}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    rows = []
    with torch.no_grad():
        for index, camera in selected:
            package = render_structgs(
                camera, gaussians, pipe, background, args.mult, compute_extra=True
            )
            rendered = package["render"].float()
            gt = camera.original_image[:3].to(rendered.device).float()
            residual = (rendered - gt).abs().mean(dim=0)
            residual_summary = qsummary(residual)
            opacity_map = package["opacity_map"].float().clamp(0.0, 1.0)
            transmittance_map = 1.0 - opacity_map
            foreground = opacity_map > 0.1
            semitransparent = (opacity_map >= 0.1) & (opacity_map <= 0.9)
            high_residual = residual >= torch.quantile(residual.reshape(-1), 0.9)

            all_mask = torch.ones_like(high_residual, dtype=torch.int32)
            all_counts = accepted_counts(
                camera, gaussians, pipe, background, args.mult, all_mask
            )
            error_counts = accepted_counts(
                camera, gaussians, pipe, background, args.mult, high_residual
            )
            semi_counts = accepted_counts(
                camera, gaussians, pipe, background, args.mult, semitransparent
            )
            total_work = int(all_counts.sum().item())
            if total_work <= 0:
                raise RuntimeError(f"no accepted contributions for {camera.image_name}")

            radii = package["radii"].reshape(-1)
            cov = package["cov2D"].reshape(-1, 7)
            visible = radii > 0
            determinant = cov[:, 0] * cov[:, 2] - cov[:, 1] * cov[:, 1]
            footprint = 9.0 * math.pi * torch.sqrt(determinant.clamp_min(0.0))
            visible_ids = torch.where(visible)[0]
            footprint_order = visible_ids[torch.argsort(footprint[visible_ids], descending=True)]
            opacity = gaussians.get_opacity_with_3D_filter.reshape(-1)
            max_front_transmittance = cov[:, 6].clamp(0.0, 1.0)

            def work_share(ids):
                return float(all_counts[ids].sum().item() / total_work)

            def top_share(fraction):
                count = max(1, math.ceil(footprint_order.numel() * fraction))
                return work_share(footprint_order[:count])

            front_translucent = visible & (opacity >= 0.1) & (opacity <= 0.9) & (max_front_transmittance >= 0.1)
            depth_map = package["depth_map"].float()
            normalized_depth = depth_map / opacity_map.clamp_min(1e-8)
            valid_depth = normalized_depth[foreground]
            rows.append({
                "sorted_train_index": index,
                "camera_name": camera.image_name,
                "width": int(camera.image_width),
                "height": int(camera.image_height),
                "residual_rgb_abs_mean": float(residual.mean().item()),
                "residual_rgb_abs_p50": residual_summary["p50"],
                "residual_rgb_abs_p90": residual_summary["p90"],
                "top10pct_residual_pixel_fraction": float(high_residual.float().mean().item()),
                "visible_gaussians": int(visible.sum().item()),
                "projected_3sigma_footprint_pixel_area_sum": float(footprint[visible].sum().item()),
                "accepted_gaussian_pixel_pairs_all": total_work,
                "accepted_work_share_top1pct_projected_footprint_gaussians": top_share(0.01),
                "accepted_work_share_top10pct_projected_footprint_gaussians": top_share(0.10),
                "accepted_work_share_front_translucent_gaussians_opacity_0p1_0p9_and_maxT_ge_0p1": work_share(torch.where(front_translucent)[0]),
                "front_translucent_gaussians_count": int(front_translucent.sum().item()),
                "accepted_work_share_top10pct_residual_pixels": float(error_counts.sum().item() / total_work),
                "accepted_work_share_semitransparent_pixels_opacity_0p1_0p9": float(semi_counts.sum().item() / total_work),
                "foreground_pixel_fraction_opacity_gt_0p1": float(foreground.float().mean().item()),
                "semitransparent_pixel_fraction_opacity_0p1_0p9": float(semitransparent.float().mean().item()),
                "opacity_map_all_pixels": qsummary(opacity_map),
                "final_transmittance_all_pixels": qsummary(transmittance_map),
                "normalized_depth_foreground_pixels": qsummary(valid_depth),
                "gaussian_opacity_visible": qsummary(opacity[visible]),
                "max_precontribution_transmittance_visible": qsummary(max_front_transmittance[visible]),
            })
            print(f"completed sorted train index {index}: {camera.image_name}", flush=True)

    output = {
        "status": "passed",
        "model_path": dataset.model_path,
        "iteration": args.iteration,
        "source_path": dataset.source_path,
        "view_selection": "sorted image_name, indices round(i*(272-1)/7), i=0..7",
        "selected_indices": selected_indices,
        "render_mult": args.mult,
        "resolution": [1600, 1066],
        "accepted_work_definition": "sum of rasterizer accum_metric_counts; one count per Gaussian-pixel pair that passes power, alpha>=1/255, and test_T>=1e-4 in the forward compositor",
        "projected_footprint_definition": "9*pi*sqrt(det(screen-space covariance)), a projected 3-sigma ellipse area in pixels",
        "foreground_definition": "rendered opacity_map > 0.1; semitransparent pixels have opacity_map in [0.1, 0.9]",
        "records": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "output": str(output_path), "selected_indices": selected_indices}, indent=2), flush=True)


if __name__ == "__main__":
    main()
