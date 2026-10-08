"""Measure training-camera frustum coverage for a saved FastGS room state.

This is a read-only opportunity diagnostic. Radii > 0 is the renderer's
projection/frustum test, not true unoccluded visibility or an Adam speedup.
"""

import argparse
import json
import math
import random
from types import SimpleNamespace

import torch

from gaussian_renderer import render_fastgs
from scene import GaussianModel, Scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--views", type=int, default=40)
    parser.add_argument("--sphere-inflate", type=float, default=0.0)
    parser.add_argument("--drift", type=float, default=0.0)
    args = parser.parse_args()

    model_args = SimpleNamespace(
        source_path=args.source,
        model_path=args.model,
        images="images",
        eval=True,
        resolution=-1,
        white_background=False,
        data_device="cuda",
        sh_degree=3,
    )
    gaussians = GaussianModel(3, "default")
    scene = Scene(model_args, gaussians, load_iteration=args.iteration, shuffle=False)
    pipe = SimpleNamespace(debug=False, convert_SHs_python=False, compute_cov3D_python=False)
    camera_pool = scene.getTrainCameras()
    cameras = random.Random(args.seed).sample(camera_pool, min(args.views, len(camera_pool)))
    background = torch.zeros(3, device="cuda")
    projected = []
    possible_fractions = []
    sphere_false_negatives = 0
    positive_behind_near = 0
    if args.sphere_inflate > 0:
        xyz = gaussians.get_xyz.detach()
        radius = 3.0 * gaussians.get_scaling.detach().max(dim=1).values * args.sphere_inflate + args.drift
    with torch.no_grad():
        for camera in cameras:
            radii = render_fastgs(camera, gaussians, pipe, background, 0.5)["radii"]
            projected.append((radii > 0).cpu())
            if args.sphere_inflate > 0:
                matrix = camera.world_view_transform
                position = xyz @ matrix[:3, :3] + matrix[3, :3]
                x, y, z = position.unbind(1)
                tx = math.tan(camera.FoVx * 0.5) * (1.0 + 4.0 / camera.image_width)
                ty = math.tan(camera.FoVy * 0.5) * (1.0 + 4.0 / camera.image_height)
                outside = (
                    (z + radius <= 0.2)
                    | (x.abs() - tx * z > radius * math.sqrt(1.0 + tx * tx))
                    | (y.abs() - ty * z > radius * math.sqrt(1.0 + ty * ty))
                )
                possible = ~outside
                possible_fractions.append(float(possible.float().mean()))
                sphere_false_negatives += int(((radii > 0) & outside).sum())
                positive_behind_near += int(((radii > 0) & (z <= 0.2)).sum())
    mask = torch.stack(projected)
    per_view = mask.float().mean(1)
    per_gaussian = mask.float().mean(0)
    last_seen = torch.full((mask.shape[1],), -1, dtype=torch.int32)
    revisit_gaps = []
    for view_index, current in enumerate(mask):
        revisit = current & (last_seen >= 0)
        revisit_gaps.append((view_index - last_seen[revisit]).float())
        last_seen[current] = view_index
    revisit_gaps = torch.cat(revisit_gaps)
    gap_quantiles = torch.quantile(revisit_gaps, torch.tensor([0.5, 0.9, 0.99])).tolist()
    result = {
        "model": args.model,
        "iteration": args.iteration,
        "effective_resolution": [cameras[0].image_width, cameras[0].image_height],
        "training_camera_count": len(camera_pool),
        "sampled_training_cameras": [c.image_name for c in cameras],
        "gaussians": int(mask.shape[1]),
        "mean_projected_fraction_per_view": float(per_view.mean()),
        "min_projected_fraction_per_view": float(per_view.min()),
        "max_projected_fraction_per_view": float(per_view.max()),
        "fraction_projected_in_no_sampled_view": float((per_gaussian == 0).float().mean()),
        "fraction_projected_in_every_sampled_view": float((per_gaussian == 1).float().mean()),
        "median_projected_view_fraction_per_gaussian": float(per_gaussian.median()),
        "revisit_gap_steps_p50_p90_p99": gap_quantiles,
        "revisit_gap_scope": "sampled-camera sequence only; not the author training camera sequence",
        "radii_semantics": "positive projected radius; not unoccluded visibility",
    }
    if args.sphere_inflate > 0:
        result.update({
            "sphere_inflate": args.sphere_inflate,
            "sphere_drift_world": args.drift,
            "sphere_possible_fraction_mean": sum(possible_fractions) / len(possible_fractions),
            "sphere_false_negatives_vs_positive_radius": sphere_false_negatives,
            "positive_radius_behind_computed_near_plane": positive_behind_near,
            "sphere_scope": "empirical proxy; not a proof of renderer-support enclosure",
        })
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
