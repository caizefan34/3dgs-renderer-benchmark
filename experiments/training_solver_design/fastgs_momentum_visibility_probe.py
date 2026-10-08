"""Audit a conservative frustum gate after real-state zero-gradient Adam drift.

This is a fixed-state diagnostic. It applies hypothetical zero-gradient Adam
steps to all Gaussian geometry/opacity rows from an owned checkpoint, then
compares positive raster radii before and after. It does not train a model.
"""

import argparse
import json
import math
import random
from types import SimpleNamespace

import torch

from gaussian_renderer import render_fastgs
from scene import GaussianModel, Scene


PARAMETER_SLOTS = {"xyz": "_xyz", "opacity": "_opacity", "scaling": "_scaling", "rotation": "_rotation"}


def drift_bound(moment, group, gap):
    beta1, beta2 = group["betas"]
    initial_m = moment["exp_avg"].double().abs()
    initial_v = moment["exp_avg_sq"].double()
    start = int(moment["step"].item())
    numerator = group["lr"] * initial_m * beta1 * (1.0 - beta1**gap) / (1.0 - beta1)
    denominator = (1.0 - beta1 ** (start + 1)) * (
        torch.sqrt(beta2**gap * initial_v / (1.0 - beta2 ** (start + gap))) + group["eps"]
    )
    return numerator / denominator


def apply_zero_gradient_debt(parameter, moment, group, gap):
    beta1, beta2 = group["betas"]
    m = moment["exp_avg"].to("cuda").clone()
    v = moment["exp_avg_sq"].to("cuda").clone()
    start = int(moment["step"].item())
    with torch.no_grad():
        for offset in range(1, gap + 1):
            m.mul_(beta1)
            v.mul_(beta2)
            step = start + offset
            parameter.sub_(
                group["lr"] * (m / (1.0 - beta1**step))
                / (torch.sqrt(v / (1.0 - beta2**step)) + group["eps"])
            )


def possible_from_sphere(camera, xyz, world_radius):
    matrix = camera.world_view_transform.double()
    position = xyz @ matrix[:3, :3] + matrix[3, :3]
    x, y, z = position.unbind(1)
    tan_x = math.tan(camera.FoVx * 0.5) * (1.0 + 4.0 / camera.image_width)
    tan_y = math.tan(camera.FoVy * 0.5) * (1.0 + 4.0 / camera.image_height)
    definitely_outside = (
        (z + world_radius <= 0.2)
        | (x.abs() - tan_x * z > world_radius * math.sqrt(1.0 + tan_x * tan_x))
        | (y.abs() - tan_y * z > world_radius * math.sqrt(1.0 + tan_y * tan_y))
    )
    return ~definitely_outside


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    assert args.gap > 0 and args.views > 0

    model_args = SimpleNamespace(
        source_path=args.source, model_path=args.model, images="images",
        eval=True, resolution=-1, white_background=False,
        data_device="cuda", sh_degree=3,
    )
    gaussians = GaussianModel(3, "default")
    scene = Scene(model_args, gaussians, load_iteration=10000, shuffle=False)
    model_state, checkpoint_iteration = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    gaussians.active_sh_degree = model_state[0]
    for slot, tensor in zip(
        ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"),
        model_state[1:7],
    ):
        setattr(gaussians, slot, torch.nn.Parameter(tensor.detach().clone().to("cuda"), requires_grad=False))
    optimizer_state = model_state[11]
    groups = {group["name"]: group for group in optimizer_state["param_groups"]}
    moments = {name: optimizer_state["state"][group["params"][0]] for name, group in groups.items()}
    xyz_start = gaussians._xyz.detach().double().clone()
    scale_start = gaussians._scaling.detach().double().clone()
    xyz_motion = torch.linalg.vector_norm(drift_bound(moments["xyz"], groups["xyz"], args.gap), dim=1).to("cuda")
    scale_motion = drift_bound(moments["scaling"], groups["scaling"], args.gap).to("cuda")
    max_scale = torch.exp(scale_start + scale_motion).max(dim=1).values

    cameras = random.Random(args.seed).sample(scene.getTrainCameras(), args.views)
    pipe = SimpleNamespace(debug=False, convert_SHs_python=False, compute_cov3D_python=False)
    background = torch.zeros(3, device="cuda")
    old_radii = []
    with torch.no_grad():
        for camera in cameras:
            old_radii.append(render_fastgs(camera, gaussians, pipe, background, 0.5)["radii"])
        for name, slot in PARAMETER_SLOTS.items():
            apply_zero_gradient_debt(getattr(gaussians, slot), moments[name], groups[name], args.gap)

        factors = (1.0, 1.5, 2.0, 4.0)
        misses = {str(factor): 0 for factor in factors}
        possible_counts = {str(factor): 0 for factor in factors}
        old_count = 0
        new_count = 0
        newly_projected = 0
        factor_two_miss_examples = []
        for camera, old_radius in zip(cameras, old_radii):
            new_radius = render_fastgs(camera, gaussians, pipe, background, 0.5)["radii"]
            old_mask, new_mask = old_radius > 0, new_radius > 0
            old_count += int(old_mask.sum())
            new_count += int(new_mask.sum())
            newly_projected += int((new_mask & ~old_mask).sum())
            for factor in factors:
                radius = factor * 3.0 * max_scale + xyz_motion
                possible = possible_from_sphere(camera, xyz_start, radius)
                misses[str(factor)] += int((new_mask & ~possible).sum())
                possible_counts[str(factor)] += int(possible.sum())
                if factor == 2.0 and len(factor_two_miss_examples) < 12:
                    indices = (new_mask & ~possible).nonzero(as_tuple=True)[0]
                    for index_tensor in indices[: 12 - len(factor_two_miss_examples)]:
                        index = int(index_tensor)
                        matrix = camera.world_view_transform.double()
                        view_position = xyz_start[index] @ matrix[:3, :3] + matrix[3, :3]
                        factor_two_miss_examples.append({
                            "camera": camera.image_name,
                            "gaussian_index": index,
                            "stale_view_position": view_position.cpu().tolist(),
                            "stale_world_position": xyz_start[index].cpu().tolist(),
                            "updated_world_position": gaussians._xyz[index].detach().double().cpu().tolist(),
                            "max_scale_after_bound": float(max_scale[index]),
                            "world_xyz_motion_bound": float(xyz_motion[index]),
                            "sphere_radius_factor2": float(radius[index]),
                            "old_raster_radius_px": float(old_radius[index]),
                            "new_raster_radius_px": float(new_radius[index]),
                        })

    total = len(cameras) * xyz_start.shape[0]
    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_iteration": checkpoint_iteration,
        "hypothetical_zero_gradient_steps": args.gap,
        "learning_rate_assumption": "each group uses its saved checkpoint lr throughout the gap",
        "source": args.source,
        "cameras": [camera.image_name for camera in cameras],
        "effective_resolution": [cameras[0].image_width, cameras[0].image_height],
        "gaussians": xyz_start.shape[0],
        "old_positive_radius_fraction": old_count / total,
        "new_positive_radius_fraction": new_count / total,
        "newly_positive_radius_cases": newly_projected,
        "sphere_possible_fraction_by_factor": {key: value / total for key, value in possible_counts.items()},
        "sphere_false_negatives_by_factor": misses,
        "factor_two_miss_examples": factor_two_miss_examples,
        "scope": "hypothetical all-zero gradients, sampled training cameras; no complete visibility proof",
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
