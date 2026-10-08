"""Fixed-checkpoint audit of a source-derived FastGS screen-space visibility bound.

The bound includes zero-gradient Adam motion, maximal scale, the projected
covariance floor, and one tile of rasterizer rounding. This is a diagnostic,
not a production culler or a proof of end-to-end training equivalence.
"""

import argparse
import json
import math
import random
from types import SimpleNamespace

import torch

from fastgs_momentum_visibility_probe import (
    PARAMETER_SLOTS,
    apply_zero_gradient_debt,
    drift_bound,
)
from gaussian_renderer import render_fastgs
from scene import GaussianModel, Scene


def possible_from_screen_envelope(camera, xyz, xyz_motion, max_scale, mult):
    """Return a conservative proxy for positive FastGS raster radius.

    Requires unit-norm rendered rotations, scale_modifier=1, BLOCK_X/Y=16,
    positive camera depth, and the author's current centered perspective.
    The +2 pixel margin covers small projection and floating-point errors;
    it is heuristic until a numerical proof/large trajectory audit exists.
    """
    matrix = camera.world_view_transform.double()
    x, y, z = (xyz @ matrix[:3, :3] + matrix[3, :3]).unbind(1)
    z_low = z - xyz_motion - 1e-6
    z_high = z + xyz_motion + 1e-6
    safe_depth = z_low > 0.2
    z_low = z_low.clamp_min(0.2)
    z_high = z_high.clamp_min(1e-6)

    tan_x = math.tan(camera.FoVx * 0.5)
    tan_y = math.tan(camera.FoVy * 0.5)
    focal_x = camera.image_width / (2 * tan_x)
    focal_y = camera.image_height / (2 * tan_y)
    t_max = 2 * math.log(255.0) * mult

    # Cov_xx <= max_scale^2 * ||J_x||^2 + 0.3, with the source's 1.3 FoV clamp.
    extent_x = torch.sqrt(
        t_max * (max_scale.square() * (focal_x / z_low).square()
                 * (1 + (1.3 * tan_x) ** 2) + 0.3)
    )
    extent_y = torch.sqrt(
        t_max * (max_scale.square() * (focal_y / z_low).square()
                 * (1 + (1.3 * tan_y) ** 2) + 0.3)
    )
    min_abs_x = (x.abs() - xyz_motion).clamp_min(0) * focal_x / z_high
    min_abs_y = (y.abs() - xyz_motion).clamp_min(0) * focal_y / z_high
    outside_x = min_abs_x > camera.image_width / 2 + extent_x + 18
    outside_y = min_abs_y > camera.image_height / 2 + extent_y + 18
    return ~(safe_depth & (outside_x | outside_y)) & (z_high > 0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--views", type=int, default=40)
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--mult", type=float, default=0.5)
    args = parser.parse_args()
    assert args.gap > 0 and args.views >= 0 and args.mult > 0

    model_args = SimpleNamespace(
        source_path=args.source, model_path=args.model, images="images",
        eval=True, resolution=-1, white_background=False,
        data_device="cuda", sh_degree=3,
    )
    gaussians = GaussianModel(3, "default")
    scene = Scene(model_args, gaussians, load_iteration=10000, shuffle=False)
    model_state, checkpoint_iteration = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    gaussians.active_sh_degree = model_state[0]
    for slot, tensor in zip(
        ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"),
        model_state[1:7],
    ):
        setattr(gaussians, slot, torch.nn.Parameter(tensor.detach().clone().cuda(), requires_grad=False))
    optimizer_state = model_state[11]
    groups = {group["name"]: group for group in optimizer_state["param_groups"]}
    moments = {name: optimizer_state["state"][group["params"][0]] for name, group in groups.items()}
    xyz_start = gaussians._xyz.detach().double().clone()
    scale_start = gaussians._scaling.detach().double().clone()
    xyz_motion = torch.linalg.vector_norm(
        drift_bound(moments["xyz"], groups["xyz"], args.gap), dim=1
    ).cuda()
    scale_motion = drift_bound(moments["scaling"], groups["scaling"], args.gap).cuda()
    max_scale = torch.exp(scale_start + scale_motion).max(dim=1).values

    with torch.no_grad():
        for name, slot in PARAMETER_SLOTS.items():
            apply_zero_gradient_debt(getattr(gaussians, slot), moments[name], groups[name], args.gap)
        train_cameras = scene.getTrainCameras()
        cameras = (train_cameras if args.views == 0
                   else random.Random(args.seed).sample(train_cameras, args.views))
        pipe = SimpleNamespace(debug=False, convert_SHs_python=False, compute_cov3D_python=False)
        background = torch.zeros(3, device="cuda")
        possible_total = positive_total = misses = 0
        examples = []
        for camera in cameras:
            possible = possible_from_screen_envelope(
                camera, xyz_start, xyz_motion, max_scale, args.mult
            )
            positive = render_fastgs(camera, gaussians, pipe, background, args.mult)["radii"] > 0
            possible_total += int(possible.sum())
            positive_total += int(positive.sum())
            missed = (positive & ~possible).nonzero(as_tuple=True)[0]
            misses += len(missed)
            for index in missed[: max(0, 12 - len(examples))]:
                examples.append({"camera": camera.image_name, "index": int(index)})

    total = len(cameras) * xyz_start.shape[0]
    print(json.dumps({
        "checkpoint_iteration": checkpoint_iteration,
        "gaussians": xyz_start.shape[0],
        "views": len(cameras),
        "gap": args.gap,
        "mult": args.mult,
        "effective_resolution": [cameras[0].image_width, cameras[0].image_height],
        "positive_radius_fraction": positive_total / total,
        "screen_envelope_possible_fraction": possible_total / total,
        "screen_envelope_false_negatives": misses,
        "miss_examples": examples,
        "scope": "sampled cameras, hypothetical all-zero gradient drift; no full-training proof",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
