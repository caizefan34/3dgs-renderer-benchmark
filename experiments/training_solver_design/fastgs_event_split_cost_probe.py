"""Native FastGS room@4100 split-event tile work and paired kernel timing.

Uses the author's renderer unchanged. The gate itself is not implemented, so
timings describe native pre/post split endpoints, not shadow-pass runtime.
"""

import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


REPO = Path("/root/codex_fastgs_audit_20260927")
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/root")

from fastgs_real_split_shadow_probe import render_and_capture  # noqa: E402
from fastgs_real_tile_reference import sha256  # noqa: E402
from fused_ssim import fused_ssim as fast_ssim  # noqa: E402
from gaussian_renderer import render_fastgs  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.general_utils import build_rotation  # noqa: E402
from utils.loss_utils import l1_loss  # noqa: E402


def make_native_post(model, selected, seed=20260927):
    torch.manual_seed(seed)
    ids = selected.nonzero().flatten().to("cuda")
    count = ids.numel()
    stds = model.get_scaling[ids].repeat(2, 1)
    samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
    rots = build_rotation(model._rotation[ids]).repeat(2, 1, 1)
    child_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + model.get_xyz[ids].repeat(2, 1)
    child_scale = model.scaling_inverse_activation(stds / 1.6)
    extensions = {
        "_xyz": child_xyz,
        "_features_dc": model._features_dc[ids].repeat(2, 1, 1),
        "_features_rest": model._features_rest[ids].repeat(2, 1, 1),
        "_scaling": child_scale,
        "_rotation": model._rotation[ids].repeat(2, 1),
        "_opacity": model._opacity[ids].repeat(2, 1),
    }
    post = GaussianModel(model.max_sh_degree, "default")
    post.active_sh_degree = model.active_sh_degree
    opacity = model._opacity.detach().clone()
    opacity[ids] = -30.0  # Suppress selected parents at the native post-split endpoint.
    for name, child in extensions.items():
        old = opacity if name == "_opacity" else getattr(model, name).detach()
        setattr(post, name, nn.Parameter(torch.cat((old, child.detach()), dim=0)))
    return post, ids, count


def one_timing(camera, model, pipe, gt, backward):
    if backward:
        for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"):
            getattr(model, name).grad = None
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    if backward:
        image = render_fastgs(camera, model, pipe, torch.zeros(3, device="cuda"), 0.5)["render"]
        loss = 0.8 * l1_loss(image, gt) + 0.2 * (1 - fast_ssim(image.unsqueeze(0), gt.unsqueeze(0)))
        loss.backward()
    else:
        with torch.no_grad():
            _ = render_fastgs(camera, model, pipe, torch.zeros(3, device="cuda"), 0.5)["render"]
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def paired_timings(camera, base, post, pipe, gt, backward):
    for _ in range(3):
        one_timing(camera, base, pipe, gt, backward)
        one_timing(camera, post, pipe, gt, backward)
    times = {"before": [], "native_post": []}
    for repeat in range(12):
        order = (("before", base), ("native_post", post)) if repeat % 2 == 0 else (("native_post", post), ("before", base))
        for label, model in order:
            times[label].append(one_timing(camera, model, pipe, gt, backward))
    return {"raw_ms": times, "median_ms": {label: statistics.median(values) for label, values in times.items()}}


def main():
    dataset = SimpleNamespace(
        source_path="/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room",
        model_path=str(REPO / "output/room_big_capture4100"),
        images="images", resolution=-1, white_background=False,
        data_device="cuda", eval=True,
    )
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    model = GaussianModel(3, "default")
    scene = Scene(dataset, model, load_iteration=4100, shuffle=False)
    camera = scene.getTrainCameras()[0]
    n = int(model.get_xyz.shape[0])
    mask_path = Path("/root/fastgs_gate_selected_4100_mask.pt")
    selected = torch.load(mask_path, map_location="cpu", weights_only=True)["selected_splits"]
    if selected.numel() != n or selected.sum().item() != 9345:
        raise AssertionError("mask does not match the pre-event model")
    _, before_instances, before_geom, _, _ = render_and_capture(camera, model, pipe)
    post, selected_ids, selected_count = make_native_post(model, selected)
    _, after_instances, after_geom, _, _ = render_and_capture(camera, post, pipe)
    parent_tiles = int(before_geom["tiles"][selected_ids].sum().item())
    suppressed_parent_tiles = int(after_geom["tiles"][selected_ids].sum().item())
    child_tiles = int(after_geom["tiles"][n:].sum().item())
    expected_after = before_instances - parent_tiles + suppressed_parent_tiles + child_tiles
    if expected_after != after_instances:
        raise AssertionError((expected_after, after_instances))
    gt = camera.original_image.cuda()
    forward = paired_timings(camera, model, post, pipe, gt, backward=False)
    forward_backward = paired_timings(camera, model, post, pipe, gt, backward=True)
    result = {
        "source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
        "camera": camera.image_name,
        "resolution": [camera.image_width, camera.image_height],
        "model_ply_sha256": sha256(REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"),
        "mask_sha256": sha256(mask_path),
        "gaussians_before": n,
        "selected_split_parents": selected_count,
        "native_children_added": 2 * selected_count,
        "selected_parents_with_tiles": int((before_geom["tiles"][selected_ids] > 0).sum().item()),
        "children_with_tiles": int((after_geom["tiles"][n:] > 0).sum().item()),
        "base_tile_instances": before_instances,
        "native_post_tile_instances": after_instances,
        "selected_parent_tile_instances": parent_tiles,
        "suppressed_parent_tile_instances": suppressed_parent_tiles,
        "native_child_tile_instances": child_tiles,
        "fixed_support_shadow_union_tile_instances": before_instances + child_tiles,
        "extra_retained_parent_tile_instances_vs_native_post": parent_tiles - suppressed_parent_tiles,
        "child_pixel_slots_upper_bound": child_tiles * 256,
        "retained_parent_pixel_slots_upper_bound": parent_tiles * 256,
        "paired_forward_cuda_ms": forward,
        "paired_forward_loss_backward_cuda_ms": forward_backward,
        "scope": "One camera and a native-rule stochastic split of the fixed selected mask; native pre/post endpoint work only. Shadow-support GPU traversal and gate arithmetic are not implemented or timed.",
    }
    print("FASTGS_EVENT_SPLIT_COST " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
