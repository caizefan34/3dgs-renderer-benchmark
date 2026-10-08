"""One-candidate full-resolution FastGS insertion formula parity probe.

Run on EPIC-05 in the pinned author environment; no author code is changed.
"""

import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import torch
from torch import nn


REPO = Path("/root/codex_fastgs_audit_20260927")
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/root")

from fastgs_event_split_cost_probe import make_native_post  # noqa: E402
from fastgs_real_split_shadow_probe import render_and_capture  # noqa: E402
from fastgs_real_tile_reference import sha256  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402


OUT = Path("/root/fastgs_birth_parity_result.json")


def append_child(model, post, child_id, q):
    result = GaussianModel(model.max_sh_degree, "default")
    result.active_sh_degree = model.active_sh_degree
    for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"):
        base = getattr(model, name).detach()
        child = getattr(post, name)[child_id:child_id + 1].detach().clone()
        if name == "_opacity":
            child.fill_(math.log(q / (1.0 - q)))
        setattr(result, name, nn.Parameter(torch.cat((base, child), dim=0)))
    return result


def raw_at(x, y, mean, conic):
    dx, dy = float(mean[0]) - x, float(mean[1]) - y
    power = -0.5 * (float(conic[0]) * dx * dx + float(conic[2]) * dy * dy) - float(conic[1]) * dx * dy
    if power > 0:
        return 0.0, 0.0
    gaussian = math.exp(power)
    return gaussian, min(0.99, float(conic[3]) * gaussian)


def context_at_pixel(x, y, width, means, conics, colors, depths, ranges, point_list, candidate_depth):
    tile = (y // 16) * ((width + 15) // 16) + x // 16
    begin, end = ranges[tile]
    prefix = np.zeros(3, dtype=np.float64)
    before_t = 1.0
    suffix = np.zeros(3, dtype=np.float64)
    behind_t = 1.0
    for j in range(int(begin), int(end)):
        i = int(point_list[j])
        _, alpha = raw_at(x, y, means[i], conics[i])
        if alpha < 1.0 / 255.0:
            continue
        if float(depths[i]) < candidate_depth:
            prefix += before_t * alpha * colors[i]
            before_t *= 1.0 - alpha
        else:
            suffix += behind_t * alpha * colors[i]
            behind_t *= 1.0 - alpha
    return prefix, before_t, suffix


def cpu_buffers(geom, img, ids, width, height):
    tw, th = (width + 15) // 16, (height + 15) // 16
    return (geom["mean2d"].detach().cpu().numpy(),
            geom["conic"].detach().cpu().numpy(),
            geom["rgb"].detach().cpu().numpy(),
            geom["depths"].detach().cpu().numpy(),
            img["ranges"][:tw * th].detach().cpu().numpy(),
            ids.detach().cpu().numpy())


def main():
    source = REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"
    mask_path = Path("/root/fastgs_gate_selected_4100_mask.pt")
    if sha256(source) != "64160970e5ea580aea4f79278a8fbe334d8d81ef6e304a903626843cc48f85bc":
        raise AssertionError("source PLY changed")
    mask = torch.load(mask_path, map_location="cpu", weights_only=True)["selected_splits"]
    dataset = SimpleNamespace(
        source_path="/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room",
        model_path=str(REPO / "output/room_big_capture4100"),
        images="images", resolution=-1, white_background=False,
        data_device="cuda", eval=True)
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    model = GaussianModel(3, "default")
    scene = Scene(dataset, model, load_iteration=4100, shuffle=False)
    camera = scene.getTrainCameras()[0]
    width, height = camera.image_width, camera.image_height
    n = int(model.get_xyz.shape[0])
    post, selected_ids, family_count = make_native_post(model, mask, seed=20260927)
    if family_count != 9345 or int(selected_ids[5311]) != 415040:
        raise AssertionError("family identity changed")
    with torch.no_grad():
        post._opacity[selected_ids] = model._opacity[selected_ids]
    child_id = n + 5311
    q = 0.4
    candidate = append_child(model, post, child_id, q)

    with torch.no_grad():
        base_image, _, base_geom, base_img, base_ids = render_and_capture(camera, model, pipe)
        _, _, post_geom, _, _ = render_and_capture(camera, post, pipe)
        candidate_image, _, candidate_geom, candidate_img, candidate_ids = render_and_capture(camera, candidate, pipe)
    means, conics, colors, depths, ranges, point_list = cpu_buffers(
        base_geom, base_img, base_ids, width, height)
    projected_mean = candidate_geom["mean2d"][n].cpu().numpy()
    projected_conic = candidate_geom["conic"][n].cpu().numpy()
    projected_depth = float(candidate_geom["depths"][n])
    post_mean = post_geom["mean2d"][child_id].cpu().numpy()
    if np.max(np.abs(post_mean - projected_mean)) > 1e-5:
        raise AssertionError("child projection differs after append")
    candidate_color = candidate_geom["rgb"][n].cpu().numpy().astype(np.float64)
    tile_ranges = candidate_img["ranges"].cpu().numpy()
    sorted_ids = candidate_ids.cpu().numpy()
    start_x = max(0, int(projected_mean[0]) - 30)
    stop_x = min(width, int(projected_mean[0]) + 31)
    start_y = max(0, int(projected_mean[1]) - 30)
    stop_y = min(height, int(projected_mean[1]) + 31)
    base = base_image.detach().cpu().permute(1, 2, 0).numpy().astype(np.float64)
    after = candidate_image.detach().cpu().permute(1, 2, 0).numpy().astype(np.float64)
    base_errors, insertion_errors, predicted_norms = [], [], []
    support_misses = 0
    for y in range(start_y, stop_y, 3):
        for x in range(start_x, stop_x, 3):
            g, alpha = raw_at(x, y, projected_mean, projected_conic)
            if alpha < 1.0 / 255.0:
                continue
            tile = (y // 16) * ((width + 15) // 16) + x // 16
            begin, end = tile_ranges[tile]
            if n not in sorted_ids[int(begin):int(end)]:
                support_misses += 1
                continue
            prefix, t, suffix = context_at_pixel(
                x, y, width, means, conics, colors, depths, ranges, point_list, projected_depth)
            predicted_base = prefix + t * suffix
            predicted_delta = t * alpha * (candidate_color - suffix)
            base_errors.append(float(np.max(np.abs(predicted_base - base[y, x]))))
            insertion_errors.append(float(np.max(np.abs(predicted_delta - (after[y, x] - base[y, x])))))
            predicted_norms.append(float(np.max(np.abs(predicted_delta))))
    if not base_errors:
        raise AssertionError("candidate had no sampled supported pixels")
    base_errors = np.array(base_errors)
    insertion_errors = np.array(insertion_errors)
    trusted = base_errors < 1e-5
    result = {
        "source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
        "source_ply_sha256": sha256(source),
        "camera": camera.image_name,
        "effective_resolution": [width, height],
        "family": 5311,
        "parent_id": int(selected_ids[5311]),
        "child_id_in_union": child_id,
        "opacity": q,
        "projected_effective_opacity": float(projected_conic[3]),
        "projected_opacity_ratio": float(projected_conic[3] / q),
        "projected_mean": projected_mean.tolist(),
        "projected_depth": projected_depth,
        "samples": len(base_errors),
        "support_misses": support_misses,
        "base_rgb_max_error": float(base_errors.max()),
        "base_rgb_median_error": float(np.median(base_errors)),
        "insertion_delta_rgb_max_error": float(insertion_errors.max()),
        "insertion_delta_rgb_median_error": float(np.median(insertion_errors)),
        "trusted_samples_base_error_lt_1e-5": int(trusted.sum()),
        "trusted_insertion_delta_rgb_max_error": float(insertion_errors[trusted].max()) if trusted.any() else None,
        "trusted_insertion_delta_rgb_median_error": float(np.median(insertion_errors[trusted])) if trusted.any() else None,
        "median_predicted_delta_absmax": float(np.median(predicted_norms)),
        "scope": "One candidate, one real full-resolution train camera; fixed geometry, no fitting or model continuation."
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("FASTGS_BIRTH_PARITY " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
