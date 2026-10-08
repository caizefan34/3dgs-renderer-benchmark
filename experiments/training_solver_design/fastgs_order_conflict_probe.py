"""Screen ray-depth ordering on sampled real FastGS pixels, without editing the trainer."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from fastgs_real_tile_reference import REPO, decode_geometry, decode_image, sha256, take
from diff_gaussian_rasterization_fastgs import _C
from gaussian_renderer import render_fastgs
from scene import GaussianModel, Scene


def accepted_rows(x, y, ids, means, conics, colors):
    rows = []
    trans = 1.0
    stopped = False
    for pos, gid in enumerate(ids):
        dx, dy = means[pos] - (x, y)
        co = conics[pos]
        power = -0.5 * (co[0] * dx * dx + co[2] * dy * dy) - co[1] * dx * dy
        if power > 0:
            continue
        alpha = min(0.99, float(co[3]) * math.exp(float(power)))
        if alpha < 1 / 255:
            continue
        if trans * (1 - alpha) < 1e-4:
            stopped = True
            break
        rows.append((int(gid), alpha, colors[pos]))
        trans *= 1 - alpha
    return rows, stopped


def composite(rows):
    rgb = np.zeros(3, dtype=np.float64)
    trans = 1.0
    for _, alpha, color in rows:
        rgb += trans * alpha * color
        trans *= 1 - alpha
    return rgb, trans


def ray_for_pixel(x, y, camera, inv_proj, center):
    clip = np.array([2 * (x + 0.5) / camera.image_width - 1,
                     2 * (y + 0.5) / camera.image_height - 1, 0.5, 1.0])
    world = clip @ inv_proj
    world = world[:3] / world[3]
    direction = world - center
    return direction / np.linalg.norm(direction)


def ray_depths(rows, direction, center, xyz, cov6):
    ids = np.array([row[0] for row in rows], dtype=np.int64)
    c = cov6[ids]
    cov = np.empty((len(ids), 3, 3), dtype=np.float64)
    cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2] = c[:, 0], c[:, 1], c[:, 2]
    cov[:, 1, 0], cov[:, 1, 1], cov[:, 1, 2] = c[:, 1], c[:, 3], c[:, 4]
    cov[:, 2, 0], cov[:, 2, 1], cov[:, 2, 2] = c[:, 2], c[:, 4], c[:, 5]
    delta = xyz[ids] - center
    inv_direction = np.linalg.solve(cov, np.broadcast_to(direction, delta.shape)[..., None])[..., 0]
    return np.sum(delta * inv_direction, axis=1) / np.sum(direction * inv_direction, axis=1)


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
    xyz = model.get_xyz.detach().cpu().numpy().astype(np.float64)
    cov6 = model.get_covariance().detach().cpu().numpy().astype(np.float64)
    rng = np.random.default_rng(20260927)
    output = {"source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
              "model_ply_sha256": sha256(REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"),
              "sampling": "256 uniform random pixels per selected full-resolution training camera; fixed native accepted support; exclude early-stopped rays",
              "views": []}
    for camera_index in (0, 17, 54, 121):
        camera = scene.getTrainCameras()[camera_index]
        captured = []
        original = _C.rasterize_gaussians

        def capture(*args):
            result = original(*args)
            captured.append(result)
            return result

        _C.rasterize_gaussians = capture
        try:
            image = render_fastgs(camera, model, pipe, torch.zeros(3, device="cuda"), 0.5)["render"]
            torch.cuda.synchronize()
        finally:
            _C.rasterize_gaussians = original
        if len(captured) != 1:
            raise AssertionError(len(captured))
        rendered, _, _, _, geom_buffer, bin_buffer, img_buffer, _, _ = captured[0]
        width, height = camera.image_width, camera.image_height
        geometry = decode_geometry(geom_buffer, len(xyz))
        image_state = decode_image(img_buffer, width * height)
        point_list, _ = take(bin_buffer, 0, rendered, torch.int32)
        point_list = point_list.cpu().numpy()
        ranges = image_state["ranges"].cpu().numpy()
        means = geometry["mean2d"].cpu().numpy().astype(np.float64)
        conics = geometry["conic"].cpu().numpy().astype(np.float64)
        colors = geometry["rgb"].cpu().numpy().astype(np.float64)
        native_image = image.detach().cpu().numpy().astype(np.float64)
        gt = camera.original_image.detach().cpu().numpy().astype(np.float64)
        inv_proj = np.linalg.inv(camera.full_proj_transform.detach().cpu().numpy().astype(np.float64))
        center = camera.camera_center.detach().cpu().numpy().astype(np.float64)
        stats = {"camera_index": camera_index, "camera": camera.image_name,
                 "resolution": [width, height], "sampled": 256, "empty": 0,
                 "early_stopped": 0, "singular": 0, "parity_failed": 0,
                 "usable": 0, "changed": 0, "better_l1": 0,
                 "mean_native_l1": 0.0, "mean_sorted_l1": 0.0,
                 "mean_rgb_change": 0.0, "max_rgb_change": 0.0,
                 "adjacent_inversions": 0, "accepted_rows": 0, "risks": []}
        for x, y in zip(rng.integers(0, width, 256), rng.integers(0, height, 256)):
            tile_id = (y // 16) * ((width + 15) // 16) + x // 16
            begin, end = ranges[tile_id]
            ids = point_list[begin:end]
            rows, stopped = accepted_rows(x, y, ids, means[ids], conics[ids], colors[ids])
            if not rows:
                stats["empty"] += 1
                continue
            if stopped:
                stats["early_stopped"] += 1
                continue
            base, _ = composite(rows)
            native = native_image[:, y, x]
            if np.max(np.abs(base - native)) > 1e-4:
                stats["parity_failed"] += 1
                continue
            direction = ray_for_pixel(x, y, camera, inv_proj, center)
            try:
                depths = ray_depths(rows, direction, center, xyz, cov6)
            except np.linalg.LinAlgError:
                stats["singular"] += 1
                continue
            if not np.all(np.isfinite(depths)):
                stats["singular"] += 1
                continue
            order = np.argsort(depths, kind="stable")
            reordered = [rows[j] for j in order]
            sorted_rgb, _ = composite(reordered)
            rgb_change = float(np.max(np.abs(sorted_rgb - base)))
            native_l1 = float(np.mean(np.abs(base - gt[:, y, x])))
            sorted_l1 = float(np.mean(np.abs(sorted_rgb - gt[:, y, x])))
            stats["usable"] += 1
            stats["changed"] += rgb_change > 1e-5
            stats["better_l1"] += sorted_l1 < native_l1
            stats["mean_native_l1"] += native_l1
            stats["mean_sorted_l1"] += sorted_l1
            stats["mean_rgb_change"] += rgb_change
            stats["max_rgb_change"] = max(stats["max_rgb_change"], rgb_change)
            stats["accepted_rows"] += len(rows)
            trans = 1.0
            for j in range(len(rows) - 1):
                a, b = rows[j][1], rows[j + 1][1]
                if depths[j] > depths[j + 1]:
                    stats["adjacent_inversions"] += 1
                    stats["risks"].append(float(trans * a * b * np.max(np.abs(rows[j][2] - rows[j + 1][2]))))
                trans *= 1 - a
        for key in ("mean_native_l1", "mean_sorted_l1", "mean_rgb_change"):
            stats[key] = stats[key] / stats["usable"] if stats["usable"] else None
        risks = np.array(stats.pop("risks"), dtype=np.float64)
        stats["risk_sum"] = float(risks.sum())
        stats["risk_top_decile_share"] = (float(np.sort(risks)[-max(1, math.ceil(len(risks) / 10)):].sum() / risks.sum())
                                          if len(risks) and risks.sum() else None)
        output["views"].append(stats)
        print("ORDER_SCREEN_VIEW " + json.dumps(stats, sort_keys=True), flush=True)
    print("ORDER_SCREEN_RESULT " + json.dumps(output, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
