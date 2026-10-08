#!/usr/bin/env python3
"""Read-only, fixed-view image diagnostics for the FastGS room PLY pair."""

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_rgb(tensor):
    array = tensor.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return np.round(array * 255).astype(np.uint8)


def tile_summary(error, tile=64):
    height, width = error.shape
    rows = [range(y, min(y + tile, height)) for y in range(0, height, tile)]
    cols = [range(x, min(x + tile, width)) for x in range(0, width, tile)]
    values = []
    for ys in rows:
        for xs in cols:
            y0, y1, x0, x1 = ys.start, ys.stop, xs.start, xs.stop
            values.append({
                "bbox_xyxy": [x0, y0, x1, y1],
                "mean_abs_error": float(error[y0:y1, x0:x1].mean()),
            })
    values.sort(key=lambda item: item["mean_abs_error"], reverse=True)
    return values[:5]


def candidate_summary(model, radii, top_k=12):
    radii = radii.detach().float()
    visible = radii > 0
    visible_radii = radii[visible]
    top_indices = torch.topk(radii, k=min(top_k, radii.numel())).indices
    xyz = model.get_xyz.detach()
    scales = model.get_scaling.detach()
    opacity = model.get_opacity.detach().flatten()
    candidates = []
    for index in top_indices.tolist():
        candidates.append({
            "ply_index": int(index),
            "projected_radius_px": float(radii[index].item()),
            "xyz": [float(v) for v in xyz[index].cpu().tolist()],
            "scale_xyz": [float(v) for v in scales[index].cpu().tolist()],
            "opacity": float(opacity[index].item()),
        })
    return {
        "visible_gaussians": int(visible.sum().item()),
        "radius_gt_32_px": int((radii > 32).sum().item()),
        "radius_gt_64_px": int((radii > 64).sum().item()),
        "visible_radius_quantiles_px": {
            str(q): float(torch.quantile(visible_radii, q).item())
            for q in (0.5, 0.9, 0.99, 0.999)
        } if visible_radii.numel() else {},
        "largest_projected_radius_candidates": candidates,
    }


def heat_rgb(values, scale):
    stops = np.asarray([
        [0, 0, 4], [38, 12, 62], [97, 25, 109], [159, 42, 99],
        [216, 75, 75], [247, 135, 55], [252, 202, 109], [252, 253, 191],
    ], dtype=np.float32)
    x = np.clip(values / max(scale, 1e-12), 0, 1) * (len(stops) - 1)
    low = np.floor(x).astype(np.int32)
    high = np.minimum(low + 1, len(stops) - 1)
    fraction = (x - low)[..., None]
    return np.round(stops[low] * (1 - fraction) + stops[high] * fraction).astype(np.uint8)


def save_panels(out_dir, name, gt, p1, p75, e1, e75, gap, scale):
    Image.fromarray(gt).save(out_dir / f"{name}_gt.png")
    Image.fromarray(p1).save(out_dir / f"{name}_p1.png")
    Image.fromarray(p75).save(out_dir / f"{name}_p75.png")
    for suffix, values in (("err_p1", e1), ("err_p75", e75), ("model_gap", gap)):
        Image.fromarray(heat_rgb(values, scale)).save(out_dir / f"{name}_{suffix}.png")

    width, height = 640, 426
    panel_gap, label_h = 8, 28
    sheet = Image.new("RGB", (3 * width + 4 * panel_gap, 2 * (height + label_h) + 3 * panel_gap), "white")
    draw = ImageDraw.Draw(sheet)
    panels = [
        (gt, "GT"), (p1, "p=1 render"), (p75, "p=0.75 render"),
        (heat_rgb(e1, scale), "|p=1 - GT|, mean RGB"),
        (heat_rgb(e75, scale), "|p=0.75 - GT|, mean RGB"),
        (heat_rgb(gap, scale), "|p=1 - p=0.75|, mean RGB"),
    ]
    for index, (array, title) in enumerate(panels):
        row, col = divmod(index, 3)
        x = panel_gap + col * (width + panel_gap)
        y = panel_gap + row * (height + label_h + panel_gap)
        draw.text((x + 3, y + 5), title, fill="black")
        tile = Image.fromarray(array).resize((width, height), Image.Resampling.LANCZOS)
        sheet.paste(tile, (x, y + label_h))
    draw.text((panel_gap, sheet.height - 19), f"Error maps use common magma-like scale 0 to {scale:.5f}", fill="black")
    sheet.save(out_dir / f"{name}_panels.png")

    height, width = gt.shape[:2]
    center_x, center_y = np.unravel_index(np.argmax(gap), gap.shape)[1], np.unravel_index(np.argmax(gap), gap.shape)[0]
    half = 128
    x0, x1 = max(0, center_x - half), min(width, center_x + half)
    y0, y1 = max(0, center_y - half), min(height, center_y + half)
    crop_images = [gt[y0:y1, x0:x1], p1[y0:y1, x0:x1], p75[y0:y1, x0:x1]]
    labels = ["GT", "p=1", "p=0.75"]
    crop = Image.new("RGB", (3 * (x1 - x0), (y1 - y0) + 28), "white")
    draw = ImageDraw.Draw(crop)
    for i, (array, label) in enumerate(zip(crop_images, labels)):
        tile = Image.fromarray(array)
        crop.paste(tile, (i * (x1 - x0), 28))
        draw.text((i * (x1 - x0) + 4, 5), label, fill="black")
    crop.save(out_dir / f"{name}_model_gap_hotspot.png")


def main():
    repo = Path(os.environ["FASTGS_REPO"]).resolve()
    dataset_root = Path(os.environ["FASTGS_DATASET"]).resolve()
    out_dir = Path(os.environ["FASTGS_OUT"]).resolve()
    p1_path = Path(os.environ["FASTGS_P1"]).resolve()
    p75_path = Path(os.environ["FASTGS_P75"]).resolve()
    names = ["DSCF4763", "DSCF4771", "DSCF4779"]

    os.chdir(repo)
    sys.path.insert(0, str(repo))
    from gaussian_renderer import render_fastgs
    from scene import GaussianModel
    from scene.dataset_readers import sceneLoadTypeCallbacks
    from utils.camera_utils import cameraList_from_camInfos

    dataset = SimpleNamespace(
        source_path=str(dataset_root), model_path=str(p1_path), images="images",
        resolution=-1, white_background=False, data_device="cuda", eval=True,
    )
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    scene_info = sceneLoadTypeCallbacks["Colmap"](str(dataset_root), "images", True)
    train_infos, test_infos = scene_info.train_cameras, scene_info.test_cameras
    if len(train_infos) != 272 or len(test_infos) != 39:
        raise RuntimeError(f"Expected split 272/39, got {len(train_infos)}/{len(test_infos)}")
    info_by_name = {camera.image_name: camera for camera in test_infos}
    if any(name not in info_by_name for name in names):
        raise RuntimeError(f"Requested view missing from held-out set: {names}")
    selected_infos = [info_by_name[name] for name in names]
    cameras = cameraList_from_camInfos(selected_infos, 1.0, dataset)
    camera_by_name = {camera.image_name: camera for camera in cameras}

    out_dir.mkdir(parents=True, exist_ok=True)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    result = {
        "source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
        "eval_script_sha256": sha256(repo / "fastgs_sparse_adjoint_eval.py"),
        "train_py_sha256": sha256(repo / "train.py"),
        "train_camera_count": len(train_infos),
        "development_camera_count": len(cameras),
        "development_camera_names": [camera.image_name for camera in test_infos],
        "evaluated_camera_names": names,
        "resolution": [camera_by_name[names[0]].image_width, camera_by_name[names[0]].image_height],
        "same_camera_objects_and_gt_reused_for_both_arms": True,
        "render_call": "render_fastgs(camera, model, pipe, zeros_cuda, 0.5), matching fastgs_sparse_adjoint_eval.py",
        "renderer_output_keys": None,
        "renderer_provides_alpha_or_depth": False,
        "fallback_diagnostic": "projected radii and largest-radius Gaussian candidates",
        "dataset_geometry_hashes": {},
        "views": {},
    }
    for relative in ("sparse/0/cameras.bin", "sparse/0/images.bin"):
        path = dataset_root / relative
        result["dataset_geometry_hashes"][relative] = sha256(path)

    camera_images = {}
    for name in names:
        matches = list((dataset_root / "images").glob(name + ".*"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one source image for {name}; found {matches}")
        camera_images[name] = {
            "path": str(matches[0]),
            "sha256": sha256(matches[0]),
        }
    result["source_ground_truth_images"] = camera_images

    model = GaussianModel(3, "default")
    arm_configs = [
        ("p1_control", p1_path),
        ("p75_sparse_adjoint", p75_path),
    ]
    for camera_name in names:
        camera = camera_by_name[camera_name]
        target = camera.original_image.cuda().clamp(0, 1)
        gt_image = tensor_rgb(target)
        predictions = {}
        view_data = {
            "camera": camera_name,
            "gt_image_sha256": camera_images[camera_name]["sha256"],
            "gt_tensor_shape_chw": list(target.shape),
            "same_target_camera_image_tensor_used_by_both_arms": True,
            "arms": {},
        }

        for arm_name, model_path in arm_configs:
            ply = model_path / "point_cloud/iteration_10000/point_cloud.ply"
            model.load_ply(str(ply))
            with torch.no_grad():
                rendered_pkg = render_fastgs(camera, model, pipe, background, 0.5)
            if result["renderer_output_keys"] is None:
                result["renderer_output_keys"] = sorted(rendered_pkg.keys())
            prediction = rendered_pkg["render"].clamp(0, 1)
            difference = (prediction - target).abs()
            mean_error = difference.mean(dim=0)
            error_map = mean_error.detach().cpu().numpy()
            mse = float(((prediction - target) ** 2).mean().item())
            l1 = float(difference.mean().item())
            psnr = float(20 * np.log10(1.0 / np.sqrt(max(mse, 1e-30))))
            pred_rgb = tensor_rgb(prediction)
            predictions[arm_name] = (pred_rgb, prediction.detach().cpu().permute(1, 2, 0).numpy(), error_map)
            per_channel_l1 = difference.mean(dim=(1, 2)).cpu().tolist()
            view_data["arms"][arm_name] = {
                "ply_path": str(ply),
                "ply_sha256": sha256(ply),
                "gaussians": int(model.get_xyz.shape[0]),
                "l1": l1,
                "mse": mse,
                "psnr_db": psnr,
                "per_channel_l1_rgb": [float(value) for value in per_channel_l1],
                "error_pixel_mean": float(error_map.mean()),
                "error_pixel_p50": float(np.quantile(error_map, 0.5)),
                "error_pixel_p90": float(np.quantile(error_map, 0.9)),
                "error_pixel_p99": float(np.quantile(error_map, 0.99)),
                "error_pixel_p99_9": float(np.quantile(error_map, 0.999)),
                "error_pixel_max": float(error_map.max()),
                "error_pixel_max_xy": [int(np.unravel_index(error_map.argmax(), error_map.shape)[1]), int(np.unravel_index(error_map.argmax(), error_map.shape)[0])],
                "highest_mean_error_tiles_64px": tile_summary(error_map),
                "radius_candidates": candidate_summary(model, rendered_pkg["radii"]),
            }

        p1_rgb, p1_float, p1_error = predictions["p1_control"]
        p75_rgb, p75_float, p75_error = predictions["p75_sparse_adjoint"]
        gap = np.abs(p1_float - p75_float).mean(axis=2)
        gap_y, gap_x = np.unravel_index(gap.argmax(), gap.shape)
        view_data["model_gap"] = {
            "mean_abs_rgb": float(gap.mean()),
            "max_mean_abs_rgb": float(gap.max()),
            "max_xy": [int(gap_x), int(gap_y)],
            "highest_mean_gap_tiles_64px": tile_summary(gap),
            "p1_minus_p75_pixel_error_mean": float(p1_error.mean() - p75_error.mean()),
            "fraction_pixels_p75_lower_mean_abs_error": float((p75_error < p1_error).mean()),
        }
        common_scale = float(max(np.quantile(p1_error, 0.995), np.quantile(p75_error, 0.995), np.quantile(gap, 0.995)))
        view_data["error_map_display"] = {"colormap": "magma", "vmin": 0.0, "vmax_common_p99_5": common_scale}
        save_panels(out_dir, camera_name, gt_image, p1_rgb, p75_rgb, p1_error, p75_error, gap, common_scale)
        view_data["pngs"] = [
            f"{camera_name}_gt.png", f"{camera_name}_p1.png", f"{camera_name}_p75.png",
            f"{camera_name}_err_p1.png", f"{camera_name}_err_p75.png", f"{camera_name}_model_gap.png",
            f"{camera_name}_panels.png", f"{camera_name}_model_gap_hotspot.png",
        ]
        result["views"][camera_name] = view_data

    (out_dir / "render_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_directory": str(out_dir),
        "views": {
            name: {
                arm: {
                    "l1": data["l1"], "mse": data["mse"], "psnr_db": data["psnr_db"],
                    "max_error_xy": data["error_pixel_max_xy"],
                }
                for arm, data in view["arms"].items()
            }
            for name, view in result["views"].items()
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
