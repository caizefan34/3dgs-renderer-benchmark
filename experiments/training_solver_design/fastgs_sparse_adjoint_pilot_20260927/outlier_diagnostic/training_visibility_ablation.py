#!/usr/bin/env python3
"""Read-only p=1 visibility scan and bounded training-view opacity probe."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_metrics(image, target, psnr_fn, fused_ssim):
    image = image.clamp(0, 1)
    target = target.clamp(0, 1)
    difference = image - target
    mse = float((difference * difference).mean().item())
    return {
        "l1": float(difference.abs().mean().item()),
        "mse": mse,
        "psnr_db": float(psnr_fn(image.unsqueeze(0), target.unsqueeze(0)).mean().item()),
        "fused_ssim": float(fused_ssim(image.unsqueeze(0), target.unsqueeze(0)).item()),
    }


def render_once(render_fastgs, camera, model, pipe, background, target, psnr_fn, fused_ssim, get_pixel_counts=False):
    torch.cuda.synchronize()
    started = time.perf_counter()
    metric_map = None
    if get_pixel_counts:
        metric_map = torch.ones(
            int(camera.image_height) * int(camera.image_width),
            dtype=torch.int32,
            device="cuda",
        )
    with torch.no_grad():
        package = render_fastgs(
            camera, model, pipe, background, 0.5,
            get_flag=True if get_pixel_counts else None,
            metric_map=metric_map,
        )
        values = image_metrics(package["render"], target, psnr_fn, fused_ssim)
    torch.cuda.synchronize()
    values["render_wall_seconds"] = time.perf_counter() - started
    return package, values


def main():
    repo = Path(os.environ["FASTGS_REPO"]).resolve()
    dataset_root = Path(os.environ["FASTGS_DATASET"]).resolve()
    out_dir = Path(os.environ["FASTGS_OUT"]).resolve()
    p1_path = Path(os.environ["FASTGS_P1"]).resolve()
    candidate_index = 689298
    fixed_neighbor_names = ["DSCF4770", "DSCF4772"]
    max_selected_training_views = 8

    os.chdir(repo)
    sys.path.insert(0, str(repo))
    from fused_ssim import fused_ssim
    from gaussian_renderer import render_fastgs
    from scene import GaussianModel
    from scene.dataset_readers import sceneLoadTypeCallbacks
    from utils.camera_utils import cameraList_from_camInfos
    from utils.image_utils import psnr

    dataset = SimpleNamespace(
        source_path=str(dataset_root), model_path=str(p1_path), images="images",
        resolution=-1, white_background=False, data_device="cuda", eval=True,
    )
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    scene_info = sceneLoadTypeCallbacks["Colmap"](str(dataset_root), "images", True)
    train_infos, test_infos = scene_info.train_cameras, scene_info.test_cameras
    if len(train_infos) != 272 or len(test_infos) != 39:
        raise RuntimeError(f"Expected split 272/39, got {len(train_infos)}/{len(test_infos)}")
    cameras = cameraList_from_camInfos(train_infos, 1.0, dataset)
    camera_by_name = {camera.image_name: camera for camera in cameras}
    if any(name not in camera_by_name for name in fixed_neighbor_names):
        raise RuntimeError(f"Expected fixed adjacent training frames {fixed_neighbor_names}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ply = p1_path / "point_cloud/iteration_10000/point_cloud.ply"
    ply_hash_before = sha256(ply)
    expected_sha = os.environ.get("FASTGS_EXPECTED_P1_SHA256")
    model = GaussianModel(3, "default")
    model.load_ply(str(ply))
    if candidate_index >= model.get_xyz.shape[0]:
        raise RuntimeError(f"Candidate {candidate_index} outside PLY with {model.get_xyz.shape[0]} rows")

    original_raw_opacity = model._opacity[candidate_index].detach().clone()
    original_opacity = float(model.get_opacity[candidate_index].item())
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    records = []
    render_times = []
    output_keys = None
    for camera_number, camera in enumerate(cameras, start=1):
        target = camera.original_image.cuda().clamp(0, 1)
        package, values = render_once(
            render_fastgs, camera, model, pipe, background, target, psnr, fused_ssim)
        if output_keys is None:
            output_keys = sorted(package.keys())
        radius = float(package["radii"][candidate_index].item())
        render_times.append(values["render_wall_seconds"])
        records.append({
            "camera": camera.image_name,
            "projected_radius_px": radius,
            "visible_radius_gt_0": radius > 0,
            "baseline_metrics": values,
        })
        if camera_number % 32 == 0 or camera_number == len(cameras):
            print(f"TRAIN_VISIBILITY_SCAN {camera_number}/{len(cameras)}", flush=True)

    record_by_name = {record["camera"]: record for record in records}
    fixed_names = list(fixed_neighbor_names)
    top_radius_names = [
        record["camera"] for record in sorted(
            (item for item in records if item["camera"] not in fixed_names),
            key=lambda item: (-item["projected_radius_px"], item["camera"]),
        )[:max_selected_training_views - len(fixed_names)]
    ]
    selected_names = fixed_names + top_radius_names
    interventions = []
    for name in selected_names:
        camera = camera_by_name[name]
        record = record_by_name[name]
        target = camera.original_image.cuda().clamp(0, 1)
        saved_raw_opacity = model._opacity[candidate_index].detach().clone()
        try:
            with torch.no_grad():
                model._opacity[candidate_index].fill_(-20.0)
            _, changed = render_once(
                render_fastgs, camera, model, pipe, background, target, psnr, fused_ssim)
        finally:
            with torch.no_grad():
                model._opacity[candidate_index].copy_(saved_raw_opacity)
        render_times.append(changed["render_wall_seconds"])
        baseline = record["baseline_metrics"]
        interventions.append({
            "camera": name,
            "selection_reason": "fixed adjacent filename" if name in fixed_names else "largest candidate radius after fixed adjacent frames",
            "candidate_radius_px": record["projected_radius_px"],
            "baseline_metrics": baseline,
            "opacity_near_zero_metrics": changed,
            "delta_ablated_minus_baseline": {
                key: changed[key] - baseline[key]
                for key in ("l1", "mse", "psnr_db", "fused_ssim")
            },
            "mse_reduction_percent": 100.0 * (baseline["mse"] - changed["mse"]) / baseline["mse"],
            "opacity_tensor_restored": bool(torch.equal(model._opacity[candidate_index].detach(), saved_raw_opacity)),
        })

    pixel_count_records = []
    for name in ["DSCF4771"] + selected_names:
        camera = camera_by_name[name] if name != "DSCF4771" else None
        if camera is None:
            test_info_by_name = {item.image_name: item for item in test_infos}
            camera = cameraList_from_camInfos([test_info_by_name[name]], 1.0, dataset)[0]
        target = camera.original_image.cuda().clamp(0, 1)
        package, values = render_once(
            render_fastgs, camera, model, pipe, background, target, psnr, fused_ssim,
            get_pixel_counts=True,
        )
        render_times.append(values["render_wall_seconds"])
        metric_counts = package["accum_metric_counts"]
        if metric_counts.numel() != model.get_xyz.shape[0]:
            raise RuntimeError(f"Expected per-Gaussian metric counts, got shape {tuple(metric_counts.shape)}")
        pixel_count_records.append({
            "camera": name,
            "candidate_ply_index": candidate_index,
            "candidate_radius_px": float(package["radii"][candidate_index].item()),
            "metric_map_selected_pixels": int(metric_map_size := int(camera.image_height) * int(camera.image_width)),
            "candidate_pixel_contribution_count": int(metric_counts[candidate_index].item()),
            "fraction_of_image_pixels_with_candidate_contribution": float(metric_counts[candidate_index].item() / max(metric_map_size, 1)),
            "baseline_metrics_from_counter_render": values,
            "counter_semantics": "Existing get_flag path atomically counts this Gaussian once for each pixel where it passes the rasterizer alpha/transmittance contribution checks; metric_map is all ones.",
        })

    radii = np.asarray([record["projected_radius_px"] for record in records], dtype=np.float64)
    positive = radii[radii > 0]
    quantiles = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    top_radius_views = sorted(records, key=lambda item: (-item["projected_radius_px"], item["camera"]))[:12]
    result = {
        "source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
        "train_camera_count": len(train_infos),
        "development_camera_count": len(test_infos),
        "resolution": [cameras[0].image_width, cameras[0].image_height],
        "render_keys": output_keys,
        "native_radius_scan": {
            "candidate_ply_index": candidate_index,
            "candidate_opacity": original_opacity,
            "camera_count": len(records),
            "radius_gt_0_count": int((radii > 0).sum()),
            "radius_gt_64_count": int((radii > 64).sum()),
            "all_camera_radius_quantiles_px_including_zeros": {
                str(q): float(np.quantile(radii, q)) for q in quantiles
            },
            "visible_radius_quantiles_px": {
                str(q): float(np.quantile(positive, q)) for q in quantiles
            } if positive.size else {},
            "adjacent_training_views": {
                name: record_by_name[name] for name in fixed_neighbor_names
            },
            "largest_radius_training_views": top_radius_views,
            "all_training_view_radii_and_baseline_metrics": records,
        },
        "intervention": {
            "candidate_ply_index": candidate_index,
            "opacity_before": original_opacity,
            "raw_opacity_replacement": -20.0,
            "raw_opacity_restored_after_each_render": all(row["opacity_tensor_restored"] for row in interventions),
            "selection_rule_fixed_before_metric_analysis": "Always include DSCF4770 and DSCF4772 by filename, then select the six largest-radius remaining training views. No GT metric enters selection.",
            "selected_training_views": selected_names,
            "selected_view_results": interventions,
        },
        "pixel_access_probe": {
            "uses_existing_fastgs_get_flag_metric_map_api": True,
            "metric_map": "all ones over the full image; no pixels excluded",
            "records": pixel_count_records,
            "available_measure": "per-Gaussian contributing pixel-instance count",
            "unavailable_measure": "exact per-Gaussian tile-instance count / bucket-list entries are not returned by the public render_fastgs output",
        },
        "model_integrity": {
            "p1_ply_path": str(ply),
            "p1_ply_sha256_before": ply_hash_before,
            "p1_ply_sha256_after": sha256(ply),
            "expected_p1_ply_sha256": expected_sha,
            "ply_unchanged_and_matches_expected": sha256(ply) == ply_hash_before and (sha256(ply) == expected_sha if expected_sha else True),
        },
        "render_cost": {
            "baseline_radius_and_metric_scan_render_calls": len(records),
            "opacity_intervention_render_calls": len(interventions),
            "pixel_count_instrumentation_render_calls": len(pixel_count_records),
            "total_render_calls": len(render_times),
            "sum_synchronized_render_wall_seconds": float(sum(render_times)),
            "per_render_wall_seconds": render_times,
        },
        "tile_instance_cost_api": {
            "renderer_output_has_per_gaussian_radius": True,
            "renderer_output_keys": output_keys,
            "direct_per_gaussian_tile_instance_or_pixel_access_count_available_from_render_fastgs": False,
            "diagnostic_limit": "Per-point tile-instance/pixel access attribution is not exposed in this render API; deriving it needs extra rasterizer instrumentation or a new kernel. This probe makes no unsupported cost claim.",
        },
        "selection_bias_note": "All ablation metrics use training views. The fixed filenames and radius-only top-view rule were set before evaluating their GT metrics. Training-view results test visibility and fit opportunity, not novel-view generalization.",
    }
    output_path = out_dir / "training_visibility_ablation.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_path": str(output_path),
        "candidate": candidate_index,
        "visible_train_view_count": result["native_radius_scan"]["radius_gt_0_count"],
        "radius_max_px": float(radii.max()),
        "selected_training_views": [
            {"camera": row["camera"], "radius_px": row["candidate_radius_px"], "delta_psnr_db": row["delta_ablated_minus_baseline"]["psnr_db"], "delta_fused_ssim": row["delta_ablated_minus_baseline"]["fused_ssim"]}
            for row in interventions
        ],
        "total_render_calls": len(render_times),
        "render_seconds": float(sum(render_times)),
        "ply_unchanged_and_matches_expected": result["model_integrity"]["ply_unchanged_and_matches_expected"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
