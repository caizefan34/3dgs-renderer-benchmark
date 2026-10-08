#!/usr/bin/env python3
"""One-at-a-time, in-memory opacity ablations for large-radius p=1 Gaussians."""

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


def metrics(rendered, target):
    diff = rendered.clamp(0, 1) - target.clamp(0, 1)
    l1 = float(diff.abs().mean().item())
    mse = float((diff * diff).mean().item())
    psnr = float(20.0 * np.log10(1.0 / np.sqrt(max(mse, 1e-30))))
    return {"l1": l1, "mse": mse, "psnr_db": psnr}


def render_once(render_fastgs, camera, model, pipe, background, target):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        package = render_fastgs(camera, model, pipe, background, 0.5)
        image = package["render"].clamp(0, 1)
        values = metrics(image, target)
    torch.cuda.synchronize()
    values["render_wall_seconds"] = time.perf_counter() - start
    return package, image, values


def candidate_record(model, radii, index):
    return {
        "ply_index": int(index),
        "projected_radius_px": float(radii[index].item()),
        "xyz": [float(value) for value in model.get_xyz[index].detach().cpu().tolist()],
        "scale_xyz": [float(value) for value in model.get_scaling[index].detach().cpu().tolist()],
        "opacity_before": float(model.get_opacity[index].item()),
        "raw_opacity_before": float(model._opacity[index].item()),
        "opacity_after": float(torch.sigmoid(torch.tensor(-20.0)).item()),
    }


def ablate_one(render_fastgs, camera, model, pipe, background, target, index, candidate):
    saved_raw_opacity = model._opacity[index].detach().clone()
    try:
        with torch.no_grad():
            model._opacity[index].fill_(-20.0)
        _, _, changed = render_once(render_fastgs, camera, model, pipe, background, target)
    finally:
        with torch.no_grad():
            model._opacity[index].copy_(saved_raw_opacity)
    return {
        "candidate": candidate,
        "ablated_metrics": changed,
        "delta_ablated_minus_baseline": {
            key: changed[key] - candidate["baseline_metrics"][key]
            for key in ("l1", "mse", "psnr_db")
        },
        "mse_reduction_percent": 100.0 * (candidate["baseline_metrics"]["mse"] - changed["mse"]) / candidate["baseline_metrics"]["mse"],
        "opacity_tensor_restored": bool(torch.equal(model._opacity[index].detach(), saved_raw_opacity)),
    }


def main():
    repo = Path(os.environ["FASTGS_REPO"]).resolve()
    dataset_root = Path(os.environ["FASTGS_DATASET"]).resolve()
    out_dir = Path(os.environ["FASTGS_OUT"]).resolve()
    p1_path = Path(os.environ["FASTGS_P1"]).resolve()
    names = ["DSCF4763", "DSCF4771", "DSCF4779"]
    target_name = "DSCF4771"
    max_candidates = 8
    opacity_raw_replacement = -20.0
    neighbor_gate_mse_reduction_percent = 1.0

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
    selected_infos = [info_by_name[name] for name in names]
    cameras = cameraList_from_camInfos(selected_infos, 1.0, dataset)
    camera_by_name = {camera.image_name: camera for camera in cameras}

    out_dir.mkdir(parents=True, exist_ok=True)
    ply = p1_path / "point_cloud/iteration_10000/point_cloud.ply"
    model = GaussianModel(3, "default")
    model.load_ply(str(ply))
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    focal_camera = camera_by_name[target_name]
    focal_target = focal_camera.original_image.cuda().clamp(0, 1)
    render_keys, baseline_package, baseline_image, baseline = None, None, None, None
    baseline_package, baseline_image, baseline = render_once(
        render_fastgs, focal_camera, model, pipe, background, focal_target)
    render_keys = sorted(baseline_package.keys())

    radii = baseline_package["radii"].detach().float()
    visible = radii > 0
    visible_indices = visible.nonzero().flatten()
    order = torch.argsort(radii[visible_indices], descending=True)[:max_candidates]
    candidate_indices = visible_indices[order].tolist()
    focal_candidates = []
    for index in candidate_indices:
        record = candidate_record(model, radii, index)
        record["baseline_metrics"] = baseline
        focal_candidates.append(record)

    all_render_times = [baseline["render_wall_seconds"]]
    interventions = []
    for index, candidate in zip(candidate_indices, focal_candidates):
        result = ablate_one(
            render_fastgs, focal_camera, model, pipe, background, focal_target, index, candidate)
        all_render_times.append(result["ablated_metrics"]["render_wall_seconds"])
        interventions.append(result)

    best = max(interventions, key=lambda item: item["delta_ablated_minus_baseline"]["psnr_db"])
    qualifies_for_neighbor_check = best["mse_reduction_percent"] >= neighbor_gate_mse_reduction_percent
    neighbor_checks = {}
    if qualifies_for_neighbor_check:
        best_index = best["candidate"]["ply_index"]
        for name in ("DSCF4763", "DSCF4779"):
            camera = camera_by_name[name]
            target = camera.original_image.cuda().clamp(0, 1)
            _, _, neighbor_baseline = render_once(render_fastgs, camera, model, pipe, background, target)
            all_render_times.append(neighbor_baseline["render_wall_seconds"])
            candidate = candidate_record(model, radii, best_index)
            candidate["baseline_metrics"] = neighbor_baseline
            intervention = ablate_one(
                render_fastgs, camera, model, pipe, background, target, best_index, candidate)
            all_render_times.append(intervention["ablated_metrics"]["render_wall_seconds"])
            neighbor_checks[name] = {
                "baseline_metrics": neighbor_baseline,
                "best_candidate_intervention": intervention,
            }

    expected_sha = os.environ.get("FASTGS_EXPECTED_P1_SHA256")
    result = {
        "source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
        "train_camera_count": len(train_infos),
        "development_camera_count": len(test_infos),
        "render_keys": render_keys,
        "renderer_has_alpha_or_depth": False,
        "ply_path": str(ply),
        "ply_sha256_before_and_after": sha256(ply),
        "expected_ply_sha256": expected_sha,
        "ply_hash_matches_expected": (sha256(ply) == expected_sha) if expected_sha else None,
        "target_view": target_name,
        "same_author_render_call": "render_fastgs(camera, model, pipe, zero_cuda, 0.5)",
        "perturbation": {
            "scope": "in-memory p=1 PLY tensor only; one Gaussian at a time",
            "attribute": "raw model._opacity[index] temporarily filled with -20, yielding sigmoid opacity near zero",
            "restore_verified_for_every_intervention": all(item["opacity_tensor_restored"] for item in interventions) and all(item["best_candidate_intervention"]["opacity_tensor_restored"] for item in neighbor_checks.values()),
            "max_focal_candidates": max_candidates,
            "candidate_selection": "largest positive projected radius among visible Gaussians in baseline DSCF4771 render",
            "neighbor_check_gate": f"best focal candidate reduces MSE by at least {neighbor_gate_mse_reduction_percent:.1f}%",
            "neighbor_check_run": qualifies_for_neighbor_check,
            "neighbor_check_reason": "threshold met" if qualifies_for_neighbor_check else "no candidate met the predeclared MSE-reduction threshold",
        },
        "focal_baseline_metrics": baseline,
        "focal_interventions": interventions,
        "best_focal_candidate": best,
        "neighbor_checks": neighbor_checks,
        "render_cost": {
            "render_call_count": len(all_render_times),
            "render_wall_seconds_each": all_render_times,
            "total_render_wall_seconds": float(sum(all_render_times)),
        },
        "selection_bias_note": "DSCF4771 is a development-retain view already selected after observing its paired metrics. This post-hoc intervention diagnoses this view only and is not independent evidence of quality gain.",
    }
    output_path = out_dir / "opacity_ablation_probe.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_path": str(output_path),
        "baseline": baseline,
        "candidates": [
            {"index": item["candidate"]["ply_index"], "radius_px": item["candidate"]["projected_radius_px"], "delta_psnr_db": item["delta_ablated_minus_baseline"]["psnr_db"], "mse_reduction_percent": item["mse_reduction_percent"]}
            for item in interventions
        ],
        "best_candidate": best["candidate"]["ply_index"],
        "neighbor_check_run": qualifies_for_neighbor_check,
        "render_calls": len(all_render_times),
        "total_render_wall_seconds": float(sum(all_render_times)),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
