#!/usr/bin/env python3
"""Run the preregistered A-only room@10K opacity-ablation gate."""

import hashlib
import contextlib
import io
import json
import os
import subprocess
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


def synchronized_render(render_fastgs, camera, model, pipe, background, target, fused_ssim, metric_counts=False):
    torch.cuda.synchronize()
    started = time.perf_counter()
    kwargs = {}
    if metric_counts:
        kwargs["get_flag"] = True
        kwargs["metric_map"] = torch.ones(
            int(camera.image_height) * int(camera.image_width),
            dtype=torch.int32,
            device="cuda",
        )
    with torch.no_grad():
        package = render_fastgs(camera, model, pipe, background, 0.5, 1.0, **kwargs)
        image = package["render"]
        l1 = float((image - target).abs().mean().item())
        ssim_value = float(fused_ssim(image.unsqueeze(0), target.unsqueeze(0)).item())
        clamped_image = image.clamp(0, 1)
        clamped_target = target.clamp(0, 1)
        mse = float(((clamped_image - clamped_target) ** 2).mean().item())
        psnr_db = float(20.0 * np.log10(1.0 / np.sqrt(max(mse, 1e-30))))
        metrics = {
            "l1": l1,
            "fused_ssim": ssim_value,
            "loss": 0.8 * l1 + 0.2 * (1.0 - ssim_value),
            "clamped_mse": mse,
            "clamped_psnr_db": psnr_db,
        }
        if metric_counts:
            counts = package["accum_metric_counts"].detach().to("cpu").numpy().copy()
            radii = package["radii"].detach().to("cpu").numpy().copy()
            if counts.shape[0] != model.get_xyz.shape[0] or radii.shape[0] != model.get_xyz.shape[0]:
                raise RuntimeError("Per-Gaussian counter/radius output has an unexpected shape")
        else:
            counts = radii = None
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return metrics, counts, radii, seconds


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise RuntimeError("This experiment is authorized to use only physical GPU 2")

    repo = Path(os.environ["FASTGS_REPO"]).resolve()
    dataset_root = Path(os.environ["FASTGS_DATASET"]).resolve()
    p1_path = Path(os.environ["FASTGS_P1"]).resolve()
    out_dir = Path(os.environ["FASTGS_OUT"]).resolve()
    prereg_path = Path(os.environ["FASTGS_PREREG"]).resolve()
    expected_prereg_sha = os.environ["FASTGS_PREREG_SHA256"]
    expected_ply_sha = os.environ["FASTGS_EXPECTED_P1_SHA256"]

    if sha256(prereg_path) != expected_prereg_sha:
        raise RuntimeError("Pre-registration SHA-256 does not match the frozen value")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    expected_script_sha = prereg["execution"]["run_a_sha256"]
    if sha256(Path(__file__).resolve()) != expected_script_sha:
        raise RuntimeError("A-stage script SHA-256 does not match the pre-registration")
    if prereg["status"] != "frozen_before_A_measurement":
        raise RuntimeError("Pre-registration is not in frozen state")

    os.chdir(repo)
    sys.path.insert(0, str(repo))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != prereg["source"]["commit"]:
        raise RuntimeError(f"Unexpected author source commit: {head}")
    train_py_sha = sha256(repo / "train.py")
    if train_py_sha != prereg["source"]["train_py_sha256"]:
        raise RuntimeError(f"Unexpected train.py SHA-256: {train_py_sha}")
    source_files = [
        "train.py",
        "gaussian_renderer/__init__.py",
        "scene/gaussian_model.py",
        "utils/fast_utils.py",
    ]
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *source_files], check=True)
    for source_file in source_files:
        if subprocess.run(["git", "diff", "--cached", "--quiet", "HEAD", "--", source_file]).returncode:
            raise RuntimeError(f"Staged source change found in {source_file}")

    ply = p1_path / "point_cloud/iteration_10000/point_cloud.ply"
    ply_sha_before = sha256(ply)
    if ply_sha_before != expected_ply_sha:
        raise RuntimeError(f"Unexpected PLY SHA-256: {ply_sha_before}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite nonempty experiment output: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    from fused_ssim import fused_ssim
    from gaussian_renderer import render_fastgs
    from scene import GaussianModel
    from scene.dataset_readers import sceneLoadTypeCallbacks
    from utils.camera_utils import cameraList_from_camInfos

    dataset = SimpleNamespace(
        source_path=str(dataset_root), model_path=str(p1_path), images="images",
        resolution=-1, white_background=False, data_device="cuda", eval=True,
    )
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        scene_info = sceneLoadTypeCallbacks["Colmap"](str(dataset_root), "images", True)
    train_infos, test_infos = scene_info.train_cameras, scene_info.test_cameras
    if len(train_infos) != 272 or len(test_infos) != 39:
        raise RuntimeError(f"Expected 272 training and 39 held-out cameras; got {len(train_infos)}/{len(test_infos)}")
    train_names = sorted(info.image_name for info in train_infos)
    actual_names_sha = hashlib.sha256(json.dumps(train_names, separators=(",", ":")).encode()).hexdigest()
    if actual_names_sha != prereg["source"]["train_names_sha256_canonical_json"]:
        raise RuntimeError(f"Training camera name set hash differs: {actual_names_sha}")
    a_names = prereg["sampling"]["a_processing_order"]
    b_names = prereg["sampling"]["b_processing_order"]
    if len(set(a_names) | set(b_names)) != 16 or not set(a_names + b_names).issubset(set(train_names)):
        raise RuntimeError("Preregistered A/B sets are not 16 unique training cameras")
    train_info_by_name = {info.image_name: info for info in train_infos}
    cameras = cameraList_from_camInfos([train_info_by_name[name] for name in a_names], 1.0, dataset)
    camera_by_name = {camera.image_name: camera for camera in cameras}
    if any(name not in camera_by_name for name in a_names):
        raise RuntimeError("An A camera could not be constructed from the training split")
    if sorted((cameras[0].image_width, cameras[0].image_height)) != sorted(prereg["render_and_loss"]["resolution_px"]):
        raise RuntimeError("Full-resolution camera dimensions differ from preregistration")

    model = GaussianModel(3, "default")
    model.load_ply(str(ply))
    if int(model.get_xyz.shape[0]) != prereg["source"]["gaussian_count"]:
        raise RuntimeError(f"Unexpected Gaussian count: {model.get_xyz.shape[0]}")
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    gpu = torch.cuda.get_device_properties(0)

    stage_started = time.perf_counter()
    baseline_rows = {}
    baseline_times = []
    for name in a_names:
        camera = camera_by_name[name]
        target = camera.original_image.cuda()
        repeats = []
        for _ in range(prereg["A_gate"]["baseline_repeat_count_per_camera"]):
            metrics, _, _, seconds = synchronized_render(
                render_fastgs, camera, model, pipe, background, target, fused_ssim)
            repeats.append(metrics)
            baseline_times.append(seconds)
        values = np.asarray([row["loss"] for row in repeats], dtype=np.float64)
        median = float(np.median(values))
        baseline_rows[name] = {
            "repeats": repeats,
            "median": median,
            "max_abs_deviation_from_median": float(np.max(np.abs(values - median))),
        }
        print(f"A_BASELINE {name} {len(repeats)}", flush=True)

    noise_max = max(row["max_abs_deviation_from_median"] for row in baseline_rows.values())
    epsilon = max(1e-6, 5.0 * noise_max)

    counter_rows = []
    counts_by_view = []
    radii_by_view = []
    counter_times = []
    for name in a_names:
        camera = camera_by_name[name]
        metrics, counts, radii, seconds = synchronized_render(
            render_fastgs, camera, model, pipe, background, camera.original_image.cuda(),
            fused_ssim, metric_counts=True)
        if counts is None:
            raise RuntimeError("Accepted-pixel counter did not return counts")
        counter_rows.append({"camera": name, "render_metrics": metrics, "render_wall_seconds": seconds})
        counts_by_view.append(counts)
        radii_by_view.append(radii)
        counter_times.append(seconds)
        print(f"A_COUNT {name}", flush=True)

    counts_matrix = np.stack(counts_by_view, axis=0)
    radii_matrix = np.stack(radii_by_view, axis=0)
    visible_count = np.count_nonzero(radii_matrix > 0, axis=0)
    eligible = np.flatnonzero(visible_count >= 2)
    costs = counts_matrix.astype(np.int64).sum(axis=0)
    candidate_indices = sorted(eligible.tolist(), key=lambda i: (-int(costs[i]), int(i)))[:64]

    scan_path = out_dir / "candidate_scan.jsonl"
    candidate_rows = []
    candidate_times = []
    with scan_path.open("x", encoding="utf-8") as scan_stream:
        for position, index in enumerate(candidate_indices, start=1):
            per_camera = []
            raw_opacity = model._opacity[index].detach().clone()
            raw_opacity_value = float(raw_opacity.item())
            for name in a_names:
                camera = camera_by_name[name]
                saved = model._opacity[index].detach().clone()
                try:
                    with torch.no_grad():
                        model._opacity[index].fill_(-20.0)
                    metrics, _, _, seconds = synchronized_render(
                        render_fastgs, camera, model, pipe, background,
                        camera.original_image.cuda(), fused_ssim)
                finally:
                    with torch.no_grad():
                        model._opacity[index].copy_(saved)
                restored = bool(torch.equal(model._opacity[index].detach(), saved))
                if not restored:
                    raise RuntimeError(f"Raw opacity was not restored for PLY row {index}")
                candidate_times.append(seconds)
                per_camera.append({
                    "camera": name,
                    "baseline_loss_median": baseline_rows[name]["median"],
                    "ablated_metrics": metrics,
                    "delta_loss": metrics["loss"] - baseline_rows[name]["median"],
                    "raw_opacity_restored": restored,
                })
            deltas = np.asarray([row["delta_loss"] for row in per_camera], dtype=np.float64)
            mean_delta = float(deltas.mean())
            strong_a_count = int(np.count_nonzero(deltas <= -1e-5))
            qualifies = bool(mean_delta <= -epsilon and strong_a_count >= 1)
            row = {
                "ply_index": int(index),
                "raw_opacity": raw_opacity_value,
                "opacity": float(torch.sigmoid(raw_opacity).item()),
                "visible_A_view_count": int(visible_count[index]),
                "A_accepted_pixel_cost": int(costs[index]),
                "accepted_pixel_counts_by_A_camera": [int(counts_matrix[j, index]) for j in range(len(a_names))],
                "projected_radii_px_by_A_camera": [float(radii_matrix[j, index]) for j in range(len(a_names))],
                "mean_A_delta_loss": mean_delta,
                "A_camera_count_delta_le_minus_1e-5": strong_a_count,
                "qualifies_A": qualifies,
                "per_camera": per_camera,
                "all_opacity_restores_verified": all(item["raw_opacity_restored"] for item in per_camera),
            }
            candidate_rows.append(row)
            scan_stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            scan_stream.flush()
            print(
                f"A_CANDIDATE {position}/64 index={index} cost={int(costs[index])} "
                f"mean_delta={mean_delta:.9g} qualifies={qualifies}", flush=True)

    qualified = [row for row in candidate_rows if row["qualifies_A"]]
    selected = sorted(qualified, key=lambda row: (-row["A_accepted_pixel_cost"], row["ply_index"]))[:16]
    a_gate_pass = len(qualified) >= 16
    if not a_gate_pass:
        selected = []

    arrays_path = out_dir / "a_counts_radii.npz"
    np.savez_compressed(
        arrays_path,
        camera_names=np.asarray(a_names),
        accepted_pixel_counts=counts_matrix,
        projected_radii_px=radii_matrix,
    )
    ply_sha_after = sha256(ply)
    if ply_sha_after != ply_sha_before:
        raise RuntimeError("Frozen p=1 PLY SHA changed during the A experiment")
    total_wall = time.perf_counter() - stage_started
    result = {
        "experiment": prereg["experiment"],
        "phase": "A",
        "preregistration_sha256": expected_prereg_sha,
        "run_a_sha256": expected_script_sha,
        "source_commit": head,
        "source_tracked_files_clean": True,
        "physical_gpu_visible_to_process": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_name": gpu.name,
        "train_camera_count": len(train_infos),
        "heldout_camera_count": len(test_infos),
        "resolution_px": [int(cameras[0].image_width), int(cameras[0].image_height)],
        "A_camera_order": a_names,
        "B_camera_order_recorded_but_not_loaded": b_names,
        "epsilon_noise_max": noise_max,
        "epsilon_frozen_from_A": epsilon,
        "baseline_repeats": baseline_rows,
        "accepted_pixel_counter_renders": counter_rows,
        "visibility_eligible_gaussian_count": int(len(eligible)),
        "A_candidate_pool": {
            "rule": prereg["A_gate"]["candidate_pool"],
            "size": len(candidate_indices),
            "indices": [int(i) for i in candidate_indices],
        },
        "candidate_rows": candidate_rows,
        "qualifying_candidate_count": len(qualified),
        "selected_16": [
            {"ply_index": row["ply_index"], "A_accepted_pixel_cost": row["A_accepted_pixel_cost"], "mean_A_delta_loss": row["mean_A_delta_loss"]}
            for row in selected
        ],
        "A_gate_pass": bool(a_gate_pass),
        "next_action": "freeze selected indices then run preregistered B" if a_gate_pass else "STOP: fewer than 16 qualified; B and conditional controls were not accessed",
        "raw_files": {
            "candidate_scan_jsonl": str(scan_path),
            "all_A_accepted_counts_and_radii_npz": str(arrays_path),
        },
        "cost_accounting": {
            "baseline_repeat_render_count": len(baseline_times),
            "accepted_pixel_counter_render_count": len(counter_times),
            "candidate_intervention_render_count": len(candidate_times),
            "baseline_repeat_wall_seconds_each": baseline_times,
            "accepted_pixel_counter_wall_seconds_each": counter_times,
            "candidate_intervention_wall_seconds_each": candidate_times,
            "stage_wall_seconds": total_wall,
        },
        "model_integrity": {
            "p1_ply_path": str(ply),
            "p1_ply_sha256_before": ply_sha_before,
            "p1_ply_sha256_after": ply_sha_after,
            "ply_unchanged": ply_sha_after == ply_sha_before,
            "no_ply_or_checkpoint_written": True,
        },
        "interpretation": "Static training-view opacity ablations only; accepted-pixel counts proxy raster work and do not establish training speed, future optimizer safety, or novel-view quality.",
    }
    result_path = out_dir / "a_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(result_path),
        "epsilon": epsilon,
        "eligible": len(eligible),
        "candidate_pool": len(candidate_indices),
        "qualified": len(qualified),
        "selected_16": [row["ply_index"] for row in selected],
        "A_gate_pass": a_gate_pass,
        "stage_wall_seconds": total_wall,
        "ply_unchanged": ply_sha_after == ply_sha_before,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
