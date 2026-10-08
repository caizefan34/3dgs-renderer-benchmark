"""Measure native accepted-hit overlap with the fixed top-L1-residual pixels.

This frozen-state diagnostic loads only the four protocol-selected eval-train
images per scene. It does not change the author renderer or model parameters.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone


REPO = Path("/root/codex_fastgs_audit_20260927")
DATA_ROOT = Path("/root/3dgs-renderer-benchmark/data/datasets/mipnerf360")
PROTOCOL_PATH = REPO / "FASTGS_RESIDUAL_COST_OVERLAP_PROTOCOL.md"
OUTPUT_PATH = Path("/root/fastgs_residual_cost_overlap_result.json")
PIN = "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f"
TRAIN_SHA256 = "1fd700c341d109f13f95ff3a3cdf1328445f27909119e7eacadb31913e34c963"
GPU_INDEX = 2
EVAL_LLFF_HOLD = 8
RGB_TOLERANCE = 1e-6

SCENES = {
    "room": {
        "score_json": Path("/root/fastgs_score_cost_state_result.json"),
        "views": ["DSCF4696", "DSCF4789", "DSCF4956", "DSCF4830"],
        "states": [
            {
                "state_id": "room_4100",
                "model_name": "room_big_capture4100",
                "iteration": 4100,
                "ply_sha256": "64160970e5ea580aea4f79278a8fbe334d8d81ef6e304a903626843cc48f85bc",
            },
            {
                "state_id": "room_30000",
                "model_name": "room_big_branch_base30k",
                "iteration": 30000,
                "ply_sha256": "c2afc615fd244d561bd790e7d7721af1b7a7515222bd5385650e7b886c58516a",
            },
        ],
    },
    "garden": {
        "score_json": Path("/root/fastgs_score_cost_garden_result.json"),
        "views": ["DSC07973", "DSC08133", "DSC08126", "DSC08050"],
        "states": [
            {
                "state_id": "garden_4100",
                "model_name": "garden_exchange_dev_4100",
                "iteration": 4100,
                "ply_sha256": "d396092ca139fa0e6aea4d17a085ca0239bf2ff0ec2bbbe0ee3c5ab539feb5fb",
            },
            {
                "state_id": "garden_30000",
                "model_name": "garden_native_stage_profile_20260927",
                "iteration": 30000,
                "ply_sha256": "ba12264768db1e09ebdb32c8c57a6be506300d4a354c459404b520d6a0977eaa",
            },
        ],
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_result(result):
    temporary = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, OUTPUT_PATH)


def new_result():
    return {
        "schema": "fastgs_residual_cost_overlap.v1",
        "status": "not_run",
        "source_commit": PIN,
        "expected_train_sha256": TRAIN_SHA256,
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": None,
        "probe_path": str(Path(__file__).resolve()),
        "probe_sha256": sha256(__file__),
        "scope": "Frozen-state spatial overlap diagnostic. It tests same-region residual/work co-location and does not assess global quality-work mechanisms.",
        "opened_training_images": [],
        "failures": [],
        "states": [
            {
                "scene": scene_name,
                "state_id": state["state_id"],
                "iteration": state["iteration"],
                "model_name": state["model_name"],
                "expected_ply_sha256": state["ply_sha256"],
                "status": "pending",
                "views": [
                    {"camera_name": camera_name, "status": "pending"}
                    for camera_name in scene["views"]
                ],
            }
            for scene_name, scene in SCENES.items()
            for state in scene["states"]
        ],
    }


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_protocol(result):
    require(PROTOCOL_PATH.is_file(), f"Frozen protocol is missing: {PROTOCOL_PATH}")
    actual = sha256(PROTOCOL_PATH)
    expected = os.environ.get("FASTGS_PREREG_SHA256", "").strip().lower()
    require(len(expected) == 64, "FASTGS_PREREG_SHA256 must contain the preregistered protocol SHA256")
    require(actual == expected, f"Protocol SHA mismatch: expected {expected}, found {actual}")
    result["protocol_sha256"] = actual
    result["preregistered_protocol_sha256"] = expected


def verify_source(result):
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    require(head == PIN, f"Author repository HEAD mismatch: {head}")
    train_path = REPO / "train.py"
    train_hash = sha256(train_path)
    require(train_hash == TRAIN_SHA256, f"train.py SHA256 mismatch: {train_hash}")
    subprocess.run(
        ["git", "diff", "--exit-code", "--", "train.py", "scene", "gaussian_renderer",
         "utils", "submodules/diff-gaussian-rasterization_fastgs"],
        cwd=REPO, check=True, stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["git", "diff", "--cached", "--exit-code", "--", "train.py", "scene", "gaussian_renderer",
         "utils", "submodules/diff-gaussian-rasterization_fastgs"],
        cwd=REPO, check=True, stdout=subprocess.DEVNULL,
    )
    result["source_commit"] = head
    result["train_sha256"] = train_hash


def verify_states(result):
    for scene_name, scene in SCENES.items():
        for state in scene["states"]:
            ply = (REPO / "output" / state["model_name"] / "point_cloud" /
                   f"iteration_{state['iteration']}" / "point_cloud.ply")
            require(ply.is_file(), f"Frozen PLY is missing: {ply}")
            actual = sha256(ply)
            require(actual == state["ply_sha256"], f"Frozen PLY SHA256 mismatch: {ply}: {actual}")
            state["ply_path"] = str(ply)
        score_path = scene["score_json"]
        require(score_path.is_file(), f"Frozen score-cost JSON is missing: {score_path}")
        score = json.loads(score_path.read_text(encoding="utf-8"))
        require(score.get("source_commit") == PIN, f"Score-cost source mismatch: {score_path}")
        require(score.get("scene", scene_name) == scene_name, f"Score-cost scene mismatch: {score_path}")
        rows = score.get("rows", [])
        require(rows, f"Score-cost camera list is empty: {score_path}")
        for row in rows:
            names = row.get("camera_names", [])
            require(names[:4] == scene["views"],
                    f"Frozen first-four camera names mismatch in {score_path}: {names[:4]}")
        scene["score_json_sha256"] = sha256(score_path)
        scene["score_json_path"] = str(score_path)
    result["frozen_score_cost_json"] = {
        name: {"path": scene["score_json_path"], "sha256": scene["score_json_sha256"]}
        for name, scene in SCENES.items()
    }


def check_gpu2_idle():
    env_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    require(env_visible == str(GPU_INDEX),
            f"Set CUDA_VISIBLE_DEVICES={GPU_INDEX}; found {env_visible!r}")
    processes = subprocess.check_output(
        ["nvidia-smi", "-i", str(GPU_INDEX), "--query-compute-apps=pid",
         "--format=csv,noheader,nounits"], text=True,
    ).strip()
    require(not processes, f"Physical GPU {GPU_INDEX} has compute processes: {processes}")
    utilization = subprocess.check_output(
        ["nvidia-smi", "-i", str(GPU_INDEX), "--query-gpu=utilization.gpu",
         "--format=csv,noheader,nounits"], text=True,
    ).strip()
    require(utilization.isdigit() and int(utilization) == 0,
            f"Physical GPU {GPU_INDEX} is not idle: utilization={utilization!r}")
    return {"physical_index": GPU_INDEX, "visible_devices": env_visible,
            "compute_processes": [], "utilization_percent_at_check": int(utilization)}


def selected_colmap_infos(scene_name, expected_names):
    """Mirror eval=True's sorted every-eighth split without opening held-out images."""
    import os.path

    import numpy as np
    from PIL import Image
    from scene.colmap_loader import (
        qvec2rotmat, read_extrinsics_binary, read_extrinsics_text,
        read_intrinsics_binary, read_intrinsics_text,
    )
    from utils.graphics_utils import focal2fov

    source = DATA_ROOT / scene_name
    sparse = source / "sparse" / "0"
    extrinsics_bin = sparse / "images.bin"
    intrinsics_bin = sparse / "cameras.bin"
    if extrinsics_bin.is_file() and intrinsics_bin.is_file():
        extrinsics = read_extrinsics_binary(str(extrinsics_bin))
        intrinsics = read_intrinsics_binary(str(intrinsics_bin))
    else:
        extrinsics = read_extrinsics_text(str(sparse / "images.txt"))
        intrinsics = read_intrinsics_text(str(sparse / "cameras.txt"))

    headers = []
    for extr in extrinsics.values():
        image_basename = os.path.basename(extr.name)
        headers.append({
            "extr": extr,
            "intr": intrinsics[extr.camera_id],
            "basename": image_basename,
            "image_name": image_basename.split(".")[0],
        })
    headers.sort(key=lambda row: row["image_name"])
    train_headers = [row for index, row in enumerate(headers) if index % EVAL_LLFF_HOLD != 0]
    test_names = [row["image_name"] for index, row in enumerate(headers)
                  if index % EVAL_LLFF_HOLD == 0]
    train_by_name = {row["image_name"]: row for row in train_headers}
    require(len(train_by_name) == len(train_headers), f"Duplicate image names in {scene_name} COLMAP metadata")
    require(all(name in train_by_name for name in expected_names),
            f"Selected camera is not in eval=True training split for {scene_name}")
    require(not set(expected_names).intersection(test_names),
            f"Selected camera falls in eval=True held-out split for {scene_name}")

    from scene.dataset_readers import CameraInfo

    image_dir = source / "images"
    selected_infos = []
    opened_paths = []
    for image_name in expected_names:
        row = train_by_name[image_name]
        extr, intr = row["extr"], row["intr"]
        if intr.model == "SIMPLE_PINHOLE":
            focal_x = intr.params[0]
            fov_y, fov_x = focal2fov(focal_x, intr.height), focal2fov(focal_x, intr.width)
        elif intr.model == "PINHOLE":
            focal_x, focal_y = intr.params[0], intr.params[1]
            fov_y, fov_x = focal2fov(focal_y, intr.height), focal2fov(focal_x, intr.width)
        else:
            raise RuntimeError(f"Unsupported COLMAP camera model {intr.model} for {image_name}")
        image_path = image_dir / row["basename"]
        require(image_path.is_file(), f"Selected training image is missing: {image_path}")
        with Image.open(image_path) as image_file:
            image = image_file.copy()
        selected_infos.append(CameraInfo(
            uid=intr.id, R=np.transpose(qvec2rotmat(extr.qvec)), T=np.array(extr.tvec),
            FovY=fov_y, FovX=fov_x, image=image, image_path=str(image_path),
            image_name=image_name, width=intr.width, height=intr.height,
        ))
        opened_paths.append(str(image_path))

    require(len(opened_paths) == 4, f"Expected exactly four opened training images for {scene_name}")
    return selected_infos, {
        "eval": True,
        "split_rule": "sorted COLMAP image_name; index % 8 == 0 is held out",
        "train_camera_names": list(expected_names),
        "heldout_camera_count": len(test_names),
        "opened_training_image_paths": opened_paths,
    }


def render_rgb_and_counts(render_fastgs, camera, model, pipe, background, get_flag, metric_map):
    package = render_fastgs(
        camera, model, pipe, background, 0.5,
        get_flag=get_flag, metric_map=metric_map,
    )
    return package["render"], package["accum_metric_counts"]


def measure_view(scene_name, state, camera, camera_name, model, render_fastgs,
                 pipe, background, torch, gpu_record, protocol_sha, train_sha, probe_sha):
    gaussian_count = int(model.get_xyz.shape[0])
    with torch.no_grad():
        original_rgb, _ = render_rgb_and_counts(
            render_fastgs, camera, model, pipe, background, None, None)
        truth = camera.original_image
        require(original_rgb.shape == truth.shape, f"RGB/GT shape mismatch for {scene_name}/{camera_name}")
        height, width = int(camera.image_height), int(camera.image_width)
        require(width == 1600, f"Expected effective width 1600, got {width} for {scene_name}/{camera_name}")
        pixels = height * width
        errors = (original_rgb - truth).abs().mean(dim=0).reshape(-1)
        k25 = (pixels * 25 + 99) // 100
        k10 = (pixels * 10 + 99) // 100
        top25_values, top25_indices = torch.topk(errors, k=k25, largest=True, sorted=True)
        top10_indices = top25_indices[:k10]
        top25_mask = torch.zeros(pixels, dtype=torch.int32, device=errors.device)
        top10_mask = torch.zeros_like(top25_mask)
        top25_mask[top25_indices] = 1
        top10_mask[top10_indices] = 1
        require(bool(torch.all(top10_mask <= top25_mask).item()), "top-10 pixel mask is not nested in top-25")

        all_mask = torch.ones(pixels, dtype=torch.int32, device=errors.device)
        all_rgb, all_counts = render_rgb_and_counts(
            render_fastgs, camera, model, pipe, background, True, all_mask)
        top25_rgb, top25_counts = render_rgb_and_counts(
            render_fastgs, camera, model, pipe, background, True, top25_mask)
        top10_rgb, top10_counts = render_rgb_and_counts(
            render_fastgs, camera, model, pipe, background, True, top10_mask)

        rgb_errors = {
            "allones": float((all_rgb - original_rgb).abs().max().item()),
            "top25": float((top25_rgb - original_rgb).abs().max().item()),
            "top10": float((top10_rgb - original_rgb).abs().max().item()),
        }
        require(max(rgb_errors.values()) <= RGB_TOLERANCE,
                f"Counter-render RGB parity exceeded {RGB_TOLERANCE}: {rgb_errors}")
        count_vectors = {"all": all_counts, "top25": top25_counts, "top10": top10_counts}
        for label, counts in count_vectors.items():
            require(counts.ndim == 1 and counts.numel() == gaussian_count,
                    f"{label} count vector shape mismatch: {tuple(counts.shape)}, N={gaussian_count}")
        require(bool(torch.all(top25_counts <= all_counts).item()),
                "top-25 accepted-hit counts exceeded all-ones counts")
        require(bool(torch.all(top10_counts <= top25_counts).item()),
                "top-10 accepted-hit counts exceeded top-25 counts")

        hits = {label: int(counts.to(torch.int64).sum().item())
                for label, counts in count_vectors.items()}
        require(hits["all"] > 0, "All-ones accepted-hit denominator is zero")
        residual_sums = {
            "all": float(errors.to(torch.float64).sum().item()),
            "top25": float(top25_values.to(torch.float64).sum().item()),
            "top10": float(errors.index_select(0, top10_indices).to(torch.float64).sum().item()),
        }

    row = {
        "scene": scene_name,
        "state_id": state["state_id"],
        "iteration": state["iteration"],
        "model_name": state["model_name"],
        "model_ply_sha256": state["ply_sha256"],
        "camera_name": camera_name,
        "eval_training_view": True,
        "resolution": [width, height],
        "gaussians": gaussian_count,
        "pixel_counts": {"all": pixels, "top25": k25, "top10": k10},
        "pixel_fractions": {"all": 1.0, "top25": k25 / pixels, "top10": k10 / pixels},
        "original_l1_residual_sums": residual_sums,
        "accepted_hit_counts_int64": hits,
        "accepted_hit_fractions": {
            "top25": hits["top25"] / hits["all"],
            "top10": hits["top10"] / hits["all"],
        },
        "counter_render_max_rgb_abs_error": rgb_errors,
        "rgb_parity_tolerance": RGB_TOLERANCE,
        "count_vector_length": gaussian_count,
        "subset_count_checks_passed": True,
        "top10_pixel_set_is_subset_of_top25": True,
        "counter_semantics": "Native get_flag accepted Gaussian-pixel contribution counts; no tile-instance or radius proxy.",
        "render": {"renderer": "author render_fastgs", "mult": 0.5,
                   "effective_width": 1600, "background": "black"},
        "source_commit": PIN,
        "train_sha256": train_sha,
        "probe_sha256": probe_sha,
        "protocol_sha256": protocol_sha,
        "gpu": gpu_record,
    }
    return row, top25_counts.detach()


def main():
    require(not OUTPUT_PATH.exists(), f"Refusing to overwrite existing result: {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = new_result()
    write_result(result)

    stage = "preflight"
    try:
        verify_protocol(result)
        verify_source(result)
        verify_states(result)
        gpu_record = check_gpu2_idle()
        os.chdir(REPO)
        sys.path.insert(0, str(REPO))
        stage = "eval_train_camera_validation"
        camera_infos_by_scene = {}
        for scene_name, scene_config in SCENES.items():
            camera_infos, camera_split = selected_colmap_infos(scene_name, scene_config["views"])
            camera_infos_by_scene[scene_name] = camera_infos
            result["opened_training_images"].extend(camera_split["opened_training_image_paths"])
            result.setdefault("camera_splits", {})[scene_name] = camera_split
        write_result(result)

        import torch
        from types import SimpleNamespace
        from gaussian_renderer import render_fastgs
        from scene import GaussianModel
        from utils.camera_utils import cameraList_from_camInfos

        require(torch.cuda.is_available(), "CUDA is unavailable after selecting physical GPU 2")
        gpu_record["name"] = torch.cuda.get_device_name(0)
        gpu_record["logical_device"] = 0
        require("A100" in gpu_record["name"], f"Expected physical GPU 2 A100, got {gpu_record['name']}")

        result["gpu"] = gpu_record
        result["render_contract"] = {
            "scene_loader": "Only protocol-selected COLMAP eval=True training cameras are opened",
            "formal_heldout_images_opened": False,
            "counter_masks": ["allones", "sorted_top25_l1", "sorted_top10_l1_prefix_of_top25"],
        }
        result["status"] = "running"
        result["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_result(result)

        pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
        background = torch.zeros(3, dtype=torch.float32, device="cuda")
        for scene_name, scene_config in SCENES.items():
            stage = f"camera_load:{scene_name}"
            camera_infos = camera_infos_by_scene[scene_name]
            cameras = cameraList_from_camInfos(camera_infos, 1.0,
                                               SimpleNamespace(resolution=-1, data_device="cuda"))
            require([camera.image_name for camera in cameras] == scene_config["views"],
                    f"Loaded camera order mismatch for {scene_name}")
            require(all(camera.image_width == 1600 for camera in cameras),
                    f"Not all {scene_name} cameras have effective width 1600")
            write_result(result)

            for state in scene_config["states"]:
                state_record = next(row for row in result["states"] if row["state_id"] == state["state_id"])
                stage = f"load_model:{state['state_id']}"
                state_record["status"] = "running"
                write_result(result)
                model = GaussianModel(3, "default")
                model.load_ply(state["ply_path"])
                require(len(model.get_xyz) > 0, f"Empty model loaded for {state['state_id']}")
                state_record["gaussians"] = int(model.get_xyz.shape[0])
                state_record["actual_ply_sha256"] = sha256(state["ply_path"])
                top25_vectors = []

                for view_record, camera in zip(state_record["views"], cameras):
                    camera_name = camera.image_name
                    stage = f"measure:{state['state_id']}:{camera_name}"
                    view_record["status"] = "running"
                    write_result(result)
                    try:
                        view_result, top25_counts = measure_view(
                            scene_name, state, camera, camera_name, model, render_fastgs,
                            pipe, background, torch, gpu_record, result["protocol_sha256"],
                            result["train_sha256"], result["probe_sha256"],
                        )
                        view_record.update(view_result)
                        view_record["status"] = "completed"
                        top25_vectors.append(top25_counts)
                        del top25_counts
                        write_result(result)
                    except Exception as error:
                        view_record["status"] = "failed"
                        view_record["failure"] = {
                            "stage": stage, "type": type(error).__name__, "message": str(error),
                        }
                        raise

                per_view_presence = sum((counts > 0).to(torch.int32) for counts in top25_vectors)
                shared_gaussian_mask = per_view_presence >= 2
                pooled_top25_counts = sum(counts.to(torch.int64) for counts in top25_vectors)
                pooled_top25_hits = int(pooled_top25_counts.sum().item())
                expected_top25_hits = sum(
                    view["accepted_hit_counts_int64"]["top25"] for view in state_record["views"]
                )
                require(pooled_top25_hits == expected_top25_hits,
                        f"Pooled top-25 vector sum mismatch for {state['state_id']}")
                shared_top25_hits = int(pooled_top25_counts[shared_gaussian_mask].sum().item())
                state_record["cross_view_shared_top25_proxy"] = {
                    "views": 4,
                    "gaussians_with_positive_top25_count_in_at_least_two_views": int(
                        shared_gaussian_mask.sum().item()),
                    "accepted_hits_from_those_gaussians": shared_top25_hits,
                    "all_top25_accepted_hits_across_views": pooled_top25_hits,
                    "fraction_of_top25_hits_from_shared_ids": (
                        shared_top25_hits / pooled_top25_hits if pooled_top25_hits else None),
                    "interpretation": "Cross-view shared-ID proxy only; does not prove the same 3D surface or error is shared.",
                }
                del top25_vectors, per_view_presence, shared_gaussian_mask, pooled_top25_counts
                state_record["status"] = "completed"
                del model
                torch.cuda.empty_cache()
                write_result(result)

            del cameras, camera_infos
            torch.cuda.empty_cache()

        result["status"] = "completed"
        result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        per_state_f25 = {
            row["state_id"]: sum(view["accepted_hit_counts_int64"]["top25"]
                                 for view in row["views"] if view["status"] == "completed") /
                             sum(view["accepted_hit_counts_int64"]["all"]
                                 for view in row["views"] if view["status"] == "completed")
            for row in result["states"]
        }
        overall_pass = all(value >= 0.40 for value in per_state_f25.values())
        result["decision_gate"] = {
            "per_state_top25_hit_fraction": per_state_f25,
            "per_state_pass": {state_id: value >= 0.40 for state_id, value in per_state_f25.items()},
            "threshold": 0.40,
            "pass": overall_pass,
            "decision": (
                "overlap_screen_pass_only_no_algorithm_authorized" if overall_pass else
                "fail_stop_high_residual_concentrated_work_direction"
            ),
            "scope": "Only high-residual/work co-location for a same-region edit; not a global quality-work mechanism gate.",
        }
        write_result(result)
    except Exception as error:
        complete_views = sum(
            view["status"] == "completed" for state in result["states"] for view in state["views"]
        )
        if result["status"] == "not_run":
            result["status"] = "preflight_failed"
        elif result["status"] == "running":
            result["status"] = "partial_failure" if complete_views else "failed"
        result["failure"] = {
            "stage": stage, "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(),
        }
        result["failures"].append(result["failure"])
        result["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_result(result)
        raise


if __name__ == "__main__":
    main()
