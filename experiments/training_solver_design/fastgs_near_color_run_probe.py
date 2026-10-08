"""Audit native FastGS accepted-hit color runs on fixed room@4100 training rays.

The script only captures the existing rasterizer's buffers. It does not alter
author code or training state. Run on EPIC-05 with CUDA_VISIBLE_DEVICES set to
one idle physical GPU.
"""

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from fastgs_real_tile_reference import REPO, decode_geometry, decode_image, sha256, take
from diff_gaussian_rasterization_fastgs import _C
from gaussian_renderer import render_fastgs
from scene import dataset_readers
from scene import GaussianModel, Scene


CAMERA_INDICES = (0, 17, 54, 121)
CAMERA_NAMES = ("DSCF4668", "DSCF4687", "DSCF4729", "DSCF4806")
SAMPLES_PER_CAMERA = 256
SAMPLE_SEED = 20260927
THRESHOLDS = (1.0 / 255.0, 2.0 / 255.0, 4.0 / 255.0)
PARITY_TOLERANCE = 1e-4


def accepted_prefix(x, y, ids, means, conics, colors):
    """Replay the forward alpha gates; the transmittance stop candidate is excluded."""
    rows = []
    trans = 1.0
    last_contributor = 0
    counters = {"power_rejected": 0, "alpha_rejected": 0, "stop_candidates": 0}
    stopped = False
    stop_candidate = None
    for pos, gid in enumerate(ids):
        dx, dy = means[pos] - (x, y)
        co = conics[pos]
        power = -0.5 * (co[0] * dx * dx + co[2] * dy * dy) - co[1] * dx * dy
        if power > 0:
            counters["power_rejected"] += 1
            continue
        alpha = min(0.99, float(co[3]) * math.exp(float(power)))
        if alpha < 1.0 / 255.0:
            counters["alpha_rejected"] += 1
            continue
        next_trans = trans * (1.0 - alpha)
        if next_trans < 1e-4:
            stopped = True
            stop_candidate = int(gid)
            counters["stop_candidates"] += 1
            break
        rows.append({"gaussian_id": int(gid), "alpha": alpha,
                     "rgb": [float(v) for v in colors[pos]]})
        trans = next_trans
        last_contributor = pos + 1
    return rows, trans, last_contributor, stopped, stop_candidate, counters


def recompose(rows, trans, background):
    rgb = np.zeros(3, dtype=np.float64)
    running_trans = 1.0
    for row in rows:
        rgb += running_trans * row["alpha"] * np.asarray(row["rgb"], dtype=np.float64)
        running_trans *= 1.0 - row["alpha"]
    if not math.isclose(running_trans, trans, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError((running_trans, trans))
    return rgb + trans * background


def greedy_run_lengths(colors, threshold):
    """Greedily extend a run while its per-channel RGB range stays in tolerance."""
    if not colors:
        return []
    color = np.asarray(colors, dtype=np.float64)
    low = color[0].copy()
    high = color[0].copy()
    lengths = []
    length = 1
    for next_color in color[1:]:
        next_low = np.minimum(low, next_color)
        next_high = np.maximum(high, next_color)
        if np.max(next_high - next_low) <= threshold:
            low, high = next_low, next_high
            length += 1
        else:
            lengths.append(length)
            low = next_color.copy()
            high = next_color.copy()
            length = 1
    lengths.append(length)
    return lengths


def weighted_run_summary(rays, threshold):
    total_hits = 0
    hits_in_runs_le2 = 0
    hits_in_runs_ge8 = 0
    run_count = 0
    histogram = {}
    for ray in rays:
        lengths = ray["run_lengths"][str(threshold)]
        run_count += len(lengths)
        for length in lengths:
            total_hits += length
            if length <= 2:
                hits_in_runs_le2 += length
            if length >= 8:
                hits_in_runs_ge8 += length
            histogram[str(length)] = histogram.get(str(length), 0) + 1
    return {
        "accepted_hits": total_hits,
        "run_count": run_count,
        "hits_in_runs_len_le_2": hits_in_runs_le2,
        "hits_in_runs_len_ge_8": hits_in_runs_ge8,
        "P_len_le_2_hit_weighted": hits_in_runs_le2 / total_hits if total_hits else None,
        "P_len_ge_8_hit_weighted": hits_in_runs_ge8 / total_hits if total_hits else None,
        "run_length_histogram": histogram,
    }


def main():
    dataset = SimpleNamespace(
        source_path="/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room",
        model_path=str(REPO / "output/room_big_capture4100"),
        images="images", resolution=-1, white_background=False,
        data_device="cuda", eval=False,
    )
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    model = GaussianModel(3, "default")
    original_camera_loader = dataset_readers.readColmapCameras

    def load_only_selected_training_cameras(cam_extrinsics, cam_intrinsics, images_folder):
        selected = {
            key: extrinsic for key, extrinsic in cam_extrinsics.items()
            if Path(extrinsic.name).stem in CAMERA_NAMES
        }
        selected_names = {Path(extrinsic.name).stem for extrinsic in selected.values()}
        if selected_names != set(CAMERA_NAMES):
            raise AssertionError(f"selected train-camera names missing: {set(CAMERA_NAMES) - selected_names}")
        return original_camera_loader(selected, cam_intrinsics, images_folder)

    dataset_readers.readColmapCameras = load_only_selected_training_cameras
    try:
        scene = Scene(dataset, model, load_iteration=4100, shuffle=False)
    finally:
        dataset_readers.readColmapCameras = original_camera_loader
    cameras_by_name = {camera.image_name: camera for camera in scene.getTrainCameras()}
    if set(cameras_by_name) != set(CAMERA_NAMES) or scene.getTestCameras():
        raise AssertionError((sorted(cameras_by_name), len(scene.getTestCameras())))
    point_count = int(model.get_xyz.shape[0])
    ply_path = REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"
    rng = np.random.default_rng(SAMPLE_SEED)
    output = {
        "source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
        "model_ply_sha256": sha256(ply_path),
        "resolution": [1600, 1066],
        "camera_indices": list(CAMERA_INDICES),
        "camera_names": list(CAMERA_NAMES),
        "camera_loading": "in-memory scene loader filtered to these four known training image names before opening images; no held-out/test images or camera objects were loaded",
        "sampling": {
            "pixels_per_camera": SAMPLES_PER_CAMERA,
            "generator": "numpy.default_rng",
            "seed": SAMPLE_SEED,
            "coordinates": "independent uniform integer x and y draws, with replacement; same fixed stream as fastgs_order_conflict_probe.py",
        },
        "accepted_semantics": "native point-list order; power<=0; alpha=min(0.99, opacity*exp(power)) >= 1/255; exclude the candidate whose next transmittance would be <1e-4 and stop that ray",
        "color_semantics": "view-evaluated FastGS per-Gaussian RGB feature from the captured native projection buffer, before alpha/transmittance weighting",
        "greedy_group_semantics": "within each run, every channel's max-minus-min RGB range must stay <= threshold; greedily extend the current run until adding the next accepted hit violates the threshold",
        "thresholds": [float(v) for v in THRESHOLDS],
        "hit_weighting": "P_len<=2 and P_len>=8 count accepted hits whose greedy run has the specified length, divided by all accepted hits in the complete sampled native prefixes",
        "views": [],
    }

    for camera_index, camera_name in zip(CAMERA_INDICES, CAMERA_NAMES):
        camera = cameras_by_name[camera_name]
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
            raise AssertionError(f"expected one rasterizer call, got {len(captured)}")

        rendered, _, _, _, geom_buffer, bin_buffer, img_buffer, _, _ = captured[0]
        width, height = camera.image_width, camera.image_height
        geometry = decode_geometry(geom_buffer, point_count)
        image_state = decode_image(img_buffer, width * height)
        point_list, _ = take(bin_buffer, 0, rendered, torch.int32)
        point_list = point_list.cpu().numpy()
        ranges = image_state["ranges"].cpu().numpy()
        native_n_contrib = image_state["n_contrib"].cpu().numpy().reshape(height, width)
        means = geometry["mean2d"].cpu().numpy().astype(np.float64)
        conics = geometry["conic"].cpu().numpy().astype(np.float64)
        colors = geometry["rgb"].cpu().numpy().astype(np.float64)
        native_image = image.detach().cpu().numpy().astype(np.float64)
        background = np.zeros(3, dtype=np.float64)

        xs = rng.integers(0, width, SAMPLES_PER_CAMERA)
        ys = rng.integers(0, height, SAMPLES_PER_CAMERA)
        rays = []
        n_contrib_matches = 0
        parity_passes = 0
        early_stopped = 0
        empty_rays = 0
        accepted_hits = 0
        power_rejected = 0
        alpha_rejected = 0
        stop_candidates = 0
        max_rgb_error = 0.0
        max_n_contrib_abs_diff = 0

        for x_raw, y_raw in zip(xs, ys):
            x, y = int(x_raw), int(y_raw)
            tile_id = (y // 16) * ((width + 15) // 16) + x // 16
            begin, end = ranges[tile_id]
            ids = point_list[int(begin):int(end)]
            rows, trans, simulated_last, stopped, stop_candidate, gate_counts = accepted_prefix(
                x, y, ids, means[ids], conics[ids], colors[ids])
            native_last = int(native_n_contrib[y, x])
            n_match = simulated_last == native_last
            n_contrib_matches += int(n_match)
            max_n_contrib_abs_diff = max(max_n_contrib_abs_diff, abs(simulated_last - native_last))
            rgb = recompose(rows, trans, background)
            native_rgb = native_image[:, y, x]
            rgb_error = float(np.max(np.abs(rgb - native_rgb)))
            max_rgb_error = max(max_rgb_error, rgb_error)
            parity_ok = rgb_error <= PARITY_TOLERANCE
            parity_passes += int(parity_ok)
            early_stopped += int(stopped)
            empty_rays += int(not rows)
            accepted_hits += len(rows)
            power_rejected += gate_counts["power_rejected"]
            alpha_rejected += gate_counts["alpha_rejected"]
            stop_candidates += gate_counts["stop_candidates"]

            colors_in_order = [row["rgb"] for row in rows]
            run_lengths = {
                str(threshold): greedy_run_lengths(colors_in_order, threshold)
                for threshold in THRESHOLDS
            }
            rays.append({
                "xy": [x, y],
                "tile_list_range": [int(begin), int(end)],
                "tile_candidate_count": int(end - begin),
                "native_n_contrib": native_last,
                "replayed_last_contributor": simulated_last,
                "n_contrib_match": n_match,
                "early_stopped": stopped,
                "stop_candidate_gaussian_id_excluded": stop_candidate,
                "accepted_hit_count": len(rows),
                "gate_rejections": gate_counts,
                "max_rgb_parity_error": rgb_error,
                "parity_pass": parity_ok,
                "accepted_hits_in_native_order": rows,
                "run_lengths": run_lengths,
            })

        view = {
            "camera_index": camera_index,
            "camera": camera.image_name,
            "resolution": [width, height],
            "sampled_rays": SAMPLES_PER_CAMERA,
            "n_contrib_matches": n_contrib_matches,
            "n_contrib_coverage": n_contrib_matches / SAMPLES_PER_CAMERA,
            "max_n_contrib_abs_diff": max_n_contrib_abs_diff,
            "rgb_parity_passes": parity_passes,
            "rgb_parity_coverage": parity_passes / SAMPLES_PER_CAMERA,
            "rgb_parity_tolerance": PARITY_TOLERANCE,
            "max_rgb_parity_error": max_rgb_error,
            "early_stopped_rays": early_stopped,
            "empty_rays": empty_rays,
            "accepted_hits": accepted_hits,
            "power_rejected_candidates_before_stop": power_rejected,
            "alpha_below_1_over_255_candidates_before_stop": alpha_rejected,
            "excluded_early_stop_candidates": stop_candidates,
            "hit_weighted_run_statistics": {
                str(threshold): weighted_run_summary(rays, threshold)
                for threshold in THRESHOLDS
            },
            "rays": rays,
        }
        output["views"].append(view)
        print("NEAR_COLOR_VIEW " + json.dumps({k: v for k, v in view.items() if k != "rays"}, sort_keys=True), flush=True)

    flat_rays = [ray for view in output["views"] for ray in view["rays"]]
    output["pooled_hit_weighted_run_statistics"] = {
        str(threshold): weighted_run_summary(flat_rays, threshold)
        for threshold in THRESHOLDS
    }
    output["coverage_pass"] = all(
        view["n_contrib_matches"] == SAMPLES_PER_CAMERA and
        view["rgb_parity_passes"] == SAMPLES_PER_CAMERA
        for view in output["views"]
    )
    output["four_view_gate"] = {
        "threshold": float(THRESHOLDS[-1]),
        "veto_if_any_view_or_pooled_P_len_le_2_ge_0_70_or_P_len_ge_8_lt_0_30": True,
    }
    gate_stats = [view["hit_weighted_run_statistics"][str(THRESHOLDS[-1])] for view in output["views"]]
    pooled = output["pooled_hit_weighted_run_statistics"][str(THRESHOLDS[-1])]
    veto = any(
        summary["P_len_le_2_hit_weighted"] >= 0.70 or
        summary["P_len_ge_8_hit_weighted"] < 0.30
        for summary in gate_stats + [pooled]
    )
    output["four_view_gate"]["veto"] = veto
    output["four_view_gate"]["disposition"] = "VETO_NO_NEW_CUDA" if veto else "SCREEN_PASSES_ONLY"
    destination = Path(os.environ.get("FASTGS_NEAR_COLOR_OUT", str(REPO / "fastgs_near_color_run_result.json")))
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("NEAR_COLOR_RESULT " + json.dumps({
        "result_path": str(destination),
        "coverage_pass": output["coverage_pass"],
        "four_view_gate": output["four_view_gate"],
        "pooled_hit_weighted_run_statistics": output["pooled_hit_weighted_run_statistics"],
    }, sort_keys=True), flush=True)
    if not output["coverage_pass"]:
        raise AssertionError("sampled-ray native n_contrib or RGB parity validation failed")


if __name__ == "__main__":
    main()
