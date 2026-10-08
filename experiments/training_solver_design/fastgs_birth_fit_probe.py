"""One-child analytic birth fit on frozen, full-resolution FastGS views.

Candidate geometry is selected with six training views. This is an offline
diagnostic, not a trainer or an independent quality evaluation.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import torch

import fastgs_birth_parity_probe_v2b as P


OUT = Path("/root/fastgs_birth_fit_multiview_result.json")
CAMERA_INDICES = (0, 30, 60, 90, 120, 150, 180, 210, 240, 270)
Q_PROBE = 0.4
Q_MAX = 0.8


def select_multiview_child(cameras, model, post, families, pipe):
    total = 2 * families
    fit_count = torch.zeros(total, dtype=torch.int32, device="cuda")
    val_count = torch.zeros_like(fit_count)
    fit_score = torch.zeros(total, dtype=torch.float32, device="cuda")
    per_camera = []
    n = len(model.get_xyz)
    for position, index in enumerate(CAMERA_INDICES):
        camera = cameras[index]
        with torch.no_grad():
            base, _, _, img, _ = P.render_and_capture(camera, model, pipe)
            _, _, projected, _, _ = P.render_and_capture(camera, post, pipe)
        width, height = camera.image_width, camera.image_height
        means = projected["mean2d"][n:]
        conics = projected["conic"][n:]
        safe = torch.nan_to_num(means, nan=-1000., posinf=-1000., neginf=-1000.)
        xs = safe[:, 0].round().long().clamp(0, width - 1)
        ys = safe[:, 1].round().long().clamp(0, height - 1)
        determinant = conics[:, 0] * conics[:, 2] - conics[:, 1].square()
        area = 2 * math.pi * math.log(255) / torch.sqrt(determinant.clamp_min(1e-20))
        t = img["alpha"].reshape(height, width)[ys, xs]
        visible = ((xs >= 30) & (xs < width - 30) & (ys >= 30) & (ys < height - 30)
                   & (conics[:, 3] > 0) & (area >= 30) & (area < 5000) & (t > 0.001))
        if position < 6:
            fit_count += visible.int()
            residual = (camera.original_image - base).abs().mean(0)[ys, xs]
            fit_score += visible.float() * t * residual * torch.sqrt(area.clamp_max(500))
        else:
            val_count += visible.int()
        per_camera.append({"camera": camera.image_name, "eligible_children": int(visible.sum())})
    eligible = (fit_count >= 3) & (val_count >= 1)
    if not bool(eligible.any()):
        eligible = (fit_count >= 2) & (val_count >= 1)
    if not bool(eligible.any()):
        raise AssertionError("no multi-view candidate with internal validation coverage")
    ranked = fit_score.masked_fill(~eligible, -1)
    choice = int(ranked.argmax())
    return choice, {"eligible_multiview_children": int(eligible.sum()),
                    "selected_fit_coverage": int(fit_count[choice]),
                    "selected_validation_coverage": int(val_count[choice]),
                    "selected_fit_score": float(fit_score[choice]),
                    "per_camera_selection": per_camera}, torch.nonzero(eligible).flatten().cpu().tolist()


def collect(camera, model, probe_model, pipe):
    with torch.no_grad():
        base, _, geom, img, ids = P.render_and_capture(camera, model, pipe)
        probe, _, pg, pi, pids = P.render_and_capture(camera, probe_model, pipe)
    width, height = camera.image_width, camera.image_height
    mean = pg["mean2d"][-1].cpu().numpy()
    conic = pg["conic"][-1].cpu().numpy()
    depth = float(pg["depths"][-1])
    if not np.all(np.isfinite(mean)) or not (30 <= mean[0] < width - 30 and 30 <= mean[1] < height - 30):
        return {"camera": camera.image_name, "samples": 0, "reason": "projection outside"}
    means, conics, colors, depths, ranges, point_list = P.cpu_buffers(geom, img, ids, width, height)
    native_ranges = pi["ranges"].cpu().numpy()
    native_ids = pids.cpu().numpy()
    base_image = base.cpu().permute(1, 2, 0).numpy().astype(np.float64)
    probe_image = probe.cpu().permute(1, 2, 0).numpy().astype(np.float64)
    gt = camera.original_image.cpu().permute(1, 2, 0).numpy().astype(np.float64)
    probe_color = pg["rgb"][-1].cpu().numpy().astype(np.float64)
    projected_opacity_factor = float(conic[3]) / Q_PROBE
    matrices, residuals, errors = [], [], []
    counters = {"no_tile": 0, "base_early": 0, "candidate_early_risk": 0,
                "base_mismatch": 0, "below_cutoff": 0}
    for y in range(int(mean[1]) - 30, int(mean[1]) + 31, 2):
        for x in range(int(mean[0]) - 30, int(mean[0]) + 31, 2):
            g, alpha = P.raw_at(x, y, mean, conic)
            if alpha < 1 / 255:
                counters["below_cutoff"] += 1
                continue
            tile = (y // 16) * ((width + 15) // 16) + x // 16
            begin, end = native_ranges[tile]
            if len(means) not in native_ids[int(begin):int(end)]:
                counters["no_tile"] += 1
                continue
            prefix, t, suffix, early, final_t = P.context_at_pixel(
                x, y, width, means, conics, colors, depths, ranges, point_list, depth)
            if early:
                counters["base_early"] += 1
                continue
            if final_t * (1 - Q_MAX * projected_opacity_factor * g) < 1e-4:
                counters["candidate_early_risk"] += 1
                continue
            if np.max(np.abs(prefix + t * suffix - base_image[y, x])) > 2e-5:
                counters["base_mismatch"] += 1
                continue
            k = t * projected_opacity_factor * g
            row = np.zeros((3, 4), dtype=np.float64)
            row[:, :3] = np.eye(3) * k
            row[:, 3] = -k * suffix
            matrices.append(row)
            residuals.append(gt[y, x] - base_image[y, x])
            expected_probe = k * Q_PROBE * (probe_color - suffix)
            errors.append(float(np.max(np.abs(
                expected_probe - (probe_image[y, x] - base_image[y, x])))))
    if not matrices:
        return {"camera": camera.image_name, "samples": 0, "mean": mean.tolist(), "counters": counters}
    return {"camera": camera.image_name, "samples": len(matrices), "mean": mean.tolist(),
            "counters": counters, "matrix": np.concatenate(matrices),
            "residual": np.concatenate(residuals), "probe_parity_max": max(errors)}


def fit_qp(matrix, residual):
    gram, rhs = matrix.T @ matrix, matrix.T @ residual
    eye = np.eye(4)
    constraints = [
        {"type": "ineq", "fun": lambda z, j=j: z[3] - z[j],
         "jac": lambda z, j=j: eye[3] - eye[j]} for j in range(3)
    ]
    result = minimize(lambda z: .5 * float(z @ gram @ z - 2 * rhs @ z),
                      np.array([.15, .15, .15, .4]),
                      jac=lambda z: gram @ z - rhs,
                      bounds=[(0, Q_MAX)] * 4, constraints=constraints,
                      method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
    if not result.success:
        raise AssertionError(result.message)
    z = result.x
    if np.any(z[:3] > z[3] + 1e-8):
        raise AssertionError("invalid premultiplied color")
    return z


def main():
    started = time.perf_counter()
    source = P.REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"
    if P.sha256(source) != "64160970e5ea580aea4f79278a8fbe334d8d81ef6e304a903626843cc48f85bc":
        raise AssertionError("source PLY changed")
    mask = torch.load("/root/fastgs_gate_selected_4100_mask.pt", map_location="cpu",
                      weights_only=True)["selected_splits"]
    dataset = P.SimpleNamespace(
        source_path="/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room",
        model_path=str(P.REPO / "output/room_big_capture4100"),
        images="images", resolution=-1, white_background=False,
        data_device="cuda", eval=True)
    pipe = P.SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    model = P.GaussianModel(3, "default")
    scene = P.Scene(dataset, model, load_iteration=4100, shuffle=False)
    post, selected_ids, families = P.make_native_post(model, mask, seed=20260927)
    if families != 9345:
        raise AssertionError("native family count changed")
    with torch.no_grad():
        post._opacity[selected_ids] = model._opacity[selected_ids]
    choice, selection, _ = select_multiview_child(cameras=scene.getTrainCameras(),
                                                  model=model, post=post, families=families,
                                                  pipe=pipe)
    family = choice % families
    child_id = len(model.get_xyz) + choice
    probe_model = P.append_child(model, post, child_id, Q_PROBE)
    views = []
    for index in CAMERA_INDICES:
        row = collect(scene.getTrainCameras()[index], model, probe_model, pipe)
        views.append(row)
        print("FASTGS_BIRTH_FIT_VIEW " + json.dumps(
            {"camera": row["camera"], "samples": row["samples"]}), flush=True)
    train = [r for r in views[:6] if r["samples"]]
    validation = [r for r in views[6:] if r["samples"]]
    output = {
        "source_ply_sha256": P.sha256(source), "source_commit":
            "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
        "family": family, "parent_id": int(selected_ids[family]),
        "child_id_in_union": child_id, "camera_indices": CAMERA_INDICES,
        "selection": selection,
        "per_view": [{k: v for k, v in r.items() if k not in ("matrix", "residual")} for r in views],
        "fit_camera_count": len(train), "validation_camera_count": len(validation),
        "fit_pixels": sum(r["samples"] for r in train),
        "validation_pixels": sum(r["samples"] for r in validation),
        "scope": "One fixed child geometry selected using six fit training views; four internal-validation training views are excluded from selection scoring and fitting. No continuation or independent test.",
    }
    if output["fit_pixels"] >= 20 and output["validation_pixels"] >= 20:
        a = np.concatenate([r["matrix"] for r in train])
        y = np.concatenate([r["residual"] for r in train])
        av = np.concatenate([r["matrix"] for r in validation])
        yv = np.concatenate([r["residual"] for r in validation])
        z = fit_qp(a, y)
        output.update({
            "status": "FITTED", "opacity": float(z[3]),
            "rgb": (z[:3] / z[3]).tolist() if z[3] > 1e-8 else None,
            "design_singular_values": np.linalg.svd(a, full_matrices=False)[1].tolist(),
            "fit_mse_before": float(np.mean(y * y)),
            "fit_mse_after": float(np.mean((a @ z - y) ** 2)),
            "validation_mse_before": float(np.mean(yv * yv)),
            "validation_mse_after": float(np.mean((av @ z - yv) ** 2)),
        })
    else:
        output["status"] = "INSUFFICIENT_MULTI_VIEW_SUPPORT"
    output["elapsed_seconds_including_scene_load"] = time.perf_counter() - started
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("FASTGS_BIRTH_FIT_RESULT " + json.dumps({
        k: output.get(k) for k in ("status", "fit_pixels", "validation_pixels",
                                  "opacity", "rgb", "fit_mse_before", "fit_mse_after",
                                  "validation_mse_before", "validation_mse_after",
                                  "elapsed_seconds_including_scene_load")}), flush=True)


if __name__ == "__main__":
    main()
