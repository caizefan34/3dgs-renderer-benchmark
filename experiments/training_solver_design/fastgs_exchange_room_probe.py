"""Delete/refit-birth opportunity test in one full-resolution FastGS room state.

Six train views choose the exchange; four other train views are read only after
selection. This is an offline fixed-state diagnostic, not a trainer.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import fastgs_birth_fit_probe as F
import fastgs_birth_native_loss_probe as N
from utils.sh_utils import RGB2SH


OUT = Path("/root/fastgs_exchange_room_result.json")


def fitted_child(model, post, child_id, q, rgb):
    result = F.P.append_child(model, post, child_id, q)
    with torch.no_grad():
        result._features_dc[-1, 0, :] = RGB2SH(torch.tensor(
            rgb, device="cuda", dtype=result._features_dc.dtype))
        result._features_rest[-1].zero_()
    return result


def losses(camera, pipe, arms):
    gt = camera.original_image
    with torch.no_grad():
        return {name: N.measure(N.native_image(camera, model, pipe), gt)["loss"]
                for name, model in arms.items()}


def nearby_deletions(cameras, model, birth, pipe, parent_id):
    n = len(model.get_xyz)
    score = torch.zeros(n, device="cuda")
    coverage = torch.zeros(n, dtype=torch.int32, device="cuda")
    for index in F.CAMERA_INDICES[:6]:
        camera = cameras[index]
        with torch.no_grad():
            _, _, base_geom, _, _ = F.P.render_and_capture(camera, model, pipe)
            _, _, new_geom, _, _ = F.P.render_and_capture(camera, birth, pipe)
        center = new_geom["mean2d"][-1]
        depth = new_geom["depths"][-1]
        mean = base_geom["mean2d"][:n]
        conic = base_geom["conic"][:n]
        old_depth = base_geom["depths"][:n]
        det = conic[:, 0] * conic[:, 2] - conic[:, 1].square()
        alpha = conic[:, 3]
        area = 2 * math.pi * torch.log((255 * alpha).clamp_min(1.00001)) / torch.sqrt(det.clamp_min(1e-20))
        dist2 = (mean[:, 0] - center[0]).square() + (mean[:, 1] - center[1]).square()
        eligible = ((dist2 < 1600) & ((old_depth - depth).abs() < 0.15 * depth.abs().clamp_min(1e-5))
                    & torch.isfinite(area) & (area >= 20) & (area < 10000)
                    & (alpha >= 0.005) & (alpha <= 0.2))
        score += eligible.float() * area
        coverage += eligible.int()
    feasible = coverage >= 2
    feasible[parent_id] = False
    ranked = torch.argsort(score.masked_fill(~feasible, -1.0), descending=True)
    choices = [int(i) for i in ranked[:6] if float(score[i]) > 0]
    choices.append(parent_id)
    info = {"eligible_count": int(feasible.sum()),
            "choices": [{"id": i, "area_score": float(score[i]),
                         "fit_view_coverage": int(coverage[i]),
                         "opacity": float(model.get_opacity[i, 0])} for i in choices]}
    return choices, info


def main():
    started = time.perf_counter()
    P = F.P
    prior = json.loads(Path("/root/fastgs_birth_multi_candidate_result.json").read_text())
    source = P.REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"
    if P.sha256(source) != prior["source_ply_sha256"]:
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
    cameras = scene.getTrainCameras()
    post, ids, families = P.make_native_post(model, mask, seed=20260927)
    chosen_birth = prior["selected"]
    if families != 9345 or int(ids[chosen_birth["family"]]) != chosen_birth["parent_id"]:
        raise AssertionError("birth identity changed")
    child_id = len(model.get_xyz) + chosen_birth["choice"]
    birth = fitted_child(model, post, child_id, chosen_birth["opacity"], chosen_birth["rgb"])
    deletion_ids, deletion_info = nearby_deletions(cameras, model, birth, pipe,
                                                    chosen_birth["parent_id"])
    baseline = [losses(cameras[i], pipe, {"base": model, "birth": birth})
                for i in F.CAMERA_INDICES[:6]]
    baseline_mean = float(np.mean([x["base"] for x in baseline]))
    records = []
    for deletion in deletion_ids:
        old = model._opacity[deletion, 0].detach().clone()
        with torch.no_grad():
            model._opacity[deletion, 0] = -30.0
        probe = P.append_child(model, post, child_id, F.Q_PROBE)
        views = [F.collect(cameras[i], model, probe, pipe) for i in F.CAMERA_INDICES[:6]]
        del probe
        valid = [v for v in views if v["samples"]]
        record = {"deletion_id": deletion,
                  "fit_pixels": sum(v["samples"] for v in valid),
                  "fit_view_count": len(valid),
                  "probe_parity_max": max((v.get("probe_parity_max", 0.0) for v in valid), default=None)}
        if record["fit_pixels"] >= 40 and len(valid) >= 2:
            a = np.concatenate([v["matrix"] for v in valid])
            y = np.concatenate([v["residual"] for v in valid])
            singular = np.linalg.svd(a, full_matrices=False)[1]
            record["min_singular_value"] = float(singular[-1])
            if singular[-1] >= 1e-3:
                try:
                    z = F.fit_qp(a, y)
                except AssertionError as error:
                    record["fit_error"] = str(error)
                else:
                    q = float(z[3])
                    rgb = (z[:3] / q).tolist() if q > 1e-8 else None
                    record.update({"opacity": q, "rgb": rgb,
                                   "local_sse_gain": float(y @ y - np.sum((a @ z - y)**2))})
                    if rgb is not None:
                        exchange = fitted_child(model, post, child_id, q, rgb)
                        measured = [losses(cameras[i], pipe, {"delete": model, "exchange": exchange})
                                    for i in F.CAMERA_INDICES[:6]]
                        record["mean_fit_delete_loss_delta"] = float(np.mean([m["delete"] for m in measured]) - baseline_mean)
                        record["mean_fit_exchange_loss_delta"] = float(np.mean([m["exchange"] for m in measured]) - baseline_mean)
                        record["mean_fit_exchange_vs_delete"] = (record["mean_fit_exchange_loss_delta"]
                                                                 - record["mean_fit_delete_loss_delta"])
                        del exchange
            else:
                record["fit_error"] = "rank deficient"
        else:
            record["fit_error"] = "insufficient stable support"
        with torch.no_grad():
            model._opacity[deletion, 0] = old
        records.append(record)
        print("FASTGS_EXCHANGE_CANDIDATE " + json.dumps(record), flush=True)

    qualified = [r for r in records if "mean_fit_exchange_loss_delta" in r]
    output = {
        "status": "NO_STABLE_EXCHANGE" if not qualified else "SELECTED",
        "source_commit": prior["source_commit"],
        "source_ply_sha256": prior["source_ply_sha256"],
        "birth_choice": chosen_birth["choice"],
        "birth_parent": chosen_birth["parent_id"],
        "deletion_selection": deletion_info,
        "baseline_mean_fit_loss": baseline_mean,
        "birth_mean_fit_loss_delta": float(np.mean([x["birth"] for x in baseline]) - baseline_mean),
        "candidates": records,
    }
    if qualified:
        best = min(qualified, key=lambda r: r["mean_fit_exchange_loss_delta"])
        deletion = best["deletion_id"]
        old = model._opacity[deletion, 0].detach().clone()
        with torch.no_grad():
            model._opacity[deletion, 0] = -30.0
        exchange = fitted_child(model, post, child_id, best["opacity"], best["rgb"])
        checks = []
        for index in F.CAMERA_INDICES[6:]:
            camera = cameras[index]
            deleted = losses(camera, pipe, {"delete": model, "birth": birth, "exchange": exchange})
            with torch.no_grad():
                model._opacity[deletion, 0] = old
            base_loss = losses(camera, pipe, {"base": model})["base"]
            with torch.no_grad():
                model._opacity[deletion, 0] = -30.0
            checks.append({"camera": camera.image_name,
                           "deltas": {name: deleted[name] - base_loss
                                      for name in ("delete", "birth", "exchange")}})
        with torch.no_grad():
            model._opacity[deletion, 0] = old
        output["selected"] = {"deletion_id": deletion, "opacity": best["opacity"],
                              "rgb": best["rgb"],
                              "mean_fit_exchange_loss_delta": best["mean_fit_exchange_loss_delta"],
                              "internal_check": checks,
                              "mean_internal_check_delta": {name: float(np.mean([
                                  row["deltas"][name] for row in checks]))
                                  for name in ("delete", "birth", "exchange")}}
    output["elapsed_seconds_including_scene_load"] = time.perf_counter() - started
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("FASTGS_EXCHANGE_RESULT " + json.dumps({
        "status": output["status"], "candidate_count": len(deletion_ids),
        "selected": output.get("selected"),
        "elapsed_seconds": output["elapsed_seconds_including_scene_load"]}), flush=True)


if __name__ == "__main__":
    main()
