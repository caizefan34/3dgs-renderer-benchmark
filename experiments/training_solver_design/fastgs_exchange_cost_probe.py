"""Native FastGS Gaussian-tile instance counts for a frozen room exchange."""

import json
import time
from pathlib import Path

import torch

import fastgs_birth_fit_probe as F
import fastgs_exchange_room_probe as E


OUT = Path("/root/fastgs_exchange_cost_result.json")


def count(camera, model, pipe):
    with torch.no_grad():
        _, rendered, _, _, _ = F.P.render_and_capture(camera, model, pipe)
    return int(rendered)


def main():
    started = time.perf_counter()
    P = F.P
    prior = json.loads(Path("/root/fastgs_exchange_room_result.json").read_text())
    fit = json.loads(Path("/root/fastgs_birth_multi_candidate_result.json").read_text())
    if prior["status"] != "SELECTED":
        raise AssertionError("exchange not selected")
    source = P.REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"
    if P.sha256(source) != prior["source_ply_sha256"]:
        raise AssertionError("source changed")
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
    post, ids, families = P.make_native_post(model, mask, seed=20260927)
    if families != 9345:
        raise AssertionError("split family count changed")
    child_id = len(model.get_xyz) + prior["birth_choice"]
    birth = E.fitted_child(model, post, child_id,
                           fit["selected"]["opacity"], fit["selected"]["rgb"])
    deletion = prior["selected"]["deletion_id"]
    old = model._opacity[deletion, 0].detach().clone()
    rows = []
    for position, index in enumerate(F.CAMERA_INDICES):
        camera = scene.getTrainCameras()[index]
        base_count = count(camera, model, pipe)
        birth_count = count(camera, birth, pipe)
        with torch.no_grad():
            model._opacity[deletion, 0] = -30.0
        delete_count = count(camera, model, pipe)
        exchange = E.fitted_child(model, post, child_id,
                                  prior["selected"]["opacity"],
                                  prior["selected"]["rgb"])
        exchange_count = count(camera, exchange, pipe)
        with torch.no_grad():
            model._opacity[deletion, 0] = old
        rows.append({"camera": camera.image_name,
                     "role": "fit" if position < 6 else "internal_check",
                     "base": base_count, "delete": delete_count,
                     "birth": birth_count, "exchange": exchange_count,
                     "delete_minus_base": delete_count - base_count,
                     "birth_minus_base": birth_count - base_count,
                     "exchange_minus_base": exchange_count - base_count})
    output = {
        "source_ply_sha256": prior["source_ply_sha256"],
        "deletion_id": deletion,
        "birth_choice": prior["birth_choice"],
        "unit": "author rasterizer Gaussian-tile instances, not accepted pixel work or time",
        "per_view": rows,
        "total_base": sum(r["base"] for r in rows),
        "total_delete_minus_base": sum(r["delete_minus_base"] for r in rows),
        "total_birth_minus_base": sum(r["birth_minus_base"] for r in rows),
        "total_exchange_minus_base": sum(r["exchange_minus_base"] for r in rows),
        "elapsed_seconds_including_scene_load": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("FASTGS_EXCHANGE_COST_RESULT " + json.dumps({
        k: output[k] for k in ("total_base", "total_delete_minus_base",
                               "total_birth_minus_base", "total_exchange_minus_base",
                               "elapsed_seconds_including_scene_load")}), flush=True)


if __name__ == "__main__":
    main()
