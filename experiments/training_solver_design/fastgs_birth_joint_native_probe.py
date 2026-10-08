"""Native full-image FastGS evaluation of all fit-positive birth candidates."""

import json
import time
from pathlib import Path

import torch
from torch import nn

import fastgs_birth_native_loss_probe as N
import fastgs_birth_fit_probe as F
from utils.sh_utils import RGB2SH


SOURCE_RESULT = Path("/root/fastgs_birth_multi_candidate_result.json")
OUT = Path("/root/fastgs_birth_joint_native_result.json")


def append_fitted_children(model, post, records):
    P = F.P
    ids = torch.tensor([len(model.get_xyz) + r["choice"] for r in records],
                       dtype=torch.long, device="cuda")
    result = P.GaussianModel(model.max_sh_degree, "default")
    result.active_sh_degree = model.active_sh_degree
    for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"):
        base = getattr(model, name).detach()
        child = getattr(post, name)[ids].detach().clone()
        if name == "_opacity":
            q = torch.tensor([r["opacity"] for r in records],
                             dtype=child.dtype, device="cuda")
            child[:, 0] = torch.logit(q)
        elif name == "_features_dc":
            rgb = torch.tensor([r["rgb"] for r in records],
                               dtype=child.dtype, device="cuda")
            child[:, 0, :] = RGB2SH(rgb)
        elif name == "_features_rest":
            child.zero_()
        setattr(result, name, nn.Parameter(torch.cat((base, child), dim=0)))
    return result


def main():
    started = time.perf_counter()
    P = F.P
    fit = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    source = P.REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"
    if P.sha256(source) != fit["source_ply_sha256"]:
        raise AssertionError("source PLY changed")
    records = [r for r in fit["candidates"]
               if r.get("fit_sse_gain", 0.0) > 1e-8 and r.get("opacity", 0.0) > 1e-3]
    if len(records) != 14:
        raise AssertionError("candidate rule changed")
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
        raise AssertionError("family count changed")
    proposal = append_fitted_children(model, post, records)
    rows = []
    with torch.no_grad():
        for position, index in enumerate(F.CAMERA_INDICES):
            camera = scene.getTrainCameras()[index]
            gt = camera.original_image
            base = N.native_image(camera, model, pipe)
            edited = N.native_image(camera, proposal, pipe)
            original = N.measure(base, gt)
            candidate = N.measure(edited, gt)
            row = {
                "camera": camera.image_name,
                "role": "fit" if position < 6 else "internal_check",
                "loss_delta_candidate_minus_base": candidate["loss"] - original["loss"],
                "psnr_delta_candidate_minus_base": candidate["psnr"] - original["psnr"],
                "max_rgb_change": float((edited - base).abs().max()),
                "changed_rgb_values_gt_1e-6": int(((edited - base).abs() > 1e-6).sum()),
            }
            rows.append(row)
            print("FASTGS_BIRTH_JOINT_VIEW " + json.dumps(row), flush=True)
    output = {
        "source_commit": fit["source_commit"],
        "source_ply_sha256": fit["source_ply_sha256"],
        "selection_rule": "fit_sse_gain > 1e-8 and fitted opacity > 1e-3; no internal-check GT used",
        "candidate_count": len(records),
        "candidate_choices": [r["choice"] for r in records],
        "camera_indices": F.CAMERA_INDICES,
        "scope": "One fixed room state, native 1600x1066 full-image render and L1+fused-SSIM, six fit and four internal-check training views. No optimization or untouched-scene test.",
        "per_view": rows,
        "mean_fit_loss_delta": sum(r["loss_delta_candidate_minus_base"] for r in rows[:6]) / 6,
        "mean_internal_check_loss_delta": sum(r["loss_delta_candidate_minus_base"] for r in rows[6:]) / 4,
        "mean_fit_psnr_delta": sum(r["psnr_delta_candidate_minus_base"] for r in rows[:6]) / 6,
        "mean_internal_check_psnr_delta": sum(r["psnr_delta_candidate_minus_base"] for r in rows[6:]) / 4,
        "elapsed_seconds_including_scene_load": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("FASTGS_BIRTH_JOINT_RESULT " + json.dumps({
        k: output[k] for k in ("candidate_count", "mean_fit_loss_delta",
                               "mean_internal_check_loss_delta", "mean_fit_psnr_delta",
                               "mean_internal_check_psnr_delta",
                               "elapsed_seconds_including_scene_load")}), flush=True)


if __name__ == "__main__":
    main()
