"""Author-rendered, full-image FastGS loss for a preselected birth candidate.

The candidate identity, opacity, and RGB are read from a frozen fitting result.
No model optimization occurs; four internal check views were not used to fit.
"""

import json
import time
from pathlib import Path

import torch

import fastgs_birth_fit_probe as F
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render_fastgs
from utils.sh_utils import RGB2SH


SOURCE_RESULT = Path("/root/fastgs_birth_multi_candidate_result.json")
OUT = Path("/root/fastgs_birth_native_loss_result.json")


def native_image(camera, model, pipe):
    return render_fastgs(camera, model, pipe,
                         torch.zeros(3, device="cuda"), 0.5)["render"]


def measure(image, gt):
    l1 = torch.mean(torch.abs(image - gt))
    ssim = fast_ssim(image.unsqueeze(0), gt.unsqueeze(0))
    mse = torch.mean((image - gt).square())
    return {"l1": float(l1), "ssim": float(ssim),
            "loss": float(0.8 * l1 + 0.2 * (1.0 - ssim)),
            "psnr": float(-10.0 * torch.log10(mse.clamp_min(1e-20)))}


def main():
    started = time.perf_counter()
    P = F.P
    fit = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    if fit["status"] != "SELECTED" or fit["source_commit"] != "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f":
        raise AssertionError("fit result identity mismatch")
    source = P.REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"
    if P.sha256(source) != fit["source_ply_sha256"]:
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
        raise AssertionError("family count changed")
    choice = fit["selected"]["choice"]
    family = choice % families
    if int(selected_ids[family]) != fit["selected"]["parent_id"]:
        raise AssertionError("parent changed")
    child_id = len(model.get_xyz) + choice
    q = fit["selected"]["opacity"]
    rgb = fit["selected"]["rgb"]
    if not (0.0 < q < 0.8 and rgb is not None):
        raise AssertionError("invalid fitted birth")
    proposal = P.append_child(model, post, child_id, q)
    with torch.no_grad():
        proposal._features_dc[-1, 0, :] = RGB2SH(
            torch.tensor(rgb, device="cuda", dtype=proposal._features_dc.dtype))
        proposal._features_rest[-1].zero_()
        rows = []
        for position, index in enumerate(F.CAMERA_INDICES):
            camera = scene.getTrainCameras()[index]
            gt = camera.original_image
            base = native_image(camera, model, pipe)
            edited = native_image(camera, proposal, pipe)
            original = measure(base, gt)
            candidate = measure(edited, gt)
            row = {
                "camera": camera.image_name, "role": "fit" if position < 6 else "internal_check",
                "base": original, "candidate": candidate,
                "loss_delta_candidate_minus_base": candidate["loss"] - original["loss"],
                "psnr_delta_candidate_minus_base": candidate["psnr"] - original["psnr"],
                "max_rgb_change": float((edited - base).abs().max()),
                "changed_rgb_values_gt_1e-6": int(((edited - base).abs() > 1e-6).sum()),
            }
            rows.append(row)
            print("FASTGS_BIRTH_NATIVE_VIEW " + json.dumps({
                k: row[k] for k in ("camera", "role", "loss_delta_candidate_minus_base",
                                    "psnr_delta_candidate_minus_base", "max_rgb_change")}), flush=True)
    output = {
        "source_commit": fit["source_commit"],
        "source_ply_sha256": fit["source_ply_sha256"],
        "selected_choice": choice, "selected_opacity": q, "selected_rgb": rgb,
        "camera_indices": F.CAMERA_INDICES,
        "scope": "Full-image author FastGS L1+fused-SSIM on six fit and four internal-check training views at one fixed room state; no continuation or official unseen-scene test.",
        "per_view": rows,
        "mean_fit_loss_delta": sum(r["loss_delta_candidate_minus_base"] for r in rows[:6]) / 6,
        "mean_internal_check_loss_delta": sum(r["loss_delta_candidate_minus_base"] for r in rows[6:]) / 4,
        "mean_fit_psnr_delta": sum(r["psnr_delta_candidate_minus_base"] for r in rows[:6]) / 6,
        "mean_internal_check_psnr_delta": sum(r["psnr_delta_candidate_minus_base"] for r in rows[6:]) / 4,
        "elapsed_seconds_including_scene_load": time.perf_counter() - started,
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("FASTGS_BIRTH_NATIVE_RESULT " + json.dumps({
        k: output[k] for k in ("mean_fit_loss_delta", "mean_internal_check_loss_delta",
                               "mean_fit_psnr_delta", "mean_internal_check_psnr_delta",
                               "elapsed_seconds_including_scene_load")}), flush=True)


if __name__ == "__main__":
    main()
