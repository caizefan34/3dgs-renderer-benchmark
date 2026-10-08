"""Evaluate frozen-state vector gates on previously consulted room development views."""

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.cpp_extension import load


REPO = Path("/root/codex_fastgs_audit_20260927")
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/root")

from fastgs_event_split_cost_probe import make_native_post  # noqa: E402
from fastgs_real_split_shadow_probe import render_and_capture  # noqa: E402
from fastgs_real_tile_reference import sha256  # noqa: E402
from fused_ssim import fused_ssim as fast_ssim  # noqa: E402
from lpipsPyTorch.modules.lpips import LPIPS  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.image_utils import psnr  # noqa: E402


OUT = Path("/root/fastgs_vector_gate_dev_eval.json")


def main():
    source = REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"
    mask_path = Path("/root/fastgs_gate_selected_4100_mask.pt")
    bundle = torch.load("/root/fastgs_vector_gate_train_20260927.pt",
                        map_location="cpu", weights_only=True)
    if bundle["ply_sha256"] != sha256(source) or bundle["mask_sha256"] != sha256(mask_path):
        raise AssertionError("frozen event identity mismatch")
    selected = torch.load(mask_path, map_location="cpu", weights_only=True)["selected_splits"]
    ext = load(name="fastgs_vector_gate_diagnostic_v1",
               sources=["/root/fastgs_vector_gate_cuda.cu"],
               extra_cuda_cflags=["-O3"], verbose=True)
    dataset = SimpleNamespace(
        source_path="/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room",
        model_path=str(REPO / "output/room_big_capture4100"),
        images="images", resolution=-1, white_background=False,
        data_device="cuda", eval=True)
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    model = GaussianModel(3, "default")
    scene = Scene(dataset, model, load_iteration=4100, shuffle=False)
    cameras = scene.getTestCameras()
    if len(cameras) != 39:
        raise AssertionError(len(cameras))
    post, selected_ids, family_count = make_native_post(model, selected, seed=bundle["child_seed"])
    if not torch.equal(selected_ids.cpu(), bundle["family_parent_ids"]):
        raise AssertionError("family order differs")
    with torch.no_grad():
        post._opacity[selected_ids] = model._opacity[selected_ids]
    n = int(model.get_xyz.shape[0])
    roles = torch.zeros(n + 2 * family_count, dtype=torch.int32, device="cuda")
    families = torch.full_like(roles, -1)
    family_ids = torch.arange(family_count, dtype=torch.int32, device="cuda")
    roles[selected_ids] = 1
    roles[n:] = 2
    families[selected_ids] = family_ids
    families[n:] = family_ids.repeat(2)
    continuous = bundle["gate_values"].float().cuda().contiguous()
    binary = bundle["binary_commit"].float().cuda().contiguous()
    variants = {"native_pre": torch.zeros_like(continuous),
                "native_post": torch.ones_like(continuous),
                "continuous": continuous, "binary": binary}
    vgg = LPIPS("vgg", "0.1").cuda().eval()
    rows = {name: [] for name in variants}
    endpoint_error = None
    with torch.no_grad():
        for camera in cameras:
            _, _, geom, img, point_list = render_and_capture(camera, post, pipe)
            width, height = camera.image_width, camera.image_height
            tw, th = (width + 15) // 16, (height + 15) // 16
            args = (geom["mean2d"], geom["conic"], geom["rgb"], roles, families,
                    img["ranges"][:tw * th].contiguous(), point_list.contiguous())
            zero_adj = torch.zeros((height, width, 3), dtype=torch.float32, device="cuda")
            gt = camera.original_image.cuda().clamp(0, 1)
            for name, gates in variants.items():
                rgb = ext.event_gate_forward(*args, zero_adj, gates,
                                             width, height, family_count, 0)[0]
                image = rgb.permute(2, 0, 1).clamp(0, 1).contiguous()
                rows[name].append({
                    "camera": camera.image_name,
                    "psnr": float(psnr(image.unsqueeze(0), gt.unsqueeze(0)).mean()),
                    "ssim": float(fast_ssim(image.unsqueeze(0), gt.unsqueeze(0))),
                    "lpips": float(vgg(image.unsqueeze(0), gt.unsqueeze(0))),
                })
            if endpoint_error is None:
                native_pre, _, _, _, _ = render_and_capture(camera, model, pipe)
                with torch.no_grad():
                    post._opacity[selected_ids] = -30.0
                native_post, _, _, _, _ = render_and_capture(camera, post, pipe)
                with torch.no_grad():
                    post._opacity[selected_ids] = model._opacity[selected_ids]
                endpoint_error = {
                    "pre": float((native_pre - ext.event_gate_forward(*args, zero_adj,
                                           variants["native_pre"], width, height,
                                           family_count, 0)[0].permute(2, 0, 1)).abs().max()),
                    "post": float((native_post - ext.event_gate_forward(*args, zero_adj,
                                            variants["native_post"], width, height,
                                            family_count, 0)[0].permute(2, 0, 1)).abs().max()),
                }
                if max(endpoint_error.values()) > 2e-5:
                    raise AssertionError(endpoint_error)
            print("FASTGS_VECTOR_GATE_DEV_VIEW " + json.dumps({
                "camera": camera.image_name,
                "psnr": {name: rows[name][-1]["psnr"] for name in rows}}, sort_keys=True), flush=True)
    output = {"source_commit": bundle["source_commit"],
              "bundle_sha256": hashlib.sha256(Path("/root/fastgs_vector_gate_train_20260927.pt").read_bytes()).hexdigest(),
              "camera_names": [c.image_name for c in cameras],
              "effective_resolution": [1600, 1066],
              "endpoint_max_rgb_error_first_dev_camera": endpoint_error,
              "arms": {name: {"mean": {metric: sum(row[metric] for row in arm) / len(arm)
                                            for metric in ("psnr", "ssim", "lpips")},
                               "per_view": arm} for name, arm in rows.items()},
              "scope": "One fixed room event; 39 development views already consulted in this project. "
                       "No model-parameter updates or continued training. Not independent quality evidence."}
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("FASTGS_VECTOR_GATE_DEV_SUMMARY " + json.dumps({
        name: value["mean"] for name, value in output["arms"].items()}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
