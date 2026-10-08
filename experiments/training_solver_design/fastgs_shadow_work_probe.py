"""Count actual pixel-list traversal in one full-resolution FastGS split event."""

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
from scene import GaussianModel, Scene  # noqa: E402


OUT = Path("/root/fastgs_shadow_work_result.json")
CAMERA_INDICES = (0, 90, 180, 270)
NAMES = ("visited_tile_rows", "raw_above_cutoff", "shadow_raw_above_cutoff",
         "nonzero_alpha_accepted", "all_accepted")


def main():
    source = REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"
    mask_path = Path("/root/fastgs_gate_selected_4100_mask.pt")
    if sha256(source) != "64160970e5ea580aea4f79278a8fbe334d8d81ef6e304a903626843cc48f85bc":
        raise AssertionError("pre-split PLY changed")
    selected = torch.load(mask_path, map_location="cpu", weights_only=True)["selected_splits"]
    if int(selected.sum()) != 9345:
        raise AssertionError("native mask changed")
    ext = load(name="fastgs_shadow_work_diagnostic_v1",
               sources=["/root/fastgs_shadow_work_cuda.cu"],
               extra_cuda_cflags=["-O3"], verbose=True)
    dataset = SimpleNamespace(
        source_path="/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room",
        model_path=str(REPO / "output/room_big_capture4100"),
        images="images", resolution=-1, white_background=False,
        data_device="cuda", eval=True)
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    model = GaussianModel(3, "default")
    scene = Scene(dataset, model, load_iteration=4100, shuffle=False)
    cameras = scene.getTrainCameras()
    if len(cameras) != 272 or selected.numel() != model.get_xyz.shape[0]:
        raise AssertionError("scene/model mismatch")
    post, selected_ids, family_count = make_native_post(model, selected, seed=20260927)
    roles = torch.zeros(model.get_xyz.shape[0] + 2 * family_count,
                        dtype=torch.int32, device="cuda")
    roles[selected_ids] = 1
    roles[model.get_xyz.shape[0]:] = 2
    rows = []
    for camera_index in CAMERA_INDICES:
        camera = cameras[camera_index]
        width, height = camera.image_width, camera.image_height
        tw, th = (width + 15) // 16, (height + 15) // 16
        _, base_instances, base_geom, base_img, base_list = render_and_capture(camera, model, pipe)
        with torch.no_grad():
            post._opacity[selected_ids] = -30.0
        _, post_instances, post_geom, post_img, post_list = render_and_capture(camera, post, pipe)
        with torch.no_grad():
            post._opacity[selected_ids] = model._opacity[selected_ids]
        _, union_instances, union_geom, union_img, union_list = render_and_capture(camera, post, pipe)
        zero_base = torch.zeros(model.get_xyz.shape[0], dtype=torch.int32, device="cuda")
        zero_post = torch.zeros(post.get_xyz.shape[0], dtype=torch.int32, device="cuda")
        configs = {
            "native_before": (base_geom, base_img, base_list, zero_base, 0.0),
            "native_after": (post_geom, post_img, post_list, zero_post, 1.0),
            "shadow_birth": (union_geom, union_img, union_list, roles, 0.0),
            "shadow_half": (union_geom, union_img, union_list, roles, 0.5),
            "shadow_release": (union_geom, union_img, union_list, roles, 1.0),
        }
        result = {"camera": camera.image_name, "resolution": [width, height],
                  "tile_instances": {"before": base_instances, "after": post_instances,
                                     "union": union_instances}, "work": {}}
        for name, (geom, img, point_list, role, gate) in configs.items():
            work = ext.count_work(geom["mean2d"], geom["conic"], role,
                                  img["ranges"][:tw * th].contiguous(),
                                  point_list.contiguous(), width, height, gate)
            totals = work.sum(dim=0, dtype=torch.int64).cpu().tolist()
            result["work"][name] = dict(zip(NAMES, totals))
        if result["work"]["native_after"]["shadow_raw_above_cutoff"] != 0:
            raise AssertionError("native control has shadow roles")
        rows.append(result)
        print("FASTGS_SHADOW_WORK_VIEW " + json.dumps(result, sort_keys=True), flush=True)
    payload = {"source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
               "ply_sha256": sha256(source), "mask_sha256": sha256(mask_path),
               "child_seed": 20260927, "family_count": family_count, "views": rows,
               "scope": "Scalar work count using author-projected and sorted tile lists; same cutoff "
                        "and early stop as isolated shadow VJP. It does not time or prove a fused "
                        "native CUDA trainer implementation."}
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
