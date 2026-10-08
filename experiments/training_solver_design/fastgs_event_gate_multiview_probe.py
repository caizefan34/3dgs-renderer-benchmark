"""Screen one real FastGS split event with full-resolution train-view gate VJPs.

This does not alter author training or select a final policy. It tests whether
one-shot counterfactual split gradients are stable across training views.
"""

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


REPO = Path("/root/codex_fastgs_audit_20260927")
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/root")

from fastgs_event_gate_vjp_probe import extension  # noqa: E402
from fastgs_event_split_cost_probe import make_native_post  # noqa: E402
from fastgs_real_split_shadow_probe import render_and_capture  # noqa: E402
from fastgs_real_tile_reference import sha256  # noqa: E402
from fused_ssim import fused_ssim as fast_ssim  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.loss_utils import l1_loss  # noqa: E402


CAMERA_INDICES = (0, 30, 60, 90, 120, 150, 180, 210, 240, 270)
SEED = int(os.environ.get("FASTGS_CHILD_SEED", "20260927"))
OUT = Path(f"/root/fastgs_event_gate_multiview_scores_{SEED}.pt")


def sync_elapsed_ms(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, (time.perf_counter() - start) * 1000


def main():
    dataset = SimpleNamespace(
        source_path="/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room",
        model_path=str(REPO / "output/room_big_capture4100"),
        images="images", resolution=-1, white_background=False,
        data_device="cuda", eval=True,
    )
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    model = GaussianModel(3, "default")
    scene = Scene(dataset, model, load_iteration=4100, shuffle=False)
    cameras = scene.getTrainCameras()
    if len(cameras) != 272:
        raise AssertionError(len(cameras))
    n = int(model.get_xyz.shape[0])
    mask_path = Path("/root/fastgs_gate_selected_4100_mask.pt")
    selected = torch.load(mask_path, map_location="cpu", weights_only=True)["selected_splits"]
    if selected.numel() != n or int(selected.sum()) != 9345:
        raise AssertionError("native selection mismatch")
    post, selected_ids, family_count = make_native_post(model, selected, seed=SEED)
    with torch.no_grad():
        post._opacity[selected_ids] = model._opacity[selected_ids]
    roles = torch.zeros(n + 2 * family_count, dtype=torch.int32, device="cuda")
    families = torch.full((n + 2 * family_count,), -1, dtype=torch.int32, device="cuda")
    family_ids = torch.arange(family_count, dtype=torch.int32, device="cuda")
    roles[selected_ids] = 1
    roles[n:] = 2
    families[selected_ids] = family_ids
    families[n:] = family_ids.repeat(2)
    ext = extension()

    rows = []
    scores = []
    for camera_index in CAMERA_INDICES:
        camera = cameras[camera_index]
        width, height = camera.image_width, camera.image_height
        tw, th = (width + 15) // 16, (height + 15) // 16
        base_result, base_render_ms = sync_elapsed_ms(lambda: render_and_capture(camera, model, pipe))
        base_image, base_instances, _, _, _ = base_result
        gt = camera.original_image.cuda()

        def make_adjoint():
            loss = 0.8 * l1_loss(base_image, gt) + 0.2 * (1 - fast_ssim(base_image.unsqueeze(0), gt.unsqueeze(0)))
            return torch.autograd.grad(loss, base_image)[0].detach().permute(1, 2, 0).contiguous()

        adjoint, adjoint_ms = sync_elapsed_ms(make_adjoint)
        union_result, union_render_ms = sync_elapsed_ms(lambda: render_and_capture(camera, post, pipe))
        _, union_instances, geom, img, point_list = union_result
        args = (geom["mean2d"], geom["conic"], geom["rgb"],
                roles, families, img["ranges"][:tw * th].contiguous(),
                point_list.contiguous(), adjoint, width, height,
                family_count, 0)
        (rgb, grad, _), gate_ms = sync_elapsed_ms(lambda: ext.event_gate_forward(*args, 0.0))
        error = float((rgb - base_image.detach().permute(1, 2, 0)).abs().max())
        if error > 2e-5 or not bool(torch.isfinite(grad).all()):
            raise AssertionError((camera.image_name, error))
        scores.append(grad.detach().cpu())
        record = {
            "camera_index": camera_index,
            "camera": camera.image_name,
            "resolution": [width, height],
            "base_tile_instances": base_instances,
            "union_tile_instances": union_instances,
            "birth_max_rgb_error": error,
            "families_nonzero_vjp": int((grad != 0).sum()),
            "base_render_wall_ms": base_render_ms,
            "original_loss_adjoint_wall_ms": adjoint_ms,
            "union_render_wall_ms": union_render_ms,
            "event_gate_wall_ms": gate_ms,
        }
        rows.append(record)
        print("FASTGS_EVENT_GATE_VIEW " + json.dumps(record, sort_keys=True), flush=True)

    matrix = torch.stack(scores)
    aggregate = matrix.mean(dim=0)
    meaningful = matrix.abs() > 1e-10
    positive = (matrix > 1e-10).sum(dim=0)
    negative = (matrix < -1e-10).sum(dim=0)
    supported_two = (meaningful.sum(dim=0) >= 2)
    opposite_signs = (positive > 0) & (negative > 0)
    torch.save({
        "source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
        "ply_sha256": sha256(REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"),
        "mask_sha256": sha256(mask_path),
        "camera_indices": CAMERA_INDICES,
        "child_seed": SEED,
        "camera_names": [row["camera"] for row in rows],
        "family_parent_ids": selected_ids.cpu(),
        "camera_family_loss_vjp": matrix,
    }, OUT)
    result = {
        "camera_count": len(rows),
        "child_seed": SEED,
        "camera_indices": CAMERA_INDICES,
        "families": family_count,
        "raw_scores_path": str(OUT),
        "ply_sha256": sha256(REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"),
        "mask_sha256": sha256(mask_path),
        "all_camera_max_birth_rgb_error": max(row["birth_max_rgb_error"] for row in rows),
        "aggregate_negative": int((aggregate < -1e-10).sum()),
        "aggregate_positive": int((aggregate > 1e-10).sum()),
        "supported_on_two_views": int(supported_two.sum()),
        "supported_on_two_with_opposite_signs": int((supported_two & opposite_signs).sum()),
        "mean_vjp_abs_quantiles": torch.quantile(aggregate.abs(), torch.tensor([0., .5, .9, .99, 1.])).tolist(),
        "sum_base_render_wall_ms": sum(row["base_render_wall_ms"] for row in rows),
        "sum_original_loss_adjoint_wall_ms": sum(row["original_loss_adjoint_wall_ms"] for row in rows),
        "sum_union_render_wall_ms": sum(row["union_render_wall_ms"] for row in rows),
        "sum_event_gate_wall_ms": sum(row["event_gate_wall_ms"] for row in rows),
        "scope": "Ten fixed training cameras, one fixed native-selected event, one candidate child sample. Rank stability and diagnostic cost only; no final training policy, complete wall or novel-view quality.",
    }
    print("FASTGS_EVENT_GATE_MULTIVIEW " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
