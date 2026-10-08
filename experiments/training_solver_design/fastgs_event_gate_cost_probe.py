"""Isolate arithmetic and atomic cost of all-family shadow-gate VJP."""

import json
import os
import statistics
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
from scene import GaussianModel, Scene  # noqa: E402
from utils.loss_utils import l1_loss  # noqa: E402


OUT = Path("/root/fastgs_event_gate_cost_result.json")
CAMERA_INDICES = (0, 90, 180, 270)


def timed_modes(ext, args):
    modes = {"rgb_only": (False, False),
             "vjp_no_atomic": (True, False),
             "vjp_with_atomic": (True, True)}
    for _ in range(3):
        for second, atomic in modes.values():
            ext.event_gate_forward(*args, .5, second, atomic)
    torch.cuda.synchronize()
    raw = {name: [] for name in modes}
    for repeat in range(15):
        ordered = list(modes.items())
        if repeat % 2:
            ordered.reverse()
        for name, (second, atomic) in ordered:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            ext.event_gate_forward(*args, .5, second, atomic)
            end.record()
            end.synchronize()
            raw[name].append(float(start.elapsed_time(end)))
    return {name: {"median_ms": statistics.median(values), "raw_ms": values}
            for name, values in raw.items()}


def main():
    source = REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"
    mask_path = Path("/root/fastgs_gate_selected_4100_mask.pt")
    if sha256(source) != "64160970e5ea580aea4f79278a8fbe334d8d81ef6e304a903626843cc48f85bc":
        raise AssertionError("pre-event PLY changed")
    selected = torch.load(mask_path, map_location="cpu", weights_only=True)["selected_splits"]
    if int(selected.sum()) != 9345:
        raise AssertionError("native split mask changed")
    ext = load(name="fastgs_event_gate_cost_diagnostic_v1",
               sources=["/root/fastgs_event_gate_cost_cuda.cu"],
               extra_cuda_cflags=["-O3"], verbose=True)
    dataset = SimpleNamespace(
        source_path="/root/3dgs-renderer-benchmark/data/datasets/mipnerf360/room",
        model_path=str(REPO / "output/room_big_capture4100"),
        images="images", resolution=-1, white_background=False,
        data_device="cuda", eval=True)
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    model = GaussianModel(3, "default")
    scene = Scene(dataset, model, load_iteration=4100, shuffle=False)
    post, selected_ids, family_count = make_native_post(model, selected, seed=20260927)
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
    rows = []
    for camera_index in CAMERA_INDICES:
        camera = scene.getTrainCameras()[camera_index]
        base_image, _, _, _, _ = render_and_capture(camera, model, pipe)
        gt = camera.original_image.cuda()
        loss = .8 * l1_loss(base_image, gt) + .2 * (1 - fast_ssim(base_image.unsqueeze(0), gt.unsqueeze(0)))
        adjoint = torch.autograd.grad(loss, base_image)[0].detach().permute(1, 2, 0).contiguous()
        _, union_instances, geom, img, point_list = render_and_capture(camera, post, pipe)
        width, height = camera.image_width, camera.image_height
        tw, th = (width + 15) // 16, (height + 15) // 16
        args = (geom["mean2d"], geom["conic"], geom["rgb"], roles, families,
                img["ranges"][:tw * th].contiguous(), point_list.contiguous(),
                adjoint, width, height, family_count, 0)
        a = ext.event_gate_forward(*args, .5, False, False)
        b = ext.event_gate_forward(*args, .5, True, False)
        c = ext.event_gate_forward(*args, .5, True, True)
        torch.cuda.synchronize()
        if not (torch.equal(a[0], b[0]) and torch.equal(b[0], c[0])
                and torch.equal(b[2], c[2]) and bool(torch.isfinite(c[1]).all())):
            raise AssertionError(f"mode output mismatch: {camera.image_name}")
        times = timed_modes(ext, args)
        row = {"camera": camera.image_name, "resolution": [width, height],
               "union_tile_instances": union_instances, "timing": times,
               "rgb_equal": True, "debug_vjp_equal": True,
               "nonzero_family_grad": int((c[1] != 0).sum())}
        rows.append(row)
        print("FASTGS_EVENT_GATE_COST_VIEW " + json.dumps(row, sort_keys=True), flush=True)
    result = {"ply_sha256": sha256(source), "mask_sha256": sha256(mask_path),
              "views": rows,
              "scope": "Isolated scalar diagnostic kernel at gate=.5 on author projection/sort. "
                       "Its cost differences do not equal the incremental cost of a fused native "
                       "FastGS backward. No training utility is measured."}
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
