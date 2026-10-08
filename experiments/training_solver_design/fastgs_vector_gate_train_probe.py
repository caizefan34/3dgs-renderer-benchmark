"""Frozen-state, train-view-only vector split-gate quality opportunity probe."""

import hashlib
import json
import math
import os
import sys
import time
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


CAMERA_INDICES = (0, 30, 60, 90, 120, 150, 180, 210, 240, 270)
EPOCHS = 12
LR = 0.1
OUT = Path("/root/fastgs_vector_gate_train_20260927.pt")
SUMMARY = Path("/root/fastgs_vector_gate_train_20260927.json")


def image_and_vjp(ext, item, gates):
    args = item["args"]
    image = ext.event_gate_forward(*args, item["zero_adjoint"], gates,
                                   item["width"], item["height"], item["families"], 0)[0]
    image = image.permute(2, 0, 1).detach().requires_grad_(True)
    gt = item["gt"]
    loss = .8 * l1_loss(image, gt) + .2 * (1 - fast_ssim(image.unsqueeze(0), gt.unsqueeze(0)))
    adjoint = torch.autograd.grad(loss, image)[0].detach().permute(1, 2, 0).contiguous()
    image2, family_grad, _ = ext.event_gate_forward(*args, adjoint, gates,
                                                    item["width"], item["height"], item["families"], 0)
    if not torch.equal(image.detach().permute(1, 2, 0), image2):
        raise AssertionError("VJP pass changed RGB")
    return float(loss.detach()), family_grad


def cpu_pixel_vjp(x, y, ids, geom, roles, families, gate_values, adjoint, target_family):
    """Independent FP64 autograd with every encountered family gate variable."""
    local = gate_values.detach().cpu().double().requires_grad_(True)
    means = geom["mean2d"][ids].detach().cpu().double()
    conics = geom["conic"][ids].detach().cpu().double()
    colors = geom["rgb"][ids].detach().cpu().double()
    roles = roles[ids].detach().cpu().tolist()
    families = families[ids].detach().cpu().tolist()
    rgb = torch.zeros(3, dtype=torch.float64)
    trans = torch.ones((), dtype=torch.float64)
    for j, role in enumerate(roles):
        dx, dy = float(means[j, 0]) - x, float(means[j, 1]) - y
        co = conics[j]
        power = -0.5 * (float(co[0]) * dx * dx + float(co[2]) * dy * dy) - float(co[1]) * dx * dy
        if power > 0:
            continue
        raw_value = min(.99, float(co[3]) * math.exp(power))
        if raw_value < 1 / 255:
            continue
        raw = torch.tensor(raw_value, dtype=torch.float64)
        if role == 1:
            alpha = 1 - torch.pow(1 - raw, 1 - local[families[j]])
        elif role == 2:
            alpha = 1 - torch.pow(1 - raw, local[families[j]])
        else:
            alpha = raw
        next_trans = trans * (1 - alpha)
        if float(next_trans.detach()) < 1e-4:
            break
        rgb = rgb + trans * alpha * colors[j]
        trans = next_trans
    scalar = (rgb * adjoint.double()).sum()
    value = torch.autograd.grad(scalar, local)[0][target_family]
    return float(value)


def main():
    source = REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"
    mask_path = Path("/root/fastgs_gate_selected_4100_mask.pt")
    if sha256(source) != "64160970e5ea580aea4f79278a8fbe334d8d81ef6e304a903626843cc48f85bc":
        raise AssertionError("pre-event PLY changed")
    selected = torch.load(mask_path, map_location="cpu", weights_only=True)["selected_splits"]
    if int(selected.sum()) != 9345:
        raise AssertionError("native parent mask changed")
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
    cameras = scene.getTrainCameras()
    if len(cameras) != 272 or selected.numel() != model.get_xyz.shape[0]:
        raise AssertionError("scene/model mismatch")
    post, selected_ids, family_count = make_native_post(model, selected, seed=20260927)
    n = int(model.get_xyz.shape[0])
    roles = torch.zeros(n + 2 * family_count, dtype=torch.int32, device="cuda")
    families = torch.full_like(roles, -1)
    family_ids = torch.arange(family_count, dtype=torch.int32, device="cuda")
    roles[selected_ids] = 1
    roles[n:] = 2
    families[selected_ids] = family_ids
    families[n:] = family_ids.repeat(2)
    zeros = torch.zeros(family_count, dtype=torch.float32, device="cuda")
    ones = torch.ones_like(zeros)
    items = []
    setup_start = time.perf_counter()
    for camera_index in CAMERA_INDICES:
        camera = cameras[camera_index]
        with torch.no_grad():
            post._opacity[selected_ids] = -30.0
        native_post, _, _, _, _ = render_and_capture(camera, post, pipe)
        native_pre, _, _, _, _ = render_and_capture(camera, model, pipe)
        with torch.no_grad():
            post._opacity[selected_ids] = model._opacity[selected_ids]
        _, _, geom, img, point_list = render_and_capture(camera, post, pipe)
        width, height = camera.image_width, camera.image_height
        tw, th = (width + 15) // 16, (height + 15) // 16
        args = (geom["mean2d"], geom["conic"], geom["rgb"], roles, families,
                img["ranges"][:tw * th].contiguous(), point_list.contiguous())
        zero_adj = torch.zeros((height, width, 3), dtype=torch.float32, device="cuda")
        pre = ext.event_gate_forward(*args, zero_adj, zeros, width, height, family_count, 0)[0]
        after = ext.event_gate_forward(*args, zero_adj, ones, width, height, family_count, 0)[0]
        pre_err = float((pre - native_pre.permute(1, 2, 0)).abs().max())
        post_err = float((after - native_post.permute(1, 2, 0)).abs().max())
        if pre_err > 2e-5 or post_err > 2e-5:
            raise AssertionError((camera.image_name, pre_err, post_err))
        items.append({"camera": camera.image_name, "camera_index": camera_index,
                      "args": args, "zero_adjoint": zero_adj, "gt": camera.original_image.cuda(),
                      "width": width, "height": height, "families": family_count,
                      "pre_max_rgb_error": pre_err, "post_max_rgb_error": post_err})
        print("FASTGS_VECTOR_GATE_CACHE " + json.dumps({
            "camera": camera.image_name, "pre_max_rgb_error": pre_err,
            "post_max_rgb_error": post_err}, sort_keys=True), flush=True)
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup_start

    # A mixed-gate pixel check exercises family-indexed gates and their VJP.
    torch.manual_seed(20260927)
    mixed = .2 + .6 * torch.rand_like(zeros)
    first = items[0]
    x, y = 790, 534
    debug_family = int((selected_ids == 415040).nonzero().flatten().item())
    test_adj = torch.zeros_like(first["zero_adjoint"])
    test_adj[y, x] = torch.tensor([1., -.3, .5], device="cuda")
    _, _, debug = ext.event_gate_forward(*first["args"], test_adj, mixed,
                                         first["width"], first["height"], family_count, debug_family)
    tile = (y // 16) * ((first["width"] + 15) // 16) + x // 16
    begin, end = first["args"][5][tile].cpu().tolist()
    ids = first["args"][6][begin:end].long()
    fp64 = cpu_pixel_vjp(x, y, ids, {"mean2d": first["args"][0],
                                    "conic": first["args"][1], "rgb": first["args"][2]},
                           roles, families, mixed, test_adj[y, x].cpu(), debug_family)
    gpu = float(debug[y, x])
    if not math.isfinite(fp64) or not math.isfinite(gpu) or abs(gpu - fp64) > 1e-6:
        raise AssertionError({"gpu": gpu, "fp64": fp64})
    oracle = {"pixel": [x, y], "family": debug_family,
              "gpu_vjp": gpu, "fp64_vjp": fp64, "abs_error": abs(gpu - fp64)}
    print("FASTGS_VECTOR_GATE_PIXEL_ORACLE " + json.dumps(oracle, sort_keys=True), flush=True)

    gates = torch.nn.Parameter(zeros.clone())
    optimizer = torch.optim.Adam([gates], lr=LR, eps=1e-12)
    history = []
    torch.cuda.synchronize()
    train_start = time.perf_counter()
    for epoch in range(EPOCHS):
        losses = []
        total_grad = torch.zeros_like(gates)
        with torch.no_grad():
            current_gates = gates.detach().clone()
        for item in items:
            loss, grad = image_and_vjp(ext, item, current_gates)
            losses.append(loss)
            total_grad += grad
        optimizer.zero_grad(set_to_none=True)
        gates.grad = (total_grad / len(items)).detach()
        gates.grad[gates.grad.abs() < 1e-10] = 0
        optimizer.step()
        with torch.no_grad():
            gates.clamp_(0, 1)
        record = {"epoch": epoch, "train_loss_before_step": sum(losses) / len(losses),
                  "gates_gt_half_after_step": int((gates > .5).sum()),
                  "gates_zero_after_step": int((gates == 0).sum()),
                  "gate_mean_after_step": float(gates.mean())}
        history.append(record)
        print("FASTGS_VECTOR_GATE_EPOCH " + json.dumps(record, sort_keys=True), flush=True)
    final_losses = [image_and_vjp(ext, item, gates.detach())[0] for item in items]
    native_post_losses = [image_and_vjp(ext, item, ones)[0] for item in items]
    native_pre_losses = [image_and_vjp(ext, item, zeros)[0] for item in items]
    binary = (gates.detach() > .5).float()
    binary_losses = [image_and_vjp(ext, item, binary)[0] for item in items]
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - train_start
    bundle = {"source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
              "ply_sha256": sha256(source), "mask_sha256": sha256(mask_path),
              "camera_indices": CAMERA_INDICES, "camera_names": [x["camera"] for x in items],
              "family_parent_ids": selected_ids.detach().cpu(),
              "gate_values": gates.detach().cpu(), "binary_commit": binary.bool().cpu(),
              "epochs": EPOCHS, "lr": LR, "child_seed": 20260927}
    torch.save(bundle, OUT)
    summary = {"source_commit": bundle["source_commit"], "ply_sha256": bundle["ply_sha256"],
               "mask_sha256": bundle["mask_sha256"], "camera_names": bundle["camera_names"],
               "families": family_count, "endpoint_max_pre_error": max(x["pre_max_rgb_error"] for x in items),
               "endpoint_max_post_error": max(x["post_max_rgb_error"] for x in items),
               "pixel_oracle": oracle, "history": history,
               "mean_train_loss": {"native_pre": sum(native_pre_losses) / len(items),
                                   "native_post": sum(native_post_losses) / len(items),
                                   "continuous": sum(final_losses) / len(items),
                                   "binary": sum(binary_losses) / len(items)},
               "binary_commit": int(binary.sum()), "gate_mean": float(gates.mean()),
               "gate_quantiles": torch.quantile(gates.detach(), torch.tensor([0., .25, .5, .75, 1.], device="cuda")).cpu().tolist(),
               "setup_seconds_excluding_scene_load": setup_seconds,
               "optimization_and_four_final_scores_seconds": training_seconds,
               "bundle_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
               "scope": "Frozen room split state, ten fixed full-resolution training cameras. "
                        "Only per-family gates are optimized; geometry/colors/Adam and camera input remain fixed. "
                        "This is an offline quality opportunity screen, not a full training algorithm or net-speed result."}
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("FASTGS_VECTOR_GATE_SUMMARY " + json.dumps({
        "mean_train_loss": summary["mean_train_loss"], "binary_commit": summary["binary_commit"],
        "gate_quantiles": summary["gate_quantiles"], "training_seconds": training_seconds,
        "bundle_sha256": summary["bundle_sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
