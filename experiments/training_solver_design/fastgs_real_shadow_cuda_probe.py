"""Validate isolated CUDA shadow gate on author FastGS real-camera tile data."""

import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPO = Path("/root/codex_fastgs_audit_20260927")
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/root")

from fastgs_real_split_shadow_probe import make_post_model, pixel_layers, render_and_capture  # noqa: E402
from fastgs_real_tile_reference import recompose, sha256  # noqa: E402
from fastgs_shadow_gate_renderer import extension  # noqa: E402
from fused_ssim import fused_ssim as fast_ssim  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.loss_utils import l1_loss  # noqa: E402


def max_error(actual, reference):
    return float((actual.detach().double() - reference.detach().double()).abs().max())


def time_kernel(ext, args, gate, repeats=6):
    for _ in range(2):
        ext.shadow_forward(*args, gate)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        _ = ext.shadow_forward(*args, gate)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return {"raw_ms": times, "median_ms": statistics.median(times)}


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
    camera = scene.getTrainCameras()[0]
    n = int(model.get_xyz.shape[0])
    parent = 415040
    mask_path = Path("/root/fastgs_gate_selected_4100_mask.pt")
    selected = torch.load(mask_path, map_location="cpu", weights_only=True)["selected_splits"]
    if selected.numel() != n or not selected[parent]:
        raise AssertionError("parent is not a native selected split")

    base_image, base_instances, base_geom, base_img, base_list = render_and_capture(camera, model, pipe)
    union = make_post_model(model, parent, 20260927)
    with torch.no_grad():
        union._opacity[parent].copy_(model._opacity[parent])
    _, union_instances, union_geom, union_img, union_list = render_and_capture(camera, union, pipe)
    with torch.no_grad():
        union._opacity[parent] = -30.0
    post_image, post_instances, _, _, _ = render_and_capture(camera, union, pipe)

    width, height = camera.image_width, camera.image_height
    tw, th = (width + 15) // 16, (height + 15) // 16
    tile = 3349
    b0, b1 = base_img["ranges"][tile].cpu().tolist()
    u0, u1 = union_img["ranges"][tile].cpu().tolist()
    base_ids = base_list[b0:b1].cpu().tolist()
    union_ids = union_list[u0:u1].cpu().tolist()
    child_ids = [index for index in union_ids if index >= n]
    if sorted(child_ids) != [n, n + 1]:
        raise AssertionError(child_ids)
    if [index for index in union_ids if index < n] != base_ids:
        raise AssertionError("union tile changed ordering of original splats")

    gt = camera.original_image.cuda()
    loss = 0.8 * l1_loss(base_image, gt) + 0.2 * (1 - fast_ssim(base_image.unsqueeze(0), gt.unsqueeze(0)))
    adjoint = torch.autograd.grad(loss, base_image)[0].detach()
    roles = torch.zeros(n + 2, dtype=torch.int32, device="cuda")
    roles[parent] = 1
    roles[n:] = 2
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    ranges = union_img["ranges"][:tw * th].contiguous()
    projected = (union_geom["mean2d"], union_geom["conic"], union_geom["rgb"])
    lists = (roles, ranges, union_list.contiguous())
    pixels = torch.tensor([[790, 534], [791, 534], [790, 535]], dtype=torch.int32, device="cuda")
    v_pix = torch.stack([adjoint[:, int(y), int(x)] for x, y in pixels.cpu().tolist()]).contiguous()
    ext = extension()
    args = (*projected, *lists, pixels, bg, v_pix, width, height)
    rgb0, vjp0 = ext.shadow_forward(*args, 0.0)
    rgb1, _ = ext.shadow_forward(*args, 1.0)
    native0 = torch.stack([base_image[:, int(y), int(x)] for x, y in pixels.cpu().tolist()])
    native1 = torch.stack([post_image[:, int(y), int(x)] for x, y in pixels.cpu().tolist()])

    # Independent FP64 reference using the same author-produced projected rows.
    from topology_shadow_vjp_oracle import render, shadow_vjp

    cpu_rows = []
    for index, (x, y) in enumerate(pixels.cpu().tolist()):
        layers = pixel_layers(x, y, base_ids, child_ids, parent, n, base_geom, union_geom)
        reference_rgb, accepted = render(layers, torch.tensor(0.0), torch.zeros(3))
        reference_vjp = shadow_vjp(accepted, torch.zeros(3), v_pix[index].cpu().double())
        cpu_rows.append({
            "xy": [x, y],
            "gpu_rgb_error_vs_cpu": max_error(rgb0[index].cpu(), reference_rgb),
            "gpu_vjp_error_vs_cpu": abs(float(vjp0[index]) - float(reference_vjp)),
            "gpu_vjp": float(vjp0[index]),
            "cpu_vjp": float(reference_vjp),
        })

    yy, xx = torch.meshgrid(torch.arange(height, device="cuda", dtype=torch.int32),
                            torch.arange(width, device="cuda", dtype=torch.int32), indexing="ij")
    all_pixels = torch.stack((xx, yy), dim=-1).reshape(-1, 2).contiguous()
    all_v = adjoint.permute(1, 2, 0).contiguous().reshape(-1, 3)
    full_args = (*projected, *lists, all_pixels, bg, all_v, width, height)
    full_rgb0, full_vjp0 = ext.shadow_forward(*full_args, 0.0)
    full_rgb1, _ = ext.shadow_forward(*full_args, 1.0)
    diff0 = (full_rgb0.reshape(height, width, 3) - base_image.permute(1, 2, 0)).abs()
    worst = []
    for flat in diff0.reshape(-1).topk(8).indices.cpu().tolist():
        pixel = flat // 3
        y, x = divmod(pixel, width)
        tile_id = (y // 16) * tw + (x // 16)
        b_start, b_end = base_img["ranges"][tile_id].cpu().tolist()
        ids = base_list[b_start:b_end].long()
        base_ref, cpu_last, cpu_accepted, cpu_stopped = recompose(
            x, y, base_geom["mean2d"][ids].cpu(),
            base_geom["conic"][ids].cpu(), base_geom["rgb"][ids].cpu(), torch.zeros(3))
        native_rgb = base_image[:, y, x].detach().cpu().tolist()
        gpu_rgb = full_rgb0[pixel].detach().cpu().tolist()
        worst.append({
            "xy": [x, y], "channel": flat % 3,
            "gpu_native_max_abs": max(abs(a - b) for a, b in zip(gpu_rgb, native_rgb)),
            "cpu_native_max_abs": max(abs(a - b) for a, b in zip(base_ref, native_rgb)),
            "native_last": int(base_img["n_contrib"][pixel]),
            "cpu_last": cpu_last,
            "cpu_accepted": cpu_accepted, "cpu_early_stop": cpu_stopped,
            "tile_splats": b_end - b_start,
        })
    result = {
        "source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
        "camera": camera.image_name,
        "resolution": [width, height],
        "model_ply_sha256": sha256(REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"),
        "mask_sha256": sha256(mask_path),
        "parent_id": parent,
        "children_in_tile": child_ids,
        "base_tile_instances": base_instances,
        "union_tile_instances": union_instances,
        "post_tile_instances": post_instances,
        "patch_birth_max_rgb_error_vs_native": max_error(rgb0, native0),
        "patch_release_max_rgb_error_vs_native": max_error(rgb1, native1),
        "patch_max_rgb_error_vs_cpu": max(row["gpu_rgb_error_vs_cpu"] for row in cpu_rows),
        "patch_max_vjp_error_vs_cpu": max(row["gpu_vjp_error_vs_cpu"] for row in cpu_rows),
        "patch": cpu_rows,
        "full_birth_max_rgb_error_vs_native": max_error(full_rgb0.reshape(height, width, 3), base_image.permute(1, 2, 0)),
        "full_release_max_rgb_error_vs_native": max_error(full_rgb1.reshape(height, width, 3), post_image.permute(1, 2, 0)),
        "full_birth_pixels_above_2e_5": int((diff0.amax(dim=2) > 2e-5).sum()),
        "full_birth_worst_pixels": worst,
        "full_gate_vjp_sum": float(full_vjp0.sum()),
        "full_kernel_birth_cuda_ms": time_kernel(ext, full_args, 0.0),
        "full_kernel_release_cuda_ms": time_kernel(ext, full_args, 1.0),
        "scope": "One selected native-rule split family, full-resolution real camera, isolated CUDA fixed-support gate compositor. Not per-family gradients for all event candidates, not an optimized trainer or quality result.",
    }
    if (result["patch_birth_max_rgb_error_vs_native"] > 2e-5 or
            result["patch_release_max_rgb_error_vs_native"] > 2e-5 or
            result["patch_max_vjp_error_vs_cpu"] > 1e-12):
        raise AssertionError(result)
    print("FASTGS_REAL_SHADOW_CUDA " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
