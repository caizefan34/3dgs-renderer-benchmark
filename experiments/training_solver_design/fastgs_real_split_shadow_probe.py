"""One native FastGS split family's fixed-support shadow reference on a real tile.

Only diagnostics are created. The author renderer/source and checkpoint stay
unchanged. Child geometry follows the author's stochastic two-child rule.
"""

import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


REPO = Path("/root/codex_fastgs_audit_20260927")
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/root")

from diff_gaussian_rasterization_fastgs import _C  # noqa: E402
from fastgs_real_tile_reference import decode_geometry, decode_image, take, sha256  # noqa: E402
from fused_ssim import fused_ssim as fast_ssim  # noqa: E402
from gaussian_renderer import render_fastgs  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.general_utils import build_rotation  # noqa: E402
from utils.loss_utils import l1_loss  # noqa: E402


def render_and_capture(camera, model, pipe):
    captured = []
    original = _C.rasterize_gaussians

    def capture(*args):
        result = original(*args)
        captured.append(result)
        return result

    _C.rasterize_gaussians = capture
    try:
        image = render_fastgs(camera, model, pipe, torch.zeros(3, device="cuda"), 0.5)["render"]
        torch.cuda.synchronize()
    finally:
        _C.rasterize_gaussians = original
    if len(captured) != 1:
        raise AssertionError(len(captured))
    rendered, _, _, _, geom_buffer, bin_buffer, img_buffer, _, _ = captured[0]
    n = int(model.get_xyz.shape[0])
    geom = decode_geometry(geom_buffer, n)
    img = decode_image(img_buffer, camera.image_width * camera.image_height)
    ids, _ = take(bin_buffer, 0, rendered, torch.int32)
    return image, rendered, geom, img, ids


def raw_at(x, y, mean, conic):
    dx = float(mean[0]) - x
    dy = float(mean[1]) - y
    power = -0.5 * (float(conic[0]) * dx * dx + float(conic[2]) * dy * dy) - float(conic[1]) * dx * dy
    return 0.0 if power > 0 else min(0.99, float(conic[3]) * math.exp(power))


def choose_parent(width, height, geom, img, ids, selected):
    tw, th = (width + 15) // 16, (height + 15) // 16
    ranges = img["ranges"][:tw * th].cpu()
    tile_len = ranges[:, 1] - ranges[:, 0]
    selected_cuda = selected.to(ids.device)
    hit = selected_cuda[ids.long()].to(torch.int64)
    prefix = torch.cat((torch.zeros(1, device=ids.device, dtype=torch.int64), hit.cumsum(0)))
    candidate_count = (prefix[img["ranges"][:tw * th, 1].long()] -
                       prefix[img["ranges"][:tw * th, 0].long()]).cpu()
    tiles = [t for t in range(tw * th) if candidate_count[t] and 20 <= tile_len[t] <= 180]
    tiles.sort(key=lambda t: ((t % tw) - tw / 2) ** 2 + ((t // tw) - th / 2) ** 2)
    for tile in tiles:
        begin, end = map(int, ranges[tile].tolist())
        tile_ids = ids[begin:end].cpu().tolist()
        candidates = [(offset, gauss) for offset, gauss in enumerate(tile_ids) if selected[gauss]]
        for y in (tile // tw * 16 + 7, tile // tw * 16 + 8):
            for x in (tile % tw * 16 + 7, tile % tw * 16 + 8):
                if x >= width or y >= height:
                    continue
                last = int(img["n_contrib"][y * width + x].item())
                for offset, gauss in candidates:
                    if offset >= last:
                        continue
                    alpha = raw_at(x, y, geom["mean2d"][gauss].cpu(), geom["conic"][gauss].cpu())
                    if alpha >= 0.05:
                        return tile, gauss, [x, y], len(tile_ids), alpha, len(tiles)
    raise AssertionError("no selected split parent contributed in eligible tiles")


def make_post_model(model, parent, seed):
    torch.manual_seed(seed)
    stds = model.get_scaling[parent:parent + 1].repeat(2, 1)
    samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
    rots = build_rotation(model._rotation[parent:parent + 1]).repeat(2, 1, 1)
    child_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + model.get_xyz[parent:parent + 1].repeat(2, 1)
    child_scale = model.scaling_inverse_activation(stds / 1.6)
    post = GaussianModel(model.max_sh_degree, "default")
    post.active_sh_degree = model.active_sh_degree
    opacity = model._opacity.detach().clone()
    opacity[parent] = -30.0  # Native renderer removes the parent at this endpoint.
    extensions = {
        "_xyz": child_xyz,
        "_features_dc": model._features_dc[parent:parent + 1].repeat(2, 1, 1),
        "_features_rest": model._features_rest[parent:parent + 1].repeat(2, 1, 1),
        "_scaling": child_scale,
        "_rotation": model._rotation[parent:parent + 1].repeat(2, 1),
        "_opacity": model._opacity[parent:parent + 1].repeat(2, 1),
    }
    for name, new in extensions.items():
        old = opacity if name == "_opacity" else getattr(model, name).detach()
        setattr(post, name, nn.Parameter(torch.cat((old, new.detach()), dim=0)))
    return post


def pixel_layers(x, y, base_ids, child_ids, parent, n, base_geom, post_geom):
    layers = []
    for gauss in base_ids + child_ids:
        geom = base_geom if gauss < n else post_geom
        mean = geom["mean2d"][gauss].cpu()
        conic = geom["conic"][gauss].cpu()
        raw = raw_at(x, y, mean, conic)
        if raw < 1 / 255:
            continue
        role = "parent" if gauss == parent else "child" if gauss >= n else "other"
        layers.append((float(geom["depths"][gauss].item()), role,
                       torch.tensor(raw, dtype=torch.float64), geom["rgb"][gauss].cpu().double()))
    return layers


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
    mask_info = torch.load("/root/fastgs_gate_selected_4100_mask.pt", map_location="cpu", weights_only=True)
    selected = mask_info["selected_splits"]
    if selected.numel() != n or selected.sum().item() != 9345:
        raise AssertionError("saved native split selection mismatches loaded model")
    base_image, base_instances, base_geom, base_img, base_list = render_and_capture(camera, model, pipe)
    width, height = camera.image_width, camera.image_height
    tile, parent, site, tile_len, parent_alpha, tiles_considered = choose_parent(
        width, height, base_geom, base_img, base_list, selected)
    tw = (width + 15) // 16
    begin, end = base_img["ranges"][tile].cpu().tolist()
    base_ids = base_list[begin:end].cpu().tolist()

    post_result = None
    for attempt in range(10):
        post_model = make_post_model(model, parent, 20260927 + attempt)
        post_result = render_and_capture(camera, post_model, pipe)
        post_image, post_instances, post_geom, post_img, post_list = post_result
        post_begin, post_end = post_img["ranges"][tile].cpu().tolist()
        post_ids = post_list[post_begin:post_end].cpu().tolist()
        child_ids = [gauss for gauss in post_ids if gauss >= n]
        if child_ids:
            break
    if not child_ids:
        raise AssertionError("native children never touched the selected parent tile")
    base_other = [gauss for gauss in base_ids if gauss != parent]
    post_other = [gauss for gauss in post_ids if gauss < n and gauss != parent]
    if base_other != post_other:
        raise AssertionError("other-splat tile order changed between endpoint renders")

    # Import only after the native renders; this scalar oracle selects FP64 by default.
    from topology_shadow_vjp_oracle import render, shadow_vjp

    gt = camera.original_image.cuda()
    loss = 0.8 * l1_loss(base_image, gt) + 0.2 * (1 - fast_ssim(base_image.unsqueeze(0), gt.unsqueeze(0)))
    pixel_adjoint = torch.autograd.grad(loss, base_image)[0].detach().cpu().double()
    best = None
    for y in range(tile // tw * 16, min((tile // tw + 1) * 16, height)):
        for x in range(tile % tw * 16, min((tile % tw + 1) * 16, width)):
            difference = (base_image[:, y, x] - post_image[:, y, x]).abs().max().item()
            if best is None or difference > best[0]:
                best = (difference, x, y)
    _, x0, y0 = best
    pixels = [(x0, y0), (min(x0 + 1, tile % tw * 16 + 15), y0),
              (x0, min(y0 + 1, tile // tw * 16 + 15))]
    records = []
    for x, y in pixels:
        layers = pixel_layers(x, y, base_ids, child_ids, parent, n, base_geom, post_geom)
        gate = torch.tensor(0.0, requires_grad=True)
        rgb0, accepted = render(layers, gate, torch.zeros(3))
        rgb1, _ = render(layers, torch.ones(()), torch.zeros(3))
        adjoint = pixel_adjoint[:, y, x]
        analytic = shadow_vjp(accepted, torch.zeros(3), adjoint)
        if rgb0.requires_grad:
            automatic = torch.autograd.grad((rgb0 * adjoint).sum(), gate, allow_unused=True)[0]
            if automatic is None:
                automatic = torch.zeros(())
        else:
            automatic = torch.zeros(())
        records.append({
            "xy": [x, y],
            "birth_max_rgb_error": float((rgb0 - base_image[:, y, x].detach().cpu().double()).abs().max()),
            "release_max_rgb_error": float((rgb1 - post_image[:, y, x].detach().cpu().double()).abs().max()),
            "shadow_vjp": float(analytic),
            "independent_autograd_vjp": float(automatic),
            "vjp_abs_error": float((analytic - automatic).abs()),
            "children_with_pixel_support": sum(row[1] == "child" for row in layers),
            "children_before_early_stop": sum(row[0] == "child" for row in accepted),
            "active_rows": len(accepted),
        })
    result = {
        "camera": camera.image_name,
        "resolution": [width, height],
        "pre_event_model_ply_sha256": sha256(REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"),
        "selected_mask_sha256": sha256(Path("/root/fastgs_gate_selected_4100_mask.pt")),
        "native_split_parent_id": parent,
        "selected_split_count": int(selected.sum()),
        "parent_ungated_alpha_at_selection_pixel": parent_alpha,
        "parent_tile": tile,
        "parent_tile_splats": tile_len,
        "selected_tiles_considered": tiles_considered,
        "child_sampling_attempts": attempt + 1,
        "children_in_tile": child_ids,
        "base_tile_instances": base_instances,
        "post_tile_instances": post_instances,
        "parent_touched_tiles": int(base_geom["tiles"][parent]),
        "children_touched_tiles": [int(post_geom["tiles"][gauss]) for gauss in (n, n + 1)],
        "native_full_view_loss": float(loss.detach()),
        "pixels": records,
        "max_birth_rgb_error": max(row["birth_max_rgb_error"] for row in records),
        "max_release_rgb_error": max(row["release_max_rgb_error"] for row in records),
        "max_vjp_abs_error": max(row["vjp_abs_error"] for row in records),
        "scope": "One real camera/tile and native-style stochastic two-child edit; fixed-support CPU reference using CUDA-produced projected rows; no CUDA gate kernel or full-training result.",
    }
    if result["max_birth_rgb_error"] > 1e-4 or result["max_release_rgb_error"] > 1e-4 or result["max_vjp_abs_error"] > 1e-12:
        raise AssertionError(result)
    print("FASTGS_REAL_SPLIT_SHADOW " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
