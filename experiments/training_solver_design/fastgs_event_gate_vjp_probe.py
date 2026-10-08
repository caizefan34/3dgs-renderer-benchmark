"""One real FastGS split event: fixed-support image parity and all-family gate VJP.

The author checkout is unchanged. This is a diagnostic pass, not a trainer.
"""

import json
import math
import os
import statistics
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

from fastgs_event_split_cost_probe import make_native_post, paired_timings  # noqa: E402
from fastgs_real_split_shadow_probe import render_and_capture  # noqa: E402
from fastgs_real_tile_reference import sha256  # noqa: E402
from fused_ssim import fused_ssim as fast_ssim  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.loss_utils import l1_loss  # noqa: E402


def extension():
    return load(
        name="fastgs_event_gate_vjp_diagnostic_v1",
        sources=["/root/fastgs_event_gate_vjp_cuda.cu"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=True,
    )


def timed_cuda(fn, repeats=8):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {"median_ms": statistics.median(samples), "raw_ms": samples}


def cpu_pixel_family_vjp(x, y, ids, geom, roles, families, target_family, adjoint):
    """Independent FP64 autograd on author-projected, exact tile-ordered rows."""
    gate = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    rgb = torch.zeros(3, dtype=torch.float64)
    trans = torch.ones((), dtype=torch.float64)
    accepted = 0
    for id in ids:
        mean = geom["mean2d"][id]
        conic = geom["conic"][id]
        dx, dy = float(mean[0]) - x, float(mean[1]) - y
        power = -0.5 * (float(conic[0]) * dx * dx + float(conic[2]) * dy * dy) - float(conic[1]) * dx * dy
        if power > 0:
            continue
        raw_value = min(0.99, float(conic[3]) * math.exp(power))
        if raw_value < 1 / 255:
            continue
        raw = torch.tensor(raw_value, dtype=torch.float64)
        role = int(roles[id])
        family = int(families[id])
        if role == 1 and family == target_family:
            alpha = -torch.expm1((1 - gate) * torch.log1p(-raw))
        elif role == 2 and family == target_family:
            alpha = -torch.expm1(gate * torch.log1p(-raw))
        elif role == 2:
            alpha = torch.zeros((), dtype=torch.float64)
        else:
            alpha = raw
        next_trans = trans * (1 - alpha)
        if float(next_trans.detach()) < 1e-4:
            break
        color = geom["rgb"][id].double()
        rgb = rgb + trans * alpha * color
        trans = next_trans
        accepted += 1
    derivative = torch.autograd.grad((rgb * adjoint.double()).sum(), gate, allow_unused=True)[0]
    return {"vjp": float(derivative if derivative is not None else 0.0),
            "accepted": accepted, "rgb": rgb.detach().tolist()}


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
    selected = torch.load("/root/fastgs_gate_selected_4100_mask.pt", map_location="cpu", weights_only=True)["selected_splits"]
    if selected.numel() != n or int(selected.sum()) != 9345:
        raise AssertionError("saved native split selection mismatch")
    base_image, base_instances, _, _, _ = render_and_capture(camera, model, pipe)
    torch.cuda.synchronize()
    construction_start = time.perf_counter()
    post, selected_ids, family_count = make_native_post(model, selected)
    torch.cuda.synchronize()
    construction_ms = (time.perf_counter() - construction_start) * 1000
    post_image, post_instances, post_geom, _, _ = render_and_capture(camera, post, pipe)
    gt = camera.original_image.cuda()
    native_post_train_cuda_ms = paired_timings(camera, model, post, pipe, gt, backward=True)
    with torch.no_grad():
        post._opacity[selected_ids] = model._opacity[selected_ids]
    torch.cuda.synchronize()
    union_start = time.perf_counter()
    _, union_instances, geom, img, point_list = render_and_capture(camera, post, pipe)
    torch.cuda.synchronize()
    union_render_wall_ms = (time.perf_counter() - union_start) * 1000
    union_train_cuda_ms = paired_timings(camera, model, post, pipe, gt, backward=True)
    child_tile_instances = int(post_geom["tiles"][n:].sum())
    if union_instances != base_instances + child_tile_instances:
        raise AssertionError((union_instances, base_instances, child_tile_instances))

    roles = torch.zeros(n + 2 * family_count, dtype=torch.int32, device="cuda")
    families = torch.full((n + 2 * family_count,), -1, dtype=torch.int32, device="cuda")
    family_ids = torch.arange(family_count, dtype=torch.int32, device="cuda")
    roles[selected_ids] = 1
    families[selected_ids] = family_ids
    roles[n:] = 2
    families[n:] = family_ids.repeat(2)
    debug_parent = 415040
    debug_family = int((selected_ids == debug_parent).nonzero().flatten().item())
    width, height = camera.image_width, camera.image_height
    tw, th = (width + 15) // 16, (height + 15) // 16
    ranges = img["ranges"][:tw * th].contiguous()
    event_hits = (roles[point_list.long()] != 0).to(torch.int32)
    event_prefix = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int64),
                              event_hits.cumsum(0)))
    affected_tiles = ((event_prefix[ranges[:, 1].long()] -
                       event_prefix[ranges[:, 0].long()]) > 0).nonzero().flatten()
    tile_x, tile_y = affected_tiles % tw, affected_tiles // tw
    affected_pixel_slots = int((torch.clamp(width - 16 * tile_x, max=16) *
                                torch.clamp(height - 16 * tile_y, max=16)).sum())
    loss = 0.8 * l1_loss(base_image, gt) + 0.2 * (1 - fast_ssim(base_image.unsqueeze(0), gt.unsqueeze(0)))
    adjoint = torch.autograd.grad(loss, base_image)[0].detach().permute(1, 2, 0).contiguous()
    ext = extension()
    args = (geom["mean2d"], geom["conic"], geom["rgb"],
            roles, families, ranges, point_list.contiguous(),
            adjoint, width, height, family_count, debug_family)
    rgb0, grad0, debug0 = ext.event_gate_forward(*args, 0.0)
    rgb1, grad1, _ = ext.event_gate_forward(*args, 1.0)
    torch.cuda.synchronize()
    base_rgb = base_image.detach().permute(1, 2, 0)
    post_rgb = post_image.detach().permute(1, 2, 0)
    birth_error = float((rgb0 - base_rgb).abs().max())
    release_error = float((rgb1 - post_rgb).abs().max())
    birth_bad = int(((rgb0 - base_rgb).abs().amax(dim=2) > 2e-5).sum())
    release_bad = int(((rgb1 - post_rgb).abs().amax(dim=2) > 2e-5).sum())

    geom_cpu = {key: geom[key].detach().cpu() for key in ("mean2d", "conic", "rgb")}
    roles_cpu, families_cpu = roles.cpu(), families.cpu()
    tile = 3349
    begin, end = img["ranges"][tile].cpu().tolist()
    ids = point_list[begin:end].cpu().tolist()
    debug_rows = []
    for x, y in ((790, 534), (791, 534), (790, 535)):
        ref = cpu_pixel_family_vjp(x, y, ids, geom_cpu, roles_cpu, families_cpu,
                                   debug_family, adjoint[y, x].cpu())
        debug_rows.append({"xy": [x, y], "gpu_vjp": float(debug0[y, x]),
                           "fp64_vjp": ref["vjp"],
                           "abs_error": abs(float(debug0[y, x]) - ref["vjp"]),
                           "gpu_rgb_fp64_error": max(abs(a - b) for a, b in zip(rgb0[y, x].cpu().tolist(), ref["rgb"])),
                           "accepted_rows": ref["accepted"]})

    def run_gate_zero():
        return ext.event_gate_forward(*args, 0.0)

    def run_gate_one():
        return ext.event_gate_forward(*args, 1.0)

    timing_zero = timed_cuda(run_gate_zero)
    timing_one = timed_cuda(run_gate_one)
    result = {
        "source_commit": "44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f",
        "camera": camera.image_name, "resolution": [width, height],
        "ply_sha256": sha256(REPO / "output/room_big_capture4100/point_cloud/iteration_4100/point_cloud.ply"),
        "mask_sha256": sha256(Path("/root/fastgs_gate_selected_4100_mask.pt")),
        "gaussians_before": n, "families": family_count,
        "base_tile_instances": base_instances, "post_tile_instances": post_instances,
        "union_tile_instances": union_instances,
        "child_tile_instances": child_tile_instances,
        "affected_tiles": int(affected_tiles.numel()),
        "total_tiles": tw * th,
        "affected_tile_pixel_slots": affected_pixel_slots,
        "birth_max_rgb_error": birth_error, "release_max_rgb_error": release_error,
        "birth_pixels_above_2e_5": birth_bad, "release_pixels_above_2e_5": release_bad,
        "debug_parent": debug_parent, "debug_family": debug_family,
        "debug_rows": debug_rows,
        "debug_family_grad_atomic": float(grad0[debug_family]),
        "debug_family_grad_pixel_sum": float(debug0.sum()),
        "family_vjp_finite": bool(torch.isfinite(grad0).all()),
        "families_nonzero_vjp": int((grad0 != 0).sum()),
        "families_negative_vjp": int((grad0 < 0).sum()),
        "families_positive_vjp": int((grad0 > 0).sum()),
        "family_vjp_abs_quantiles": torch.quantile(grad0.abs(), torch.tensor([0, .5, .9, .99, 1.], device="cuda")).cpu().tolist(),
        "construction_wall_ms": construction_ms,
        "union_render_and_capture_wall_ms": union_render_wall_ms,
        "native_post_train_cuda_ms": native_post_train_cuda_ms,
        "union_raw_opacity_train_cuda_ms": union_train_cuda_ms,
        "event_gate_zero_cuda_ms": timing_zero,
        "event_gate_one_cuda_ms": timing_one,
        "scope": "One full-resolution real camera and one native-selected 9345-family split event. Diagnostic fixed-support forward and all-family image-loss VJP only; not a trainer, not full wall savings, not reconstruction-quality evidence.",
    }
    if birth_error > 2e-5 or release_error > 2e-5 or birth_bad or release_bad:
        raise AssertionError("endpoint mismatch: " + json.dumps(result, sort_keys=True))
    if not result["family_vjp_finite"] or max(row["abs_error"] for row in debug_rows) > 1e-12:
        raise AssertionError("VJP mismatch: " + json.dumps(result, sort_keys=True))
    print("FASTGS_EVENT_GATE_VJP " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
