#!/usr/bin/env python3
"""Compare B1A and B1A_R6A rasterizer-forward/backward tensors on one camera.

Run this file twice in separate Python processes: ``--mode save`` with B1A,
then ``--mode compare`` with B1A_R6A.  Separate processes prevent a cached
``gsplat.csrc`` extension from invalidating the A/B comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


def viewmat_k(camera: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    rotation = torch.tensor(camera["rotation"], dtype=torch.float32, device="cuda")
    position = torch.tensor(camera["position"], dtype=torch.float32, device="cuda")
    viewmat = torch.eye(4, dtype=torch.float32, device="cuda")
    viewmat[:3, :3] = rotation
    viewmat[:3, 3] = -rotation @ position
    width, height = int(camera["width"]), int(camera["height"])
    k = torch.tensor(
        [[camera["fx"], 0.0, width / 2], [0.0, camera["fy"], height / 2], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device="cuda",
    )[None]
    return viewmat[None], k, width, height


def collect(gsplat_path: Path, checkpoint: Path, cameras: Path, camera_index: int, count_events: bool) -> dict[str, torch.Tensor | int]:
    sys.path.insert(0, str(gsplat_path))
    from gsplat import fully_fused_projection, rasterization
    from gsplat.cuda._wrapper import _make_lazy_cuda_func

    state = torch.load(checkpoint, map_location="cpu", weights_only=False).get("model_state")
    if state is None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    camera = json.loads(cameras.read_text())[camera_index]
    viewmats, ks, width, height = viewmat_k(camera)
    means = state["xyz"].cuda().contiguous()
    quats = state["rotations"].cuda().contiguous()
    scales = torch.exp(state["scales"].cuda()).contiguous()
    opacities = torch.sigmoid(state["opacity"].cuda().flatten()).contiguous()
    # The rasterizer is indifferent to how RGB was produced.  Using degree-0
    # coefficients gives a deterministic direct VJP test for all five outputs.
    colors = state["shs"][:, 0, :].cuda().float().contiguous()
    sh_degree = int(state.get("sh_degree", 3))
    with torch.no_grad():
        _, _, meta = rasterization(
            means, quats, scales, opacities, state["shs"].cuda().contiguous(), viewmats, ks,
            width, height, sh_degree=sh_degree, absgrad=True, tile_size=16,
            packed=False, render_mode="RGB", accutile=True,
        )
        projected = fully_fused_projection(
            means, None, quats, scales, viewmats, ks, width, height, eps2d=0.1,
            packed=False, sparse_grad=False, calc_compensations=False,
            camera_model="pinhole", opacities=opacities,
        )
        means2d, conics = projected[1].float().contiguous(), projected[3].float().contiguous()
        offsets = meta["isect_offsets"].contiguous()
        flatten_ids = meta["flatten_ids"].contiguous()
        fwd = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_fwd")
        rgb, alpha, last_ids = fwd(
            means2d, conics, colors, opacities, None, None, width, height, 16, offsets, flatten_ids
        )
        torch.manual_seed(20260919)
        v_rgb = torch.randn_like(rgb)
        v_alpha = torch.randn_like(alpha)
        bwd = _make_lazy_cuda_func("rasterize_to_pixels_3dgs_bwd")
        event_counter = block_counter = None
        if count_events:
            event_counter = _make_lazy_cuda_func("r6a_get_global_writer_events")
            block_counter = _make_lazy_cuda_func("r6a_get_block_reduction_events")
            _make_lazy_cuda_func("r6a_reset_global_writer_events")()
        v_abs, v_means, v_conics, v_colors, v_opacity = bwd(
            means2d, conics, colors, opacities, None, None, width, height, 16,
            offsets, flatten_ids, alpha, last_ids, v_rgb, v_alpha, True,
        )
    torch.cuda.synchronize()
    result = {
        "render_rgb": rgb.cpu(), "render_alpha": alpha.cpu(),
        "intersections": int(flatten_ids.numel()),
        "v_colors": v_colors.cpu(), "v_conics": v_conics.cpu(),
        "v_means2d": v_means.cpu(), "v_means2d_abs": v_abs.cpu(),
        "v_opacity": v_opacity.cpu(),
    }
    if event_counter is not None:
        result["global_writer_events"] = int(event_counter())
        result["block_local_reduction_events"] = int(block_counter())
    return result


def stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    a, b = reference.float(), candidate.float()
    diff = (a - b).abs()
    denom = torch.linalg.vector_norm(a).item()
    norm_b = torch.linalg.vector_norm(b).item()
    cosine = max(-1.0, min(1.0, torch.sum(a * b).item() / max(denom * norm_b, 1e-30)))
    return {
        "shape_identical": list(reference.shape) == list(candidate.shape),
        "dtype_identical": str(reference.dtype) == str(candidate.dtype),
        "nan": int(torch.isnan(candidate).sum().item()),
        "inf": int(torch.isinf(candidate).sum().item()),
        "max_abs": float(diff.max().item()), "mean_abs": float(diff.mean().item()),
        "relative_l2": float(torch.linalg.vector_norm(a - b).item() / max(denom, 1e-30)),
        "cosine_similarity": float(cosine),
        "nonzero_mismatch_count": int((diff != 0).sum().item()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("save", "compare"), required=True)
    p.add_argument("--gsplat-path", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cameras", type=Path, required=True)
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count-events", action="store_true")
    args = p.parse_args()
    current = collect(args.gsplat_path, args.checkpoint, args.cameras, args.camera_index, args.count_events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "save":
        torch.save(current, args.reference)
        args.output.write_text(json.dumps({"saved": str(args.reference), "intersections": current["intersections"]}, indent=2))
        return
    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    result = {"intersections_identical": reference["intersections"] == current["intersections"], "tensors": {}}
    if "global_writer_events" in current:
        result["global_writer_events"] = current["global_writer_events"]
        result["global_atomic_add_calls_absgrad"] = current["global_writer_events"] * 11
        result["block_local_reduction_events"] = current["block_local_reduction_events"]
    for name in ("render_rgb", "render_alpha", "v_colors", "v_conics", "v_means2d", "v_means2d_abs", "v_opacity"):
        result["tensors"][name] = stats(reference[name], current[name])
    result["gradient_pass_relative_l2_le_1e-4"] = all(
        result["tensors"][k]["relative_l2"] <= 1e-4
        for k in ("v_colors", "v_conics", "v_means2d", "v_means2d_abs", "v_opacity")
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
