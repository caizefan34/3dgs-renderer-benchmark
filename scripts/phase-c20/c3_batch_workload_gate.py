#!/usr/bin/env python3
"""C20/C3 fixed-CTA workload characterization and batch-size implementation gate.

Stock gsplat does not expose the rasterizer shared-memory batch size. This
script measures valid workload sensitivity with tile_size/CTA fixed at 16 and
records that C3 cannot claim configuration or mechanism gain without a custom
CUDA implementation exposing an independent batch parameter.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import gsplat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras


def time_cuda(fn, repeats: int) -> dict:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        values.append(float(start.elapsed_time(end)))
    a = np.asarray(values)
    return {
        "samples_ms": values,
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "std_ms": float(a.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--caps", type=int, nargs="+", default=[96, 128, 160, 192, 224, 256, 320])
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    width, height, tile_size = 1920, 1080, 16
    scene = load_ply(str(ROOT / "data/official/mipnerf360/room/point_cloud.ply"), device="cuda")
    camera = resize_cameras(
        load_cameras_from_json(str(ROOT / "data/official/mipnerf360/room/cameras.json"), device="cuda"),
        width,
        height,
    )[0]
    means = scene["xyz"].contiguous()
    quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
    scales = scene["scales"].exp().contiguous()
    opacities = torch.sigmoid(scene["opacity"]).contiguous()
    shs = scene["shs"].contiguous()
    backgrounds = torch.zeros(1, 3, device="cuda")
    kwargs = dict(
        means=means, quats=quats, scales=scales, opacities=opacities, colors=shs,
        viewmats=camera.world_view_transform.unsqueeze(0).contiguous(),
        Ks=camera.K.unsqueeze(0).contiguous(), width=width, height=height,
        near_plane=0.01, far_plane=1e10, radius_clip=0.0, eps2d=0.3,
        sh_degree=3, packed=False, tile_size=tile_size, backgrounds=backgrounds,
        render_mode="RGB", sparse_grad=False, absgrad=False, rasterize_mode="classic",
    )
    _, _, meta = gsplat.rasterization(**kwargs)
    torch.cuda.synchronize()
    means2d = meta["means2d"].contiguous()
    conics = meta["conics"].contiguous()
    radii = meta["radii"].contiguous()
    depths = meta["depths"].contiguous()
    opacities2d = meta["opacities"].contiguous()
    dirs = camera.camera_center.to("cuda") - means
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    colors = gsplat.spherical_harmonics(3, dirs, shs).unsqueeze(0)
    tile_width, tile_height = math.ceil(width / tile_size), math.ceil(height / tile_size)
    _, isect_ids, flatten_ids = gsplat.isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height, sort=True)
    offsets = gsplat.isect_offset_encode(isect_ids, 1, tile_width, tile_height)
    original_count = int(flatten_ids.numel())
    flat_offsets = offsets[0].reshape(-1).cpu().tolist()

    records = []
    for cap in args.caps:
        chunks, starts = [], [0]
        for index, low in enumerate(flat_offsets):
            high = flat_offsets[index + 1] if index + 1 < len(flat_offsets) else original_count
            chosen = flatten_ids[low: low + min(high - low, cap)]
            chunks.append(chosen)
            starts.append(starts[-1] + int(chosen.numel()))
        replay_ids = torch.cat(chunks)
        replay_offsets = torch.tensor(starts[:-1], dtype=torch.int32, device="cuda").reshape(1, tile_height, tile_width)
        fn = lambda: gsplat.rasterize_to_pixels(means2d, conics, colors, opacities2d, width, height, tile_size, replay_offsets, replay_ids, backgrounds)
        for _ in range(4):
            fn()
        timing = time_cuda(fn, args.repeats)
        records.append({
            "per_tile_intersection_cap": cap,
            "total_replayed_intersections": int(replay_ids.numel()),
            "mean_replayed_intersections_per_tile": float(replay_ids.numel() / (tile_width * tile_height)),
            "fixed_tile_size": tile_size,
            "fixed_cta": "16x16x1",
            "fixed_threads_per_block": 256,
            "rasterizer_timing": timing,
            "ns_per_replayed_intersection": float(timing["mean_ms"] * 1e6 / max(int(replay_ids.numel()), 1)),
            "correctness_status": "not_a_correctness_candidate: truncation intentionally changes rendered intersections",
        })

    output = {
        "schema_version": 1,
        "candidate": "C3",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "gsplat": gsplat.__version__},
        "protocol": {
            "scene": "room official Mip-NeRF360",
            "resolution": "1920x1080",
            "fixed_tile_size": 16,
            "fixed_cta": "16x16x1",
            "canonical_intersections": original_count,
            "repeats": args.repeats,
        },
        "parameter_audit": {
            "batch_size_exposed_by_stock_gsplat_api": False,
            "compiled_cuda_kernel_modified": False,
            "configuration_gain": "not measurable: no independent batch configuration changed",
            "mechanism_gain": "not established",
            "system_level_gain": "not applicable",
        },
        "workload_characterization": records,
        "failure_boundary": "C3 cannot be accepted or tuned through stock gsplat. A separately compiled CUDA variant with an independently parameterized shared-memory batch is required before claiming a batch-size mechanism or speedup.",
        "verdict": "DROP",
        "evidence_strength": "strong for the implementation/API boundary; descriptive only for replay workload sensitivity",
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
