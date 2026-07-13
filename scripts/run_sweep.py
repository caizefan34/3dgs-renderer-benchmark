#!/usr/bin/env python
"""
Parameter sweep runner for benchmark matrix experiments.
"""
import argparse
import os
import sys
from dataclasses import replace

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

from benchmark_framework import (  # noqa: E402
    BenchmarkConfig,
    export_aggregate_csv,
    load_cameras_from_json,
    load_ply,
    run_renderer_benchmark,
    set_global_seed,
)
from benchmark_framework.cameras import Camera  # noqa: E402
from renderers import get_renderer, list_available  # noqa: E402


def parse_csv_ints(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_resolutions(text: str):
    out = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        w, h = part.split("x")
        out.append((int(w), int(h)))
    return out


def make_resolution_cameras(cameras, width, height):
    return [
        replace(cam, image_width=width, image_height=height)
        if isinstance(cam, Camera)
        else cam
        for cam in cameras
    ]


def subset_scene(scene_data, gs_count):
    if gs_count <= 0 or gs_count >= scene_data["num_points"]:
        return scene_data
    subset = {}
    for k, v in scene_data.items():
        if torch.is_tensor(v) and v.shape and v.shape[0] == scene_data["num_points"]:
            subset[k] = v[:gs_count]
        else:
            subset[k] = v
    subset["num_points"] = gs_count
    return subset


def parse_args():
    p = argparse.ArgumentParser(description="Run benchmark parameter sweep")
    p.add_argument("--scene", type=str, default=os.path.join(REPO_ROOT, "data", "scene.ply"))
    p.add_argument("--cameras", type=str, default=os.path.join(REPO_ROOT, "data", "cameras.json"))
    p.add_argument("--renderers", nargs="+", default=None, help="Renderers to include")
    p.add_argument("--gs-counts", type=str, default="100000,400000", help="Comma-separated gaussian counts")
    p.add_argument("--resolutions", type=str, default="1280x720,1920x1080", help="Comma-separated WxH list")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=os.path.join(REPO_ROOT, "results"))
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--amp-dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"])
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    raw_dir = os.path.join(args.output, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    cfg = BenchmarkConfig.create_default()
    cfg.warmup_frames = args.warmup
    cfg.benchmark_frames = args.frames
    cfg.repeats = args.repeats
    cfg.seed = args.seed
    set_global_seed(cfg.seed)

    scene_data = load_ply(args.scene, device="cuda")
    base_cameras = load_cameras_from_json(args.cameras, device="cuda")
    renderers = args.renderers or list_available()
    gs_counts = parse_csv_ints(args.gs_counts)
    resolutions = parse_resolutions(args.resolutions)

    summary_rows = []
    for renderer_name in renderers:
        renderer = get_renderer(renderer_name)
        if not renderer:
            continue
        for gs_count in gs_counts:
            scene_subset = subset_scene(scene_data, gs_count)
            prep_data = renderer.prepare_scene(scene_subset)
            for width, height in resolutions:
                cameras = make_resolution_cameras(base_cameras, width, height)
                result = run_renderer_benchmark(
                    renderer_name=f"{renderer_name}_g{gs_count}_{width}x{height}",
                    renderer=renderer,
                    prep_data=prep_data,
                    cameras=cameras,
                    warmup_iters=cfg.warmup_frames,
                    measured_iters=cfg.benchmark_frames,
                    repeats=cfg.repeats,
                    raw_output_dir=raw_dir,
                    seed=cfg.seed,
                    use_mixed_precision=args.mixed_precision,
                    amp_dtype=args.amp_dtype,
                    use_compile=args.compile,
                    compile_mode=args.compile_mode,
                )
                agg = result["aggregate"]
                summary_rows.append({
                    "renderer": renderer_name,
                    "gs_count": gs_count,
                    "resolution": f"{width}x{height}",
                    "warmup_iters": cfg.warmup_frames,
                    "measured_iters": cfg.benchmark_frames,
                    "repeats": cfg.repeats,
                    "seed": cfg.seed,
                    "mean_fps": round(agg["mean_fps"], 4),
                    "median_fps": round(agg["median_fps"], 4),
                    "std_fps": round(agg["std_fps"], 4),
                    "p95_latency_ms": round(agg["p95_latency_ms"], 4),
                    "peak_vram_mb": round(agg["peak_vram_mb"], 2),
                })

    export_aggregate_csv(summary_rows, os.path.join(args.output, "summary.csv"))
    print(f"Sweep complete. Results in {args.output}")


if __name__ == "__main__":
    main()
