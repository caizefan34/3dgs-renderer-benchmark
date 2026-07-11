#!/usr/bin/env python
"""
3DGS Renderer Benchmark Runner
"""
import sys
import os
import time
import json
import argparse
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_HOME = os.path.dirname(os.path.dirname(PROJECT_ROOT))
sys.path.insert(0, PROJECT_ROOT)

from benchmark_framework import (
    load_ply, load_cameras_from_json, generate_cameras,
    Timer, RendererMetrics, FrameMetrics, ResultsManager
)
from renderers import get_renderer, list_renderers, list_available
from config import BenchmarkConfig


def parse_args():
    parser = argparse.ArgumentParser(description="3DGS Renderer Benchmark")
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--cameras", type=str, default=None)
    parser.add_argument("--renderers", type=str, nargs="+", default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--list-renderers", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = BenchmarkConfig.create_default()
    
    if args.renderers: cfg.renderers = args.renderers
    if args.frames: cfg.benchmark_frames = args.frames
    if args.warmup: cfg.warmup_frames = args.warmup
    if args.width: cfg.image_width = args.width
    if args.height: cfg.image_height = args.height
    if args.output: cfg.output_dir = args.output
    
    # Resolve paths
    data_dir = os.path.join(PROJECT_HOME, "data")
    scene_path = args.scene if args.scene else os.path.join(data_dir, "scene.ply")
    cameras_path = args.cameras if args.cameras else os.path.join(data_dir, "cameras.json")
    output_dir = args.output if args.output else os.path.join(PROJECT_HOME, "outputs")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if args.list_renderers:
        print("Registered:", list_renderers())
        print("Available:", list_available())
        return
    
    print("=" * 70)
    print("  3DGS Renderer Benchmark ¡ª Phase 1")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA: {torch.version.cuda}  |  PyTorch: {torch.__version__}")
    print(f"  Scene: {scene_path}")
    print(f"  Resolution: {cfg.image_width}x{cfg.image_height}")
    print(f"  Frames: {cfg.benchmark_frames} (+ {cfg.warmup_frames} warmup)")
    print("=" * 70)
    
    # 1. Load scene
    print("\n[1/4] Loading scene...")
    assert os.path.exists(scene_path), f"Scene not found: {scene_path}"
    scene_data = load_ply(scene_path, device="cuda")
    N = scene_data["num_points"]
    
    # 2. Load cameras
    print("\n[2/4] Loading camera poses...")
    if os.path.exists(cameras_path):
        cameras = load_cameras_from_json(cameras_path, device="cuda")
        print(f"  Loaded {len(cameras)} cameras from {cameras_path}")
    else:
        cameras = generate_cameras(50, cfg.image_width, cfg.image_height, device="cuda")
        print(f"  Generated {len(cameras)} cameras (no cameras.json found)")
    
    # 3. Check renderers
    print("\n[3/4] Checking renderers...")
    available = list_available()
    print(f"  Available: {available}")
    renderers = [r for r in cfg.renderers if r in available]
    if not renderers:
        print("  No renderers available!")
        sys.exit(1)
    
    # 4. Run benchmarks
    print("\n[4/4] Running benchmarks...")
    results_mgr = ResultsManager()
    
    for rname in renderers:
        print(f"\n  --- {rname} ---")
        renderer = get_renderer(rname)
        if not renderer:
            continue
        
        prep_data = renderer.prepare_scene(scene_data)
        frame_times = []
        
        # Warmup
        print(f"  Warmup ({cfg.warmup_frames})...", end=" ", flush=True)
        for f in range(cfg.warmup_frames):
            with torch.no_grad():
                renderer.render(prep_data, cameras[f % len(cameras)])
        torch.cuda.synchronize()
        print("done")
        
        # Benchmark
        print(f"  Benchmark ({cfg.benchmark_frames})...", end=" ", flush=True)
        for f in range(cfg.benchmark_frames):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                renderer.render(prep_data, cameras[f % len(cameras)])
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            frame_times.append(ms)
        
        t_arr = np.array(frame_times)
        t_sorted = np.sort(t_arr)
        trim = max(1, cfg.benchmark_frames // 10)
        t_stable = t_sorted[trim:-trim] if cfg.benchmark_frames > 2 * trim else t_arr
        
        mean_ms = float(t_stable.mean())
        median_ms = float(np.median(t_arr))
        fps = 1000.0 / mean_ms
        fps_med = 1000.0 / median_ms
        
        print("done")
        print(f"  Mean: {mean_ms:.1f}ms ({fps:.1f}FPS)  Median: {median_ms:.1f}ms ({fps_med:.1f}FPS)")
        print(f"  P10/P50/P90: {np.percentile(t_arr, 10):.1f}/{median_ms:.1f}/{np.percentile(t_arr, 90):.1f}ms")
        
        metrics = RendererMetrics(
            renderer_name=rname,
            mean_fps=round(fps, 1),
            mean_latency_ms=round(mean_ms, 2),
            median_latency_ms=round(median_ms, 2),
            min_latency_ms=round(float(t_arr.min()), 2),
            max_latency_ms=round(float(t_arr.max()), 2),
            std_latency_ms=round(float(t_arr.std()), 2),
            num_frames=cfg.benchmark_frames,
            warmup_frames=cfg.warmup_frames,
            image_width=cfg.image_width,
            image_height=cfg.image_height,
            frame_times_ms=[round(x, 2) for x in frame_times],
        )
        results_mgr.add_result(rname, metrics)
    
    # Summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    rankings = results_mgr.get_ranking()
    for i, (name, fps, lat) in enumerate(rankings, 1):
        tag = " ¡ï FASTEST" if i == 1 else ""
        print(f"  #{i}: {name:20s}  {fps:8.1f} FPS  {lat:8.2f} ms{tag}")
    
    results_mgr.export_json(os.path.join(output_dir, "benchmark_results.json"))
    print(f"\nResults saved to {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
