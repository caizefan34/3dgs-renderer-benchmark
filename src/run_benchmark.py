#!/usr/bin/env python
"""
3DGS Renderer Benchmark — Unified Command-Line Interface.

Implements the standardized benchmark protocol: load scene data, evaluate
all specified renderers on identical camera paths, collect comprehensive
metrics (FPS, latency distribution, VRAM consumption, scene loading time),
and export results in multiple formats (JSON, CSV, Markdown, HTML).

Usage:
    python run_benchmark.py --scene data/scene.ply --cameras data/cameras.json
    python run_benchmark.py --renderers speedy_splat diff_gaussian --frames 100
    python run_benchmark.py --list-renderers

References:
    Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023).
    3D Gaussian Splatting for Real-Time Radiance Field Rendering.
    ACM Transactions on Graphics, 42(4).
"""
import sys, os, time, json, argparse, gc, subprocess
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from benchmark_framework import (
    load_ply, load_cameras_from_json, generate_cameras,
    validate_cameras_facing_point, RendererMetrics, ResultsManager, BenchmarkConfig
)
from renderers import get_renderer, list_renderers, list_available
from benchmark.difficulty import (
    DifficultyConfig,
    DifficultyInputs,
    calculate_difficulty,
)
from benchmark_suite import BENCHMARK_SUITE_VERSION


def _uniform_camera_resolution(cameras):
    resolutions = {(camera.image_width, camera.image_height) for camera in cameras}
    if len(resolutions) != 1:
        raise ValueError(f"Benchmark requires one camera resolution, got {sorted(resolutions)}")
    return next(iter(resolutions))


def _git_commit_hash(repo_root):
    try:
        return subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _cuda_driver_version():
    try:
        raw = torch._C._cuda_getDriverVersion()
    except Exception:
        return None
    return str(raw) if raw else None


def _hardware_metadata(device_index=0):
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    props = torch.cuda.get_device_properties(device_index)
    return {
        "cuda_available": True,
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_vram_mb": round(props.total_memory / (1024 * 1024), 1),
        "multi_processor_count": props.multi_processor_count,
    }


def parse_args():
    p = argparse.ArgumentParser(description="3DGS Renderer Benchmark")
    p.add_argument("--scene", type=str, default=None, help="Path to .ply scene file")
    p.add_argument("--cameras", type=str, default=None, help="Path to cameras.json")
    p.add_argument("--renderers", type=str, nargs="+", default=None, help="Renderers to benchmark")
    p.add_argument("--frames", type=int, default=None, help="Number of benchmark frames")
    p.add_argument("--warmup", type=int, default=None, help="Number of warmup frames")
    p.add_argument("--repeats", type=int, default=None, help="Number of measurement repeats")
    p.add_argument("--clock-lock", action="store_true", default=None, help="Enable GPU clock lock")
    p.add_argument("--width", type=int, default=None, help="Image width in pixels")
    p.add_argument("--height", type=int, default=None, help="Image height in pixels")
    p.add_argument("--output", type=str, default=None, help="Output directory for results")
    p.add_argument("--camera-path", type=str, default=None, choices=["spiral", "circle", "flythrough", "random_walk"],
                   help="Standard camera path preset (from data/camera_presets/)")
    p.add_argument("--list-renderers", action="store_true", help="List registered renderers and exit")
    p.add_argument("--allow-backfacing-cameras", action="store_true",
                   help="Allow camera paths whose scene center has z <= 0")
    p.add_argument(
        "--benchmark-type",
        choices=["synthetic_stress", "real_scene_speed"],
        default=None,
        help="Result taxonomy (default: synthetic_stress)",
    )
    p.add_argument(
        "--difficulty-metrics",
        default=None,
        help="JSON containing measured visibility, overlap, tile density, and depth complexity",
    )
    return p.parse_args()


def _load_difficulty(path):
    """Load explicitly measured scene factors without inventing missing values."""
    if path is None:
        return None
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    raw_inputs = data.get("inputs", data)
    raw_config = data.get("normalization", {})
    config_names = {
        "visible_gaussian_count": "visible_gaussian_scale",
        "overlap_ratio": "overlap_ratio_scale",
        "average_tile_density": "average_tile_density_scale",
        "depth_complexity": "depth_complexity_scale",
    }
    config = DifficultyConfig(**{
        config_names[name]: value
        for name, value in raw_config.items()
        if name in config_names
    })
    inputs = DifficultyInputs(**{
        name: raw_inputs[name]
        for name in (
            "visible_gaussian_count",
            "overlap_ratio",
            "average_tile_density",
            "depth_complexity",
        )
    })
    return calculate_difficulty(inputs, config)


def main():
    """Execute the standardized 3DGS renderer benchmark.

    Pipeline:
        1. Load scene from PLY file
        2. Load or generate camera poses
        3. Check renderer availability
        4. Run benchmark for each renderer with warmup and measurement phases
        5. Export results in JSON, CSV, Markdown, and HTML formats
    """
    args = parse_args()
    cfg = BenchmarkConfig.create_default()

    if args.renderers: cfg.renderers = args.renderers
    if args.frames: cfg.benchmark_frames = args.frames
    if args.warmup: cfg.warmup_frames = args.warmup
    if args.repeats: cfg.repeats = args.repeats
    if args.clock_lock is not None: cfg.clock_lock = args.clock_lock
    if args.width: cfg.image_width = args.width
    if args.height: cfg.image_height = args.height
    if args.output: cfg.output_dir = args.output
    if args.benchmark_type: cfg.benchmark_type = args.benchmark_type
    difficulty = _load_difficulty(args.difficulty_metrics)

    repo_root = os.path.dirname(PROJECT_ROOT)
    benchmark_commit_hash = _git_commit_hash(repo_root)
    hardware_metadata = _hardware_metadata(0)
    driver_version = _cuda_driver_version()
    data_dir = os.path.join(repo_root, "data")
    default_data_dirs = [data_dir, os.path.join(PROJECT_ROOT, "data")]
    scene_path = args.scene or next(
        (os.path.join(d, "scene.ply") for d in default_data_dirs
         if os.path.exists(os.path.join(d, "scene.ply"))),
        os.path.join(data_dir, "scene.ply"),
    )
    cameras_path = args.cameras or next(
        (os.path.join(d, "cameras.json") for d in default_data_dirs
         if os.path.exists(os.path.join(d, "cameras.json"))),
        os.path.join(data_dir, "cameras.json"),
    )
    output_dir = args.output or os.path.join(repo_root, "results")
    os.makedirs(output_dir, exist_ok=True)

    if args.list_renderers:
        print("Registered:", list_renderers())
        print("Available:", list_available())
        return

    # Resolve camera path preset
    if args.camera_path:
        preset = os.path.join(data_dir, "camera_presets", f"{args.camera_path}.json")
        if os.path.exists(preset):
            cameras_path = preset
            print(f"  Using camera preset: {args.camera_path}")

    print("=" * 70)
    print("  3DGS Renderer Benchmark")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA: {torch.version.cuda}  |  PyTorch: {torch.__version__}")
    print(f"  Scene: {scene_path}")
    print(f"  Cameras: {cameras_path}")
    resolution_label = (
        "from camera file" if os.path.exists(cameras_path)
        else f"{cfg.image_width}x{cfg.image_height}"
    )
    print(f"  Resolution: {resolution_label}")
    print(f"  Frames: {cfg.benchmark_frames} (+ {cfg.warmup_frames} warmup)")
    print(f"  Repeats: {cfg.repeats}  |  Clock Lock: {cfg.clock_lock}")
    print("=" * 70)

    # Phase 1: Load scene
    print("\n[1/5] Loading scene data...")
    assert os.path.exists(scene_path), f"Scene not found: {scene_path}"
    t0 = time.perf_counter()
    scene_data = load_ply(scene_path, device="cuda")
    scene_load_ms = (time.perf_counter() - t0) * 1000
    N = scene_data["num_points"]
    file_size_mb = os.path.getsize(scene_path) / (1024 * 1024)

    # Phase 2: Load cameras
    print("\n[2/5] Loading camera poses...")
    if os.path.exists(cameras_path):
        t0 = time.perf_counter()
        cameras = load_cameras_from_json(cameras_path, device="cuda")
        cam_load_ms = (time.perf_counter() - t0) * 1000
        print(f"  Loaded {len(cameras)} cameras ({cam_load_ms:.0f}ms)")
    else:
        t0 = time.perf_counter()
        cameras = generate_cameras(50, cfg.image_width, cfg.image_height, device="cuda")
        cam_load_ms = (time.perf_counter() - t0) * 1000
        print(f"  Generated {len(cameras)} cameras ({cam_load_ms:.0f}ms)")

    image_width, image_height = _uniform_camera_resolution(cameras)
    print(f"  Camera resolution: {image_width}x{image_height}")

    scene_center = (scene_data["xyz"].amin(dim=0) + scene_data["xyz"].amax(dim=0)) * 0.5
    if not args.allow_backfacing_cameras:
        validate_cameras_facing_point(cameras, scene_center)

    # Phase 3: Check renderer availability
    print("\n[3/5] Checking renderer availability...")
    available = list_available()
    print(f"  Available: {available}")
    renderers = [r for r in cfg.renderers if r in available]
    if not renderers:
        print("  No renderers available!")
        sys.exit(1)

    # Phase 4: Run benchmarks
    print("\n[4/5] Running benchmarks...")
    results_mgr = ResultsManager()
    gpu_name = torch.cuda.get_device_name(0)

    for rname in renderers:
        print(f"\n  --- {rname} ---")
        renderer = get_renderer(rname)
        if not renderer:
            continue

        prep_data = renderer.prepare_scene(scene_data)
        all_frame_times = []
        all_wall_times = []
        all_peak_mem = 0
        all_mem_samples = []

        for repeat_idx in range(cfg.repeats):
            if cfg.repeats > 1:
                print(f"  Repeat {repeat_idx + 1}/{cfg.repeats}...")

            frame_times = []
            wall_times = []
            peak_mem = 0
            mem_samples = []

            # Warmup phase (excluded from measurement)
            print(f"  Warmup ({cfg.warmup_frames} frames)...", end=" ", flush=True)
            for f in range(cfg.warmup_frames):
                with torch.no_grad():
                    renderer.render(prep_data, cameras[f % len(cameras)])
            torch.cuda.synchronize()
            print("done")

            # Measurement phase
            torch.cuda.reset_peak_memory_stats()
            print(f"  Benchmark ({cfg.benchmark_frames} frames)...", end=" ", flush=True)
            for f in range(cfg.benchmark_frames):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                start_event.record()
                with torch.no_grad():
                    renderer.render(prep_data, cameras[f % len(cameras)])
                end_event.record()
                end_event.synchronize()
                frame_times.append(start_event.elapsed_time(end_event))
                wall_times.append((time.perf_counter() - wall_start) * 1000.0)

                mem = torch.cuda.memory_allocated() / (1024 * 1024)
                mem_samples.append(mem)
                if mem > peak_mem:
                    peak_mem = mem

                if (f + 1) % 50 == 0:
                    print(f"{f+1}..", end="", flush=True)
            peak_mem = max(
                peak_mem,
                torch.cuda.max_memory_allocated() / (1024 * 1024),
            )
            print(" done")

            all_frame_times.extend(frame_times)
            all_wall_times.extend(wall_times)
            all_mem_samples.extend(mem_samples)
            if peak_mem > all_peak_mem:
                all_peak_mem = peak_mem

        t_arr = np.array(all_frame_times)
        renderer_meta = renderer.metadata()
        metrics = RendererMetrics(
            renderer_name=rname,
            num_frames=cfg.benchmark_frames * cfg.repeats,
            warmup_frames=cfg.warmup_frames,
            image_width=image_width,
            image_height=image_height,
            num_gaussians=N,
            gpu_name=gpu_name,
            renderer_implementation=renderer_meta["implementation"],
            renderer_version=renderer_meta["version"],
            renderer_source_url=renderer_meta["source_url"],
            renderer_commit_hash=renderer_meta.get("commit_hash"),
            timing_method="torch.cuda.Event elapsed time; per-frame synchronization",
            benchmark_suite_version=BENCHMARK_SUITE_VERSION,
            benchmark_commit_hash=benchmark_commit_hash,
            driver_version=driver_version,
            cuda_version=torch.version.cuda,
            hardware_metadata=hardware_metadata,
            peak_vram_mb=all_peak_mem,
            avg_vram_mb=float(np.mean(all_mem_samples)),
            scene_load_time_ms=scene_load_ms,
            scene_parse_time_ms=0.0,
            file_size_mb=file_size_mb,
            benchmark_type=cfg.benchmark_type,
            difficulty_score=difficulty.score if difficulty else None,
            difficulty_formula=difficulty.formula_id if difficulty else None,
            difficulty_inputs=difficulty.to_dict()["inputs"] if difficulty else None,
            difficulty_normalization=difficulty.to_dict()["normalization"] if difficulty else None,
            frame_times_ms=[round(x, 2) for x in all_frame_times],
            wall_frame_times_ms=[round(x, 2) for x in all_wall_times],
        )
        metrics.compute()

        mean_std = np.std(t_arr) / np.sqrt(len(t_arr))
        print(f"  Mean: {metrics.mean_latency_ms:.1f} +/- {mean_std:.1f}ms ({metrics.mean_fps:.1f}FPS)  "
              f"Median: {metrics.median_latency_ms:.1f}ms")
        print(f"  End-to-end: mean={metrics.mean_wall_latency_ms:.1f}ms "
              f"median={metrics.median_wall_latency_ms:.1f}ms ({metrics.wall_fps:.1f}FPS)")
        print(f"  P1/P5/P50/P95/P99: {metrics.p1_latency_ms:.1f}/{metrics.p5_latency_ms:.1f}/"
              f"{metrics.median_latency_ms:.1f}/{metrics.p95_latency_ms:.1f}/{metrics.p99_latency_ms:.1f}ms")
        print(f"  VRAM: peak={all_peak_mem:.0f}MB  avg={metrics.avg_vram_mb:.0f}MB")

        results_mgr.add_result(rname, metrics)

    # Phase 5: Export results
    print("\n[5/5] Exporting results...")
    rankings = results_mgr.get_ranking()
    fastest = rankings[0][0] if rankings else ""

    results_mgr.export_json(os.path.join(output_dir, "benchmark_results.json"))
    results_mgr.export_csv(os.path.join(output_dir, "benchmark_results.csv"))
    results_mgr.export_markdown(os.path.join(output_dir, "benchmark_report.md"))
    results_mgr.export_html(os.path.join(output_dir, "benchmark_report.html"),
                            title="3DGS Renderer Benchmark")
    results_mgr.export_analysis(output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    for i, (name, fps, lat) in enumerate(rankings, 1):
        tag = "  (FASTEST)" if i == 1 else ""
        m = results_mgr.results[name]
        t_arr = np.array(m.frame_times_ms)
        sem = np.std(t_arr) / np.sqrt(len(t_arr))
        print(f"  #{i}: {name:20s}  {fps:8.1f} FPS  {lat:8.2f} +/- {sem:.2f} ms  "
              f"P99={m.p99_latency_ms:6.2f}ms  VRAM={m.peak_vram_mb:.0f}MB{tag}")

    print(f"\nResults saved to {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
