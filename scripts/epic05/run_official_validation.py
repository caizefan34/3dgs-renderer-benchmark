#!/usr/bin/env python3
"""
EPIC-05 Official Dataset Validation Runner.

Benchmarks tile_size={8,16,32} on each official real-scene checkpoint from the
Mip-NeRF 360 cohort, following the strict fair-comparison protocol:

    Same checkpoint ⟹ same cameras ⟹ same resolution ⟹ same renderer version
    ⟹ same CUDA/driver/GPU ⟹ same dtype ⟹ same SH ⟹ same packed/eps2d/radius_clip
    ⟹ only tile_size varies.

Protocol:
    - Cold-start CUDA context prep
    - Explicit warmup (excluded from measurement)
    - Steady-state measurement with torch.cuda.Event per frame
    - Multiple repeats with VRAM tracking

Usage:
    python scripts/epic05/run_official_validation.py \\
        --scene bicycle --resolution 1080p \\
        --tile-sizes 16 32 --repeats 5

    python scripts/epic05/run_official_validation.py \\
        --all --resolution 1080p --tile-sizes 8 16 32 --repeats 3
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark_suite import BENCHMARK_SUITE_VERSION, resolve_suite_case
from benchmark_framework import (
    load_ply,
    load_cameras_from_json,
    resize_cameras,
    validate_cameras_facing_point,
    RendererMetrics,
    ResultsManager,
)


# ---------------------------------------------------------------------------
# Resolution presets
# ---------------------------------------------------------------------------
RESOLUTION_PRESETS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

# ---------------------------------------------------------------------------
# GPU timer utility
# ---------------------------------------------------------------------------
def _gpu_timer() -> Tuple[Any, Any, Any]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    return start, end, lambda: start.elapsed_time(end)


class GsplatTileRenderer:
    """Thin wrapper around gsplat rasterization with configurable tile_size."""

    def __init__(self, tile_size: int = 16, packed: bool = True,
                 device: str = "cuda"):
        self.tile_size = tile_size
        self.packed = packed
        self.device = device
        self._available = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from gsplat import rasterization  # noqa: F401
                self._available = True
            except (ImportError, OSError):
                self._available = False
        return self._available

    def prepare_scene(self, scene_data: dict) -> dict:
        return {
            **scene_data,
            "quats": torch.nn.functional.normalize(
                scene_data["rotations"], dim=-1
            ).contiguous(),
            "scales_activated": torch.exp(scene_data["scales"]).contiguous(),
            "opacities_activated": torch.sigmoid(
                scene_data["opacity"]
            ).contiguous(),
            "shs": scene_data["shs"].contiguous(),
        }

    def render(self, scene_data: dict, camera) -> torch.Tensor:
        from gsplat import rasterization

        rendered, _, _ = rasterization(
            means=scene_data["xyz"],
            quats=scene_data["quats"],
            scales=scene_data["scales_activated"],
            opacities=scene_data["opacities_activated"],
            colors=scene_data["shs"],
            viewmats=camera.viewmatrix.unsqueeze(0),
            Ks=camera.K.unsqueeze(0),
            width=camera.image_width,
            height=camera.image_height,
            tile_size=self.tile_size,
            sh_degree=scene_data.get("sh_degree", 3),
            packed=self.packed,
            render_mode="RGB",
        )
        return rendered[0].clamp(0, 1)

    def metadata(self) -> dict:
        return {
            "implementation": (
                f"nerfstudio-project/gsplat rasterization (tile {self.tile_size}, "
                f"{'packed' if self.packed else 'dense'})"
            ),
            "version": "gsplat",
            "source_url": "https://github.com/nerfstudio-project/gsplat",
        }


# ---------------------------------------------------------------------------
# Scene registry (mirrors suite.json for Mip-NeRF 360 scenes)
# ---------------------------------------------------------------------------
OFFICIAL_SCENES = {
    "bicycle": {
        "scene_id": "bicycle",
        "dataset_family": "Mip-NeRF 360",
        "scene_path": "data/official/mipnerf360/bicycle/point_cloud.ply",
        "camera_path": "data/official/mipnerf360/bicycle/cameras.json",
        "num_gaussians": 6131954,
        "camera_count": 281,
    },
    "garden": {
        "scene_id": "garden",
        "dataset_family": "Mip-NeRF 360",
        "scene_path": "data/official/mipnerf360/garden/point_cloud.ply",
        "camera_path": "data/official/mipnerf360/garden/cameras.json",
        "num_gaussians": 5834784,
        "camera_count": 250,
    },
    "room": {
        "scene_id": "room",
        "dataset_family": "Mip-NeRF 360",
        "scene_path": "data/official/mipnerf360/room/point_cloud.ply",
        "camera_path": "data/official/mipnerf360/room/cameras.json",
        "num_gaussians": 1593376,
        "camera_count": 217,
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _sha256(path: str) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit_hash() -> Optional[str]:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _hardware_metadata():
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_vram_mb": round(props.total_memory / (1024 * 1024), 1),
        "multi_processor_count": props.multi_processor_count,
    }


def _check_steady_state(renderer, prep_data, cameras, num_check: int = 10) -> float:
    """Verify that frames after warmup are stable (no JIT/compilation spikes)."""
    times = []
    for f in range(num_check):
        cam = cameras[f % len(cameras)]
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            renderer.render(prep_data, cam)
        end_event.record()
        end_event.synchronize()
        times.append(start_event.elapsed_time(end_event))
    t = np.array(times)
    cv = t.std() / t.mean() if t.mean() > 0 else 0.0
    return float(cv)


def run_single_benchmark(
    scene_data: dict,
    cameras_list,
    tile_size: int,
    warmup_frames: int = 30,
    measured_frames: int = 100,
    repeats: int = 5,
    packed: bool = True,
) -> Dict[str, Any]:
    """Run one tile_size benchmark on an already-loaded scene and cameras.

    Returns a dict with timing statistics, VRAM usage, and stability info.
    """
    renderer = GsplatTileRenderer(tile_size=tile_size, packed=packed)
    if not renderer.is_available():
        raise RuntimeError(f"gsplat renderer not available (tile_size={tile_size})")

    prep_data = renderer.prepare_scene(scene_data)
    N = scene_data["num_points"]
    gpu_name = torch.cuda.get_device_name(0)

    all_frame_times = []
    all_wall_times = []
    all_peak_mem = 0
    all_mem_samples = []

    for repeat_idx in range(repeats):
        if repeats > 1:
            print(f"    Repeat {repeat_idx + 1}/{repeats}...", end=" ", flush=True)

        frame_times = []
        wall_times = []
        peak_mem = 0
        mem_samples = []

        # --- Warmup phase ---
        for f in range(warmup_frames):
            cam = cameras_list[f % len(cameras_list)]
            with torch.no_grad():
                renderer.render(prep_data, cam)
        torch.cuda.synchronize()

        # Check steady state after warmup
        stability_cv = _check_steady_state(renderer, prep_data, cameras_list, 10)
        if stability_cv > 0.15:
            # Additional warmup if instability detected
            print(f"(CV={stability_cv:.3f}→extra warmup) ", end="", flush=True)
            for _ in range(10):
                for f in range(len(cameras_list)):
                    with torch.no_grad():
                        renderer.render(prep_data, cameras_list[f])
            torch.cuda.synchronize()
            stability_cv = _check_steady_state(
                renderer, prep_data, cameras_list, 10
            )

        # --- Measurement phase ---
        torch.cuda.reset_peak_memory_stats()
        for f in range(measured_frames):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start_event.record()
            cam = cameras_list[f % len(cameras_list)]
            with torch.no_grad():
                renderer.render(prep_data, cam)
            end_event.record()
            end_event.synchronize()
            frame_times.append(start_event.elapsed_time(end_event))
            wall_times.append((time.perf_counter() - wall_start) * 1000.0)

            mem = torch.cuda.memory_allocated() / (1024 * 1024)
            mem_samples.append(mem)
            if mem > peak_mem:
                peak_mem = mem

        peak_mem = max(
            peak_mem,
            torch.cuda.max_memory_allocated() / (1024 * 1024),
        )

        all_frame_times.extend(frame_times)
        all_wall_times.extend(wall_times)
        all_mem_samples.extend(mem_samples)
        if peak_mem > all_peak_mem:
            all_peak_mem = peak_mem

        if repeats > 1:
            print("done")

    t_arr = np.array(all_frame_times)
    wall_arr = np.array(all_wall_times)

    result = {
        "tile_size": tile_size,
        "packed": packed,
        "num_gaussians": N,
        "gpu": gpu_name,
        "num_frames": measured_frames * repeats,
        "warmup_frames": warmup_frames,
        "measured_frames_per_repeat": measured_frames,
        "repeats": repeats,
        "stability_cv_after_warmup": round(stability_cv, 6),
        "mean_ms": round(float(t_arr.mean()), 4),
        "median_ms": round(float(np.median(t_arr)), 4),
        "std_ms": round(float(t_arr.std()), 4),
        "min_ms": round(float(t_arr.min()), 4),
        "max_ms": round(float(t_arr.max()), 4),
        "p1_ms": round(float(np.percentile(t_arr, 1)), 4),
        "p5_ms": round(float(np.percentile(t_arr, 5)), 4),
        "p95_ms": round(float(np.percentile(t_arr, 95)), 4),
        "p99_ms": round(float(np.percentile(t_arr, 99)), 4),
        "mean_wall_ms": round(float(wall_arr.mean()), 4),
        "median_wall_ms": round(float(np.median(wall_arr)), 4),
        "mean_fps": round(1000.0 / float(t_arr.mean()), 2) if t_arr.mean() > 0 else 0.0,
        "median_fps": round(1000.0 / float(np.median(t_arr)), 2) if np.median(t_arr) > 0 else 0.0,
        "peak_vram_mb": round(all_peak_mem, 1),
        "avg_vram_mb": round(float(np.mean(all_mem_samples)), 1),
        "frame_times_ms": [round(x, 2) for x in all_frame_times],
    }
    return result


# ---------------------------------------------------------------------------
# Main validation runner
# ---------------------------------------------------------------------------
def run_validation(args) -> None:
    """Run the full official validation for specified scenes and tile sizes."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_base = REPO_ROOT / "results" / "epic05" / "official" / "raw"
    output_base.mkdir(parents=True, exist_ok=True)

    # Determine scenes to process
    if args.all:
        scene_ids = list(OFFICIAL_SCENES.keys())
    elif args.scene:
        scene_ids = args.scene
    else:
        raise ValueError("Specify --scene or --all")

    # Determine tile sizes
    tile_sizes = args.tile_sizes or [8, 16, 32]

    # Resolution
    if args.resolution in RESOLUTION_PRESETS:
        target_resolution = RESOLUTION_PRESETS[args.resolution]
    else:
        # Use native camera resolution
        target_resolution = None
    resolution_label = args.resolution or "native"

    repeats = args.repeats
    warmup = args.warmup
    measured = args.measured

    # Protocol summary
    protocol = {
        "warmup_frames": warmup,
        "measured_frames_per_repeat": measured,
        "repeats": repeats,
        "total_measured_frames": measured * repeats,
        "timing": "torch.cuda.Event elapsed time; per-frame synchronization",
        "steady_state_check": "CV threshold 0.15 with auto-extension",
    }

    results_data: Dict[str, Any] = {
        "experiment_id": "epic05-official-validation-v1",
        "benchmark_suite_version": BENCHMARK_SUITE_VERSION,
        "benchmark_type": "real_scene_speed",
        "date": date.today().isoformat(),
        "protocol": protocol,
        "resolution": target_resolution,
        "resolution_label": resolution_label,
        "tile_sizes_compared": tile_sizes,
        "hardware": _hardware_metadata(),
        "scenes": {},
    }

    for scene_id in scene_ids:
        if scene_id not in OFFICIAL_SCENES:
            print(f"WARNING: Unknown scene {scene_id!r}, skipping")
            continue

        scene_info = OFFICIAL_SCENES[scene_id]
        scene_path = REPO_ROOT / scene_info["scene_path"]
        camera_path = REPO_ROOT / scene_info["camera_path"]

        if not scene_path.exists():
            print(f"ERROR: Scene not found: {scene_path}")
            results_data["scenes"][scene_id] = {"error": f"Scene not found: {scene_path}"}
            continue
        if not camera_path.exists():
            print(f"ERROR: Cameras not found: {camera_path}")
            results_data["scenes"][scene_id] = {"error": f"Cameras not found: {camera_path}"}
            continue

        print(f"\n{'='*70}")
        print(f"  Scene: {scene_id} ({scene_info['dataset_family']})")
        print(f"  Gaussians: {scene_info['num_gaussians']:,}")
        print(f"  Cameras: {scene_info['camera_count']}")
        print(f"  Resolution: {resolution_label} ({target_resolution})")
        print(f"  Tile sizes: {tile_sizes}")
        print(f"{'='*70}\n")

        # --- Load scene ---
        print(f"  Loading scene...", end=" ", flush=True)
        t0 = time.perf_counter()
        scene_data = load_ply(str(scene_path), device="cuda")
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"done ({load_ms:.0f}ms)")

        # --- Load cameras ---
        print(f"  Loading cameras...", end=" ", flush=True)
        cameras = load_cameras_from_json(str(camera_path), device="cuda")
        print(f"loaded {len(cameras)} cameras")

        # Resolve resolution
        if target_resolution is not None:
            cameras = resize_cameras(cameras, *target_resolution)
            print(f"  Resized to {target_resolution[0]}x{target_resolution[1]}")
        else:
            target_resolution = (cameras[0].image_width, cameras[0].image_height)
            print(f"  Using native resolution: {target_resolution[0]}x{target_resolution[1]}")

        # Validate cameras
        scene_center = (
            scene_data["xyz"].amin(dim=0) + scene_data["xyz"].amax(dim=0)
        ) * 0.5
        validate_cameras_facing_point(cameras, scene_center)

        print(f"  Scene loaded: {scene_data['num_points']:,} gaussians")
        print(f"  Camera resolution: {cameras[0].image_width}x{cameras[0].image_height}")
        print()

        # --- Cold-start preparation ---
        print("  Cold-start preparation...", end=" ", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # One forward pass to initialize CUDA context and load gsplat kernels
        _prep = GsplatTileRenderer(tile_size=tile_sizes[0], packed=True)
        _prep_data = _prep.prepare_scene(scene_data)
        with torch.no_grad():
            _ = _prep.render(_prep_data, cameras[0])
        torch.cuda.synchronize()
        del _prep, _prep_data
        gc.collect()
        torch.cuda.empty_cache()
        print("done")

        # --- Run benchmarks for each tile size ---
        scene_results = {}
        for ts in tile_sizes:
            print(f"\n  --- tile_size={ts} ---")
            result = run_single_benchmark(
                scene_data,
                cameras,
                tile_size=ts,
                warmup_frames=warmup,
                measured_frames=measured,
                repeats=repeats,
                packed=True,
            )
            scene_results[f"tile{ts}"] = result

            mean = result["mean_ms"]
            fps = result["mean_fps"]
            p99 = result["p99_ms"]
            vram = result["peak_vram_mb"]
            cv = result["stability_cv_after_warmup"]
            print(f"    Mean: {mean:.2f}ms = {fps:.1f}FPS  P99: {p99:.2f}ms  "
                  f"VRAM: {vram:.0f}MB  CV: {cv:.4f}")

            # Compute speedup vs tile16
            if ts == 16:
                tile16_mean = mean

        scene_info_output = {
            **scene_info,
            "scene_path": str(scene_path),
            "camera_path": str(camera_path),
            "model_sha256": _sha256(str(scene_path)),
            "camera_sha256": _sha256(str(camera_path)),
            "resolution_used": list(target_resolution),
            "resolution_label": resolution_label,
        }
        results_data["scenes"][scene_id] = {
            "scene_info": scene_info_output,
            "tile_results": scene_results,
        }

        # Per-scene speedup summary
        print(f"\n  --- Speedup Summary (tile32 vs tile16) ---")
        if "tile16" in scene_results and "tile32" in scene_results:
            speedup = scene_results["tile16"]["mean_ms"] / scene_results["tile32"]["mean_ms"]
            print(f"  tile16: {scene_results['tile16']['mean_ms']:.2f}ms  "
                  f"tile32: {scene_results['tile32']['mean_ms']:.2f}ms  "
                  f"Speedup: {speedup:.4f}x")
        if "tile8" in scene_results and "tile16" in scene_results:
            speedup_8 = scene_results["tile8"]["mean_ms"] / scene_results["tile16"]["mean_ms"]
            print(f"  tile8:  {scene_results['tile8']['mean_ms']:.2f}ms  "
                  f"tile16: {scene_results['tile16']['mean_ms']:.2f}ms  "
                  f"Ratio: {speedup_8:.4f}x")

        # Cleanup
        del scene_data, cameras
        gc.collect()
        torch.cuda.empty_cache()

    # --- Export raw results ---
    timestamp_suffix = f"{timestamp}"
    output_path = output_base / f"official_validation_{timestamp_suffix}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"\n{'='*70}")
    print(f"  Raw results saved: {output_path}")
    print(f"{'='*70}")

    return results_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scene", nargs="+",
        choices=list(OFFICIAL_SCENES.keys()),
        default=None,
        help="Scene(s) to validate"
    )
    p.add_argument("--all", action="store_true", help="Validate all official scenes")
    p.add_argument(
        "--resolution",
        choices=list(RESOLUTION_PRESETS.keys()),
        default="1080p",
        help="Target resolution (default: 1080p)",
    )
    p.add_argument(
        "--tile-sizes", nargs="+", type=int,
        default=[8, 16, 32],
        help="Tile sizes to compare (default: 8 16 32)",
    )
    p.add_argument("--repeats", type=int, default=5, help="Measurement repeats")
    p.add_argument("--warmup", type=int, default=30, help="Warmup frames")
    p.add_argument("--measured", type=int, default=100, help="Measured frames per repeat")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.scene and not args.all:
        raise SystemExit("Specify --scene or --all")
    run_validation(args)


if __name__ == "__main__":
    main()
