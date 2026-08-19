#!/usr/bin/env python3
"""
EPIC-05 Phase 4: Bicycle Anomaly Rerun + Backend Validation.

STEP 1: Rerun bicycle with tile8/16/32 × 5 repeats × 100 measured frames.
    - Enhanced outlier detection
    - Per-repeat statistics
    - Distinguish steady-state vs sync anomaly

STEP 2: Validate backend binary/path fairness.
    - Record extension_hash, source_commit, compiler_version, CUDA version
    - Verify same binary used for all tile sizes

Output:
    results/epic05/phase4/bicycle_rerun_<timestamp>.json
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
import torch.utils.cpp_extension as cpp_ext
cpp_ext.SUBPROCESS_DECODE_ARGS = ('utf-8', 'ignore')

_msvc_dir = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
for _p in [_msvc_dir, _cuda_bin]:
    if _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")
os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
os.environ["CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sha256(path: str) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def _git_commit_hash() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _hardware_metadata() -> dict:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    props = torch.cuda.get_device_properties(0)
    # SM shared memory: use prop.sharedMemPerBlock (per block) and
    # prop.sharedMemPerMultiprocessor (per SM)
    return {
        "cuda_available": True,
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_vram_mb": round(props.total_memory / (1024 * 1024), 1),
        "multi_processor_count": props.multi_processor_count,
        "shared_mem_per_block_kb": round(props.shared_memory_per_block / 1024, 1),
        "shared_mem_per_sm_kb": round(props.shared_memory_per_multiprocessor / 1024, 1),
        "l2_cache_size_bytes": props.L2_cache_size if hasattr(props, 'L2_cache_size') else "N/A",
        "l2_cache_size_mb": round(props.L2_cache_size / (1024 * 1024), 2) if hasattr(props, 'L2_cache_size') else "N/A",
        "regs_per_multiprocessor": props.regs_per_multiprocessor if hasattr(props, 'regs_per_multiprocessor') else "N/A",
        "max_threads_per_multi_processor": props.max_threads_per_multi_processor,
        "warp_size": props.warp_size,
        "shared_mem_per_block_optin_kb": round(props.shared_memory_per_block_optin / 1024, 1) if hasattr(props, 'shared_memory_per_block_optin') else "N/A",
    }


def _cuda_toolkit_version() -> str:
    try:
        return torch.version.cuda or "unknown"
    except Exception:
        return "unknown"


def _msvc_version() -> str:
    try:
        r = subprocess.run(["cl.exe", "--version"], capture_output=True, text=True, timeout=5)
        for line in r.stderr.splitlines():
            if "Version" in line:
                return line.strip()
        # Try stdout
        for line in r.stdout.splitlines():
            if "Version" in line or "Microsoft" in line:
                return line.strip()
        return r.stdout.splitlines()[0] if r.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def _extension_info() -> dict:
    """Record the gsplat CUDA extension binary path and hash."""
    info = {
        "gsplat_version": "unknown",
        "backend_module": None,
        "backend_path": None,
        "backend_hash": None,
    }
    try:
        import gsplat
        info["gsplat_version"] = getattr(gsplat, "__version__", "unknown")
    except ImportError:
        pass
    try:
        from gsplat.cuda._backend import _C as backend
        info["backend_module"] = type(backend).__name__
        p = getattr(backend, "__file__", None)
        if p:
            info["backend_path"] = str(p)
            info["backend_hash"] = _sha256(p)
    except Exception as e:
        info["backend_error"] = str(e)
    # Check if cached extension is being used
    try:
        ext_cache = os.environ.get(
            "TORCH_EXTENSIONS_DIR",
            str(Path.home() / ".cache" / "torch_extensions" / "torch_extensions" / "Cache")
        )
        cp = Path(ext_cache)
        if cp.exists():
            for f in cp.iterdir():
                if f.is_dir() and "gsplat" in f.name.lower():
                    info["cached_ext_dir"] = str(f)
                    for ext_f in f.iterdir():
                        if ext_f.suffix in (".pyd", ".so", ".dll"):
                            info["cached_ext_path"] = str(ext_f)
                            info["cached_ext_hash"] = _sha256(str(ext_f))
                            break
                    break
    except Exception:
        pass
    return info


def _check_steady_state(renderer, prep_data, cameras, num_check=10):
    times = []
    for f in range(num_check):
        cam = cameras[f % len(cameras)]
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad():
            renderer.render(prep_data, cam)
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    t = np.array(times)
    cv = float(t.std() / t.mean()) if t.mean() > 0 else 0.0
    return cv


class GsplatTileRenderer:
    def __init__(self, tile_size=16, packed=True, device="cuda"):
        self.tile_size = tile_size
        self.packed = packed
        self.device = device
        self._available = None

    def is_available(self):
        if self._available is None:
            try:
                from gsplat import rasterization
                self._available = True
            except (ImportError, OSError):
                self._available = False
        return self._available

    def prepare_scene(self, scene_data):
        return {
            **scene_data,
            "quats": torch.nn.functional.normalize(scene_data["rotations"], dim=-1).contiguous(),
            "scales_activated": torch.exp(scene_data["scales"]).contiguous(),
            "opacities_activated": torch.sigmoid(scene_data["opacity"]).contiguous(),
            "shs": scene_data["shs"].contiguous(),
        }

    def render(self, scene_data, camera):
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


def analyze_outliers(times_ms: List[float]) -> dict:
    """Analyze outliers in frame time series."""
    t = np.array(times_ms)
    q1, q3 = np.percentile(t, 25), np.percentile(t, 75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_mask = (t < lower) | (t > upper)
    severe_mask = t > 100  # >100ms is clearly anomalous for rasterization
    return {
        "count_outliers_iqr": int(outlier_mask.sum()),
        "count_severe_outliers": int(severe_mask.sum()),
        "outlier_threshold_lower_ms": round(float(lower), 2),
        "outlier_threshold_upper_ms": round(float(upper), 2),
        "severe_outlier_indices": [int(i) for i in np.where(severe_mask)[0].tolist()],
        "severe_outlier_times": [round(float(t[i]), 2) for i in np.where(severe_mask)[0].tolist()],
        "steady_state_count": int((~severe_mask).sum()),
    }


def run_single_benchmark(scene_data, cameras_list, tile_size,
                         warmup_frames=30, measured_frames=100, repeats=5,
                         packed=True):
    """Enhanced benchmark with per-repeat statistics and outlier analysis."""
    renderer = GsplatTileRenderer(tile_size=tile_size, packed=packed)
    if not renderer.is_available():
        raise RuntimeError(f"gsplat renderer not available (tile_size={tile_size})")

    prep_data = renderer.prepare_scene(scene_data)
    N = scene_data["num_points"]
    gpu_name = torch.cuda.get_device_name(0)

    all_frame_times = []
    all_stable_times = []
    repeat_stats = []

    for repeat_idx in range(repeats):
        if repeats > 1:
            print(f"    Repeat {repeat_idx + 1}/{repeats}...", end=" ", flush=True)

        frame_times = []
        peak_mem = 0

        # Warmup
        for f in range(warmup_frames):
            cam = cameras_list[f % len(cameras_list)]
            with torch.no_grad():
                renderer.render(prep_data, cam)
        torch.cuda.synchronize()

        # Steady-state check
        stability_cv = _check_steady_state(renderer, prep_data, cameras_list, 10)
        if stability_cv > 0.15:
            print(f"(CV={stability_cv:.3f}→extra warmup) ", end="", flush=True)
            for _ in range(10):
                for f in range(len(cameras_list)):
                    with torch.no_grad():
                        renderer.render(prep_data, cameras_list[f])
            torch.cuda.synchronize()
            stability_cv = _check_steady_state(renderer, prep_data, cameras_list, 10)

        # Measurement
        torch.cuda.reset_peak_memory_stats()
        for f in range(measured_frames):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            cam = cameras_list[f % len(cameras_list)]
            with torch.no_grad():
                renderer.render(prep_data, cam)
            end_event.record()
            end_event.synchronize()
            t_ms = start_event.elapsed_time(end_event)
            frame_times.append(t_ms)

        peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1024 * 1024))

        t_arr = np.array(frame_times)
        rep = {
            "repeat": repeat_idx,
            "mean_ms": round(float(t_arr.mean()), 4),
            "median_ms": round(float(np.median(t_arr)), 4),
            "std_ms": round(float(t_arr.std()), 4),
            "min_ms": round(float(t_arr.min()), 4),
            "max_ms": round(float(t_arr.max()), 4),
            "p99_ms": round(float(np.percentile(t_arr, 99)), 4),
        }
        repeat_stats.append(rep)

        all_frame_times.extend(frame_times)
        # Track steady-state frames (only non-severe outliers for clean analysis)
        clean = [t for t in frame_times if t <= 100]
        all_stable_times.extend(clean)

        if repeats > 1:
            print(f"  mean={rep['mean_ms']:.2f}ms median={rep['median_ms']:.2f}ms max={rep['max_ms']:.2f}ms")

    t_all = np.array(all_frame_times)
    t_stable = np.array(all_stable_times) if all_stable_times else t_all

    analysis = analyze_outliers(all_frame_times)

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
        # Full statistics (including outliers)
        "mean_ms": round(float(t_all.mean()), 4),
        "median_ms": round(float(np.median(t_all)), 4),
        "std_ms": round(float(t_all.std()), 4),
        "min_ms": round(float(t_all.min()), 4),
        "max_ms": round(float(t_all.max()), 4),
        "p1_ms": round(float(np.percentile(t_all, 1)), 4),
        "p5_ms": round(float(np.percentile(t_all, 5)), 4),
        "p95_ms": round(float(np.percentile(t_all, 95)), 4),
        "p99_ms": round(float(np.percentile(t_all, 99)), 4),
        # Steady-state statistics (filtered)
        "stable_mean_ms": round(float(t_stable.mean()), 4),
        "stable_median_ms": round(float(np.median(t_stable)), 4),
        "stable_std_ms": round(float(t_stable.std()), 4),
        "stable_min_ms": round(float(t_stable.min()), 4),
        "stable_max_ms": round(float(t_stable.max()), 4),
        "stable_count": len(t_stable),
        # FPS
        "mean_fps": round(1000.0 / float(t_all.mean()), 2) if t_all.mean() > 0 else 0.0,
        "stable_fps": round(1000.0 / float(t_stable.mean()), 2) if t_stable.mean() > 0 else 0.0,
        # VRAM
        "peak_vram_mb": round(peak_mem, 1),
        # Outlier analysis
        "outlier_analysis": analysis,
        # Per-repeat
        "repeat_statistics": repeat_stats,
        # Full frame times
        "frame_times_ms": [round(x, 2) for x in all_frame_times],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Bicycle rerun + backend validation")
    parser.add_argument("--scene", nargs="+", default=["bicycle"],
                        help="Scenes to benchmark (default: bicycle)")
    parser.add_argument("--tile-sizes", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--measured", type=int, default=100)
    parser.add_argument("--resolution", default="1080p")
    parser.add_argument("--validate-backend", action="store_true", default=True)
    args = parser.parse_args()

    RESOLUTION_PRESETS = {"720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160)}
    target_res = RESOLUTION_PRESETS.get(args.resolution, (1920, 1080))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = REPO_ROOT / "results" / "epic05" / "phase4"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Backend validation (STEP 2)
    # -----------------------------------------------------------------------
    backend_info = _extension_info() if args.validate_backend else {}
    # Add source and environment info
    backend_info.update({
        "git_commit": _git_commit_hash(),
        "cuda_toolkit_version": _cuda_toolkit_version(),
        "msvc_version": _msvc_version(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
    })

    print("=" * 70)
    print("  Phase 4: Bicycle Anomaly Rerun + Backend Validation")
    print("=" * 70)
    print()
    print("Backend Info:")
    for k, v in backend_info.items():
        print(f"  {k}: {v}")
    print()

    # -----------------------------------------------------------------------
    # Scene paths
    # -----------------------------------------------------------------------
    OFFICIAL_SCENES = {
        "bicycle": {
            "scene_path": str(REPO_ROOT / "data" / "official" / "mipnerf360" / "bicycle" / "point_cloud.ply"),
            "camera_path": str(REPO_ROOT / "data" / "official" / "mipnerf360" / "bicycle" / "cameras.json"),
            "num_gaussians": 6131954,
        },
        "garden": {
            "scene_path": str(REPO_ROOT / "data" / "official" / "mipnerf360" / "garden" / "point_cloud.ply"),
            "camera_path": str(REPO_ROOT / "data" / "official" / "mipnerf360" / "garden" / "cameras.json"),
            "num_gaussians": 5834784,
        },
        "room": {
            "scene_path": str(REPO_ROOT / "data" / "official" / "mipnerf360" / "room" / "point_cloud.ply"),
            "camera_path": str(REPO_ROOT / "data" / "official" / "mipnerf360" / "room" / "cameras.json"),
            "num_gaussians": 1593376,
        },
    }

    results = {
        "experiment_id": "epic05-phase4-bicycle-rerun-v1",
        "benchmark_type": "phase4_rerun",
        "date": date.today().isoformat(),
        "timestamp": timestamp,
        "resolution": list(target_res),
        "resolution_label": args.resolution,
        "protocol": {
            "warmup_frames": args.warmup,
            "measured_frames_per_repeat": args.measured,
            "repeats": args.repeats,
            "total_measured_frames": args.measured * args.repeats,
            "timing": "torch.cuda.Event elapsed time; per-frame synchronization",
        },
        "backend_validation": backend_info,
        "hardware": _hardware_metadata(),
        "scenes": {},
    }

    # -----------------------------------------------------------------------
    # Run for each scene
    # -----------------------------------------------------------------------
    for scene_id in args.scene:
        if scene_id not in OFFICIAL_SCENES:
            print(f"ERROR: Unknown scene {scene_id}")
            continue
        sinfo = OFFICIAL_SCENES[scene_id]
        scene_path, camera_path = sinfo["scene_path"], sinfo["camera_path"]

        if not os.path.exists(scene_path):
            print(f"ERROR: Scene not found: {scene_path}")
            continue
        if not os.path.exists(camera_path):
            print(f"ERROR: Cameras not found: {camera_path}")
            continue

        print(f"\n{'='*70}")
        print(f"  Scene: {scene_id}")
        print(f"  Gaussians: {sinfo['num_gaussians']:,}")
        print(f"  Resolution: {args.resolution} ({target_res})")
        print(f"  Tile sizes: {args.tile_sizes}")
        print(f"  Repeats: {args.repeats} × {args.measured} frames")
        print(f"{'='*70}\n")

        # Load scene
        print(f"  Loading scene...", end=" ", flush=True)
        t0 = time.perf_counter()
        scene_data = load_ply(scene_path, device="cuda")
        print(f"done ({ (time.perf_counter()-t0)*1000:.0f}ms)")

        # Load cameras
        cameras = load_cameras_from_json(camera_path, device="cuda")
        print(f"  Loaded {len(cameras)} cameras")

        # Resize
        cameras = resize_cameras(cameras, *target_res)

        # Cold start
        print("  Cold-start prep...", end=" ", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _prep = GsplatTileRenderer(tile_size=args.tile_sizes[0], packed=True)
        _prep_data = _prep.prepare_scene(scene_data)
        with torch.no_grad():
            _ = _prep.render(_prep_data, cameras[0])
        torch.cuda.synchronize()
        del _prep, _prep_data
        gc.collect()
        torch.cuda.empty_cache()
        print("done")

        # Benchmark each tile size
        tile_results = {}
        for ts in args.tile_sizes:
            print(f"\n  --- tile_size={ts} ---")
            result = run_single_benchmark(
                scene_data, cameras, tile_size=ts,
                warmup_frames=args.warmup,
                measured_frames=args.measured,
                repeats=args.repeats,
                packed=True,
            )
            tile_results[f"tile{ts}"] = result

            mean = result["mean_ms"]
            s_mean = result["stable_mean_ms"]
            fps = result["mean_fps"]
            s_fps = result["stable_fps"]
            p99 = result["p99_ms"]
            outliers = result["outlier_analysis"]["count_severe_outliers"]
            print(f"    Mean: {mean:.2f}ms ({fps:.1f}FPS)")
            print(f"    Steady-state: {s_mean:.2f}ms ({s_fps:.1f}FPS)")
            print(f"    P99: {p99:.2f}ms  Outliers: {outliers}")

        # Summary
        print(f"\n  --- Speedup Summary ---")
        if "tile16" in tile_results and "tile32" in tile_results:
            r16 = tile_results["tile16"]["stable_mean_ms"]
            r32 = tile_results["tile32"]["stable_mean_ms"]
            print(f"  tile16: {r16:.2f}ms  tile32: {r32:.2f}ms  Ratio: {r16/r32:.4f}x (steady-state)")
        if "tile8" in tile_results and "tile16" in tile_results:
            r8 = tile_results["tile8"]["stable_mean_ms"]
            r16 = tile_results["tile16"]["stable_mean_ms"]
            print(f"  tile8:  {r8:.2f}ms  tile16: {r16:.2f}ms  Ratio: {r16/r8:.4f}x (steady-state)")

        results["scenes"][scene_id] = {
            "scene_info": {
                "scene_id": scene_id,
                "scene_path": scene_path,
                "camera_path": camera_path,
                "num_gaussians": sinfo["num_gaussians"],
                "resolution_used": list(target_res),
            },
            "tile_results": tile_results,
        }

        del scene_data, cameras
        gc.collect()
        torch.cuda.empty_cache()

    # Save
    output_path = output_dir / f"bicycle_rerun_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"\n{'='*70}")
    print(f"  Saved: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
