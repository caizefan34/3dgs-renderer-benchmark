#!/usr/bin/env python3
"""
EPIC-05 Complete Optimization Study Runner.

Orchestrates the full experimental pipeline:
1. Environment recording
2. Scene generation
3. Baseline establishment
4. Ablation experiments (single modules)
5. Forward/backward profiling
6. Interaction experiments (2-way, 3-way)
7. Statistical aggregation
8. Report generation

Usage:
    python scripts/epic05/run_optimization_all.py --stage all
    python scripts/epic05/run_optimization_all.py --stage ablation
    python scripts/epic05/run_optimization_all.py --output results/epic05/raw
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import date
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
import os
os.environ["PATH"] = os.path.dirname(sys.executable) + ":" + os.environ.get("PATH", "")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

OUTPUT_DIR = REPO_ROOT / "results" / "epic05"
RAW_DIR = OUTPUT_DIR / "raw"
AGG_DIR = OUTPUT_DIR / "aggregated"
PROFILE_DIR = OUTPUT_DIR / "profiles"

# ---------------------------------------------------------------------------
# Scene definitions
# ---------------------------------------------------------------------------
SCENES = {
    "50k": {"gaussians": 50000, "label": "50K"},
    "200k": {"gaussians": 200000, "label": "200K"},
    "400k": {"gaussians": 400000, "label": "400K"},
}

# ---------------------------------------------------------------------------
# Optimization module definitions
# Each module has a unique ID, name, description, and a function that
# returns the benchmark arguments for that module.
# ---------------------------------------------------------------------------
BASELINE_PARAMS = {
    "tile_size": 16,
    "packed": True,
    "sh_degree": 3,
    "radius_clip": 0.0,
    "eps2d": 0.1,
}

OPTIMIZATION_MODULES: Dict[str, Dict[str, Any]] = {
    "M0": {
        "name": "baseline",
        "description": "gsplat baseline (tile16, packed, SH3)",
        "params": BASELINE_PARAMS.copy(),
    },
    # --- Tile size variants ---
    "M1a": {
        "name": "tile8",
        "description": "tile_size=8 (fine-grained tiles)",
        "params": {**BASELINE_PARAMS, "tile_size": 8},
    },
    "M1b": {
        "name": "tile16",
        "description": "tile_size=16 (default)",
        "params": {**BASELINE_PARAMS, "tile_size": 16},
    },
    "M1c": {
        "name": "tile32",
        "description": "tile_size=32 (coarse tiles)",
        "params": {**BASELINE_PARAMS, "tile_size": 32},
    },
    # --- Packed mode ---
    "M2a": {
        "name": "packed_true",
        "description": "packed=True",
        "params": {**BASELINE_PARAMS, "packed": True},
    },
    "M2b": {
        "name": "packed_false",
        "description": "packed=False (dense)",
        "params": {**BASELINE_PARAMS, "packed": False},
    },
    # --- SH degree ---
    "M3a": {
        "name": "sh0",
        "description": "SH degree=0 (DC only)",
        "params": {**BASELINE_PARAMS, "sh_degree": 0},
    },
    "M3b": {
        "name": "sh1",
        "description": "SH degree=1 (4 coeffs)",
        "params": {**BASELINE_PARAMS, "sh_degree": 1},
    },
    "M3c": {
        "name": "sh3",
        "description": "SH degree=3 (16 coeffs, full)",
        "params": {**BASELINE_PARAMS, "sh_degree": 3},
    },
    # --- Radius clip ---
    "M4a": {
        "name": "rclip_none",
        "description": "radius_clip=0 (none)",
        "params": {**BASELINE_PARAMS, "radius_clip": 0.0},
    },
    "M4b": {
        "name": "rclip_0_001",
        "description": "radius_clip=0.001 (aggressive)",
        "params": {**BASELINE_PARAMS, "radius_clip": 0.001},
    },
    "M4c": {
        "name": "rclip_0_01",
        "description": "radius_clip=0.01",
        "params": {**BASELINE_PARAMS, "radius_clip": 0.01},
    },
    # --- Epsilon 2D ---
    "M5a": {
        "name": "eps2d_0_01",
        "description": "eps2d=0.01 (sharp)",
        "params": {**BASELINE_PARAMS, "eps2d": 0.01},
    },
    "M5b": {
        "name": "eps2d_0_1",
        "description": "eps2d=0.1 (default)",
        "params": {**BASELINE_PARAMS, "eps2d": 0.1},
    },
    "M5c": {
        "name": "eps2d_0_5",
        "description": "eps2d=0.5 (blurrier)",
        "params": {**BASELINE_PARAMS, "eps2d": 0.5},
    },
}

# Interaction experiments
INTERACTION_EXPERIMENTS = [
    {"id": "I1", "modules": ["M0"], "description": "baseline"},
    {"id": "I2", "modules": ["M1a", "M2b"], "description": "tile8 + packed"},
    {"id": "I3", "modules": ["M1b", "M2b"], "description": "tile16 + packed"},
    {"id": "I4", "modules": ["M1c", "M2b"], "description": "tile32 + packed"},
    {"id": "I5", "modules": ["M1a", "M3a"], "description": "tile8 + SH0"},
    {"id": "I6", "modules": ["M1b", "M3a"], "description": "tile16 + SH0"},
    {"id": "I7", "modules": ["M2b", "M3a"], "description": "dense + SH0"},
    {"id": "I8", "modules": ["M1a", "M2b", "M3a"], "description": "tile8 + dense + SH0"},
    {"id": "I9", "modules": ["M1b", "M2b", "M3a"], "description": "tile16 + dense + SH0"},
    {"id": "I10", "modules": ["M1a", "M4b"], "description": "tile8 + rclip"},
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _get_hardware_info() -> Dict[str, Any]:
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["compute_capability"] = f"{props.major}.{props.minor}"
        info["total_vram_mb"] = round(props.total_memory / (1024 * 1024), 1)
        info["multi_processor_count"] = props.multi_processor_count
    # Try nvidia-smi for driver version
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        info["driver_version"] = r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        info["driver_version"] = "unknown"
    return info


def make_scene(gaussians: int) -> str:
    """Generate a synthetic PLY scene."""
    label = f"{gaussians // 1000}k"
    path = REPO_ROOT / "data" / f"scene_{label}.ply"
    if not path.exists():
        print(f"  Generating {label} scene ({gaussians} Gaussians)...")
        subprocess.run(
            [sys.executable, "src/scripts/generate_scene.py",
             "--gaussians", str(gaussians), "--output", str(path)],
            cwd=REPO_ROOT, check=True, timeout=60, capture_output=True
        )
        print(f"  Created {path}")
    return str(path)


def build_gsplat_call_params(scene_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Build a parameter dict for the custom benchmark."""
    return {
        "scene_path": scene_path,
        "tile_size": params["tile_size"],
        "packed": params["packed"],
        "sh_degree": params["sh_degree"],
        "radius_clip": params["radius_clip"],
        "eps2d": params["eps2d"],
    }


def run_custom_benchmark(
    call_params: Dict[str, Any],
    resolution: str = "1080p",
    frames: int = 100,
    warmup: int = 30,
    repeats: int = 3,
) -> Dict[str, Any]:
    """
    Run a custom benchmark through direct gsplat API calls.
    This gives us precise control over all parameters including
    tile_size, packed mode, SH degree, etc.
    """
    from gsplat import rasterization
    from src.benchmark_framework import load_ply, load_cameras_from_json, resize_cameras

    device = "cuda"

    # Load scene
    scene_data = load_ply(call_params["scene_path"], device=device)

    # Activate parameters
    xyz = scene_data["xyz"]
    quats = torch.nn.functional.normalize(scene_data["rotations"], dim=-1).contiguous()
    scales_activated = torch.exp(scene_data["scales"]).contiguous()
    opacities_activated = torch.sigmoid(scene_data["opacity"]).squeeze(-1).contiguous()
    shs = scene_data["shs"].contiguous()  # Shape [N, K, 3]

    # Handle SH degree
    sh_degree = call_params["sh_degree"]
    if sh_degree < 3:
        # Keep only the first (sh_degree+1)^2 coefficients
        n_coeffs = (sh_degree + 1) ** 2
        shs = shs[:, :n_coeffs, :].contiguous()  # Keep [N, K, 3] shape
    # After slicing, reshape to flat channels

    # Load cameras
    cameras_path = REPO_ROOT / "data" / "camera_presets" / "circle.json"
    cameras = load_cameras_from_json(str(cameras_path), device=device)

    # Set resolution
    res_map = {"720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160)}
    width, height = res_map.get(resolution, (1920, 1080))
    cameras = resize_cameras(cameras, width, height)

    # Warmup
    cam = cameras[0]
    for _ in range(warmup):
        with torch.no_grad():
            rendered, alpha, info = rasterization(
                means=xyz, quats=quats, scales=scales_activated,
                opacities=opacities_activated, colors=shs,
                viewmats=cam.viewmatrix.unsqueeze(0),
                Ks=cam.K.unsqueeze(0),
                width=width, height=height,
                tile_size=call_params["tile_size"],
                packed=call_params["packed"],
                radius_clip=call_params["radius_clip"],
                eps2d=call_params["eps2d"],
                sh_degree=sh_degree,
                render_mode="RGB",
            )
    torch.cuda.synchronize()

    # Timed runs
    all_times = []
    all_peak_mem = 0
    all_mem_samples = []

    for repeat in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        for frame_idx in range(frames):
            cam = cameras[frame_idx % len(cameras)]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                rendered, alpha, info = rasterization(
                    means=xyz, quats=quats, scales=scales_activated,
                    opacities=opacities_activated, colors=shs,
                    viewmats=cam.viewmatrix.unsqueeze(0),
                    Ks=cam.K.unsqueeze(0),
                    width=width, height=height,
                    tile_size=call_params["tile_size"],
                    packed=call_params["packed"],
                    radius_clip=call_params["radius_clip"],
                    eps2d=call_params["eps2d"],
                    sh_degree=sh_degree,
                    render_mode="RGB",
                )
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end)
            all_times.append(elapsed)
            mem = torch.cuda.memory_allocated() / (1024 * 1024)
            all_mem_samples.append(mem)
            if mem > all_peak_mem:
                all_peak_mem = mem
        all_peak_mem = max(all_peak_mem, torch.cuda.max_memory_allocated() / (1024 * 1024))

    t_arr = np.array(all_times)
    results = {
        "num_frames": frames * repeats,
        "warmup": warmup,
        "repeats": repeats,
        "resolution": f"{width}x{height}",
        "mean_ms": float(np.mean(t_arr)),
        "median_ms": float(np.median(t_arr)),
        "std_ms": float(np.std(t_arr)),
        "min_ms": float(np.min(t_arr)),
        "max_ms": float(np.max(t_arr)),
        "p1_ms": float(np.percentile(t_arr, 1)),
        "p5_ms": float(np.percentile(t_arr, 5)),
        "p95_ms": float(np.percentile(t_arr, 95)),
        "p99_ms": float(np.percentile(t_arr, 99)),
        "mean_fps": float(1000.0 / np.mean(t_arr)),
        "median_fps": float(1000.0 / np.median(t_arr)),
        "peak_vram_mb": round(all_peak_mem, 1),
        "avg_vram_mb": round(float(np.mean(all_mem_samples)), 1),
        "num_gaussians": xyz.shape[0],
        "config": call_params.copy(),
    }
    return results


def run_single_ablation(
    module_id: str,
    module_config: Dict[str, Any],
    scene_path: str,
    output_dir: Path,
    resolution: str = "1080p",
    frames: int = 100,
    warmup: int = 30,
    repeats: int = 3,
) -> Dict[str, Any]:
    """Run a single ablation experiment."""
    label = f"{module_id}_{module_config['name']}"
    print(f"\n  [{label}] {module_config['description']}")
    print(f"  Params: tile_size={module_config['params']['tile_size']}, "
          f"packed={module_config['params']['packed']}, "
          f"sh_degree={module_config['params']['sh_degree']}, "
          f"eps2d={module_config['params']['eps2d']}, "
          f"rclip={module_config['params']['radius_clip']}")

    call_params = build_gsplat_call_params(scene_path, module_config["params"])
    result = run_custom_benchmark(
        call_params, resolution=resolution,
        frames=frames, warmup=warmup, repeats=repeats,
    )
    result["module_id"] = module_id
    result["module_name"] = module_config["name"]
    result["description"] = module_config["description"]
    result["label"] = label

    # Save raw
    raw_path = output_dir / f"{label}.json"
    with open(raw_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Mean: {result['mean_ms']:.2f}ms ({result['mean_fps']:.1f}FPS)  "
          f"VRAM: {result['peak_vram_mb']:.0f}MB  P99: {result['p99_ms']:.2f}ms")

    return result


# ---------------------------------------------------------------------------
# Main experiment stages
# ---------------------------------------------------------------------------

STAGE_ABLATION = "ablation"
STAGE_INTERACTION = "interaction"
STAGE_ALL = "all"
STAGE_ENV = "env"


def stage_environment(output_dir: Path) -> Dict[str, Any]:
    """Record environment metadata."""
    print("\n=== Recording Environment ===")
    env_info = _get_hardware_info()
    env_info["git_commit"] = _get_git_commit()
    env_info["date"] = date.today().isoformat()
    env_info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    env_info["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS", "1")

    # Add nvidia-smi topo
    try:
        r = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=10
        )
        env_info["gpu_topology"] = r.stdout
    except Exception:
        env_info["gpu_topology"] = "unavailable"

    env_path = output_dir / "environment.json"
    with open(env_path, "w") as f:
        json.dump(env_info, f, indent=2)
    print(f"  Environment saved to {env_path}")
    print(f"  GPU: {env_info.get('gpu_name', 'N/A')}")
    print(f"  CUDA: {env_info.get('cuda_version', 'N/A')}")
    print(f"  Driver: {env_info.get('driver_version', 'N/A')}")
    print(f"  PyTorch: {env_info.get('pytorch_version', 'N/A')}")
    print(f"  Commit: {env_info.get('git_commit', 'N/A')}")
    return env_info


def stage_ablation(
    scene_key: str = "50k",
    resolution: str = "1080p",
    frames: int = 100,
    warmup: int = 30,
    repeats: int = 3,
    output_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run all single-module ablation experiments."""
    if output_dir is None:
        output_dir = REPO_ROOT / "results" / "epic05" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_config = SCENES[scene_key]
    scene_path = make_scene(scene_config["gaussians"])

    print(f"\n{'=' * 70}")
    print(f"  Ablation Experiments: {scene_key} ({resolution})")
    print(f"  Scene: {scene_path}")
    print(f"  Protocol: {warmup} warmup, {frames} frames, {repeats} repeats")
    print(f"{'=' * 70}")

    results = []
    # Run baseline first, then all modules
    module_keys = list(OPTIMIZATION_MODULES.keys())

    for module_id in module_keys:
        result = run_single_ablation(
            module_id, OPTIMIZATION_MODULES[module_id],
            scene_path, output_dir,
            resolution=resolution, frames=frames, warmup=warmup, repeats=repeats,
        )
        results.append(result)

    # Create aggregated ablation table
    agg = {
        "experiment_type": "ablation",
        "scene_key": scene_key,
        "resolution": resolution,
        "protocol": {"warmup": warmup, "frames": frames, "repeats": repeats},
        "date": date.today().isoformat(),
        "baseline_id": "M0",
        "baseline_name": "baseline",
        "results": results,
    }

    agg_path = output_dir.parent / "aggregated" / f"ablation_{scene_key}_{resolution}.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n  Ablation aggregated results saved to {agg_path}")

    return results


def stage_interaction(
    scene_key: str = "50k",
    resolution: str = "1080p",
    frames: int = 100,
    warmup: int = 30,
    repeats: int = 3,
    output_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Run interaction experiments (combinations of modules)."""
    if output_dir is None:
        output_dir = REPO_ROOT / "results" / "epic05" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_config = SCENES[scene_key]
    scene_path = make_scene(scene_config["gaussians"])

    print(f"\n{'=' * 70}")
    print(f"  Interaction Experiments: {scene_key} ({resolution})")
    print(f"  Protocol: {warmup} warmup, {frames} frames, {repeats} repeats")
    print(f"{'=' * 70}")

    results = []
    for exp in INTERACTION_EXPERIMENTS:
        # Merge params from all modules
        merged_params = BASELINE_PARAMS.copy()
        module_names = []
        for mod_id in exp["modules"]:
            if mod_id in OPTIMIZATION_MODULES:
                merged_params.update(OPTIMIZATION_MODULES[mod_id]["params"])
                module_names.append(OPTIMIZATION_MODULES[mod_id]["name"])

        label = f"I{exp['id']}_{'_'.join(module_names)}"
        print(f"\n  [{exp['id']}] {exp['description']}")
        print(f"  Combined params: tile_size={merged_params['tile_size']}, "
              f"packed={merged_params['packed']}, "
              f"sh_degree={merged_params['sh_degree']}")

        call_params = build_gsplat_call_params(scene_path, merged_params)
        result = run_custom_benchmark(
            call_params, resolution=resolution,
            frames=frames, warmup=warmup, repeats=repeats,
        )
        result["experiment_id"] = exp["id"]
        result["description"] = exp["description"]
        result["modules"] = exp["modules"]
        result["module_names"] = module_names
        result["label"] = label

        raw_path = output_dir / f"{label}.json"
        with open(raw_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Mean: {result['mean_ms']:.2f}ms ({result['mean_fps']:.1f}FPS)  "
              f"VRAM: {result['peak_vram_mb']:.0f}MB")
        results.append(result)

    # Create aggregated interaction table
    agg = {
        "experiment_type": "interaction",
        "scene_key": scene_key,
        "resolution": resolution,
        "protocol": {"warmup": warmup, "frames": frames, "repeats": repeats},
        "date": date.today().isoformat(),
        "results": results,
    }
    agg_path = output_dir.parent / "aggregated" / f"interaction_{scene_key}_{resolution}.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n  Interaction results saved to {agg_path}")

    return results


def stage_profile_forward_backward(
    scene_key: str = "50k",
    resolution: str = "1080p",
    frames: int = 50,
    warmup: int = 20,
) -> Dict[str, Any]:
    """Profile forward vs backward pass timing."""
    scene_config = SCENES[scene_key]
    scene_path = make_scene(scene_config["gaussians"])

    print(f"\n{'=' * 70}")
    print(f"  Forward/Backward Profiling: {scene_key} ({resolution})")
    print(f"{'=' * 70}")

    from gsplat import rasterization
    from src.benchmark_framework import load_ply, load_cameras_from_json, resize_cameras

    device = "cuda"

    scene_data = load_ply(scene_path, device=device)
    xyz = scene_data["xyz"].requires_grad_(True)
    quats = torch.nn.functional.normalize(scene_data["rotations"], dim=-1).contiguous().requires_grad_(True)
    scales = scene_data["scales"].contiguous().requires_grad_(True)
    opacities = scene_data["opacity"].contiguous().requires_grad_(True)
    shs = scene_data["shs"].contiguous()  # Shape [N, K, 3].requires_grad_(True)

    # Make scales activatable for autograd
    scales_activated = torch.exp(scales)
    opacities_activated = torch.sigmoid(opacities)
    quats_normed = torch.nn.functional.normalize(quats, dim=-1)

    cameras_path = REPO_ROOT / "data" / "camera_presets" / "circle.json"
    cameras = load_cameras_from_json(str(cameras_path), device=device)
    res_map = {"720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160)}
    width, height = res_map.get(resolution, (1920, 1080))
    cameras = resize_cameras(cameras, width, height)

    all_fwd_times = []
    all_bwd_times = []
    all_total_times = []

    for tile_size in [8, 16, 32]:
        fwd_times = []
        bwd_times = []
        total_times = []

        print(f"\n  tile_size={tile_size}:")

        # Warmup
        for _ in range(warmup):
            cam = cameras[0]
            with torch.no_grad():
                rendered, alpha, info = rasterization(
                means=xyz, quats=quats_normed, scales=scales_activated,
                opacities=opacities_activated, colors=shs,
                viewmats=cam.viewmatrix.unsqueeze(0),
                Ks=cam.K.unsqueeze(0),
                width=width, height=height,
                tile_size=tile_size, packed=True, render_mode="RGB", sh_degree=3,
            )
            # Warmup - no backward needed
            pass
        torch.cuda.synchronize()

        # Timed
        for i in range(frames):
            cam = cameras[i % len(cameras)]

            # Create fresh tensors every iteration
            fresh_xyz = xyz.detach().clone().requires_grad_(True)
            fresh_quats = quats.detach().clone().requires_grad_(True)
            fresh_scales = scales.detach().clone().requires_grad_(True)
            fresh_opacities = opacities.detach().clone().requires_grad_(True)
            fresh_shs = shs.detach().clone().requires_grad_(True)
            
            fresh_qn = torch.nn.functional.normalize(fresh_quats, dim=-1)
            fresh_sa = torch.exp(fresh_scales)
            fresh_oa = torch.sigmoid(fresh_opacities).squeeze(-1)

            # Forward
            torch.cuda.synchronize()
            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            fwd_start.record()
            rendered, alpha, info = rasterization(
                means=fresh_xyz, quats=fresh_qn, scales=fresh_sa,
                opacities=fresh_oa, colors=fresh_shs,
                viewmats=cam.viewmatrix.unsqueeze(0),
                Ks=cam.K.unsqueeze(0),
                width=width, height=height,
                tile_size=tile_size, packed=True, render_mode="RGB", sh_degree=3,
            )
            fwd_end.record()
            fwd_end.synchronize()
            fwd_time = fwd_start.elapsed_time(fwd_end)
            fwd_times.append(fwd_time)

            # Backward
            bwd_start = torch.cuda.Event(enable_timing=True)
            bwd_end = torch.cuda.Event(enable_timing=True)
            loss = rendered.sum()
            bwd_start.record()
            loss.backward()
            bwd_end.record()
            bwd_end.synchronize()
            bwd_time = bwd_start.elapsed_time(bwd_end)
            bwd_times.append(bwd_time)

            total_times.append(fwd_time + bwd_time)

        avg_fwd = float(np.mean(fwd_times))
        avg_bwd = float(np.mean(bwd_times))
        avg_total = float(np.mean(total_times))
        print(f"    Forward:  mean={avg_fwd:.2f}ms ({avg_fwd/avg_total*100:.1f}%)")
        print(f"    Backward: mean={avg_bwd:.2f}ms ({avg_bwd/avg_total*100:.1f}%)")
        print(f"    Total:    mean={avg_total:.2f}ms")

        all_fwd_times.append({"tile_size": tile_size, "mean_ms": avg_fwd, "times": fwd_times})
        all_bwd_times.append({"tile_size": tile_size, "mean_ms": avg_bwd, "times": bwd_times})
        all_total_times.append({"tile_size": tile_size, "mean_ms": avg_total})

    profile = {
        "scene_key": scene_key,
        "resolution": resolution,
        "num_gaussians": xyz.shape[0],
        "fwd": all_fwd_times,
        "bwd": all_bwd_times,
        "total": all_total_times,
    }

    prof_path = PROFILE_DIR / f"fwd_bwd_profile_{scene_key}_{resolution}.json"
    prof_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prof_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"\n  Profile saved to {prof_path}")

    return profile


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EPIC-05 Optimization Study")
    parser.add_argument("--stage", choices=[STAGE_ABLATION, STAGE_INTERACTION, STAGE_ALL, STAGE_ENV],
                        default=STAGE_ALL, help="Experiment stage to run")
    parser.add_argument("--scene", choices=list(SCENES.keys()), default="50k",
                        help="Scene size")
    parser.add_argument("--resolution", choices=["720p", "1080p", "4k"], default="1080p")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output) if args.output else RAW_DIR

    stages = []
    if args.stage == STAGE_ALL:
        stages = [STAGE_ENV, STAGE_ABLATION, STAGE_INTERACTION]
    else:
        stages = [args.stage]

    results = {}
    for stage in stages:
        if stage == STAGE_ENV:
            results["env"] = stage_environment(output_dir)
        elif stage == STAGE_ABLATION:
            results["ablation"] = stage_ablation(
                args.scene, args.resolution, args.frames, args.warmup, args.repeats, output_dir
            )
            # Also profile forward/backward
            profile = stage_profile_forward_backward(args.scene, args.resolution, args.frames // 2, args.warmup)
            results["profile"] = profile
        elif stage == STAGE_INTERACTION:
            results["interaction"] = stage_interaction(
                args.scene, args.resolution, args.frames, args.warmup, args.repeats, output_dir
            )

    # Summary
    print(f"\n{'=' * 70}")
    print("  EPIC-05 Optimization Study Complete")
    print(f"{'=' * 70}")

    if "ablation" in results:
        print(f"\n  Ablation: {len(results['ablation'])} experiments")
    if "interaction" in results:
        print(f"\n  Interaction: {len(results['interaction'])} experiments")
    print(f"\n  Results saved to {output_dir}/")
    print(f"  Aggregated results saved to {AGG_DIR}/")


if __name__ == "__main__":
    main()
