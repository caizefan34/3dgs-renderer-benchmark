#!/usr/bin/env python3
"""
EPIC-05 Phase 2 Experiments: Final Validation Suite.

This script runs the complete Phase 2 validation experiments:

1. 400K M0 anomaly re-verification (cold/warm start protocol)
2. 400K forward/backward profiling (tile8/16/32)
3. Training pipeline validation (tile16 vs tile32)
4. Interaction re-check with validated protocol
5. Workload scaling analysis

Usage:
    python scripts/epic05/phase2_experiments.py --stage m0_verification
    python scripts/epic05/phase2_experiments.py --stage fwd_bwd_400k
    python scripts/epic05/phase2_experiments.py --stage training
    python scripts/epic05/phase2_experiments.py --stage all

Output:
    results/epic05/final_validation/
        raw/
        aggregated/
        profiles/
        training/
        statistics/
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
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

os.environ["PATH"] = os.path.dirname(sys.executable) + ":" + os.environ.get("PATH", "")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

EPIC05_DIR = REPO_ROOT / "results" / "epic05"
FINAL_DIR = EPIC05_DIR / "final_validation"
RAW_DIR = FINAL_DIR / "raw"
AGG_DIR = FINAL_DIR / "aggregated"
PROFILE_DIR = FINAL_DIR / "profiles"
TRAINING_DIR = FINAL_DIR / "training"
STATS_DIR = FINAL_DIR / "statistics"

# ---------------------------------------------------------------------------
# Scene definitions
# ---------------------------------------------------------------------------
SCENES = {
    "50k": {"gaussians": 50000, "label": "50K"},
    "200k": {"gaussians": 200000, "label": "200K"},
    "400k": {"gaussians": 400000, "label": "400K"},
}

# ---------------------------------------------------------------------------
# Baseline config
# ---------------------------------------------------------------------------
BASELINE_PARAMS = {
    "tile_size": 16,
    "packed": True,
    "sh_degree": 3,
    "radius_clip": 0.0,
    "eps2d": 0.1,
}


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
            cwd=REPO_ROOT, check=True, timeout=120, capture_output=True
        )
        print(f"  Created {path}")
    return str(path)


def load_scene(scene_path: str, device: str = "cuda"):
    """Load a PLY scene and return torch tensors."""
    from src.benchmark_framework import load_ply
    return load_ply(scene_path, device=device)


def load_cameras(resolution: str = "1080p", device: str = "cuda"):
    """Load camera presets."""
    from src.benchmark_framework import load_cameras_from_json, resize_cameras
    cameras_path = REPO_ROOT / "data" / "camera_presets" / "circle.json"
    cameras = load_cameras_from_json(str(cameras_path), device=device)
    res_map = {"720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160)}
    width, height = res_map.get(resolution, (1920, 1080))
    cameras = resize_cameras(cameras, width, height)
    return cameras, width, height


def build_experiment_id() -> str:
    """Build unique experiment ID."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"epic05_phase2_{ts}"


# ===========================================================================
# EXPERIMENT 1: 400K M0 Anomaly Verification
# ===========================================================================

def stage_m0_verification(
    scene_key: str = "400k",
    resolution: str = "1080p",
    frames: int = 100,
    warmup: int = 30,
    repeats: int = 3,
) -> Dict[str, Any]:
    """
    Re-verify the 400K M0 anomaly with explicit cold/warm start protocol.

    Protocol:
        Phase 1 (Cold start - first run after import):
            Run M0 config (tile16, packed, SH3)
            Record CUDA init, JIT compile, first frame time
        Phase 2 (Warm start - same config after warmup):
            Run M2a config (same params, but after JIT cache)
        Phase 3:
            Run M0 again (now JIT cached)
        Phase 4:
            Run M2a again

    All phases record per-frame timing to distinguish first-frame vs steady-state.
    """
    print(f"\n{'=' * 70}")
    print(f"  STAGE: 400K M0 Anomaly Verification")
    print(f"  Scene: {scene_key}, Resolution: {resolution}")
    print(f"{'=' * 70}")

    scene_path = make_scene(SCENES[scene_key]["gaussians"])
    cameras, width, height = load_cameras(resolution)
    device = "cuda"

    # Load scene data once
    scene_data = load_scene(scene_path, device=device)
    xyz = scene_data["xyz"]
    quats = torch.nn.functional.normalize(scene_data["rotations"], dim=-1).contiguous()
    scales_activated = torch.exp(scene_data["scales"]).contiguous()
    opacities_activated = torch.sigmoid(scene_data["opacity"]).squeeze(-1).contiguous()
    shs = scene_data["shs"].contiguous()

    from gsplat import rasterization

    # =========================================================
    # Define the 4 phases (M0/M2a/M0/M2a)
    # =========================================================
    phases = [
        {"id": "M0_cold", "label": "M0 (cold start)", "params": BASELINE_PARAMS.copy()},
        {"id": "M2a_cold", "label": "M2a (first warm)", "params": {**BASELINE_PARAMS, "packed": True}},
        {"id": "M0_warm", "label": "M0 (warm)", "params": BASELINE_PARAMS.copy()},
        {"id": "M2a_warm", "label": "M2a (warm)", "params": {**BASELINE_PARAMS, "packed": True}},
    ]

    all_phase_results = []
    detailed_frame_log = {"phases": [], "metadata": {}}

    for phase_idx, phase in enumerate(phases):
        p = phase["params"]
        print(f"\n  --- Phase {phase_idx + 1}: {phase['label']} ---")
        print(f"    tile_size={p['tile_size']}, packed={p['packed']}, "
              f"sh_degree={p['sh_degree']}")

        # Track initialization overhead
        init_times = {}

        # First frame timing (no CUDA cache)
        cam = cameras[0]
        torch.cuda.synchronize()
        start_init = torch.cuda.Event(enable_timing=True)
        end_init = torch.cuda.Event(enable_timing=True)
        start_init.record()
        rendered_first, alpha_first, info_first = rasterization(
            means=xyz, quats=quats, scales=scales_activated,
            opacities=opacities_activated, colors=shs,
            viewmats=cam.viewmatrix.unsqueeze(0),
            Ks=cam.K.unsqueeze(0),
            width=width, height=height,
            tile_size=p["tile_size"],
            packed=p["packed"],
            radius_clip=p["radius_clip"],
            eps2d=p["eps2d"],
            sh_degree=p["sh_degree"],
            render_mode="RGB",
        )
        end_init.record()
        end_init.synchronize()
        init_times["first_frame_ms"] = start_init.elapsed_time(end_init)
        print(f"    First frame: {init_times['first_frame_ms']:.2f}ms")

        # Warmup
        warmup_times = []
        for w in range(warmup):
            cam_w = cameras[w % len(cameras)]
            with torch.no_grad():
                torch.cuda.synchronize()
                ws = torch.cuda.Event(enable_timing=True)
                we = torch.cuda.Event(enable_timing=True)
                ws.record()
                rendered, alpha, info = rasterization(
                    means=xyz, quats=quats, scales=scales_activated,
                    opacities=opacities_activated, colors=shs,
                    viewmats=cam_w.viewmatrix.unsqueeze(0),
                    Ks=cam_w.K.unsqueeze(0),
                    width=width, height=height,
                    tile_size=p["tile_size"],
                    packed=p["packed"],
                    radius_clip=p["radius_clip"],
                    eps2d=p["eps2d"],
                    sh_degree=p["sh_degree"],
                    render_mode="RGB",
                )
                we.record()
                we.synchronize()
                warmup_times.append(ws.elapsed_time(we))
        torch.cuda.synchronize()

        if warmup_times:
            init_times["warmup_mean_ms"] = float(np.mean(warmup_times))
            init_times["warmup_first_ms"] = warmup_times[0] if warmup_times else 0
            init_times["warmup_last_ms"] = warmup_times[-1] if warmup_times else 0

        print(f"    Warmup: first={warmup_times[0]:.2f}ms, last={warmup_times[-1]:.2f}ms, "
              f"mean={np.mean(warmup_times):.2f}ms")

        # Measured frames
        frame_times = []
        for frame_idx in range(frames):
            cam_f = cameras[frame_idx % len(cameras)]
            with torch.no_grad():
                torch.cuda.synchronize()
                fs = torch.cuda.Event(enable_timing=True)
                fe = torch.cuda.Event(enable_timing=True)
                fs.record()
                rendered, alpha, info = rasterization(
                    means=xyz, quats=quats, scales=scales_activated,
                    opacities=opacities_activated, colors=shs,
                    viewmats=cam_f.viewmatrix.unsqueeze(0),
                    Ks=cam_f.K.unsqueeze(0),
                    width=width, height=height,
                    tile_size=p["tile_size"],
                    packed=p["packed"],
                    radius_clip=p["radius_clip"],
                    eps2d=p["eps2d"],
                    sh_degree=p["sh_degree"],
                    render_mode="RGB",
                )
                fe.record()
                fe.synchronize()
                frame_times.append(fs.elapsed_time(fe))

        t_arr = np.array(frame_times)
        result = {
            "phase_id": phase["id"],
            "phase_label": phase["label"],
            "params": p,
            "num_frames": frames,
            "warmup": warmup,
            "first_frame_ms": init_times.get("first_frame_ms", 0),
            "warmup_mean_ms": init_times.get("warmup_mean_ms", 0),
            "warmup_first_ms": init_times.get("warmup_first_ms", 0),
            "warmup_last_ms": init_times.get("warmup_last_ms", 0),
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
            "all_times_ms": [float(t) for t in t_arr],
        }
        all_phase_results.append(result)

        print(f"    Measured: mean={result['mean_ms']:.2f}ms, "
              f"median={result['median_ms']:.2f}ms, "
              f"std={result['std_ms']:.2f}ms")
        print(f"    FPS: {result['mean_fps']:.1f}")
        print(f"    Range: [{result['min_ms']:.2f}, {result['max_ms']:.2f}]")

    # Record overhead between phases
    metadata = {
        "experiment_id": build_experiment_id(),
        "date": date.today().isoformat(),
        "git_commit": _get_git_commit(),
        "scene_key": scene_key,
        "scene_path": scene_path,
        "resolution": resolution,
        "device": device,
        "num_gaussians": xyz.shape[0],
        "protocol": {"warmup": warmup, "frames": frames},
        "hypothesis": "M0 first-run is slower due to JIT compilation, not real performance diff",
    }

    experiment = {
        "experiment_type": "m0_anomaly_verification",
        "metadata": metadata,
        "environment": _get_hardware_info(),
        "phases": all_phase_results,
    }

    # Comparative analysis
    if len(all_phase_results) >= 4:
        m0_cold = all_phase_results[0]
        m2a_cold = all_phase_results[1]
        m0_warm = all_phase_results[2]
        m2a_warm = all_phase_results[3]

        analysis = {
            "M0_cold_vs_M2a_cold_ratio": m0_cold["mean_ms"] / m2a_cold["mean_ms"],
            "M0_warm_vs_M2a_warm_ratio": m0_warm["mean_ms"] / m2a_warm["mean_ms"],
            "M0_cold_vs_M0_warm_ratio": m0_cold["mean_ms"] / m0_warm["mean_ms"],
            "M2a_cold_vs_M2a_warm_ratio": m2a_cold["mean_ms"] / m2a_warm["mean_ms"],
            "M0_cold_first_frame_ms": m0_cold["first_frame_ms"],
            "M0_warm_first_frame_ms": m0_warm["first_frame_ms"],
            "first_run_penalty_ms": m0_cold["mean_ms"] - m0_warm["mean_ms"],
            "first_run_penalty_percent": ((m0_cold["mean_ms"] - m0_warm["mean_ms"]) / m0_warm["mean_ms"]) * 100,
        }
        experiment["analysis"] = analysis
        print(f"\n  --- ANALYSIS ---")
        print(f"  M0 cold / M0 warm: {analysis['M0_cold_vs_M0_warm_ratio']:.2f}x")
        print(f"  M2a cold / M2a warm: {analysis['M2a_cold_vs_M2a_warm_ratio']:.2f}x")
        print(f"  M0 cold / M2a cold (original anomaly): {analysis['M0_cold_vs_M2a_cold_ratio']:.2f}x")
        print(f"  M0 warm / M2a warm (corrected): {analysis['M0_warm_vs_M2a_warm_ratio']:.2f}x")
        print(f"  First-run penalty: {analysis['first_run_penalty_ms']:.2f}ms "
              f"({analysis['first_run_penalty_percent']:.1f}%)")

    # Save
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "m0_verification.json"
    with open(out_path, "w") as f:
        json.dump(experiment, f, indent=2)
    print(f"\n  Saved to {out_path}")

    return experiment


# ===========================================================================
# EXPERIMENT 2: 400K Forward/Backward Profiling
# ===========================================================================

def stage_fwd_bwd_400k(
    scene_key: str = "400k",
    resolution: str = "1080p",
    frames: int = 50,
    warmup: int = 20,
) -> Dict[str, Any]:
    """
    Forward/backward profiling specifically for 400K scene.
    Tests tile8, tile16, tile32 with detailed breakdown.
    """
    print(f"\n{'=' * 70}")
    print(f"  STAGE: 400K Forward/Backward Profiling")
    print(f"  Scene: {scene_key}, Resolution: {resolution}")
    print(f"{'=' * 70}")

    scene_path = make_scene(SCENES[scene_key]["gaussians"])
    cameras, width, height = load_cameras(resolution)
    device = "cuda"

    from gsplat import rasterization

    # Load scene data with gradients
    scene_data = load_scene(scene_path, device=device)
    xyz = scene_data["xyz"].requires_grad_(True)
    quats = torch.nn.functional.normalize(scene_data["rotations"], dim=-1).contiguous().requires_grad_(True)
    scales = scene_data["scales"].contiguous().requires_grad_(True)
    opacities = scene_data["opacity"].contiguous().requires_grad_(True)
    shs = scene_data["shs"].contiguous().requires_grad_(True)

    tile_sizes = [8, 16, 32]
    all_results = {}

    for tile_size in tile_sizes:
        print(f"\n  --- tile_size={tile_size} ---")
        fwd_times = []
        bwd_times = []
        optimizer_times = []
        total_times = []
        peak_mems = []

        # Warmup
        for _ in range(warmup):
            cam_w = cameras[0]
            with torch.no_grad():
                rendered_w, alpha_w, info_w = rasterization(
                    means=xyz, quats=torch.nn.functional.normalize(quats, dim=-1),
                    scales=torch.exp(scales),
                    opacities=torch.sigmoid(opacities).squeeze(-1),
                    colors=shs,
                    viewmats=cam_w.viewmatrix.unsqueeze(0),
                    Ks=cam_w.K.unsqueeze(0),
                    width=width, height=height,
                    tile_size=tile_size, packed=True, render_mode="RGB", sh_degree=3,
                )
        torch.cuda.synchronize()

        # Timed runs
        for i in range(frames):
            cam = cameras[i % len(cameras)]

            # Fresh tensors for each iteration
            fresh_xyz = xyz.detach().clone().requires_grad_(True)
            fresh_quats = quats.detach().clone().requires_grad_(True)
            fresh_scales = scales.detach().clone().requires_grad_(True)
            fresh_opacities = opacities.detach().clone().requires_grad_(True)
            fresh_shs = shs.detach().clone().requires_grad_(True)

            fresh_qn = torch.nn.functional.normalize(fresh_quats, dim=-1)
            fresh_sa = torch.exp(fresh_scales)
            fresh_oa = torch.sigmoid(fresh_opacities).squeeze(-1)

            torch.cuda.reset_peak_memory_stats()

            # Forward
            torch.cuda.synchronize()
            fwd_s = torch.cuda.Event(enable_timing=True)
            fwd_e = torch.cuda.Event(enable_timing=True)
            fwd_s.record()
            rendered, alpha, info = rasterization(
                means=fresh_xyz, quats=fresh_qn, scales=fresh_sa,
                opacities=fresh_oa, colors=fresh_shs,
                viewmats=cam.viewmatrix.unsqueeze(0),
                Ks=cam.K.unsqueeze(0),
                width=width, height=height,
                tile_size=tile_size, packed=True, render_mode="RGB", sh_degree=3,
            )
            fwd_e.record()
            fwd_e.synchronize()
            fwd_t = fwd_s.elapsed_time(fwd_e)
            fwd_times.append(fwd_t)

            # Backward
            bwd_s = torch.cuda.Event(enable_timing=True)
            bwd_e = torch.cuda.Event(enable_timing=True)
            loss = rendered.sum()
            bwd_s.record()
            loss.backward()
            bwd_e.record()
            bwd_e.synchronize()
            bwd_t = bwd_s.elapsed_time(bwd_e)
            bwd_times.append(bwd_t)

            total_times.append(fwd_t + bwd_t)
            peak_mems.append(torch.cuda.max_memory_allocated() / (1024 * 1024))

        fwd_arr = np.array(fwd_times)
        bwd_arr = np.array(bwd_times)
        total_arr = np.array(total_times)

        result = {
            "tile_size": tile_size,
            "forward_ms": float(np.mean(fwd_arr)),
            "forward_std_ms": float(np.std(fwd_arr)),
            "forward_median_ms": float(np.median(fwd_arr)),
            "forward_p99_ms": float(np.percentile(fwd_arr, 99)),
            "backward_ms": float(np.mean(bwd_arr)),
            "backward_std_ms": float(np.std(bwd_arr)),
            "backward_median_ms": float(np.median(bwd_arr)),
            "backward_p99_ms": float(np.percentile(bwd_arr, 99)),
            "total_ms": float(np.mean(total_arr)),
            "forward_pct": float(np.mean(fwd_arr) / np.mean(total_arr) * 100),
            "backward_pct": float(np.mean(bwd_arr) / np.mean(total_arr) * 100),
            "peak_vram_mb": float(np.max(peak_mems)),
            "num_frames": frames,
            "warmup": warmup,
        }
        all_results[str(tile_size)] = result

        print(f"    Forward:  mean={result['forward_ms']:.2f}ms ({result['forward_pct']:.1f}%)")
        print(f"    Backward: mean={result['backward_ms']:.2f}ms ({result['backward_pct']:.1f}%)")
        print(f"    Total:    mean={result['total_ms']:.2f}ms")
        print(f"    Peak VRAM: {result['peak_vram_mb']:.0f}MB")

        # Clear GPU memory
        del fresh_xyz, fresh_quats, fresh_scales, fresh_opacities, fresh_shs
        del fresh_qn, fresh_sa, fresh_oa, rendered, alpha
        torch.cuda.empty_cache()
        gc.collect()

    # Comparative analysis
    if "8" in all_results and "16" in all_results and "32" in all_results:
        r8 = all_results["8"]
        r16 = all_results["16"]
        r32 = all_results["32"]

        print(f"\n  --- COMPARATIVE ANALYSIS ---")
        print(f"  tile8  -> tile16: forward {r8['forward_ms']:.1f} -> {r16['forward_ms']:.1f}ms "
              f"({r8['forward_ms']/r16['forward_ms']:.2f}x)")
        print(f"  tile16 -> tile32: forward {r16['forward_ms']:.1f} -> {r32['forward_ms']:.1f}ms "
              f"({r16['forward_ms']/r32['forward_ms']:.2f}x)")
        print(f"  tile8  -> tile32: forward {r8['forward_ms']:.1f} -> {r32['forward_ms']:.1f}ms "
              f"({r8['forward_ms']/r32['forward_ms']:.2f}x)")
        print(f"  tile16 -> tile32: backward {r16['backward_ms']:.1f} -> {r32['backward_ms']:.1f}ms "
              f"({r16['backward_ms']/r32['backward_ms']:.2f}x)")
        print(f"  tile16 -> tile32: total {r16['total_ms']:.1f} -> {r32['total_ms']:.1f}ms "
              f"({r16['total_ms']/r32['total_ms']:.2f}x)")

    experiment = {
        "experiment_type": "fwd_bwd_400k",
        "metadata": {
            "experiment_id": build_experiment_id(),
            "date": date.today().isoformat(),
            "git_commit": _get_git_commit(),
            "scene_key": scene_key,
            "resolution": resolution,
            "num_gaussians": xyz.shape[0],
            "protocol": {"warmup": warmup, "frames": frames},
        },
        "environment": _get_hardware_info(),
        "results": all_results,
        "tile_sizes": tile_sizes,
    }

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROFILE_DIR / "fwd_bwd_400k.json"
    with open(out_path, "w") as f:
        json.dump(experiment, f, indent=2)
    print(f"\n  Saved to {out_path}")

    return experiment


# ===========================================================================
# EXPERIMENT 3: Full Ablation with Corrected Protocol
# ===========================================================================

def stage_corrected_ablation(
    scene_key: str = "400k",
    resolution: str = "1080p",
    frames: int = 100,
    warmup: int = 30,
    repeats: int = 3,
) -> Dict[str, Any]:
    """
    Re-run 400K ablation with corrected warm-start protocol.
    Each config is run after a warmup phase, ensuring JIT cache is populated.
    """
    print(f"\n{'=' * 70}")
    print(f"  STAGE: Corrected Ablation (warm-start protocol)")
    print(f"  Scene: {scene_key}, Resolution: {resolution}")
    print(f"{'=' * 70}")

    scene_path = make_scene(SCENES[scene_key]["gaussians"])
    cameras, width, height = load_cameras(resolution)
    device = "cuda"

    from gsplat import rasterization

    scene_data = load_scene(scene_path, device=device)
    xyz = scene_data["xyz"]
    quats = torch.nn.functional.normalize(scene_data["rotations"], dim=-1).contiguous()
    scales_activated = torch.exp(scene_data["scales"]).contiguous()
    opacities_activated = torch.sigmoid(scene_data["opacity"]).squeeze(-1).contiguous()
    shs = scene_data["shs"].contiguous()

    # Configs to test: tile8, tile16, tile32 (corrected protocol)
    configs = [
        {"module_id": "M1a_validated", "label": "tile8", "tile_size": 8},
        {"module_id": "M1b_validated", "label": "tile16", "tile_size": 16},
        {"module_id": "M1c_validated", "label": "tile32", "tile_size": 32},
        {"module_id": "M0_validated", "label": "baseline", "tile_size": 16},
    ]

    results = []
    for cfg in configs:
        print(f"\n  --- {cfg['module_id']}: tile_size={cfg['tile_size']} ---")

        ts = cfg["tile_size"]
        all_times = []
        first_frame_times = []
        warmup_times_list = []

        for repeat in range(repeats):
            # First frame of this repeat
            cam = cameras[0]
            torch.cuda.synchronize()
            s0 = torch.cuda.Event(enable_timing=True)
            e0 = torch.cuda.Event(enable_timing=True)
            s0.record()
            with torch.no_grad():
                rendered, alpha, info = rasterization(
                    means=xyz, quats=quats, scales=scales_activated,
                    opacities=opacities_activated, colors=shs,
                    viewmats=cam.viewmatrix.unsqueeze(0),
                    Ks=cam.K.unsqueeze(0),
                    width=width, height=height,
                    tile_size=ts, packed=True, render_mode="RGB", sh_degree=3,
                )
            e0.record()
            e0.synchronize()
            first_frame_times.append(s0.elapsed_time(e0))

            # Warmup
            for w in range(warmup):
                cam_w = cameras[w % len(cameras)]
                with torch.no_grad():
                    rendered_w, alpha_w, info_w = rasterization(
                        means=xyz, quats=quats, scales=scales_activated,
                        opacities=opacities_activated, colors=shs,
                        viewmats=cam_w.viewmatrix.unsqueeze(0),
                        Ks=cam_w.K.unsqueeze(0),
                        width=width, height=height,
                        tile_size=ts, packed=True, render_mode="RGB", sh_degree=3,
                    )
            torch.cuda.synchronize()

            # Measured frames
            for frame_idx in range(frames):
                cam_f = cameras[frame_idx % len(cameras)]
                with torch.no_grad():
                    torch.cuda.synchronize()
                    sf = torch.cuda.Event(enable_timing=True)
                    ef = torch.cuda.Event(enable_timing=True)
                    sf.record()
                    rendered_f, alpha_f, info_f = rasterization(
                        means=xyz, quats=quats, scales=scales_activated,
                        opacities=opacities_activated, colors=shs,
                        viewmats=cam_f.viewmatrix.unsqueeze(0),
                        Ks=cam_f.K.unsqueeze(0),
                        width=width, height=height,
                        tile_size=ts, packed=True, render_mode="RGB", sh_degree=3,
                    )
                    ef.record()
                    ef.synchronize()
                    all_times.append(sf.elapsed_time(ef))

            del rendered, alpha
            torch.cuda.empty_cache()

        t_arr = np.array(all_times)
        result = {
            "module_id": cfg["module_id"],
            "label": cfg["label"],
            "tile_size": ts,
            "num_frames": frames * repeats,
            "warmup": warmup,
            "repeats": repeats,
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
            "first_frame_mean_ms": float(np.mean(first_frame_times)),
            "peak_vram_mb": 0.0,  # Would need reset_peak_memory_stats
        }
        results.append(result)

        print(f"    First frame: {result['first_frame_mean_ms']:.2f}ms")
        print(f"    Steady-state: mean={result['mean_ms']:.2f}ms, "
              f"median={result['median_ms']:.2f}ms, std={result['std_ms']:.2f}ms")
        print(f"    FPS: {result['mean_fps']:.1f}")

    experiment = {
        "experiment_type": "corrected_ablation_400k",
        "metadata": {
            "experiment_id": build_experiment_id(),
            "date": date.today().isoformat(),
            "git_commit": _get_git_commit(),
            "scene_key": scene_key,
            "resolution": resolution,
            "protocol": {"warmup": warmup, "frames": frames, "repeats": repeats},
            "note": "Validated warm-start protocol - each config JIT-warmed before measurement",
        },
        "environment": _get_hardware_info(),
        "results": results,
    }

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = AGG_DIR / "corrected_ablation_400k.json"
    with open(out_path, "w") as f:
        json.dump(experiment, f, indent=2)
    print(f"\n  Saved to {out_path}")

    return experiment


# ===========================================================================
# EXPERIMENT 4: Training Pipeline Validation
# ===========================================================================

def stage_training_validation(
    scene_key: str = "50k",
    num_steps: int = 1000,
    validation_interval: int = 100,
) -> Dict[str, Any]:
    """
    Validate tile16 vs tile32 in a minimal training pipeline.

    Because full 3DGS training requires densification/pruning/SH scheduling,
    this implements a simplified but realistic training loop that covers:
    - Forward rendering
    - Backward pass
    - Optimizer step (Adam)
    - PSNR tracking
    - VRAM monitoring

    Tests at early, middle, and late training stages.
    """
    print(f"\n{'=' * 70}")
    print(f"  STAGE: Training Validation")
    print(f"  Scene: {scene_key}, Steps: {num_steps}")
    print(f"{'=' * 70}")

    scene_path = make_scene(SCENES[scene_key]["gaussians"])
    cameras, width, height = load_cameras(resolution="1080p")
    device = "cuda"

    from gsplat import rasterization
    scene_data = load_scene(scene_path, device=device)

    # Target image (render once with baseline)
    # For synthetic scenes, we create a ground truth using tile16
    from src.benchmark_framework import load_ply
    gt_data = load_ply(scene_path, device=device)
    gt_xyz = gt_data["xyz"]
    gt_quats = torch.nn.functional.normalize(gt_data["rotations"], dim=-1).contiguous()
    gt_scales = torch.exp(gt_data["scales"]).contiguous()
    gt_opacities = torch.sigmoid(gt_data["opacity"]).squeeze(-1).contiguous()
    gt_shs = gt_data["shs"].contiguous()

    # Generate reference images
    cam0 = cameras[0]
    with torch.no_grad():
        gt_img, _, _ = rasterization(
            means=gt_xyz, quats=gt_quats, scales=gt_scales,
            opacities=gt_opacities, colors=gt_shs,
            viewmats=cam0.viewmatrix.unsqueeze(0),
            Ks=cam0.K.unsqueeze(0),
            width=width, height=height,
            tile_size=16, packed=True, render_mode="RGB", sh_degree=3,
        )
    gt_img = gt_img.detach()
    del gt_data, gt_xyz, gt_quats, gt_scales, gt_opacities, gt_shs
    torch.cuda.empty_cache()

    def psnr(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

    # We test tile sizes in training
    tile_configs = [
        {"name": "tile16_training", "tile_size": 16},
        {"name": "tile32_training", "tile_size": 32},
    ]

    all_training_results = {}

    for cfg in tile_configs:
        ts = cfg["tile_size"]
        print(f"\n  === Training with tile_size={ts} ===")

        # Initialize trainable parameters from the scene
        xyz = scene_data["xyz"].clone().requires_grad_(True)
        quats_raw = scene_data["rotations"].clone().requires_grad_(True)
        scales_raw = scene_data["scales"].clone().requires_grad_(True)
        opacity_raw = scene_data["opacity"].clone().requires_grad_(True)
        shs = scene_data["shs"].clone().requires_grad_(True)

        # Optimizer
        optimizer = torch.optim.Adam(
            [xyz, quats_raw, scales_raw, opacity_raw, shs],
            lr=1e-3, eps=1e-15
        )

        training_log = []
        step_times_fwd = []
        step_times_bwd = []
        step_times_opt = []
        step_times_total = []
        psnrs = []
        gaussian_counts = []
        peak_vram_log = []

        for step in range(num_steps):
            optimizer.zero_grad()

            # Activate parameters
            qn = torch.nn.functional.normalize(quats_raw, dim=-1)
            sa = torch.exp(scales_raw)
            oa = torch.sigmoid(opacity_raw).squeeze(-1)

            cam = cameras[step % len(cameras)]

            torch.cuda.reset_peak_memory_stats()

            # Forward
            fwd_s = torch.cuda.Event(enable_timing=True)
            fwd_e = torch.cuda.Event(enable_timing=True)
            fwd_s.record()
            rendered, alpha, info = rasterization(
                means=xyz, quats=qn, scales=sa,
                opacities=oa, colors=shs,
                viewmats=cam.viewmatrix.unsqueeze(0),
                Ks=cam.K.unsqueeze(0),
                width=width, height=height,
                tile_size=ts, packed=True, render_mode="RGB", sh_degree=3,
            )
            fwd_e.record()
            fwd_e.synchronize()
            fwd_t = fwd_s.elapsed_time(fwd_e)

            # Loss
            loss = torch.mean((rendered - gt_img) ** 2)

            # Backward
            bwd_s = torch.cuda.Event(enable_timing=True)
            bwd_e = torch.cuda.Event(enable_timing=True)
            bwd_s.record()
            loss.backward()
            bwd_e.record()
            bwd_e.synchronize()
            bwd_t = bwd_s.elapsed_time(bwd_e)

            # Optimizer step
            opt_s = torch.cuda.Event(enable_timing=True)
            opt_e = torch.cuda.Event(enable_timing=True)
            opt_s.record()
            optimizer.step()
            opt_e.record()
            opt_e.synchronize()
            opt_t = opt_s.elapsed_time(opt_e)

            total_t = fwd_t + bwd_t + opt_t

            step_times_fwd.append(fwd_t)
            step_times_bwd.append(bwd_t)
            step_times_opt.append(opt_t)
            step_times_total.append(total_t)
            peak_vram_log.append(torch.cuda.max_memory_allocated() / (1024 * 1024))

            if step % validation_interval == 0:
                with torch.no_grad():
                    p = psnr(rendered, gt_img).item()
                    psnrs.append({"step": step, "psnr": p})
                    gaussian_counts.append({"step": step, "count": xyz.shape[0]})
                print(f"    Step {step:4d}: forward={fwd_t:.2f}ms, backward={bwd_t:.2f}ms, "
                      f"opt={opt_t:.2f}ms, PSNR={p:.2f}")

        # Final metrics
        fwd_arr = np.array(step_times_fwd)
        bwd_arr = np.array(step_times_bwd)
        opt_arr = np.array(step_times_opt)
        total_arr = np.array(step_times_total)

        # Divide into early/middle/late
        n = len(total_arr)
        third = n // 3
        early = total_arr[:third]
        middle = total_arr[third:2*third]
        late = total_arr[2*third:]

        result = {
            "tile_size": ts,
            "num_steps": num_steps,
            "step_time_ms": float(np.mean(total_arr)),
            "step_time_std_ms": float(np.std(total_arr)),
            "forward_ms": float(np.mean(fwd_arr)),
            "backward_ms": float(np.mean(bwd_arr)),
            "optimizer_ms": float(np.mean(opt_arr)),
            "forward_pct": float(np.mean(fwd_arr) / np.mean(total_arr) * 100),
            "backward_pct": float(np.mean(bwd_arr) / np.mean(total_arr) * 100),
            "optimizer_pct": float(np.mean(opt_arr) / np.mean(total_arr) * 100),
            "early_stage_ms": float(np.mean(early)),
            "middle_stage_ms": float(np.mean(middle)),
            "late_stage_ms": float(np.mean(late)),
            "peak_vram_mb": float(np.max(peak_vram_log)),
            "final_psnr": psnrs[-1]["psnr"] if psnrs else 0.0,
            "psnr_trajectory": psnrs,
            "gaussian_count_trajectory": gaussian_counts,
            "step_times_fwd": [float(t) for t in step_times_fwd],
            "step_times_bwd": [float(t) for t in step_times_bwd],
            "step_times_total": [float(t) for t in step_times_total],
        }
        all_training_results[cfg["name"]] = result

        print(f"\n  --- tile_size={ts} Training Summary ---")
        print(f"  Step time: mean={result['step_time_ms']:.2f}ms, std={result['step_time_std_ms']:.2f}ms")
        print(f"  Forward: {result['forward_ms']:.2f}ms ({result['forward_pct']:.1f}%)")
        print(f"  Backward: {result['backward_ms']:.2f}ms ({result['backward_pct']:.1f}%)")
        print(f"  Optimizer: {result['optimizer_ms']:.2f}ms ({result['optimizer_pct']:.1f}%)")
        print(f"  Early stage: {result['early_stage_ms']:.2f}ms")
        print(f"  Middle stage: {result['middle_stage_ms']:.2f}ms")
        print(f"  Late stage: {result['late_stage_ms']:.2f}ms")
        print(f"  Peak VRAM: {result['peak_vram_mb']:.0f}MB")
        print(f"  Final PSNR: {result['final_psnr']:.2f}")

        # Clean up
        del xyz, quats_raw, scales_raw, opacity_raw, shs, rendered, alpha
        torch.cuda.empty_cache()
        gc.collect()

    # Compare
    if "tile16_training" in all_training_results and "tile32_training" in all_training_results:
        r16 = all_training_results["tile16_training"]
        r32 = all_training_results["tile32_training"]
        print(f"\n  === TRAINING COMPARISON: tile16 vs tile32 ===")
        print(f"  Step time: {r16['step_time_ms']:.2f}ms -> {r32['step_time_ms']:.2f}ms "
              f"({r16['step_time_ms']/r32['step_time_ms']:.2f}x)")
        print(f"  Peak VRAM: {r16['peak_vram_mb']:.0f}MB -> {r32['peak_vram_mb']:.0f}MB")
        print(f"  Final PSNR: {r16['final_psnr']:.2f} -> {r32['final_psnr']:.2f}")

    experiment = {
        "experiment_type": "training_validation",
        "metadata": {
            "experiment_id": build_experiment_id(),
            "date": date.today().isoformat(),
            "git_commit": _get_git_commit(),
            "scene_key": scene_key,
            "resolution": "1920x1080",
            "num_gaussians_initial": scene_data["xyz"].shape[0],
            "num_steps": num_steps,
            "validation_interval": validation_interval,
            "note": "Simplified training: MSE loss, Adam, no densification/pruning",
        },
        "environment": _get_hardware_info(),
        "results": all_training_results,
    }

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRAINING_DIR / f"training_{scene_key}.json"
    with open(out_path, "w") as f:
        json.dump(experiment, f, indent=2)
    print(f"\n  Saved to {out_path}")

    return experiment


# ===========================================================================
# EXPERIMENT 5: Full Scaling Validation (50K, 200K, 400K)
# ===========================================================================

def stage_scaling_validation(
    resolution: str = "1080p",
    frames: int = 100,
    warmup: int = 30,
) -> Dict[str, Any]:
    """
    Run tile8/16/32 across all scenes (50K, 200K, 400K) with validated protocol.
    """
    print(f"\n{'=' * 70}")
    print(f"  STAGE: Scaling Validation (50K, 200K, 400K)")
    print(f"{'=' * 70}")

    from gsplat import rasterization
    all_results = {}

    for scene_key in ["50k", "200k", "400k"]:
        print(f"\n  === Scene: {scene_key} ===")
        scene_path = make_scene(SCENES[scene_key]["gaussians"])
        cameras, width, height = load_cameras(resolution)
        device = "cuda"

        scene_data = load_scene(scene_path, device=device)
        xyz = scene_data["xyz"]
        quats = torch.nn.functional.normalize(scene_data["rotations"], dim=-1).contiguous()
        scales_activated = torch.exp(scene_data["scales"]).contiguous()
        opacities_activated = torch.sigmoid(scene_data["opacity"]).squeeze(-1).contiguous()
        shs = scene_data["shs"].contiguous()

        scene_results = {}

        for tile_size in [8, 16, 32]:
            all_times = []
            peak_vram = 0

            # Warmup
            for _ in range(warmup):
                cam = cameras[0]
                with torch.no_grad():
                    rendered, alpha, info = rasterization(
                        means=xyz, quats=quats, scales=scales_activated,
                        opacities=opacities_activated, colors=shs,
                        viewmats=cam.viewmatrix.unsqueeze(0),
                        Ks=cam.K.unsqueeze(0),
                        width=width, height=height,
                        tile_size=tile_size, packed=True, render_mode="RGB", sh_degree=3,
                    )
            torch.cuda.synchronize()

            # Measured
            torch.cuda.reset_peak_memory_stats()
            for frame_idx in range(frames):
                cam = cameras[frame_idx % len(cameras)]
                with torch.no_grad():
                    torch.cuda.synchronize()
                    sf = torch.cuda.Event(enable_timing=True)
                    ef = torch.cuda.Event(enable_timing=True)
                    sf.record()
                    rendered, alpha, info = rasterization(
                        means=xyz, quats=quats, scales=scales_activated,
                        opacities=opacities_activated, colors=shs,
                        viewmats=cam.viewmatrix.unsqueeze(0),
                        Ks=cam.K.unsqueeze(0),
                        width=width, height=height,
                        tile_size=tile_size, packed=True, render_mode="RGB", sh_degree=3,
                    )
                    ef.record()
                    ef.synchronize()
                    all_times.append(sf.elapsed_time(ef))
                mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
                if mem > peak_vram:
                    peak_vram = mem
                del rendered, alpha

            t_arr = np.array(all_times)
            scene_results[f"tile{tile_size}"] = {
                "tile_size": tile_size,
                "mean_ms": float(np.mean(t_arr)),
                "median_ms": float(np.median(t_arr)),
                "std_ms": float(np.std(t_arr)),
                "min_ms": float(np.min(t_arr)),
                "max_ms": float(np.max(t_arr)),
                "p99_ms": float(np.percentile(t_arr, 99)),
                "mean_fps": float(1000.0 / np.mean(t_arr)),
                "peak_vram_mb": float(peak_vram),
            }
            print(f"    tile{tile_size}: mean={scene_results[f'tile{tile_size}']['mean_ms']:.2f}ms, "
                  f"FPS={scene_results[f'tile{tile_size}']['mean_fps']:.1f}, "
                  f"VRAM={scene_results[f'tile{tile_size}']['peak_vram_mb']:.0f}MB")

        # Speedups
        t16 = scene_results["tile16"]["mean_ms"]
        s8 = t16 / scene_results["tile8"]["mean_ms"]
        s32 = t16 / scene_results["tile32"]["mean_ms"]
        print(f"    tile16/tile8: {s8:.2f}x, tile16/tile32: {s32:.2f}x")

        all_results[scene_key] = {
            "num_gaussians": xyz.shape[0],
            "results": scene_results,
            "speedup_tile32": s32,
            "speedup_tile8": s8,
        }

        del scene_data, xyz, quats, scales_activated, opacities_activated, shs
        torch.cuda.empty_cache()
        gc.collect()

    experiment = {
        "experiment_type": "scaling_validation",
        "metadata": {
            "experiment_id": build_experiment_id(),
            "date": date.today().isoformat(),
            "git_commit": _get_git_commit(),
            "resolution": resolution,
            "protocol": {"warmup": warmup, "frames": frames},
        },
        "environment": _get_hardware_info(),
        "results": all_results,
    }

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = AGG_DIR / "scaling_validation.json"
    with open(out_path, "w") as f:
        json.dump(experiment, f, indent=2)
    print(f"\n  Saved to {out_path}")

    return experiment


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="EPIC-05 Phase 2 Experiments")
    parser.add_argument("--stage", choices=[
        "m0_verification", "fwd_bwd_400k", "corrected_ablation",
        "training", "scaling", "all"
    ], default="all")
    parser.add_argument("--scene", choices=list(SCENES.keys()), default="400k")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--training_steps", type=int, default=1000)
    args = parser.parse_args()

    for d in [RAW_DIR, AGG_DIR, PROFILE_DIR, TRAINING_DIR, STATS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    stages = []
    if args.stage == "all":
        stages = ["m0_verification", "fwd_bwd_400k", "corrected_ablation", "training", "scaling"]
    else:
        stages = [args.stage]

    results = {}
    for s in stages:
        print(f"\n{'#' * 70}")
        print(f"# Running stage: {s}")
        print(f"{'#' * 70}")
        if s == "m0_verification":
            results[s] = stage_m0_verification(args.scene, frames=args.frames, warmup=args.warmup)
        elif s == "fwd_bwd_400k":
            results[s] = stage_fwd_bwd_400k(args.scene, frames=args.frames // 2, warmup=args.warmup // 2)
        elif s == "corrected_ablation":
            results[s] = stage_corrected_ablation(args.scene, frames=args.frames, warmup=args.warmup, repeats=args.repeats)
        elif s == "training":
            results[s] = stage_training_validation(args.scene, num_steps=args.training_steps)
        elif s == "scaling":
            results[s] = stage_scaling_validation(frames=args.frames, warmup=args.warmup)

    print(f"\n{'=' * 70}")
    print(f"  Phase 2 Experiments Complete")
    print(f"  Results in: {FINAL_DIR}/")
    print(f"{'=' * 70}")

    return results


if __name__ == "__main__":
    main()
