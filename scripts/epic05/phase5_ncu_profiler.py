#!/usr/bin/env python3
"""
Phase 5: NCU Profiling Orchestrator.

Profiles gsplat forward rasterization kernels with ncu for tile16 vs tile32
on RTX 5070 Laptop GPU. Captures:
  - Duration
  - Registers per thread
  - Shared memory per block
  - Achieved occupancy
  - Active blocks/warps per SM
  - Stalls (long scoreboard, short scoreboard, wait, etc.)
  - Memory throughput
  - L2 hit rate

Usage:
    python scripts/epic05/phase5_ncu_profiler.py --scene room --tile-sizes 16 32
    python scripts/epic05/phase5_ncu_profiler.py --scene garden --tile-sizes 16 32
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
_MSVC_DIR = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
_CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
for _p in [_MSVC_DIR, _CUDA_BIN]:
    if _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")
os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras

# ---------------------------------------------------------------------------
# Kernel short names for NCU filtering
# ---------------------------------------------------------------------------
KERNEL_PATTERNS = {
    "rasterize_to_pixels": "rasterize_to_pixels_3dgs_fwd_kernel",
    "spherical_harmonics": "spherical_harmonics_fwd_kernel",
    "projection_ewa": "projection_ewa_3dgs_packed_fwd_kernel",
    "intersect_tile": "intersect_tile_kernel",
    "intersect_offset": "intersect_offset_kernel",
}

NCU_METRICS = (
    "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
    "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
    "lts__t_sectors_srcunit_tex_op_read.ratio,"
    "launch__registers_per_thread,"
    "launch__shared_mem_per_block_dynamic,"
    "launch__shared_mem_per_block_static,"
    "launch__block_size,"
    "launch__occupancy_per_block_size,"
    "sm__warps_active.avg.pct_of_peak_sustained_active,"
    "sm__occupancy.avg.pct,"
    "sm__warps_active.avg,"
    "sm__warps_active.avg.per_cycle_elapsed,"
    "smsp__warps_launched.avg.per_cycle_elapsed,"
    "smsp__average_warps_launched.avg.pct_of_peak_sustained_active,"
    # Stall reasons
    "smsp__issue_active.avg.pct_of_peak_sustained_elapsed,"
    "smsp__warps_idle.avg.pct_of_peak_sustained_elapsed,"
    "smsp__warps_stall_long_scoreboard.avg.pct_of_peak_sustained_elapsed,"
    "smsp__warps_stall_short_scoreboard.avg.pct_of_peak_sustained_elapsed,"
    "smsp__warps_stall_wait.avg.pct_of_peak_sustained_elapsed,"
    "smsp__warps_stall_math_pipe_throttle.avg.pct_of_peak_sustained_elapsed,"
    "smsp__warps_stall_membar.avg.pct_of_peak_sustained_elapsed,"
    "smsp__warps_stall_not_selected.avg.pct_of_peak_sustained_elapsed,"
    "smsp__warps_stall_sleeping.avg.pct_of_peak_sustained_elapsed,"
    "smsp__warps_stall_other.avg.pct_of_peak_sustained_elapsed,"
    # Memory
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg.pct_of_peak_sustained_elapsed,"
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.avg.pct_of_peak_sustained_elapsed,"
    "lts__t_sectors_srcunit_tex_op_read.sum,"
    "lts__t_sectors_srcunit_tex_op_write.sum,"
    "lts__t_sectors_op_read.sum,"
    "lts__t_sectors_op_write.sum,"
)


def run_ncu_profile(
    scene_name: str,
    tile_size: int,
    kernel_pattern: str,
    output_path: Path,
    num_warmup: int = 5,
    num_measure: int = 10,
    timeout: int = 600,
) -> Dict:
    """Run ncu profiling for a specific kernel pattern.

    Creates a Python wrapper script that ncu will profile.
    Returns parsed JSON metrics.
    """
    # Create the measurement script
    script_path = REPO_ROOT / "scripts" / "epic05" / f"_ncu_measure_{scene_name}_t{tile_size}.py"
    _create_ncu_measurement_script(script_path, scene_name, tile_size, num_warmup, num_measure)

    # Build ncu command
    ncu_cmd = [
        "ncu",
        "--target-processes", "all",
        "--kernel-name", kernel_pattern,
        "--kernel-name-base", "function",
        "--set", "full",
        "--metrics", NCU_METRICS,
        "--csv",
        "--print-summary", "per-kernel",
        "--page", "details",
        "--log-file", str(output_path),
        "--nvtx", "--nvtx-include", "gsplat",
        "python", str(script_path),
    ]

    print(f"\n  Running ncu for tile{tile_size}, kernel={kernel_pattern}...")
    print(f"  Command: {' '.join(ncu_cmd)}")
    print(f"  Output: {output_path}")

    try:
        result = subprocess.run(
            ncu_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT),
        )
        print(f"  ncu stdout ({len(result.stdout)} chars):")
        for line in result.stdout.split("\n")[-30:]:
            if line.strip():
                print(f"    {line.strip()}")
        if result.stderr:
            stderr_lines = [l for l in result.stderr.split("\n") if l.strip() and "Warning" not in l and "===" not in l]
            if stderr_lines:
                print(f"  ncu stderr (last 10 lines):")
                for line in stderr_lines[-10:]:
                    print(f"    {line.strip()}")
    except subprocess.TimeoutExpired:
        print(f"  ncu TIMEOUT after {timeout}s")
        return {"status": "TIMEOUT", "tile_size": tile_size, "kernel": kernel_pattern}
    except FileNotFoundError:
        print(f"  ncu NOT FOUND")
        return {"status": "NCU_NOT_FOUND", "tile_size": tile_size, "kernel": kernel_pattern}
    except Exception as e:
        print(f"  ncu ERROR: {e}")
        return {"status": "ERROR", "tile_size": tile_size, "kernel": kernel_pattern, "error": str(e)}

    # Parse CSV output
    if output_path.exists():
        return _parse_ncu_csv(output_path, tile_size, kernel_pattern)
    else:
        return {"status": "NO_OUTPUT", "tile_size": tile_size, "kernel": kernel_pattern}


def _create_ncu_measurement_script(script_path: Path, scene_name: str, tile_size: int,
                                    num_warmup: int, num_measure: int):
    """Create a Python script that ncu will profile."""
    content = f'''#!/usr/bin/env python3
"""NCU measurement script: {scene_name}, tile_size={tile_size}"""
import os, sys
sys.path.insert(0, r"{REPO_ROOT}")
sys.path.insert(0, r"{REPO_ROOT / "src"}")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import gc

from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
from gsplat import rasterization

scene_path = r"{REPO_ROOT / "data" / "official" / "mipnerf360" / scene_name / "point_cloud.ply"}"
cam_path = r"{REPO_ROOT / "data" / "official" / "mipnerf360" / scene_name / "cameras.json"}"
scene = load_ply(scene_path, device="cuda")
cams = load_cameras_from_json(cam_path, device="cuda")
cams = resize_cameras(cams, 1920, 1080)
cam = cams[0]

quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
scales_activated = torch.exp(scene["scales"]).contiguous()
opacities_activated = torch.sigmoid(scene["opacity"]).squeeze(-1).contiguous()
shs = scene["shs"].contiguous()
xyz = scene["xyz"].contiguous()

# Warmup
for _ in range({num_warmup}):
    rendered, _, _ = rasterization(
        means=xyz, quats=quats, scales=scales_activated,
        opacities=opacities_activated, colors=shs,
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size={tile_size}, sh_degree=3, packed=True, render_mode="RGB",
    )
    torch.cuda.synchronize()

# Measurement
for _ in range({num_measure}):
    rendered, _, _ = rasterization(
        means=xyz, quats=quats, scales=scales_activated,
        opacities=opacities_activated, colors=shs,
        viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
        width=cam.image_width, height=cam.image_height,
        tile_size={tile_size}, sh_degree=3, packed=True, render_mode="RGB",
    )
    torch.cuda.synchronize()

del rendered, scene, cams
gc.collect()
torch.cuda.empty_cache()
print("PROFILING COMPLETE")
'''
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)


def _parse_ncu_csv(csv_path: Path, tile_size: int, kernel_pattern: str) -> Dict:
    """Parse ncu CSV output into structured metrics."""
    import csv
    result = {
        "status": "PARSED",
        "tile_size": tile_size,
        "kernel_pattern": kernel_pattern,
        "metrics": {},
        "raw_rows": [],
    }

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result["raw_rows"].append(dict(row))

    # Extract key metrics from the CSV
    key_metrics = [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__t_sectors_srcunit_tex_op_read.ratio",
        "launch__registers_per_thread",
        "launch__shared_mem_per_block_dynamic",
        "launch__shared_mem_per_block_static",
        "launch__block_size",
        "launch__occupancy_per_block_size",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__occupancy.avg.pct",
        "sm__warps_active.avg",
        "sm__warps_active.avg.per_cycle_elapsed",
        "smsp__warps_launched.avg.per_cycle_elapsed",
        "smsp__average_warps_launched.avg.pct_of_peak_sustained_active",
        "smsp__issue_active.avg.pct_of_peak_sustained_elapsed",
        "smsp__warps_idle.avg.pct_of_peak_sustained_elapsed",
        "smsp__warps_stall_long_scoreboard.avg.pct_of_peak_sustained_elapsed",
        "smsp__warps_stall_short_scoreboard.avg.pct_of_peak_sustained_elapsed",
        "smsp__warps_stall_wait.avg.pct_of_peak_sustained_elapsed",
        "smsp__warps_stall_math_pipe_throttle.avg.pct_of_peak_sustained_elapsed",
        "smsp__warps_stall_membar.avg.pct_of_peak_sustained_elapsed",
        "smsp__warps_stall_not_selected.avg.pct_of_peak_sustained_elapsed",
        "smsp__warps_stall_sleeping.avg.pct_of_peak_sustained_elapsed",
        "smsp__warps_stall_other.avg.pct_of_peak_sustained_elapsed",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg.pct_of_peak_sustained_elapsed",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.avg.pct_of_peak_sustained_elapsed",
        "lts__t_sectors_srcunit_tex_op_read.sum",
        "lts__t_sectors_srcunit_tex_op_write.sum",
        "lts__t_sectors_op_read.sum",
        "lts__t_sectors_op_write.sum",
    ]

    if result["raw_rows"]:
        # Use the last row (summary) or first data row
        summary_row = result["raw_rows"][-1]
        for metric in key_metrics:
            # Try different column name formats
            for col_key in summary_row:
                if metric in col_key:
                    val = summary_row[col_key]
                    try:
                        result["metrics"][metric] = float(val)
                    except (ValueError, TypeError):
                        result["metrics"][metric] = val
                    break

    return result


def get_tile_metrics_cuda(scene_name: str, tile_size: int, num_measure: int = 100) -> Dict:
    """Get tile metrics using pure PyTorch CUDA events (faster than ncu)."""
    from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
    from gsplat import rasterization

    scene_path = REPO_ROOT / "data" / "official" / "mipnerf360" / scene_name / "point_cloud.ply"
    cam_path = REPO_ROOT / "data" / "official" / "mipnerf360" / scene_name / "cameras.json"

    scene = load_ply(str(scene_path), device="cuda")
    cams = load_cameras_from_json(str(cam_path), device="cuda")
    cams = resize_cameras(cams, 1920, 1080)
    cam = cams[0]

    quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
    scales_activated = torch.exp(scene["scales"]).contiguous()
    opacities_activated = torch.sigmoid(scene["opacity"]).squeeze(-1).contiguous()
    shs = scene["shs"].contiguous()
    xyz = scene["xyz"].contiguous()

    # Warmup
    for _ in range(10):
        rendered, _, _ = rasterization(
            means=xyz, quats=quats, scales=scales_activated,
            opacities=opacities_activated, colors=shs,
            viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            tile_size=tile_size, sh_degree=3, packed=True, render_mode="RGB",
        )
        torch.cuda.synchronize()

    # Precision measurement with CUDA events
    times = []
    for _ in range(num_measure):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        rendered, _, _ = rasterization(
            means=xyz, quats=quats, scales=scales_activated,
            opacities=opacities_activated, colors=shs,
            viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
            width=cam.image_width, height=cam.image_height,
            tile_size=tile_size, sh_degree=3, packed=True, render_mode="RGB",
        )
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
        del rendered

    import numpy as np
    times_np = np.array(times)
    return {
        "tile_size": tile_size,
        "scene": scene_name,
        "num_gaussians": scene["num_points"],
        "num_measurements": num_measure,
        "mean_ms": round(float(times_np.mean()), 4),
        "median_ms": round(float(np.median(times_np)), 4),
        "std_ms": round(float(times_np.std()), 4),
        "min_ms": round(float(times_np.min()), 4),
        "max_ms": round(float(times_np.max()), 4),
        "p99_ms": round(float(np.percentile(times_np, 99)), 4),
        "cv": round(float(times_np.std() / times_np.mean()), 4),
    }


def run_pytorch_profiler_kernel_timing(scene_name: str, tile_size: int) -> Dict:
    """Use PyTorch profiler for per-kernel breakdown."""
    import torch.profiler as profiler

    from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras
    from gsplat import rasterization

    scene_path = REPO_ROOT / "data" / "official" / "mipnerf360" / scene_name / "point_cloud.ply"
    cam_path = REPO_ROOT / "data" / "official" / "mipnerf360" / scene_name / "cameras.json"

    scene = load_ply(str(scene_path), device="cuda")
    cams = load_cameras_from_json(str(cam_path), device="cuda")
    cams = resize_cameras(cams, 1920, 1080)
    cam = cams[0]

    quats = torch.nn.functional.normalize(scene["rotations"], dim=-1).contiguous()
    scales_activated = torch.exp(scene["scales"]).contiguous()
    opacities_activated = torch.sigmoid(scene["opacity"]).squeeze(-1).contiguous()
    shs = scene["shs"].contiguous()
    xyz = scene["xyz"].contiguous()

    with profiler.profile(
        activities=[profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(5):
            rendered, _, _ = rasterization(
                means=xyz, quats=quats, scales=scales_activated,
                opacities=opacities_activated, colors=shs,
                viewmats=cam.viewmatrix.unsqueeze(0), Ks=cam.K.unsqueeze(0),
                width=cam.image_width, height=cam.image_height,
                tile_size=tile_size, sh_degree=3, packed=True, render_mode="RGB",
            )
            torch.cuda.synchronize()

    # Parse kernel events - keep only gsplat kernels
    gsplat_kernels = {}
    for e in prof.events():
        name = str(e.key)
        if "gsplat" in name:
            kernel_key = name.split("gsplat")[1].split("If")[0] if "If" in name else name[:60]
            gsplat_kernels[name] = {
                "device_time_us": e.device_time,
                "count": e.count,
                "avg_us": e.device_time / e.count if e.count else 0,
            }

    return {
        "tile_size": tile_size,
        "scene": scene_name,
        "num_gaussians": scene["num_points"],
        "kernels": gsplat_kernels,
        "total_kernel_time_us": sum(k["device_time_us"] for k in gsplat_kernels.values()),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 5: NCU Profiling")
    parser.add_argument("--scene", default="room", choices=["room", "garden", "bicycle"])
    parser.add_argument("--tile-sizes", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--mode", choices=["ncu", "cuda_events", "profiler", "all"],
                        default="all")
    parser.add_argument("--num-measure", type=int, default=100,
                        help="CUDA event measurements per tile")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  Phase 5: Kernel Profiling — Scene={args.scene}")
    print("=" * 70)

    output_dir = REPO_ROOT / "results" / "epic05" / "phase5"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    all_results = {
        "experiment_id": "epic05-phase5-kernel-profiling-v1",
        "date": date.today().isoformat(),
        "timestamp": timestamp,
        "gpu": torch.cuda.get_device_name(0),
        "scene": args.scene,
        "tile_sizes": args.tile_sizes,
        "methods": [],
    }

    # 1. CUDA Event timing (fast, reliable)
    if args.mode in ("cuda_events", "all"):
        print(f"\n{'='*60}")
        print(f"  CUDA Event Timing")
        print(f"{'='*60}")
        for ts in args.tile_sizes:
            result = get_tile_metrics_cuda(args.scene, ts, num_measure=args.num_measure)
            all_results.setdefault("cuda_event_timing", {})[f"tile{ts}"] = result
            print(f"  tile{ts}: mean={result['mean_ms']:.3f}ms, "
                  f"median={result['median_ms']:.3f}ms, "
                  f"std={result['std_ms']:.3f}ms")
        all_results["methods"].append("cuda_events")

    # 2. PyTorch Profiler kernel breakdown
    if args.mode in ("profiler", "all"):
        print(f"\n{'='*60}")
        print(f"  PyTorch Profiler — Kernel Breakdown")
        print(f"{'='*60}")
        for ts in args.tile_sizes:
            result = run_pytorch_profiler_kernel_timing(args.scene, ts)
            all_results.setdefault("profiler_kernel_breakdown", {})[f"tile{ts}"] = result
            print(f"\n  tile{ts}:")
            for kname, kinfo in sorted(result["kernels"].items(),
                                        key=lambda x: x[1]["device_time_us"], reverse=True):
                kshort = kname.split("gsplat")[1][:60] if "gsplat" in kname else kname[:60]
                print(f"    {kshort:65s} avg={kinfo['avg_us']:8.1f}us "
                      f"count={kinfo['count']:3d} total={kinfo['device_time_us']:8.1f}us")
        all_results["methods"].append("profiler")

    # 3. NCU profiling (per-kernel hardware counters)
    if args.mode in ("ncu", "all"):
        print(f"\n{'='*60}")
        print(f"  NCU Hardware Counter Profiling")
        print(f"{'='*60}")
        for ts in args.tile_sizes:
            for kernel_short, kernel_pattern in KERNEL_PATTERNS.items():
                ncu_out = output_dir / f"ncu_{args.scene}_t{ts}_{kernel_short}_{timestamp}.csv"
                result = run_ncu_profile(
                    scene_name=args.scene,
                    tile_size=ts,
                    kernel_pattern=kernel_pattern,
                    output_path=ncu_out,
                    num_warmup=5,
                    num_measure=10,
                    timeout=600,
                )
                all_results.setdefault("ncu_profiling", {}).setdefault(f"tile{ts}", {})[kernel_short] = result
        all_results["methods"].append("ncu")

    # Save all results
    output_path = output_dir / f"profiling_{args.scene}_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False,
                  default=str)
    print(f"\n\nResults saved: {output_path}")
    return all_results


if __name__ == "__main__":
    main()
