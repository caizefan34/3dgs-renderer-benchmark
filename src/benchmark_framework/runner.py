"""
Unified benchmark runner with reproducible timing and machine-readable outputs.
"""
import csv
import json
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class BenchmarkRunResult:
    run_index: int
    frame_times_ms: List[float]
    mean_fps: float
    median_fps: float
    std_fps: float
    p95_latency_ms: float
    peak_vram_mb: float


def set_global_seed(seed: Optional[int]) -> None:
    """Set Python/NumPy/Torch seeds for reproducible benchmark setup."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_fps_latency_stats(frame_times_ms: List[float]) -> Dict[str, float]:
    """Compute benchmark stats required for reporting."""
    if not frame_times_ms:
        return {
            "mean_fps": 0.0,
            "median_fps": 0.0,
            "std_fps": 0.0,
            "p95_latency_ms": 0.0,
            "mean_latency_ms": 0.0,
            "median_latency_ms": 0.0,
        }
    t = np.asarray(frame_times_ms, dtype=np.float64)
    fps = 1000.0 / t
    return {
        "mean_fps": float(1000.0 / t.mean()),
        "median_fps": float(1000.0 / np.median(t)),
        "std_fps": float(fps.std()),
        "p95_latency_ms": float(np.percentile(t, 95)),
        "mean_latency_ms": float(t.mean()),
        "median_latency_ms": float(np.median(t)),
    }


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def run_renderer_benchmark(
    renderer_name: str,
    renderer: Any,
    prep_data: Dict[str, Any],
    cameras: List[Any],
    warmup_iters: int = 100,
    measured_iters: int = 500,
    repeats: int = 5,
    raw_output_dir: Optional[str] = None,
    seed: Optional[int] = 42,
    use_mixed_precision: bool = False,
    amp_dtype: str = "float16",
    use_compile: bool = False,
    compile_mode: str = "default",
) -> Dict[str, Any]:
    """Run benchmark for one renderer with reproducible per-run artifacts."""
    set_global_seed(seed)
    os.makedirs(raw_output_dir or "", exist_ok=True) if raw_output_dir else None

    render_fn = lambda cam: renderer.render(prep_data, cam)
    if use_compile and hasattr(torch, "compile"):
        try:
            render_fn = torch.compile(render_fn, mode=compile_mode)
        except Exception as e:
            print(f"  torch.compile failed for {renderer_name}, fallback to eager: {e}")

    runs: List[BenchmarkRunResult] = []
    all_frame_times_ms: List[float] = []
    dtype = _dtype_from_name(amp_dtype)
    autocast_context = (
        (lambda: torch.autocast(device_type="cuda", dtype=dtype))
        if use_mixed_precision else nullcontext
    )

    for repeat_idx in range(repeats):
        if repeats > 1:
            print(f"  Repeat {repeat_idx + 1}/{repeats}...")

        with torch.no_grad():
            for wi in range(warmup_iters):
                with autocast_context():
                    render_fn(cameras[wi % len(cameras)])
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        frame_times = np.empty(measured_iters, dtype=np.float64)

        print(f"  Benchmark ({measured_iters})...", end=" ", flush=True)
        with torch.no_grad():
            for fi in range(measured_iters):
                start_event.record()
                with autocast_context():
                    render_fn(cameras[fi % len(cameras)])
                end_event.record()
                end_event.synchronize()
                frame_times[fi] = start_event.elapsed_time(end_event)
                if (fi + 1) % 50 == 0:
                    print(f"{fi + 1}..", end="", flush=True)
        print(" done")

        frame_times_list = frame_times.tolist()
        run_stats = compute_fps_latency_stats(frame_times_list)
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        run_result = BenchmarkRunResult(
            run_index=repeat_idx + 1,
            frame_times_ms=[round(x, 4) for x in frame_times_list],
            mean_fps=run_stats["mean_fps"],
            median_fps=run_stats["median_fps"],
            std_fps=run_stats["std_fps"],
            p95_latency_ms=run_stats["p95_latency_ms"],
            peak_vram_mb=peak_vram_mb,
        )
        runs.append(run_result)
        all_frame_times_ms.extend(frame_times_list)

        if raw_output_dir:
            out_path = os.path.join(raw_output_dir, f"{renderer_name}_run{repeat_idx + 1}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "renderer": renderer_name,
                        "run_index": run_result.run_index,
                        "warmup_iters": warmup_iters,
                        "measured_iters": measured_iters,
                        "seed": seed,
                        "mean_fps": round(run_result.mean_fps, 4),
                        "median_fps": round(run_result.median_fps, 4),
                        "std_fps": round(run_result.std_fps, 4),
                        "p95_latency_ms": round(run_result.p95_latency_ms, 4),
                        "peak_vram_mb": round(run_result.peak_vram_mb, 2),
                        "frame_times_ms": run_result.frame_times_ms,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

    aggregate = compute_fps_latency_stats(all_frame_times_ms)
    per_run_mean_fps = [r.mean_fps for r in runs]
    aggregate["std_mean_fps_across_runs"] = float(np.std(per_run_mean_fps)) if per_run_mean_fps else 0.0
    aggregate["peak_vram_mb"] = max((r.peak_vram_mb for r in runs), default=0.0)
    aggregate["avg_vram_mb"] = float(np.mean([r.peak_vram_mb for r in runs])) if runs else 0.0

    return {
        "renderer": renderer_name,
        "runs": runs,
        "all_frame_times_ms": [round(x, 4) for x in all_frame_times_ms],
        "aggregate": aggregate,
        "warmup_iters": warmup_iters,
        "measured_iters": measured_iters,
        "repeats": repeats,
    }


def export_aggregate_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Exported: {path}")
