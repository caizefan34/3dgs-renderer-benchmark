#!/usr/bin/env python3
"""
Trainable-HiGS-SCALING0-R-MX: Frozen 4-cell fixed-state benchmark.

Executes fresh MX measurements for:
- W1: LOW-N (~1M) / 540p / B4
- W2: HIGH-N (~6.1M) / 540p / B4  
- W3: HIGH-N (~6.1M) / 1080p / B4
- W4: HIGH-N (~6.1M) / 1080p / B1

Baseline: std_ll (low-level standard gsplat)
Candidate: higs_native (HiGS native backward)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_benchmark_cell(
    cell_id: str,
    scene: str,
    width: int,
    height: int,
    n_train: int,
    out_dir: Path,
    base_dir: str,
) -> dict:
    """Run one benchmark cell with fresh MX measurement."""
    
    print(f"\n{'='*80}")
    print(f"CELL {cell_id}: {scene} @ {width}x{height} B{n_train}")
    print(f"{'='*80}\n")
    
    out_file = out_dir / f"{cell_id}.json"
    
    # Build command
    cmd = [
        "python",
        "/mnt/storage_pool/liaoyuanjun/tmp_scaling0r/run_higs_train_benchmark.py",  # Fixed version without cull_cache_key
        "--base-dir", base_dir,
        "--scene", scene,
        "--backends", "std_ll", "higs_native",
        "--n-train", str(n_train),
        "--n-eval", "3",  # Minimal eval cameras for speed
        "--steps", "100",  # Fixed-state: no training, just timing
        "--width", str(width),
        "--height", str(height),
        "--seed", "0",
        "--out", str(out_file),
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    # Set environment
    env = os.environ.copy()
    env["PYTHONPATH"] = "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx:" + env.get("PYTHONPATH", "")
    env["TMPDIR"] = "/mnt/storage_pool/liaoyuanjun/tmp_scaling0r/tmp"
    env["TEMP"] = "/mnt/storage_pool/liaoyuanjun/tmp_scaling0r/tmp"
    
    # Run benchmark
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per cell
        )
        elapsed = time.time() - start_time
        
        print(f"\n--- STDOUT ---")
        print(result.stdout)
        if result.stderr:
            print(f"\n--- STDERR ---")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"\n✗ FAILED with return code {result.returncode}")
            return {
                "cell_id": cell_id,
                "status": "FAILED",
                "returncode": result.returncode,
                "elapsed_seconds": elapsed,
                "timestamp": _now(),
            }
        
        # Load results
        if out_file.exists():
            with open(out_file) as f:
                data = json.load(f)
            data["status"] = "COMPLETED"
            data["elapsed_seconds"] = elapsed
            data["timestamp"] = _now()
            _write_json(out_file, data)
            print(f"\n✓ COMPLETED in {elapsed:.1f}s")
            return data
        else:
            print(f"\n✗ Output file not created: {out_file}")
            return {
                "cell_id": cell_id,
                "status": "FAILED",
                "reason": "output file not created",
                "elapsed_seconds": elapsed,
                "timestamp": _now(),
            }
            
    except subprocess.TimeoutExpired:
        print(f"\n✗ TIMEOUT after 600s")
        return {
            "cell_id": cell_id,
            "status": "TIMEOUT",
            "elapsed_seconds": 600.0,
            "timestamp": _now(),
        }
    except Exception as e:
        print(f"\n✗ EXCEPTION: {e}")
        return {
            "cell_id": cell_id,
            "status": "ERROR",
            "error": str(e),
            "timestamp": _now(),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="output directory for results")
    ap.add_argument("--base-dir", default="/mnt/storage_pool/liaoyuanjun/benchmark_data",
                    help="root of processed datasets")
    args = ap.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Trainable-HiGS-SCALING0-R-MX: Frozen 4-cell Fixed-State Benchmark")
    print("="*80)
    print(f"Output directory: {out_dir}")
    print(f"Base directory: {args.base_dir}")
    print(f"Timestamp: {_now()}")
    
    # Define frozen cells
    # Note: W1 (LOW-N) skipped due to missing eval_cameras.json for train scene
    # Only HIGH-N cells (bicycle ~6.1M Gaussians) are executed
    cells = [
        {
            "cell_id": "W2_HIGH_540p_B4",
            "scene": "mipnerf360/bicycle",
            "width": 960,
            "height": 540,
            "n_train": 4,
            "description": "HIGH-N (~6.1M) / 540p / B4"
        },
        {
            "cell_id": "W3_HIGH_1080p_B4",
            "scene": "mipnerf360/bicycle",
            "width": 1920,
            "height": 1080,
            "n_train": 4,
            "description": "HIGH-N (~6.1M) / 1080p / B4"
        },
        {
            "cell_id": "W4_HIGH_1080p_B1",
            "scene": "mipnerf360/bicycle",
            "width": 1920,
            "height": 1080,
            "n_train": 1,
            "description": "HIGH-N (~6.1M) / 1080p / B1"
        },
    ]
    
    results = []
    for cell in cells:
        print(f"\n\n{'#'*80}")
        print(f"# {cell['description']}")
        print(f"{'#'*80}")
        
        result = run_benchmark_cell(
            cell_id=cell["cell_id"],
            scene=cell["scene"],
            width=cell["width"],
            height=cell["height"],
            n_train=cell["n_train"],
            out_dir=out_dir,
            base_dir=args.base_dir,
        )
        result["description"] = cell["description"]
        results.append(result)
    
    # Write summary
    summary = {
        "experiment": "Trainable-HiGS-SCALING0-R-MX",
        "timestamp": _now(),
        "baseline": "std_ll",
        "candidate": "higs_native",
        "cells": results,
    }
    summary_file = out_dir / "SCALING0R_SUMMARY.json"
    _write_json(summary_file, summary)
    
    print(f"\n\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Summary written to: {summary_file}")
    
    # Print quick summary
    print(f"\n{'='*80}")
    print("QUICK SUMMARY")
    print(f"{'='*80}")
    for result in results:
        status = result.get("status", "UNKNOWN")
        cell_id = result["cell_id"]
        if status == "COMPLETED":
            print(f"✓ {cell_id}: COMPLETED")
        else:
            print(f"✗ {cell_id}: {status}")


if __name__ == "__main__":
    main()
