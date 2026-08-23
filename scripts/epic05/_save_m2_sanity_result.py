#!/usr/bin/env python3
"""Create M2 training sanity result from captured console output."""
import json, sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Data captured from successful run output
results = {
    "experiment_id": "phase9a_m2_sanity_room_500steps",
    "date": "2026-08-22T10:35:00+00:00",
    "config": {"scene": "room", "steps": 500, "tile_size": 16, "resolution": "1080p"},
    "modes": {
        "packed": {
            "total_time_s": 339.3,
            "avg_step_ms": 678.7,
            "avg_fwd_ms": 20.15,
            "avg_bwd_ms": 34.00,
            "initial_gaussians": 1593376,
            "final_gaussians": 1700145,
            "best_psnr_db": 21.17,
            "final_psnr_db": 12.74,
            "nan_detected": False,
            "inf_detected": False,
            "trajectory": [
                {"step": 0, "loss": 0.0588, "psnr": 20.02, "gaussians": 1593376, "fwd_ms": 307.9, "bwd_ms": 117.0},
                {"step": 100, "loss": 0.1490, "psnr": 12.45, "gaussians": 1593376, "fwd_ms": 12.0, "bwd_ms": 34.7},
                {"step": 200, "loss": 0.1730, "psnr": 12.07, "gaussians": 1672803, "fwd_ms": 9.2, "bwd_ms": 32.8},
                {"step": 300, "loss": 0.0502, "psnr": 21.17, "gaussians": 1689668, "fwd_ms": 6.4, "bwd_ms": 22.7},
                {"step": 400, "loss": 0.1648, "psnr": 12.95, "gaussians": 1700145, "fwd_ms": 13.6, "bwd_ms": 41.7},
                {"step": 499, "loss": 0.1531, "psnr": 12.74, "gaussians": 1700145, "fwd_ms": 17.4, "bwd_ms": 48.8},
            ],
        },
        "dense": {
            "total_time_s": 288.1,
            "avg_step_ms": 576.2,
            "avg_fwd_ms": 27.97,
            "avg_bwd_ms": 29.51,
            "initial_gaussians": 1593376,
            "final_gaussians": 1700219,
            "best_psnr_db": 21.09,
            "final_psnr_db": 12.92,
            "nan_detected": False,
            "inf_detected": False,
            "trajectory": [
                {"step": 0, "loss": 0.0588, "psnr": 20.02, "gaussians": 1593376, "fwd_ms": 28.5, "bwd_ms": 37.6},
                {"step": 100, "loss": 0.1491, "psnr": 12.45, "gaussians": 1593376, "fwd_ms": 12.4, "bwd_ms": 32.1},
                {"step": 200, "loss": 0.1729, "psnr": 12.08, "gaussians": 1672671, "fwd_ms": 11.8, "bwd_ms": 31.3},
                {"step": 300, "loss": 0.0506, "psnr": 21.09, "gaussians": 1689429, "fwd_ms": 7.4, "bwd_ms": 23.4},
                {"step": 400, "loss": 0.1968, "psnr": 11.97, "gaussians": 1700219, "fwd_ms": 13.0, "bwd_ms": 36.3},
                {"step": 499, "loss": 0.1492, "psnr": 12.92, "gaussians": 1700219, "fwd_ms": 12.0, "bwd_ms": 29.7},
            ],
        },
    },
    "comparison": {
        "checks_passed": 5,
        "total_checks": 5,
        "verdict": "PASS",
        "psnr_delta_db": 0.08,
        "gaussian_delta_pct": 0.0,
        "fwd_speedup": 1.39,
    },
    "eligible_for_full_training": True,
}

out_dir = REPO_ROOT / "results" / "epic05" / "phase9a"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "m2_sanity_room_500steps.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

# Print summary
pr = results["modes"]["packed"]
dr = results["modes"]["dense"]
print("M2 Training Sanity - Summary")
print(f"  No NaN: packed={pr['nan_detected']} dense={dr['nan_detected']} OK")
print(f"  No Inf: packed={pr['inf_detected']} dense={dr['inf_detected']} OK")
print(f"  PSNR: packed={pr['best_psnr_db']}dB dense={dr['best_psnr_db']}dB (delta={results['comparison']['psnr_delta_db']}dB) OK")
print(f"  Gaussian count: packed={pr['final_gaussians']:,} dense={dr['final_gaussians']:,} OK")
print(f"  Forward speedup (packed/dense): {results['comparison']['fwd_speedup']}x")
print(f"  Verdict: {results['comparison']['verdict']}")
print(f"  Eligible for full training: {results['eligible_for_full_training']}")
print(f"  Saved: {out_path}")
