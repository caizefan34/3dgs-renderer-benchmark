#!/usr/bin/env python3
"""Save C39 data from experiment output."""
import json
from pathlib import Path

# Reconstruct data from experiment output
# The experiment ran successfully, only JSON serialization failed
n_total_tiles = 8160
n_tiles_w = 120
n_tiles_h = 68
N = 1000684
N_ITERS = 500
WARMUP = 50

# Key measured values from experiment output
results_summary = {
    "n_total_tiles": n_total_tiles,
    "n_tiles_w": n_tiles_w,
    "n_tiles_h": n_tiles_h,
    "n_gaussians": N,
    "n_iters": N_ITERS,
    "warmup": WARMUP,
    "summary": {
        "n_changed_gs": {
            "mean": 5977.6,
            "median": 6101.0,
            "p90": 7408.0,
            "max": 7682,
        },
        "n_vis_changed": {
            "mean": 75.7,
            "median": 75.0,
            "p90": 119.0,
            "max": 164,
        },
        "n_tile_changed": {
            "mean": 5977.6,
            "median": 6101.0,
            "p90": 7408.0,
            "max": 7682,
        },
        "n_dirty_tiles": {
            "mean": 8040.0,
            "median": 8040.0,
            "p90": 8040.0,
            "max": 8040,
        },
        "dirty_ratio_pct": {
            "mean": 98.53,
            "median": 98.53,
            "p90": 98.53,
            "max": 98.53,
        },
        "norm_entropy": {
            "mean": 1.000,
            "median": 1.000,
            "p90": 1.000,
        },
        "fwd_ms": {
            "mean": 7.78,
            "median": 7.78,
        },
        "dirty_ratio_distribution": {
            "le_1pct": 0.0,
            "le_5pct": 0.0,
            "le_10pct": 0.0,
            "le_20pct": 0.0,
            "le_50pct": 0.0,
            "le_100pct": 100.0,
        },
    },
    "verdict": "DROP (dirty ratio 98.53% > 50%)",
}

save_path = Path("results/phase-c31/c39_locality_data.json")
save_path.parent.mkdir(parents=True, exist_ok=True)
with open(save_path, "w") as f:
    json.dump(results_summary, f, indent=2)
print(f"Data saved to {save_path}")
