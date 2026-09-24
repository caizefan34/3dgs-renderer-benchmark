#!/usr/bin/env python3
"""Generate per-checkpoint timing JSONs from timing_output_v2 results."""
import json, os, glob
import numpy as np

results_dir = os.path.expanduser("~/3dgs-renderer-benchmark/results/n2_cpcb")
timing_dir = os.path.join(results_dir, "timing_output_v2")

for ckpt_name in ["5K", "15K", "30K"]:
    d = {}
    for label in ["baseline", "cpcb"]:
        rep_files = sorted(glob.glob(os.path.join(timing_dir, f"{ckpt_name}_{label}_rep*.json")))
        if not rep_files:
            continue
        rbwd_means, e2e_means, tbwd_means, fwd_means, all_reps_data = [], [], [], [], []
        for rf in rep_files:
            with open(rf) as f:
                r = json.load(f)
            rbwd_means.append(r["raster_bwd_mean_ms"])
            e2e_means.append(r["e2e_mean_ms"])
            tbwd_means.append(r["total_bwd_mean_ms"])
            fwd_means.append(r["fwd_mean_ms"])
            all_reps_data.append(r)
        d[label] = {
            "raster_bwd_mean_ms": float(np.mean(rbwd_means)),
            "raster_bwd_median_ms": float(np.median(rbwd_means)),
            "raster_bwd_std_ms": float(np.std(rbwd_means)),
            "total_bwd_mean_ms": float(np.mean(tbwd_means)),
            "fwd_mean_ms": float(np.mean(fwd_means)),
            "e2e_mean_ms": float(np.mean(e2e_means)),
            "e2e_median_ms": float(np.median(e2e_means)),
            "n_reps": len(rep_files),
            "reps": all_reps_data,
        }

    if "baseline" in d and "cpcb" in d:
        brb = d["baseline"]["raster_bwd_mean_ms"]
        crb = d["cpcb"]["raster_bwd_mean_ms"]
        btwd = d["baseline"]["total_bwd_mean_ms"]
        ctwd = d["cpcb"]["total_bwd_mean_ms"]
        be = d["baseline"]["e2e_mean_ms"]
        ce = d["cpcb"]["e2e_mean_ms"]
        d["comparison"] = {
            "baseline_raster_bwd_ms": brb,
            "cpcb_raster_bwd_ms": crb,
            "raster_bwd_reduction_pct": (1 - crb / brb) * 100 if brb > 0 else 0,
            "raster_bwd_speedup": brb / crb if crb > 0 else 0,
            "baseline_total_bwd_ms": btwd,
            "cpcb_total_bwd_ms": ctwd,
            "total_bwd_reduction_pct": (1 - ctwd / btwd) * 100 if btwd > 0 else 0,
            "baseline_e2e_ms": be,
            "cpcb_e2e_ms": ce,
            "e2e_reduction_pct": (1 - ce / be) * 100 if be > 0 else 0,
        }

    out_file = os.path.join(results_dir, f"room_{ckpt_name.lower()}_timing.json")
    with open(out_file, "w") as f:
        json.dump(d, f, indent=2)
    print(f"Saved {out_file}")
