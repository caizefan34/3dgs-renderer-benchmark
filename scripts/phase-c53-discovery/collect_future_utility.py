#!/usr/bin/env python3
"""
Phase C53-Discovery — Future Utility Collection (post-processing)

This script is a post-processing wrapper that loads the raw .npz data
collected by collect_signals.py and produces structured future-utility
JSON outputs. The actual future-utility accumulation happens during
training in collect_signals.py (it must be done online because per-iteration
gradient norms and parameter updates are not saved).

This script extracts and organizes the future-utility data from the raw
.npz files into the JSON format specified by the deliverables.
"""
import json
from pathlib import Path
import numpy as np

RESULT_DIR = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c53-discovery")
DELTAS = [10, 50, 100]
CHECKPOINTS = [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]


def extract_future_utility(scene, suffix=""):
    """Extract future utility data from raw .npz file."""
    filepath = RESULT_DIR / f"{scene}{suffix}_raw.npz"
    if not filepath.exists():
        return None

    data = np.load(filepath, allow_pickle=True)
    results = {"scene": scene, "checkpoints": {}}

    for cp in CHECKPOINTS:
        prefix = f"cp{cp}"
        if f"{prefix}_ids" not in data:
            continue

        cp_data = {"iter": cp, "n_total": int(data[f"{prefix}_n_total"])}

        for d in DELTAS:
            d_prefix = f"{prefix}_d{d}"
            if f"{d_prefix}_alive" not in data:
                continue

            alive = data[f"{d_prefix}_alive"]
            d_data = {
                "delta": d,
                "n_alive": int(alive.sum()),
                "n_dead": int((~alive).sum()),
                "grad_sum": {"mean": float(data[f"{d_prefix}_grad_sum"][alive].mean()) if alive.any() else 0,
                             "std": float(data[f"{d_prefix}_grad_sum"][alive].std()) if alive.any() else 0},
                "grad_max": {"mean": float(data[f"{d_prefix}_grad_max"][alive].mean()) if alive.any() else 0},
                "update_xyz": {"mean": float(data[f"{d_prefix}_update_xyz"][alive].mean()) if alive.any() else 0},
                "update_opacity": {"mean": float(data[f"{d_prefix}_update_opacity"][alive].mean()) if alive.any() else 0},
                "update_scale": {"mean": float(data[f"{d_prefix}_update_scale"][alive].mean()) if alive.any() else 0},
                "update_rot": {"mean": float(data[f"{d_prefix}_update_rot"][alive].mean()) if alive.any() else 0},
                "update_shs": {"mean": float(data[f"{d_prefix}_update_shs"][alive].mean()) if alive.any() else 0},
                "vis_count": {"mean": float(data[f"{d_prefix}_vis_count"][alive].mean()) if alive.any() else 0},
                "radius_mean": {"mean": float(data[f"{d_prefix}_radius_mean"][alive].mean()) if alive.any() else 0},
                "area_mean": {"mean": float(data[f"{d_prefix}_area_mean"][alive].mean()) if alive.any() else 0},
                "tiles_mean": {"mean": float(data[f"{d_prefix}_tiles_mean"][alive].mean()) if alive.any() else 0},
            }

            # Outcome distribution at this delta
            if f"{prefix}_outcome" in data:
                outcome = data[f"{prefix}_outcome"]
                d_data["outcomes"] = {
                    "unchanged": int((outcome == 0).sum()),
                    "cloned": int((outcome == 1).sum()),
                    "split": int((outcome == 2).sum()),
                    "pruned": int((outcome == 3).sum()),
                }

            cp_data[f"delta_{d}"] = d_data

        results["checkpoints"][str(cp)] = cp_data

    return results


def main():
    all_results = {}
    for scene in ["room", "garden", "bicycle"]:
        result = extract_future_utility(scene)
        if result:
            all_results[scene] = result
        result_ctrl = extract_future_utility(scene, "_seed123")
        if result_ctrl:
            all_results[f"{scene}_seed123"] = result_ctrl

    # Save future_gradient.json
    grad_results = {}
    for scene, r in all_results.items():
        grad_results[scene] = {}
        for cp_key, cp_data in r["checkpoints"].items():
            grad_results[scene][cp_key] = {}
            for d in DELTAS:
                if f"delta_{d}" in cp_data:
                    grad_results[scene][cp_key][f"delta_{d}"] = {
                        "grad_sum": cp_data[f"delta_{d}"]["grad_sum"],
                        "grad_max": cp_data[f"delta_{d}"]["grad_max"],
                        "n_alive": cp_data[f"delta_{d}"]["n_alive"],
                    }
    with open(RESULT_DIR / "future_gradient.json", 'w') as f:
        json.dump(grad_results, f, indent=2)

    # Save future_update.json
    update_results = {}
    for scene, r in all_results.items():
        update_results[scene] = {}
        for cp_key, cp_data in r["checkpoints"].items():
            update_results[scene][cp_key] = {}
            for d in DELTAS:
                if f"delta_{d}" in cp_data:
                    update_results[scene][cp_key][f"delta_{d}"] = {
                        k: cp_data[f"delta_{d}"][k] for k in
                        ["update_xyz", "update_opacity", "update_scale", "update_rot", "update_shs"]
                    }
    with open(RESULT_DIR / "future_update.json", 'w') as f:
        json.dump(update_results, f, indent=2)

    # Save future_render_proxy.json
    render_results = {}
    for scene, r in all_results.items():
        render_results[scene] = {}
        for cp_key, cp_data in r["checkpoints"].items():
            render_results[scene][cp_key] = {}
            for d in DELTAS:
                if f"delta_{d}" in cp_data:
                    render_results[scene][cp_key][f"delta_{d}"] = {
                        k: cp_data[f"delta_{d}"][k] for k in
                        ["vis_count", "radius_mean", "area_mean", "tiles_mean"]
                    }
    with open(RESULT_DIR / "future_render_proxy.json", 'w') as f:
        json.dump(render_results, f, indent=2)

    print("Future utility extraction complete.")
    print(f"  future_gradient.json, future_update.json, future_render_proxy.json saved")


if __name__ == "__main__":
    main()
