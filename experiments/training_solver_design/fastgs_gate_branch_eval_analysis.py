"""Descriptive paired-view uncertainty for the room 4100 split-veto pilot."""

import json
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent
CHILD_SEED = int(os.environ.get("FASTGS_CHILD_SEED", "20260927"))
DATA = HERE / ("fastgs_gate_branch_64_512_eval.json" if CHILD_SEED == 20260927
               else f"fastgs_gate_branch_64_512_eval_{CHILD_SEED}.json")
OUT = HERE / ("fastgs_gate_branch_eval_analysis.json" if CHILD_SEED == 20260927
              else f"fastgs_gate_branch_eval_analysis_{CHILD_SEED}.json")


def main():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    rng = np.random.default_rng(20260927)
    result = {}
    result["child_seed"] = CHILD_SEED
    for iteration in (4164, 4612):
        c = data["arms"][f"control_{iteration}"]["per_view"]
        v = data["arms"][f"veto_{iteration}"]["per_view"]
        if [row["camera"] for row in c] != [row["camera"] for row in v]:
            raise AssertionError("unpaired cameras")
        metrics = {}
        for metric in ("psnr", "ssim", "lpips"):
            delta = np.array([vr[metric] - cr[metric] for cr, vr in zip(c, v)], dtype=np.float64)
            sample = rng.integers(len(delta), size=(10000, len(delta)))
            means = delta[sample].mean(axis=1)
            metrics[metric] = {
                "mean_veto_minus_control": float(delta.mean()),
                "median_veto_minus_control": float(np.median(delta)),
                "view_bootstrap_95_interval": np.quantile(means, [.025, .975]).tolist(),
                "min": float(delta.min()),
                "max": float(delta.max()),
            }
        result[str(iteration)] = metrics
    result["scope"] = ("Descriptive bootstrap over 39 room development views only. "
                       "Cameras are correlated and there is one scene/child seed, "
                       "so this is not a confidence interval for cross-scene or cross-seed quality.")
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
