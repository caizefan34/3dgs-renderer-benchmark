"""Descriptive paired-view analysis of frozen-state vector-gate renders."""

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent
data = json.loads((HERE / "fastgs_vector_gate_dev_eval.json").read_text(encoding="utf-8"))
rng = np.random.default_rng(20260927)
result = {"comparisons": {}}
for arm in ("continuous", "binary"):
    a = data["arms"][arm]["per_view"]
    for base in ("native_pre", "native_post"):
        b = data["arms"][base]["per_view"]
        if [x["camera"] for x in a] != [x["camera"] for x in b]:
            raise AssertionError("unpaired cameras")
        metrics = {}
        for key in ("psnr", "ssim", "lpips"):
            delta = np.array([x[key] - y[key] for x, y in zip(a, b)])
            indices = rng.integers(len(delta), size=(10000, len(delta)))
            metrics[key] = {
                "mean_delta": float(delta.mean()),
                "views_better": int((delta < 0).sum() if key == "lpips" else (delta > 0).sum()),
                "view_bootstrap_95_interval": np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist(),
            }
        result["comparisons"][f"{arm}_minus_{base}"] = metrics
result["scope"] = ("One fixed room event, 39 previously consulted development views. "
                   "No model training after gate selection; view bootstrap is descriptive only.")
(HERE / "fastgs_vector_gate_dev_analysis.json").write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
