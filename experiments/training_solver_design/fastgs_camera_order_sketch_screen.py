"""Test whether low-dimensional frozen gradient sketches preserve order quality."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parent
GRAM_PATH = ROOT / "fastgs_camera_xyz_gram.npz"
RESULT_PATH = ROOT / "fastgs_camera_order_sketch_result.json"


def greedy_order(gram):
    count = len(gram)
    available = np.ones(count, dtype=bool)
    prefix_dot = np.zeros(count)
    diagonal = np.diag(gram)
    order = []
    for _ in range(count):
        scores = 2 * prefix_dot + diagonal
        index = int(np.argmin(np.where(available, scores, np.inf)))
        available[index] = False
        order.append(index)
        prefix_dot += gram[:, index]
    return order


def max_prefix_norm(gram, order):
    prefix_dot = np.zeros(len(gram))
    square_norm = 0.0
    maximum = 0.0
    for index in order:
        square_norm += 2 * prefix_dot[index] + gram[index, index]
        prefix_dot += gram[:, index]
        maximum = max(maximum, np.sqrt(max(0.0, square_norm)))
    return float(maximum)


def summarize(values):
    return {"min": float(np.min(values)), "p10": float(np.percentile(values, 10)),
            "p50": float(np.median(values)), "p90": float(np.percentile(values, 90)),
            "max": float(np.max(values))}


def main():
    with np.load(GRAM_PATH) as source:
        gram = source["gram"].astype(np.float64)
    if not np.all(np.isfinite(gram)) or not np.allclose(gram, gram.T, atol=1e-8):
        raise ValueError("Invalid gradient Gram matrix")
    centered = gram - gram.mean(axis=0)[None, :] - gram.mean(axis=1)[:, None] + gram.mean()
    centered = (centered + centered.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(centered)
    positive = np.maximum(eigenvalues, 0.0)
    coordinates = eigenvectors * np.sqrt(positive)
    reconstructed = coordinates @ coordinates.T
    relative_reconstruction_error = float(np.linalg.norm(reconstructed - centered) /
                                          np.linalg.norm(centered))
    if relative_reconstruction_error > 1e-4:
        raise AssertionError(relative_reconstruction_error)
    full_order = greedy_order(centered)
    full_score = max_prefix_norm(centered, full_order)
    rng = np.random.default_rng(20260927)
    rows = {}
    for dimension in (8, 16, 32, 64, 128, 256, 512):
        scores = []
        for _ in range(16):
            sketch = coordinates @ rng.standard_normal((len(gram), dimension)) / np.sqrt(dimension)
            order = greedy_order(sketch @ sketch.T)
            scores.append(max_prefix_norm(centered, order))
        rows[str(dimension)] = {"max_prefix_norm": summarize(scores),
                                "ratio_to_full_oracle": summarize(np.asarray(scores) / full_score)}
    result = {"camera_count": len(gram), "full_gradient_oracle_max_prefix_norm": full_score,
              "negative_eigenvalue_count": int(np.sum(eigenvalues < -1e-12)),
              "relative_psd_reconstruction_error": relative_reconstruction_error,
              "sketch_dimensions": rows,
              "scope": "Frozen room@4100 xyz gradients; projections use the complete offline Gram. No online sketch, refreshed order, Adam training, speed, or novel-view quality result."}
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
