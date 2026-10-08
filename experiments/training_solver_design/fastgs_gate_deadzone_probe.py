"""Measure the native alpha-cutoff dead zone of a zero-born split gate.

This is a center-pixel lower bound, not a renderer or training benchmark.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from plyfile import PlyData


CUTOFF = 1.0 / 255.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", type=Path)
    args = parser.parse_args()

    logits = np.asarray(PlyData.read(args.ply)["vertex"]["opacity"], dtype=np.float64)
    positive = logits >= 0
    opacity = np.empty_like(logits)
    opacity[positive] = 1 / (1 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    opacity[~positive] = exp_logits / (1 + exp_logits)

    # The pixel kernel clamps raw alpha at 0.99 before the proposed gate.
    # Every off-center pixel has alpha <= the center alpha and a later onset.
    center_alpha = np.minimum(0.99, opacity)
    viable = center_alpha >= CUTOFF
    threshold = np.full(opacity.shape, np.inf)
    threshold[viable] = math.log1p(-CUTOFF) / np.log1p(-center_alpha[viable])
    finite = threshold[np.isfinite(threshold)]

    result = {
        "ply_sha256": sha256(args.ply),
        "gaussians": int(len(opacity)),
        "native_alpha_cutoff": CUTOFF,
        "native_pixel_alpha_cap": 0.99,
        "never_reaches_cutoff_at_center": int((~viable).sum()),
        "center_gate_threshold_quantiles": {
            str(p): float(np.quantile(finite, p))
            for p in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        },
        "center_gate_threshold_above": {
            str(g): int((threshold > g).sum())
            for g in (0.01, 0.05, 0.1, 0.25, 0.5)
        },
        "example_thresholds": {
            str(a): math.log1p(-CUTOFF) / math.log1p(-a)
            for a in (0.01, 0.05, 0.25, 0.5, 0.9)
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
