#!/usr/bin/env python3
"""Deterministic short-training A/B runner for R6-B."""
from __future__ import annotations

import argparse
import json
import sys
from functools import wraps
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "baseline" / "reference_v1"))
from runtime import configure, invalidate  # noqa: E402
from gaussian_model import GaussianModel  # noqa: E402
from config import ReferenceV1Config  # noqa: E402
from trainer import run_training  # noqa: E402


def install_class_topology_guard() -> None:
    if getattr(GaussianModel, "_r6b_topology_guard", False):
        return
    for name in ("densification_postfix", "prune_points"):
        original = getattr(GaussianModel, name)

        @wraps(original)
        def guarded(self, *args, __original=original, **kwargs):
            result = __original(self, *args, **kwargs)
            invalidate()
            return result

        setattr(GaussianModel, name, guarded)
    GaussianModel._r6b_topology_guard = True


def run(mode: str, scene: str, iterations: int, output: Path) -> dict:
    configure(mode)
    config = ReferenceV1Config(scene=scene, iterations=iterations)
    config.repo_root = str(REPO)
    run_training(config, str(output), allow_dirty=True)
    metrics = json.loads((output / "training_metrics.json").read_text())
    timing = json.loads((output / "timing.json").read_text())
    topology = json.loads((output / "topology_events.json").read_text())
    return {"metrics": metrics, "timing": timing, "topology": topology}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--iterations", type=int, default=600,
                        help=">=600 exercises the first densification interval")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    install_class_topology_guard()
    root = Path(args.output)
    baseline = run("baseline", args.scene, args.iterations, root / "baseline")
    b1 = run("b1", args.scene, args.iterations, root / "b1")
    result = {"scene": args.scene, "iterations": args.iterations,
              "baseline": baseline, "b1": b1,
              "checks": ["loss/PSNR checkpoints", "Gaussian count", "clone/split/prune events",
                         "parameter gradients (use correctness_r6b.py)"]}
    (root / "r6-b-training-sanity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
