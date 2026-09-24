#!/usr/bin/env python3
"""Extract exact source evidence from the pinned official Graphdeco checkout."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OFFICIAL_SHA, git, sha256, write_json

FUNCTIONS = (
    "densify_and_split", "densify_and_clone", "densification_postfix",
    "prune_points", "_prune_optimizer", "cat_tensors_to_optimizer",
    "replace_tensor_to_optimizer", "reset_opacity", "add_densification_stats",
)


def excerpt(path: Path, start: int, end: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(f"{number}: {lines[number - 1]}" for number in range(start, end + 1))


def functions(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in FUNCTIONS:
            found[node.name] = {
                "file": path.relative_to(path.parents[1]).as_posix(),
                "lines": [node.lineno, node.end_lineno],
                "excerpt": excerpt(path, node.lineno, node.end_lineno),
            }
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, default=ROOT / ".c0-official-reference")
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    actual = git("-C", str(checkout), "rev-parse", "HEAD")
    if actual != OFFICIAL_SHA:
        raise RuntimeError(f"official checkout is {actual}, expected {OFFICIAL_SHA}")
    model = checkout / "scene" / "gaussian_model.py"
    training = checkout / "train.py"
    payload = {
        "status": "PASS",
        "repository": "https://github.com/graphdeco-inria/gaussian-splatting",
        "commit": actual,
        "files": {
            "scene/gaussian_model.py": sha256(model),
            "train.py": sha256(training),
        },
        "functions": functions(model),
        "training_loop": {
            "file": "train.py",
            "lines": [91, 186],
            "excerpt": excerpt(training, 91, 186),
        },
    }
    print(write_json("official_source_audit.json", payload))


if __name__ == "__main__":
    main()
