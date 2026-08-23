#!/usr/bin/env python3
"""Block-bootstrap analysis for paired per-scene/per-seed experiments.

Reads two experiment-summary JSON files (baseline and method), matches runs
into pairs by scene and seed, and reports the paired delta distribution with
a scene-level block bootstrap. This is the analysis tool referenced by the
paper's statistical protocol.

Input JSON shape (both files):

    {
      "runs": {
        "r60_train_720p_ma_s0": {"train_ms": 8.0, "psnr": 27.0, ...},
        ...
      }
    }

Run IDs must contain the scene and seed. Use --arm-regex to select one arm per
file when a summary mixes baseline and method runs (for example round-60
contains both ``r59_*_k1_*`` baseline and ``r60_*_ma_*`` method runs). Pass
--key-regex for other run-id conventions; it must define ``scene`` and
``seed`` named groups.

Usage::

    python src/scripts/bootstrap_analysis.py \
        --baseline results/higs-round60/r60-summary.json --baseline-arm k1 \
        --method results/higs-round60/r60-summary.json --method-arm ma \
        --metric train_ms \
        --out paper/tables/round60-train-ms-bootstrap.json
"""

import argparse
import json
import re
from pathlib import Path
from statistics import mean, stdev

import numpy as np

DEFAULT_KEY_REGEX = r"(?P<scene>garden|bicycle|bonsai|train|truck)_[A-Za-z0-9_.]+_s(?P<seed>\d+)$"


def load_runs(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, dict) and "runs" in document:
        return document["runs"]
    raise ValueError(f"{path}: expected a JSON object with a 'runs' mapping")


def filter_arm(runs: dict, arm_regex: str | None) -> dict:
    if arm_regex is None:
        return runs
    pattern = re.compile(arm_regex)
    return {run_id: record for run_id, record in runs.items() if pattern.search(run_id)}


def extract_metric(record: dict, metric: str) -> float:
    value = record.get(metric)
    if value is None:
        raise KeyError(f"metric {metric!r} missing from record keys: {sorted(record)}")
    return float(value)


def parse_key(run_id: str, key_regex: str) -> tuple[str, str]:
    match = re.search(key_regex, run_id)
    if not match:
        raise ValueError(f"run id {run_id!r} does not match key regex {key_regex!r}")
    return (match.group("scene"), match.group("seed"))


def load_pairs(
    baseline: dict,
    method: dict,
    metric: str,
    key_regex: str,
    strict: bool,
) -> tuple[list[str], dict[str, list[float]], list[str]]:
    baseline_map: dict[tuple[str, str], float] = {}
    for run_id, record in baseline.items():
        key = parse_key(run_id, key_regex)
        if key in baseline_map:
            raise ValueError(f"duplicate pair key {key} in baseline")
        baseline_map[key] = extract_metric(record, metric)

    method_map: dict[tuple[str, str], float] = {}
    for run_id, record in method.items():
        key = parse_key(run_id, key_regex)
        if key in method_map:
            raise ValueError(f"duplicate pair key {key} in method")
        method_map[key] = extract_metric(record, metric)

    shared = set(baseline_map) & set(method_map)
    unmatched = sorted(set(baseline_map) ^ set(method_map))
    if unmatched and strict:
        raise ValueError(f"unpaired runs for keys: {unmatched}")

    scenes = sorted({scene for scene, _ in shared})
    per_scene: dict[str, list[float]] = {scene: [] for scene in scenes}
    for (scene, _seed) in shared:
        per_scene[scene].append(method_map[(scene, _seed)] - baseline_map[(scene, _seed)])
    return scenes, per_scene, unmatched


def block_bootstrap(
    per_scene: dict[str, list[float]],
    replicates: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[float], float]:
    """Bootstrap mean delta by resampling scenes (blocks) with replacement."""
    scene_names = list(per_scene)
    observed = [value for values in per_scene.values() for value in values]
    observed_mean = mean(observed)

    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = []
        for _ in range(len(scene_names)):
            scene = scene_names[rng.integers(0, len(scene_names))]
            sampled.extend(per_scene[scene])
        estimates[index] = mean(sampled)
    return estimates, observed, observed_mean


def percentile_interval(estimates: np.ndarray, level: float) -> list[float]:
    alpha = (1.0 - level) / 2.0
    low, high = np.percentile(estimates, [100 * alpha, 100 * (1.0 - alpha)])
    return [float(low), float(high)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--baseline-arm", default=None, help="regex selecting baseline runs")
    parser.add_argument("--method-arm", default=None, help="regex selecting method runs")
    parser.add_argument("--metric", required=True)
    parser.add_argument("--key-regex", default=DEFAULT_KEY_REGEX)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict", action="store_true",
                        help="fail when baseline/method pair keys are unbalanced")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline_runs = filter_arm(load_runs(args.baseline), args.baseline_arm)
    method_runs = filter_arm(load_runs(args.method), args.method_arm)
    scenes, per_scene, unmatched = load_pairs(
        baseline_runs, method_runs, args.metric, args.key_regex, args.strict
    )

    rng = np.random.default_rng(args.seed)
    estimates, observed, observed_mean = block_bootstrap(per_scene, args.replicates, rng)

    deltas = [value for values in per_scene.values() for value in values]
    delta_sd = stdev(deltas) if len(deltas) > 1 else 0.0
    effect_size = (observed_mean / delta_sd) if delta_sd > 0 else None

    output = {
        "tool": "bootstrap_analysis",
        "baseline": str(args.baseline),
        "method": str(args.method),
        "baseline_arm": args.baseline_arm,
        "method_arm": args.method_arm,
        "metric": args.metric,
        "key_regex": args.key_regex,
        "replicates": args.replicates,
        "level": args.level,
        "seed": args.seed,
        "n_scenes": len(scenes),
        "n_pairs": len(deltas),
        "unmatched_pair_keys": unmatched,
        "scenes": scenes,
        "per_scene_mean_delta": {scene: mean(values) for scene, values in per_scene.items()},
        "observed_mean_delta": observed_mean,
        "observed_sd_delta": delta_sd,
        "effect_size_d": effect_size,
        "percentile_ci": percentile_interval(estimates, args.level),
        "method_faster_when_lower_is_better": observed_mean < 0.0,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())