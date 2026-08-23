"""Unit tests for scripts/bootstrap_analysis.py (paper statistical tool).

The tool computes paired per-scene/per-seed deltas between a baseline and a
method arm and summarizes the distribution with a scene-level block
bootstrap. These tests cover the pure functions and a CLI end-to-end run;
they are CPU-only and deterministic.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.bootstrap_analysis import (
    block_bootstrap,
    filter_arm,
    load_pairs,
    load_runs,
    percentile_interval,
)

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SCRIPT = ROOT / "src" / "scripts" / "bootstrap_analysis.py"
METRIC = "train_ms"
KEY_REGEX = r"(?P<scene>garden|bicycle|bonsai|train|truck)_[A-Za-z0-9_.]+_s(?P<seed>\d+)$"


def make_runs(scene_seed_values, arm="ma"):
    """Build a runs mapping like results/higs-round60/r60-summary.json."""
    runs = {}
    for scene, seed, value in scene_seed_values:
        runs[f"{scene}_1080p_{arm}_s{seed}"] = {METRIC: value}
    return runs


class LoadRunsTest(unittest.TestCase):
    def test_load_runs_accepts_runs_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "summary.json"
            path.write_text(
                json.dumps({"runs": {"garden_1080p_ma_s0": {METRIC: 1.0}}}),
                encoding="utf-8",
            )
            self.assertEqual(load_runs(path), {"garden_1080p_ma_s0": {METRIC: 1.0}})

    def test_load_runs_rejects_document_without_runs_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "summary.json"
            path.write_text(json.dumps({"other": 1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_runs(path)


class FilterArmTest(unittest.TestCase):
    def test_filter_arm_selects_matching_run_ids(self):
        runs = make_runs([("garden", 0, 1.0), ("garden", 1, 2.0)])
        selected = filter_arm(runs, r"_ma_")
        self.assertEqual(set(selected), set(runs))

    def test_filter_arm_without_regex_returns_all(self):
        runs = make_runs([("garden", 0, 1.0)])
        self.assertEqual(filter_arm(runs, None), runs)


class LoadPairsTest(unittest.TestCase):
    def test_matches_scene_and_seed_and_reports_unmatched(self):
        baseline = make_runs([("garden", 0, 10.0), ("bicycle", 0, 20.0), ("train", 0, 30.0)], arm="k1")
        method = make_runs([("garden", 0, 8.0), ("bicycle", 0, 18.0), ("bicycle", 1, 17.0)], arm="ma")
        scenes, per_scene, unmatched = load_pairs(
            baseline, method, METRIC, KEY_REGEX, strict=False
        )
        self.assertEqual(scenes, ["bicycle", "garden"])
        self.assertEqual(per_scene["garden"], [-2.0])
        self.assertEqual(per_scene["bicycle"], [-2.0])
        self.assertEqual(
            unmatched,
            sorted([("bicycle", "1"), ("train", "0")]),
        )

    def test_strict_mode_raises_on_unbalanced_pairs(self):
        baseline = make_runs([("garden", 0, 10.0)], arm="k1")
        method = make_runs([("garden", 0, 8.0), ("garden", 1, 7.0)], arm="ma")
        with self.assertRaises(ValueError):
            load_pairs(baseline, method, METRIC, KEY_REGEX, strict=True)

    def test_missing_metric_raises_key_error(self):
        baseline = make_runs([("garden", 0, 10.0)], arm="k1")
        method = make_runs([("garden", 0, 8.0)], arm="ma")
        with self.assertRaises(KeyError):
            load_pairs(baseline, method, "psnr", KEY_REGEX, strict=False)


class BlockBootstrapTest(unittest.TestCase):
    def test_recovers_constant_delta_with_exact_interval(self):
        per_scene = {"garden": [-2.0, -2.0], "bicycle": [-2.0]}
        estimates, observed, observed_mean = block_bootstrap(
            per_scene, replicates=500, rng=np.random.default_rng(0)
        )
        self.assertEqual(observed_mean, -2.0)
        self.assertEqual(list(estimates), [-2.0] * 500)
        self.assertEqual(percentile_interval(estimates, 0.95), [-2.0, -2.0])

    def test_is_deterministic_for_fixed_seed(self):
        per_scene = {"garden": [-4.0, -3.0], "bicycle": [-6.0, -5.0], "train": [-1.0, -0.5]}
        first = block_bootstrap(per_scene, 200, np.random.default_rng(7))[0]
        second = block_bootstrap(per_scene, 200, np.random.default_rng(7))[0]
        np.testing.assert_array_equal(first, second)

    def test_interval_bounds_are_ordered_and_cover_observed_mean(self):
        rng = np.random.default_rng(3)
        per_scene = {
            scene: [rng.normal(-2.0, 0.5) for _ in range(2)]
            for scene in ("garden", "bicycle", "train")
        }
        estimates, observed, observed_mean = block_bootstrap(per_scene, 2000, rng)
        low, high = percentile_interval(estimates, 0.95)
        self.assertLessEqual(low, high)
        self.assertGreaterEqual(high, observed_mean)
        self.assertLessEqual(low, observed_mean)


class BootstrapCliTest(unittest.TestCase):
    def test_cli_end_to_end_writes_paired_bootstrap_json(self):
        baseline = make_runs(
            [("garden", 0, 10.0), ("bicycle", 0, 20.0), ("train", 0, 30.0),
             ("garden", 1, 11.0), ("bicycle", 1, 21.0), ("train", 1, 31.0)],
            arm="k1",
        )
        method = make_runs(
            [("garden", 0, 8.0), ("bicycle", 0, 17.0), ("train", 0, 29.0),
             ("garden", 1, 9.0), ("bicycle", 1, 18.0), ("train", 1, 30.0)],
            arm="ma",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            baseline_path = temp / "baseline.json"
            method_path = temp / "method.json"
            out_path = temp / "out" / "bootstrap.json"
            baseline_path.write_text(json.dumps({"runs": baseline}), encoding="utf-8")
            method_path.write_text(json.dumps({"runs": method}), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable, str(BOOTSTRAP_SCRIPT),
                    "--baseline", str(baseline_path), "--baseline-arm", r"_k1_",
                    "--method", str(method_path), "--method-arm", r"_ma_",
                    "--metric", METRIC,
                    "--replicates", "200", "--seed", "0",
                    "--out", str(out_path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            output = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(output["tool"], "bootstrap_analysis")
            self.assertEqual(output["metric"], METRIC)
            self.assertEqual(output["n_scenes"], 3)
            self.assertEqual(output["n_pairs"], 6)
            self.assertEqual(output["unmatched_pair_keys"], [])
            self.assertEqual(set(output["scenes"]), {"garden", "bicycle", "train"})
            self.assertAlmostEqual(output["observed_mean_delta"], -2.0)
            self.assertTrue(output["method_faster_when_lower_is_better"])
            self.assertEqual(len(output["percentile_ci"]), 2)
            self.assertLess(output["effect_size_d"], 0.0)


if __name__ == "__main__":
    unittest.main()
