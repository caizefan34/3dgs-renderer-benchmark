import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_statistics import summarize_repeat_throughput


class RepeatThroughputStatisticsTest(unittest.TestCase):
    def test_summarizes_independent_repeats_without_counting_frames_as_replicates(self):
        summary = summarize_repeat_throughput([
            [10.0, 10.0],
            [20.0, 20.0],
            [12.5, 12.5],
            [25.0, 25.0],
            [10.0, 10.0],
        ])

        self.assertEqual(summary["repeat_count"], 5)
        self.assertEqual(summary["frames_per_repeat"], 2)
        self.assertAlmostEqual(summary["pooled_fps"], 1000.0 / 15.5)
        self.assertAlmostEqual(summary["repeat_mean_fps"], 74.0)
        self.assertAlmostEqual(summary["repeat_median_fps"], 80.0)
        self.assertAlmostEqual(summary["repeat_cv"], 0.3774118931)
        self.assertEqual(summary["ci95_method"], "student_t_over_repeat_fps")
        self.assertLess(summary["fps_ci95_low"], summary["repeat_mean_fps"])
        self.assertGreater(summary["fps_ci95_high"], summary["repeat_mean_fps"])

    def test_rejects_invalid_repeat_structure(self):
        for repeats in ([], [[]], [[1.0], [1.0, 2.0]], [[0.0]], [[math.nan]]):
            with self.subTest(repeats=repeats):
                with self.assertRaises(ValueError):
                    summarize_repeat_throughput(repeats)


if __name__ == "__main__":
    unittest.main()
