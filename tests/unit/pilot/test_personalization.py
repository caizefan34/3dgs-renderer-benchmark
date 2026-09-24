"""Unit tests for pilot module: personalization gain."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pilot.personalization import PersonalizationGainTracker, PersonalizationScore


class TestPersonalizationScore(unittest.TestCase):
    def test_create_score(self):
        score = PersonalizationScore(
            domain="learning",
            generic_baseline_score=3.0,
            personalized_score=4.5,
        )
        score.compute_gain()
        self.assertEqual(score.gain, 1.5)

    def test_negative_gain(self):
        score = PersonalizationScore(
            domain="research",
            generic_baseline_score=4.0,
            personalized_score=2.0,
        )
        score.compute_gain()
        self.assertEqual(score.gain, -2.0)

    def test_zero_gain(self):
        score = PersonalizationScore(
            domain="knowledge",
            generic_baseline_score=3.5,
            personalized_score=3.5,
        )
        score.compute_gain()
        self.assertEqual(score.gain, 0.0)


class TestPersonalizationGainTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = PersonalizationGainTracker()

    def test_record_and_get(self):
        score = PersonalizationScore(
            domain="learning",
            generic_baseline_score=3.0,
            personalized_score=4.5,
        )
        self.tracker.record_score(score)
        retrieved = self.tracker.get_score("learning")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.gain, 1.5)

    def test_average_gain(self):
        s1 = PersonalizationScore(
            domain="learning", generic_baseline_score=3.0, personalized_score=4.0
        )
        s2 = PersonalizationScore(
            domain="research", generic_baseline_score=4.0, personalized_score=5.0
        )
        self.tracker.record_score(s1)
        self.tracker.record_score(s2)
        self.assertEqual(self.tracker.average_gain(), 1.0)

    def test_average_gain_empty(self):
        self.assertEqual(self.tracker.average_gain(), 0.0)

    def test_overall_summary(self):
        s1 = PersonalizationScore(
            domain="learning", generic_baseline_score=3.0, personalized_score=4.0
        )
        s2 = PersonalizationScore(
            domain="research", generic_baseline_score=4.0, personalized_score=5.0
        )
        self.tracker.record_score(s1)
        self.tracker.record_score(s2)
        summary = self.tracker.overall_summary()
        self.assertEqual(len(summary["domains_tested"]), 2)
        self.assertAlmostEqual(summary["average_gain"], 1.0)

    def test_save_and_load_json(self):
        self.tracker.record_score(
            PersonalizationScore(
                domain="learning",
                generic_baseline_score=3.0,
                personalized_score=4.5,
            )
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name
            self.tracker.save_json(path)

        loaded = PersonalizationGainTracker.load_json(path)
        score = loaded.get_score("learning")
        self.assertIsNotNone(score)
        self.assertAlmostEqual(score.gain, 1.5)
        os.unlink(path)

    def test_clear(self):
        self.tracker.record_score(
            PersonalizationScore(domain="learning")
        )
        self.tracker.clear()
        self.assertEqual(len(self.tracker.list_all()), 0)


if __name__ == "__main__":
    unittest.main()
