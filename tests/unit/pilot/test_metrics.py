"""Unit tests for pilot module: context, learning, research, router, memory, daily review, recommendations."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


# ─── Context Usefulness ────────────────────────────────────────────────
from pilot.context_usefulness import ContextUsefulnessTracker, ContextFeedbackRecord


class TestContextUsefulness(unittest.TestCase):
    def setUp(self):
        self.tracker = ContextUsefulnessTracker()

    def test_record_and_get(self):
        record = ContextFeedbackRecord(
            event_id="evt_001", num_context_items=5,
            helpful_count=3, irrelevant_count=1,
        )
        self.tracker.record(record)
        retrieved = self.tracker.get("evt_001")
        self.assertIsNotNone(retrieved)
        self.assertAlmostEqual(retrieved.helpful_ratio, 0.6)

    def test_overall_success_rate_empty(self):
        self.assertEqual(self.tracker.overall_success_rate(), 0.0)

    def test_save_and_load(self):
        self.tracker.record(
            ContextFeedbackRecord(
                event_id="evt_001", num_context_items=4, helpful_count=3,
            )
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name
            self.tracker.save_json(path)

        loaded = ContextUsefulnessTracker.load_json(path)
        record = loaded.get("evt_001")
        self.assertIsNotNone(record)
        self.assertAlmostEqual(record.helpful_ratio, 0.75)
        os.unlink(path)


# ─── Learning Usefulness ───────────────────────────────────────────────
from pilot.learning_usefulness import (
    LearningUsefulnessTracker,
    LearningFeedbackRecord,
)


class TestLearningUsefulness(unittest.TestCase):
    def setUp(self):
        self.tracker = LearningUsefulnessTracker()

    def test_record_and_summary(self):
        r = LearningFeedbackRecord(
            record_id="lr_001", date="2026-08-30",
            learning_question="What is tile size?",
            answer_usefulness="useful",
            recommendation_accepted=True,
        )
        self.tracker.record(r)
        self.assertAlmostEqual(self.tracker.recommendation_acceptance_rate(), 1.0)

    def test_usefulness_rate(self):
        self.tracker.record(LearningFeedbackRecord(
            record_id="lr_001", date="2026-08-30",
            answer_usefulness="useful",
        ))
        self.tracker.record(LearningFeedbackRecord(
            record_id="lr_002", date="2026-08-30",
            answer_usefulness="not_useful",
        ))
        rates = self.tracker.usefulness_rate()
        self.assertAlmostEqual(rates["useful"], 0.5)
        self.assertAlmostEqual(rates["not_useful"], 0.5)

    def test_gap_detection_rate(self):
        self.tracker.record(LearningFeedbackRecord(
            record_id="lr_001", date="2026-08-30", learning_gap_detected=True,
        ))
        self.tracker.record(LearningFeedbackRecord(
            record_id="lr_002", date="2026-08-30", learning_gap_detected=False,
        ))
        self.assertAlmostEqual(self.tracker.gap_detection_rate(), 0.5)


# ─── Research Usefulness ───────────────────────────────────────────────
from pilot.research_usefulness import (
    ResearchUsefulnessTracker,
    ResearchFeedbackRecord,
)


class TestResearchUsefulness(unittest.TestCase):
    def setUp(self):
        self.tracker = ResearchUsefulnessTracker()

    def test_record_and_summary(self):
        self.tracker.record(ResearchFeedbackRecord(
            record_id="rr_001", user_feedback="useful",
            alternative_explanations_provided=True,
        ))
        self.tracker.record(ResearchFeedbackRecord(
            record_id="rr_002", user_feedback="not_useful",
        ))
        rates = self.tracker.usefulness_rate()
        self.assertAlmostEqual(rates["useful"], 0.5)
        self.assertAlmostEqual(rates["not_useful"], 0.5)
        self.assertAlmostEqual(self.tracker.alternative_explanations_rate(), 0.5)


# ─── Router Usefulness ─────────────────────────────────────────────────
from pilot.router_usefulness import (
    RouterUsefulnessTracker,
    RouterDecisionRecord,
)


class TestRouterUsefulness(unittest.TestCase):
    def setUp(self):
        self.tracker = RouterUsefulnessTracker()

    def test_good_decision_rate(self):
        self.tracker.record(RouterDecisionRecord(record_id="rd_001"))
        self.tracker.record(RouterDecisionRecord(
            record_id="rd_002", wrong_mode=True,
        ))
        self.assertAlmostEqual(self.tracker.good_decision_rate(), 0.5)

    def test_issue_rate(self):
        self.tracker.record(RouterDecisionRecord(
            record_id="rd_001", unnecessary_cloud=True,
        ))
        self.tracker.record(RouterDecisionRecord(record_id="rd_002"))
        rate = self.tracker.issue_rate("unnecessary_cloud")
        self.assertAlmostEqual(rate, 0.5)

    def test_correction_rate(self):
        self.tracker.record(RouterDecisionRecord(
            record_id="rd_001", manual_correction="should be local",
        ))
        self.tracker.record(RouterDecisionRecord(record_id="rd_002"))
        self.assertAlmostEqual(self.tracker.correction_rate(), 0.5)

    def test_has_issue_property(self):
        r = RouterDecisionRecord(record_id="rd_001")
        self.assertFalse(r.has_issue)
        r.wrong_retrieval = True
        self.assertTrue(r.has_issue)


# ─── Memory Feedback ───────────────────────────────────────────────────
from pilot.memory_feedback import MemoryFeedbackTracker, MemoryFeedbackRecord


class TestMemoryFeedback(unittest.TestCase):
    def setUp(self):
        self.tracker = MemoryFeedbackTracker()

    def test_count_by_tag(self):
        self.tracker.record(MemoryFeedbackRecord(
            record_id="mf_001", feedback_tag="CORRECT_MEMORY",
        ))
        self.tracker.record(MemoryFeedbackRecord(
            record_id="mf_002", feedback_tag="INCORRECT_MEMORY",
        ))
        self.tracker.record(MemoryFeedbackRecord(
            record_id="mf_003", feedback_tag="INCORRECT_MEMORY",
        ))
        self.assertEqual(self.tracker.count_by_tag("CORRECT_MEMORY"), 1)
        self.assertEqual(self.tracker.count_by_tag("INCORRECT_MEMORY"), 2)

    def test_correction_rate(self):
        self.tracker.record(MemoryFeedbackRecord(
            record_id="mf_001", feedback_tag="CORRECT_MEMORY",
        ))
        self.tracker.record(MemoryFeedbackRecord(
            record_id="mf_002", feedback_tag="INCORRECT_MEMORY",
        ))
        self.tracker.record(MemoryFeedbackRecord(
            record_id="mf_003", feedback_tag="OUTDATED_MEMORY",
        ))
        self.assertAlmostEqual(self.tracker.correction_rate(), 2 / 3)


# ─── Daily Review Feedback ─────────────────────────────────────────────
from pilot.daily_review_feedback import (
    DailyReviewFeedbackTracker,
    DailyReviewFeedbackRecord,
)


class TestDailyReviewFeedback(unittest.TestCase):
    def setUp(self):
        self.tracker = DailyReviewFeedbackTracker()

    def test_usefulness_by_component(self):
        self.tracker.record(DailyReviewFeedbackRecord(
            record_id="dr_001", component="today_focus",
            usefulness="useful",
        ))
        self.tracker.record(DailyReviewFeedbackRecord(
            record_id="dr_002", component="today_focus",
            usefulness="not_useful",
        ))
        rates = self.tracker.usefulness_by_component("today_focus")
        self.assertAlmostEqual(rates["useful"], 0.5)
        self.assertAlmostEqual(rates["not_useful"], 0.5)

    def test_overall_summary(self):
        self.tracker.record(DailyReviewFeedbackRecord(
            record_id="dr_001", component="today_focus",
            usefulness="useful",
        ))
        summary = self.tracker.overall_summary()
        self.assertIn("today_focus", summary)


# ─── Recommendation Utility ────────────────────────────────────────────
from pilot.recommendation_utility import (
    RecommendationUtilityTracker,
    RecommendationRecord,
)


class TestRecommendationUtility(unittest.TestCase):
    def setUp(self):
        self.tracker = RecommendationUtilityTracker()

    def test_acceptance_rate(self):
        self.tracker.record(RecommendationRecord(
            record_id="rec_001", accepted=True,
        ))
        self.tracker.record(RecommendationRecord(
            record_id="rec_002", accepted=False,
        ))
        self.assertAlmostEqual(self.tracker.acceptance_rate(), 0.5)

    def test_success_rate(self):
        self.tracker.record(RecommendationRecord(
            record_id="rec_001", accepted=True, successful=True,
        ))
        self.tracker.record(RecommendationRecord(
            record_id="rec_002", accepted=True, successful=False,
        ))
        self.assertAlmostEqual(self.tracker.success_rate(), 0.5)

    def test_pending_evaluation_count(self):
        self.tracker.record(RecommendationRecord(
            record_id="rec_001", accepted=True, successful=None,
        ))
        self.tracker.record(RecommendationRecord(
            record_id="rec_002", accepted=True, successful=True,
        ))
        self.assertEqual(self.tracker.pending_evaluation_count(), 1)

    def test_does_not_assume_accepted_is_successful(self):
        """accepted must not be automatically treated as successful."""
        self.tracker.record(RecommendationRecord(
            record_id="rec_001", accepted=True, successful=None,
        ))
        self.assertIsNone(
            self.tracker.get("rec_001").successful
        )
        self.assertEqual(self.tracker.success_rate(), 0.0)


if __name__ == "__main__":
    unittest.main()
