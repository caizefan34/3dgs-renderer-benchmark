"""Integration tests for the complete pilot pipeline.

Covers:
- Full event → feedback → analysis pipeline
- REAL_USER / DEMO / EVAL separation
- PilotDataStore save/load/export
- Personalization measurement flow
- Router correction flow
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pilot.store import PilotDataStore
from pilot.usage_event import (
    CopilotUsageEvent,
    TaskType,
    UsageMode,
    RetrievalMode,
    DataSource,
)
from pilot.feedback import (
    UserFeedback,
    FeedbackRating,
    FeedbackReason,
    ManualCorrection,
    DailyReviewUsefulness,
)
from pilot.personalization import PersonalizationScore
from pilot.context_usefulness import ContextFeedbackRecord
from pilot.learning_usefulness import LearningFeedbackRecord
from pilot.research_usefulness import ResearchFeedbackRecord
from pilot.router_usefulness import RouterDecisionRecord
from pilot.memory_feedback import MemoryFeedbackRecord
from pilot.daily_review_feedback import DailyReviewFeedbackRecord
from pilot.recommendation_utility import RecommendationRecord


class TestPilotDataStoreIntegration(unittest.TestCase):
    """Full pipeline integration test."""

    def setUp(self):
        self.store = PilotDataStore()

    def test_full_event_feedback_pipeline(self):
        """Event → feedback → analysis pipeline."""
        # 1. Record usage events
        e1 = CopilotUsageEvent(
            conversation_id="conv_001",
            task_type=TaskType.LEARNING.value,
            mode=UsageMode.CLOUD.value,
            selected_provider="openai",
            selected_model="gpt-4",
            retrieval_mode=RetrievalMode.VECTOR.value,
            context_types=["personal_memory", "research_notes"],
            personal_context_used=True,
            relevant_context_ratio=0.8,
        )
        e2 = CopilotUsageEvent(
            conversation_id="conv_002",
            task_type=TaskType.RESEARCH.value,
            mode=UsageMode.HYBRID.value,
            personal_context_used=True,
        )
        e3 = CopilotUsageEvent(
            conversation_id="conv_003",
            task_type=TaskType.GENERAL.value,
            mode=UsageMode.LOCAL.value,
            personal_context_used=False,
        )
        self.store.record_event(e1)
        self.store.record_event(e2)
        self.store.record_event(e3)

        # 2. Record feedback
        fb = UserFeedback(
            event_id=e1.id,
            rating=FeedbackRating.GOOD.value,
            reason_codes=[],
        )
        self.store.record_feedback(fb)

        # 3. Record a manual correction
        corr = ManualCorrection(
            event_id=e2.id,
            original_text="这里应该用 Cloud。",
            correction_type=FeedbackReason.WRONG_MODEL.value,
        )
        self.store.record_correction(corr)

        # 4. Analyze
        analysis = self.store.analysis
        summary = analysis.daily_summary()

        # Verify summary contents
        self.assertEqual(summary["queries"]["total"], 3)
        self.assertEqual(summary["queries"]["learning"], 1)
        self.assertEqual(summary["queries"]["research"], 1)
        self.assertEqual(summary["queries"]["personalized"], 2)
        self.assertEqual(summary["feedback"]["total"], 1)
        self.assertEqual(summary["feedback"]["good"], 1)

        # 5. Quality dashboard
        dashboard = analysis.quality_dashboard()
        self.assertIn("personalization_gain", dashboard)
        self.assertIn("context_success_rate", dashboard)

    def test_real_user_demo_eval_separation(self):
        """REAL_USER / DEMO / EVAL separation."""
        # REAL_USER data gets recorded
        real = CopilotUsageEvent(
            conversation_id="real_001",
            data_source=DataSource.REAL_USER.value,
        )
        self.store.record_event(real)

        # DEMO and EVAL data are NOT recorded (rejected by record_event)
        demo = CopilotUsageEvent(
            conversation_id="demo_001",
            data_source=DataSource.DEMO.value,
        )
        self.store.record_event(demo)

        eval_evt = CopilotUsageEvent(
            conversation_id="eval_001",
            data_source=DataSource.EVAL.value,
        )
        self.store.record_event(eval_evt)

        # Only REAL_USER events should be in the store
        all_events = self.store.events.list_all()
        self.assertEqual(len(all_events), 1)
        self.assertEqual(all_events[0].data_source, DataSource.REAL_USER.value)

    def test_feedback_separation(self):
        """REAL_USER feedback gets recorded; DEMO does not."""
        fb_real = UserFeedback(data_source="REAL_USER")
        fb_demo = UserFeedback(data_source="DEMO")
        self.store.record_feedback(fb_real)
        self.store.record_feedback(fb_demo)
        self.assertEqual(len(self.store.feedback.list_feedback()), 1)
        self.assertEqual(len(self.store.feedback.list_feedback("REAL_USER")), 1)

    def test_personalization_measurement_flow(self):
        """Personalization gain measurement flow."""
        score = PersonalizationScore(
            domain="learning",
            generic_answer="Generic answer about tile size",
            personalized_answer="Personalized answer with your research context",
            generic_baseline_score=3.0,
            personalized_score=4.5,
            context_used=["tile16_results", "tile32_results"],
        )
        self.store.personalization.record_score(score)

        summary = self.store.personalization.overall_summary()
        self.assertAlmostEqual(summary["average_gain"], 1.5)
        self.assertIn("learning", summary["domains_tested"])

    def test_router_correction_flow(self):
        """Router decision recording and correction flow."""
        record = RouterDecisionRecord(
            record_id="rd_001",
            event_id="evt_001",
            selected_route="local_llama3",
            selected_mode="local",
            wrong_mode=True,
            manual_correction="should use cloud for this task",
        )
        self.store.router.record(record)

        summary = self.store.router.summary()
        self.assertAlmostEqual(summary["good_decision_rate"], 0.0)
        self.assertAlmostEqual(summary["correction_rate"], 1.0)

    def test_recommendation_does_not_assume_success(self):
        """accepted does not automatically mean successful."""
        rec = RecommendationRecord(
            record_id="rec_001",
            accepted=True,
            successful=None,
        )
        self.store.recommendations.record(rec)
        self.assertAlmostEqual(self.store.recommendations.acceptance_rate(), 1.0)
        self.assertAlmostEqual(self.store.recommendations.success_rate(), 0.0)

    def test_save_and_load_all(self):
        """Save all stores and reload them."""
        self.store.record_event(
            CopilotUsageEvent(
                conversation_id="conv_001",
                task_type=TaskType.LEARNING.value,
            )
        )
        self.store.record_feedback(
            UserFeedback(rating=FeedbackRating.GOOD.value)
        )
        self.store.personalization.record_score(
            PersonalizationScore(
                domain="learning",
                generic_baseline_score=3.0,
                personalized_score=4.5,
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            self.store.save_all(tmpdir)
            loaded = PilotDataStore.load_all(tmpdir)

            self.assertEqual(loaded.events.count(), 1)
            self.assertEqual(
                len(loaded.feedback.list_feedback()), 1
            )
            score = loaded.personalization.get_score("learning")
            self.assertIsNotNone(score)
            self.assertAlmostEqual(score.gain, 1.5)

    def test_export_real_user_pilot(self):
        """Export to real_user_pilot.json."""
        self.store.record_event(
            CopilotUsageEvent(
                conversation_id="conv_001",
                task_type=TaskType.LEARNING.value,
            )
        )
        self.store.record_feedback(
            UserFeedback(rating=FeedbackRating.GOOD.value)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "real_user_pilot.json")
            result = self.store.export_real_user_pilot(export_path)

            self.assertIn("exported_at", result)
            self.assertEqual(result["data_source"], "REAL_USER")
            self.assertEqual(result["events_count"], 1)
            self.assertEqual(result["feedback_count"], 1)
            self.assertIn("quality_dashboard", result)
            self.assertTrue(os.path.exists(export_path))

    def test_daily_and_weekly_analysis(self):
        """Daily and weekly analysis generation."""
        self.store.record_event(
            CopilotUsageEvent(
                conversation_id="conv_001",
                task_type=TaskType.LEARNING.value,
            )
        )
        self.store.record_event(
            CopilotUsageEvent(
                conversation_id="conv_002",
                task_type=TaskType.RESEARCH.value,
            )
        )
        self.store.record_feedback(
            UserFeedback(rating=FeedbackRating.GOOD.value)
        )

        analysis = self.store.analysis
        report = analysis.full_report()
        self.assertIn("summary", report)
        self.assertEqual(report["summary"]["total_events"], 2)
        self.assertEqual(report["summary"]["total_feedback"], 1)

        dashboard = analysis.quality_dashboard()
        self.assertIn("personalization_gain", dashboard)
        self.assertIn("recommendation_acceptance_rate", dashboard)


class TestPrivacyInPipeline(unittest.TestCase):
    """Privacy protection in the full pipeline."""

    def setUp(self):
        self.store = PilotDataStore()

    def test_feedback_with_sensitive_content(self):
        """Sensitive data in feedback free_text should be handled."""
        fb = UserFeedback(
            event_id="evt_001",
            rating=FeedbackRating.GOOD.value,
            free_text="My API key is sk-abcdefghijklmnopqrst",
        )
        self.store.record_feedback(fb)
        # The raw feedback has the key, but export sanitizes it
        export = self.store.get_real_user_data()
        # free_text is sanitized before export
        for fb_item in export.get("feedback", []):
            self.assertNotIn("sk-abcdefghijklmnopqrst", fb_item.get("free_text", ""))


if __name__ == "__main__":
    unittest.main()
