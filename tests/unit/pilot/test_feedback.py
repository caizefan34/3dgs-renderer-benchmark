"""Unit tests for pilot module: feedback."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pilot.feedback import (
    UserFeedback,
    FeedbackStore,
    FeedbackRating,
    FeedbackReason,
    ContextUsefulnessTag,
    ManualCorrection,
    DailyReviewUsefulness,
)


class TestUserFeedback(unittest.TestCase):
    def test_create_good_feedback(self):
        fb = UserFeedback(rating=FeedbackRating.GOOD.value)
        self.assertTrue(fb.feedback_id.startswith("fb_"))
        self.assertEqual(fb.rating, FeedbackRating.GOOD.value)
        self.assertEqual(fb.data_source, "REAL_USER")

    def test_create_feedback_with_reasons(self):
        fb = UserFeedback(
            rating=FeedbackRating.BAD.value,
            reason_codes=[
                FeedbackReason.WRONG_MODEL.value,
                FeedbackReason.BAD_RETRIEVAL.value,
            ],
        )
        self.assertEqual(fb.rating, FeedbackRating.BAD.value)
        self.assertIn(FeedbackReason.WRONG_MODEL.value, fb.reason_codes)

    def test_feedback_with_context_tags(self):
        fb = UserFeedback(
            rating=FeedbackRating.PARTIAL.value,
            context_tags=[
                ContextUsefulnessTag.HELPFUL_CONTEXT.value,
                ContextUsefulnessTag.MISSING_CONTEXT.value,
            ],
        )
        self.assertIn(
            ContextUsefulnessTag.HELPFUL_CONTEXT.value, fb.context_tags
        )
        self.assertIn(
            ContextUsefulnessTag.MISSING_CONTEXT.value, fb.context_tags
        )

    def test_to_dict(self):
        fb = UserFeedback(
            event_id="evt_001",
            rating=FeedbackRating.GOOD.value,
            free_text="Great answer!",
        )
        d = fb.to_dict()
        self.assertEqual(d["event_id"], "evt_001")
        self.assertEqual(d["rating"], "GOOD")
        self.assertEqual(d["free_text"], "Great answer!")
        self.assertEqual(d["data_source"], "REAL_USER")


class TestManualCorrection(unittest.TestCase):
    def test_create_correction(self):
        corr = ManualCorrection(
            event_id="evt_001",
            original_text="这里应该用 Cloud。",
            correction_type=FeedbackReason.WRONG_MODEL.value,
            context="User corrected model selection",
        )
        self.assertTrue(corr.correction_id.startswith("corr_"))
        self.assertEqual(corr.original_text, "这里应该用 Cloud。")
        self.assertEqual(
            corr.correction_type, FeedbackReason.WRONG_MODEL.value
        )

    def test_to_dict(self):
        corr = ManualCorrection(
            event_id="evt_001",
            original_text="不应该查研究记录",
            correction_type=FeedbackReason.BAD_RETRIEVAL.value,
        )
        d = corr.to_dict()
        self.assertEqual(d["correction_type"], "BAD_RETRIEVAL")
        self.assertEqual(d["original_text"], "不应该查研究记录")


class TestFeedbackStore(unittest.TestCase):
    def setUp(self):
        self.store = FeedbackStore()

    def test_add_feedback(self):
        fb = UserFeedback(rating=FeedbackRating.GOOD.value)
        self.store.add_feedback(fb)
        self.assertEqual(len(self.store.list_feedback()), 1)

    def test_add_correction(self):
        corr = ManualCorrection(correction_type="WRONG_MODEL")
        self.store.add_correction(corr)
        self.assertEqual(len(self.store.list_corrections()), 1)

    def test_count_by_rating(self):
        self.store.add_feedback(UserFeedback(rating=FeedbackRating.GOOD.value))
        self.store.add_feedback(UserFeedback(rating=FeedbackRating.GOOD.value))
        self.store.add_feedback(UserFeedback(rating=FeedbackRating.BAD.value))
        self.assertEqual(self.store.count_feedback_by_rating("GOOD"), 2)
        self.assertEqual(self.store.count_feedback_by_rating("BAD"), 1)

    def test_count_by_reason(self):
        fb = UserFeedback(
            rating=FeedbackRating.BAD.value,
            reason_codes=[FeedbackReason.WRONG_MODEL.value],
        )
        self.store.add_feedback(fb)
        self.store.add_feedback(
            UserFeedback(rating=FeedbackRating.BAD.value, reason_codes=["WRONG_MODEL"])
        )
        self.assertEqual(
            self.store.count_feedback_by_reason(
                FeedbackReason.WRONG_MODEL.value
            ),
            2,
        )

    def test_count_corrections_by_type(self):
        self.store.add_correction(
            ManualCorrection(correction_type="WRONG_MODEL")
        )
        self.store.add_correction(
            ManualCorrection(correction_type="BAD_RETRIEVAL")
        )
        self.store.add_correction(
            ManualCorrection(correction_type="WRONG_MODEL")
        )
        self.assertEqual(
            self.store.count_corrections_by_type("WRONG_MODEL"), 2
        )
        self.assertEqual(
            self.store.count_corrections_by_type("BAD_RETRIEVAL"), 1
        )

    def test_save_and_load_json(self):
        self.store.add_feedback(
            UserFeedback(event_id="evt_001", rating=FeedbackRating.GOOD.value)
        )
        self.store.add_correction(
            ManualCorrection(
                event_id="evt_001", correction_type="WRONG_MODEL"
            )
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name
            self.store.save_json(path)

        loaded = FeedbackStore.load_json(path)
        self.assertEqual(len(loaded.list_feedback()), 1)
        self.assertEqual(len(loaded.list_corrections()), 1)
        os.unlink(path)

    def test_filter_by_data_source(self):
        fb_real = UserFeedback(data_source="REAL_USER")
        fb_demo = UserFeedback(data_source="DEMO")
        self.store.add_feedback(fb_real)
        self.store.add_feedback(fb_demo)
        self.assertEqual(len(self.store.list_feedback("REAL_USER")), 1)
        self.assertEqual(len(self.store.list_feedback("DEMO")), 1)

    def test_clear(self):
        self.store.add_feedback(UserFeedback())
        self.store.clear()
        self.assertEqual(len(self.store.list_feedback()), 0)


if __name__ == "__main__":
    unittest.main()
