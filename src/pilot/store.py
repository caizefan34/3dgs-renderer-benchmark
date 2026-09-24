"""PilotDataStore — centralized store for all pilot telemetry.

Enforces REAL_USER / DEMO / EVAL separation.
Provides unified export to data/eval/real_user_pilot.json.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .usage_event import (
    CopilotUsageEvent,
    UsageEventStore,
    DataSource,
)
from .feedback import (
    FeedbackStore,
    UserFeedback,
    ManualCorrection,
)
from .personalization import (
    PersonalizationGainTracker,
    PersonalizationScore,
)
from .context_usefulness import ContextUsefulnessTracker
from .learning_usefulness import LearningUsefulnessTracker
from .research_usefulness import ResearchUsefulnessTracker
from .router_usefulness import RouterUsefulnessTracker
from .memory_feedback import MemoryFeedbackTracker
from .daily_review_feedback import DailyReviewFeedbackTracker
from .recommendation_utility import RecommendationUtilityTracker
from .analysis import PilotAnalysisEngine
from .privacy import PrivacySanitizer


class PilotDataStore:
    """Central store for all pilot telemetry with source separation."""

    def __init__(self) -> None:
        self.events = UsageEventStore()
        self.feedback = FeedbackStore()
        self.personalization = PersonalizationGainTracker()
        self.context = ContextUsefulnessTracker()
        self.learning = LearningUsefulnessTracker()
        self.research = ResearchUsefulnessTracker()
        self.router = RouterUsefulnessTracker()
        self.memory = MemoryFeedbackTracker()
        self.daily_review = DailyReviewFeedbackTracker()
        self.recommendations = RecommendationUtilityTracker()
        self._sanitizer = PrivacySanitizer()

    @property
    def analysis(self) -> PilotAnalysisEngine:
        return PilotAnalysisEngine(
            event_store=self.events,
            feedback_store=self.feedback,
            personalization=self.personalization,
            context=self.context,
            learning=self.learning,
            research=self.research,
            router=self.router,
            memory=self.memory,
            daily_review=self.daily_review,
            recommendations=self.recommendations,
        )

    def record_event(self, event: CopilotUsageEvent) -> None:
        """Record a usage event (only REAL_USER data stored)."""
        if event.data_source != DataSource.REAL_USER.value:
            return  # Only store REAL_USER data in pilot
        self.events.record(event)

    def record_feedback(self, feedback: UserFeedback) -> None:
        """Record user feedback."""
        if feedback.data_source != DataSource.REAL_USER.value:
            return
        self.feedback.add_feedback(feedback)

    def record_correction(self, correction: ManualCorrection) -> None:
        """Record manual correction without modifying authoritative state."""
        if correction.data_source != DataSource.REAL_USER.value:
            return
        self.feedback.add_correction(correction)

    # ─── Source Separation ──────────────────────────────────────

    def get_real_user_data(self) -> dict[str, Any]:
        """Get only REAL_USER data."""
        return {
            "events": [
                self._sanitizer.sanitize_event(e.to_dict())
                for e in self.events.find_by_data_source(
                    DataSource.REAL_USER.value
                )
            ],
            "feedback": [
                self._sanitizer.sanitize_feedback(f.to_dict())
                for f in self.feedback.list_feedback(
                    DataSource.REAL_USER.value
                )
            ],
            "corrections": [
                c.to_dict()
                for c in self.feedback.list_corrections(
                    DataSource.REAL_USER.value
                )
            ],
        }

    def get_demo_data(self) -> dict[str, Any]:
        """Get only DEMO data."""
        return {
            "events": [
                e.to_dict()
                for e in self.events.find_by_data_source(
                    DataSource.DEMO.value
                )
            ],
        }

    def get_eval_data(self) -> dict[str, Any]:
        """Get only EVAL data."""
        return {
            "events": [
                e.to_dict()
                for e in self.events.find_by_data_source(
                    DataSource.EVAL.value
                )
            ],
        }

    # ─── Export ─────────────────────────────────────────────────

    def export_real_user_pilot(
        self, path: str | None = None
    ) -> dict[str, Any]:
        """Export real user pilot data.

        Default path: data/eval/real_user_pilot.json
        """
        if path is None:
            path = str(
                Path.cwd() / "data" / "eval" / "real_user_pilot.json"
            )

        export_data = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "data_source": DataSource.REAL_USER.value,
            "personalization_gain": {
                "summary": self.personalization.overall_summary(),
                "scores": [
                    s.to_dict() for s in self.personalization.list_all()
                ],
            },
            "context_quality": self.context.summary(),
            "learning_feedback": self.learning.summary(),
            "research_feedback": self.research.summary(),
            "router_summary": self.router.summary(),
            "memory_feedback": self.memory.summary(),
            "recommendation_summary": self.recommendations.summary(),
            "quality_dashboard": self.analysis.quality_dashboard(),
            "events_count": len(
                self.events.find_by_data_source(
                    DataSource.REAL_USER.value
                )
            ),
            "feedback_count": len(
                self.feedback.list_feedback(DataSource.REAL_USER.value)
            ),
            "correction_count": len(
                self.feedback.list_corrections(
                    DataSource.REAL_USER.value
                )
            ),
        }

        # Ensure directory exists
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return export_data

    # ─── Persistence ─────────────────────────────────────────────

    def save_all(self, base_dir: str) -> None:
        """Save all stores to JSON files under base_dir."""
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)

        self.events.save_json(str(base / "usage_events.json"))
        self.feedback.save_json(str(base / "feedback.json"))
        self.personalization.save_json(
            str(base / "personalization.json")
        )
        self.context.save_json(str(base / "context_usefulness.json"))
        self.learning.save_json(str(base / "learning_usefulness.json"))
        self.research.save_json(str(base / "research_usefulness.json"))
        self.router.save_json(str(base / "router_usefulness.json"))
        self.memory.save_json(str(base / "memory_feedback.json"))
        self.daily_review.save_json(
            str(base / "daily_review_feedback.json")
        )
        self.recommendations.save_json(
            str(base / "recommendation_utility.json")
        )

    @classmethod
    def load_all(cls, base_dir: str) -> PilotDataStore:
        """Load all stores from JSON files under base_dir."""
        store = cls()
        base = Path(base_dir)

        events_path = base / "usage_events.json"
        if events_path.exists():
            store.events = UsageEventStore.load_json(str(events_path))

        feedback_path = base / "feedback.json"
        if feedback_path.exists():
            store.feedback = FeedbackStore.load_json(str(feedback_path))

        pg_path = base / "personalization.json"
        if pg_path.exists():
            store.personalization = PersonalizationGainTracker.load_json(
                str(pg_path)
            )

        ctx_path = base / "context_usefulness.json"
        if ctx_path.exists():
            store.context = ContextUsefulnessTracker.load_json(str(ctx_path))

        learn_path = base / "learning_usefulness.json"
        if learn_path.exists():
            store.learning = LearningUsefulnessTracker.load_json(
                str(learn_path)
            )

        research_path = base / "research_usefulness.json"
        if research_path.exists():
            store.research = ResearchUsefulnessTracker.load_json(
                str(research_path)
            )

        router_path = base / "router_usefulness.json"
        if router_path.exists():
            store.router = RouterUsefulnessTracker.load_json(
                str(router_path)
            )

        memory_path = base / "memory_feedback.json"
        if memory_path.exists():
            store.memory = MemoryFeedbackTracker.load_json(str(memory_path))

        dr_path = base / "daily_review_feedback.json"
        if dr_path.exists():
            store.daily_review = DailyReviewFeedbackTracker.load_json(
                str(dr_path)
            )

        rec_path = base / "recommendation_utility.json"
        if rec_path.exists():
            store.recommendations = RecommendationUtilityTracker.load_json(
                str(rec_path)
            )

        return store

    def clear_all(self) -> None:
        """Clear all stores (test helper)."""
        self.events.clear()
        self.feedback.clear()
        self.personalization.clear()
        self.context.clear()
        self.learning.clear()
        self.research.clear()
        self.router.clear()
        self.memory.clear()
        self.daily_review.clear()
        self.recommendations.clear()
