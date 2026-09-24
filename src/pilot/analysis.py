"""PilotAnalysis — weekly/daily summaries and quality dashboard.

Provides:
  GET /daily/pilot-summary style analysis (queries, learning, research,
  personalized, feedback, router corrections, retrieval corrections,
  memory corrections, recommendation acceptance).

Quality dashboard metrics:
  personalization_gain, context_success_rate, router_correction_rate,
  retrieval_correction_rate, learning_recommendation_acceptance,
  research_context_usefulness
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from .usage_event import UsageEventStore, TaskType, DataSource
from .feedback import FeedbackStore, FeedbackRating, FeedbackReason
from .personalization import PersonalizationGainTracker
from .context_usefulness import ContextUsefulnessTracker
from .learning_usefulness import LearningUsefulnessTracker
from .research_usefulness import ResearchUsefulnessTracker
from .router_usefulness import RouterUsefulnessTracker
from .memory_feedback import MemoryFeedbackTracker
from .daily_review_feedback import DailyReviewFeedbackTracker
from .recommendation_utility import RecommendationUtilityTracker


class PilotAnalysisEngine:
    """Aggregates pilot data into summaries and quality metrics."""

    def __init__(
        self,
        event_store: UsageEventStore | None = None,
        feedback_store: FeedbackStore | None = None,
        personalization: PersonalizationGainTracker | None = None,
        context: ContextUsefulnessTracker | None = None,
        learning: LearningUsefulnessTracker | None = None,
        research: ResearchUsefulnessTracker | None = None,
        router: RouterUsefulnessTracker | None = None,
        memory: MemoryFeedbackTracker | None = None,
        daily_review: DailyReviewFeedbackTracker | None = None,
        recommendations: RecommendationUtilityTracker | None = None,
    ) -> None:
        self.events = event_store or UsageEventStore()
        self.feedback = feedback_store or FeedbackStore()
        self.personalization = personalization or PersonalizationGainTracker()
        self.context = context or ContextUsefulnessTracker()
        self.learning = learning or LearningUsefulnessTracker()
        self.research = research or ResearchUsefulnessTracker()
        self.router = router or RouterUsefulnessTracker()
        self.memory = memory or MemoryFeedbackTracker()
        self.daily_review = daily_review or DailyReviewFeedbackTracker()
        self.recommendations = (
            recommendations or RecommendationUtilityTracker()
        )

    def daily_summary(
        self, date_str: str | None = None
    ) -> dict[str, Any]:
        """Generate a daily pilot summary report."""
        if date_str is None:
            date_str = date.today().isoformat()

        day_events = [
            e for e in self.events.list_all()
            if e.timestamp.startswith(date_str)
        ]
        day_feedback = [
            f for f in self.feedback.list_feedback()
            if f.created_at.startswith(date_str)
        ]

        return {
            "date": date_str,
            "queries": {
                "total": len(day_events),
                "learning": sum(
                    1 for e in day_events
                    if e.task_type == TaskType.LEARNING.value
                ),
                "research": sum(
                    1 for e in day_events
                    if e.task_type == TaskType.RESEARCH.value
                ),
                "knowledge": sum(
                    1 for e in day_events
                    if e.task_type == TaskType.KNOWLEDGE.value
                ),
                "planning": sum(
                    1 for e in day_events
                    if e.task_type == TaskType.PLANNING.value
                ),
                "general": sum(
                    1 for e in day_events
                    if e.task_type == TaskType.GENERAL.value
                ),
                "personalized": sum(
                    1 for e in day_events
                    if e.personal_context_used
                ),
            },
            "feedback": {
                "total": len(day_feedback),
                "good": sum(
                    1 for f in day_feedback
                    if f.rating == FeedbackRating.GOOD.value
                ),
                "partial": sum(
                    1 for f in day_feedback
                    if f.rating == FeedbackRating.PARTIAL.value
                ),
                "bad": sum(
                    1 for f in day_feedback
                    if f.rating == FeedbackRating.BAD.value
                ),
                "reason_breakdown": {
                    reason.value: self.feedback.count_feedback_by_reason(
                        reason.value
                    )
                    for reason in FeedbackReason
                },
            },
            "router_corrections": len(
                self.feedback.list_corrections()
            ),
            "retrieval_corrections": self.feedback.count_corrections_by_type(
                "BAD_RETRIEVAL"
            ),
            "memory_corrections": self.memory.summary(),
            "recommendation_acceptance": (
                self.recommendations.summary()
            ),
            "learning_feedback": self.learning.summary(),
            "research_feedback": self.research.summary(),
            "daily_review_feedback": self.daily_review.overall_summary(),
        }

    def quality_dashboard(self) -> dict[str, Any]:
        """Generate quality dashboard JSON."""
        # Personalization gain
        pg = self.personalization.overall_summary()

        # Context success rate
        ctx = self.context.summary()

        # Router correction rate
        router_summary = self.router.summary()

        # Learning recommendation acceptance
        learning_summary = self.learning.summary()

        # Research context usefulness
        research_summary = self.research.summary()

        return {
            "personalization_gain": {
                "average_gain": pg.get("average_gain", 0.0),
                "domains_tested": pg.get("domains_tested", []),
                "domain_gains": pg.get("domain_gains", {}),
                "num_measurements": pg.get("num_measurements", 0),
            },
            "context_success_rate": ctx.get("overall_success_rate", 0.0),
            "context_missing_rate": ctx.get("missing_context_rate", 0.0),
            "context_conflicting_rate": ctx.get(
                "conflicting_context_rate", 0.0
            ),
            "router_correction_rate": router_summary.get(
                "correction_rate", 0.0
            ),
            "router_good_decision_rate": router_summary.get(
                "good_decision_rate", 0.0
            ),
            "retrieval_correction_rate": (
                self.feedback.count_feedback_by_reason(
                    FeedbackReason.BAD_RETRIEVAL.value
                )
                / max(self.events.count(), 1)
                if self.events.count() > 0
                else 0.0
            ),
            "learning_recommendation_acceptance": (
                learning_summary.get(
                    "recommendation_acceptance_rate", 0.0
                )
            ),
            "research_context_usefulness": (
                research_summary.get("usefulness_distribution", {})
            ),
            "recommendation_acceptance_rate": (
                self.recommendations.acceptance_rate()
            ),
            "recommendation_success_rate": (
                self.recommendations.success_rate()
            ),
        }

    def weekly_trend(
        self, end_date: str | None = None, days: int = 7
    ) -> list[dict[str, Any]]:
        """Generate daily summaries for the last N days."""
        if end_date is None:
            end = date.today()
        else:
            end = date.fromisoformat(end_date)

        summaries = []
        for i in range(days):
            d = (end - timedelta(days=i)).isoformat()
            summaries.append(self.daily_summary(d))

        return summaries

    def full_report(self) -> dict[str, Any]:
        """Generate a complete pilot report."""
        return {
            "summary": {
                "total_events": self.events.count(),
                "total_feedback": len(self.feedback.list_feedback()),
                "total_corrections": len(self.feedback.list_corrections()),
            },
            "daily_summary": self.daily_summary(),
            "quality_dashboard": self.quality_dashboard(),
            "personalization_gain": self.personalization.overall_summary(),
            "context_quality": self.context.summary(),
            "router_summary": self.router.summary(),
            "memory_summary": self.memory.summary(),
            "recommendation_summary": self.recommendations.summary(),
        }
