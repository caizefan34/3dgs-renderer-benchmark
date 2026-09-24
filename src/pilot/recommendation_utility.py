"""RecommendationUtility — tracks recommendation acceptance and outcome.

Computes recommendation acceptance rate.
Does NOT automatically treat "accepted" as "successful" —
needs to observe actual results.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class RecommendationRecord:
    """Record of a recommendation and its outcome."""

    record_id: str
    event_id: str = ""
    recommendation_type: str = ""  # learning_focus, research_direction, knowledge_area, action_item
    title: str = ""
    description: str = ""
    accepted: bool | None = None  # None = not yet acted upon
    successful: bool | None = None  # None = not yet evaluated
    follow_up_result: str | None = None
    user_feedback: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    evaluated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RecommendationUtilityTracker:
    """Tracks recommendation acceptance and follow-up."""

    def __init__(self) -> None:
        self._records: dict[str, RecommendationRecord] = {}

    def record(self, record: RecommendationRecord) -> None:
        self._records[record.record_id] = record

    def get(self, record_id: str) -> RecommendationRecord | None:
        return self._records.get(record_id)

    def list_all(self) -> list[RecommendationRecord]:
        return list(self._records.values())

    def acceptance_rate(self) -> float:
        decided = [
            r for r in self._records.values()
            if r.accepted is not None
        ]
        if not decided:
            return 0.0
        return (
            sum(1 for r in decided if r.accepted) / len(decided)
        )

    def success_rate(self) -> float:
        """Only among accepted recommendations that have been evaluated."""
        evaluated = [
            r for r in self._records.values()
            if r.accepted is True and r.successful is not None
        ]
        if not evaluated:
            return 0.0
        return (
            sum(1 for r in evaluated if r.successful) / len(evaluated)
        )

    def pending_evaluation_count(self) -> int:
        """Recommendations accepted but not yet evaluated for success."""
        return sum(
            1 for r in self._records.values()
            if r.accepted is True and r.successful is None
        )

    def acceptance_rate_by_type(
        self, rec_type: str
    ) -> float:
        records = [
            r for r in self._records.values()
            if r.recommendation_type == rec_type
            and r.accepted is not None
        ]
        if not records:
            return 0.0
        return sum(1 for r in records if r.accepted) / len(records)

    def summary(self) -> dict[str, Any]:
        return {
            "total_recommendations": len(self._records),
            "acceptance_rate": self.acceptance_rate(),
            "success_rate": self.success_rate(),
            "pending_evaluation": self.pending_evaluation_count(),
            "by_type": {
                t: self.acceptance_rate_by_type(t)
                for t in set(
                    r.recommendation_type for r in self._records.values()
                )
            },
        }

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [r.to_dict() for r in self._records.values()],
                f,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def load_json(cls, path: str) -> RecommendationUtilityTracker:
        tracker = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            record = RecommendationRecord(
                record_id=item["record_id"],
                event_id=item.get("event_id", ""),
                recommendation_type=item.get("recommendation_type", ""),
                title=item.get("title", ""),
                description=item.get("description", ""),
                accepted=item.get("accepted"),
                successful=item.get("successful"),
                follow_up_result=item.get("follow_up_result"),
                user_feedback=item.get("user_feedback"),
                created_at=item.get("created_at", ""),
                evaluated_at=item.get("evaluated_at"),
            )
            tracker._records[record.record_id] = record
        return tracker

    def clear(self) -> None:
        self._records.clear()
