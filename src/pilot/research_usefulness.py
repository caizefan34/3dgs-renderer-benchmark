"""ResearchUsefulness — tracks whether PAI-OS research context helps.

Records: research_question, context used, evidence used,
alternative explanations, review used, user feedback.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class ResearchFeedbackRecord:
    """Record of research interaction usefulness."""

    record_id: str
    event_id: str = ""
    research_question: str = ""
    research_context_used: str = ""
    evidence_used: list[str] = field(default_factory=list)
    alternative_explanations_provided: bool = False
    review_used: bool = False
    user_feedback: str | None = None  # useful / partial / not_useful
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResearchUsefulnessTracker:
    """Tracks research usefulness feedback."""

    def __init__(self) -> None:
        self._records: dict[str, ResearchFeedbackRecord] = {}

    def record(self, record: ResearchFeedbackRecord) -> None:
        self._records[record.record_id] = record

    def get(self, record_id: str) -> ResearchFeedbackRecord | None:
        return self._records.get(record_id)

    def list_all(self) -> list[ResearchFeedbackRecord]:
        return list(self._records.values())

    def usefulness_rate(self) -> dict[str, float]:
        rated = [r for r in self._records.values() if r.user_feedback]
        total = len(rated)
        if total == 0:
            return {"useful": 0.0, "partial": 0.0, "not_useful": 0.0}
        return {
            "useful": sum(1 for r in rated if r.user_feedback == "useful") / total,
            "partial": sum(1 for r in rated if r.user_feedback == "partial") / total,
            "not_useful": sum(1 for r in rated if r.user_feedback == "not_useful") / total,
        }

    def alternative_explanations_rate(self) -> float:
        total = len(self._records)
        if total == 0:
            return 0.0
        return (
            sum(1 for r in self._records.values()
                if r.alternative_explanations_provided)
            / total
        )

    def summary(self) -> dict[str, Any]:
        return {
            "total_records": len(self._records),
            "usefulness_distribution": self.usefulness_rate(),
            "alternative_explanations_rate": self.alternative_explanations_rate(),
            "review_used_count": sum(1 for r in self._records.values() if r.review_used),
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
    def load_json(cls, path: str) -> ResearchUsefulnessTracker:
        tracker = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            record = ResearchFeedbackRecord(
                record_id=item["record_id"],
                event_id=item.get("event_id", ""),
                research_question=item.get("research_question", ""),
                research_context_used=item.get("research_context_used", ""),
                evidence_used=item.get("evidence_used", []),
                alternative_explanations_provided=item.get(
                    "alternative_explanations_provided", False
                ),
                review_used=item.get("review_used", False),
                user_feedback=item.get("user_feedback"),
                created_at=item.get("created_at", ""),
            )
            tracker._records[record.record_id] = record
        return tracker

    def clear(self) -> None:
        self._records.clear()
