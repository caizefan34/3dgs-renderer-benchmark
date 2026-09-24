"""LearningUsefulness — tracks whether PAI-OS learning recommendations help.

Daily records: learning_question, answer_usefulness, gap_detected,
focus_recommendation, recommendation_accepted.

Does NOT modify Learning State. Existing Evidence Engine rules preserved.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class LearningFeedbackRecord:
    """Daily record of learning interaction usefulness."""

    record_id: str
    date: str
    learning_question: str = ""
    answer_usefulness: str | None = None  # useful / partial / not_useful
    learning_gap_detected: bool = False
    learning_gap_description: str = ""
    focus_recommendation: str = ""
    recommendation_accepted: bool | None = None
    user_feedback: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LearningUsefulnessTracker:
    """Tracks learning usefulness feedback."""

    def __init__(self) -> None:
        self._records: dict[str, LearningFeedbackRecord] = {}

    def record(self, record: LearningFeedbackRecord) -> None:
        self._records[record.record_id] = record

    def get(self, record_id: str) -> LearningFeedbackRecord | None:
        return self._records.get(record_id)

    def list_by_date(self, date: str) -> list[LearningFeedbackRecord]:
        return [
            r for r in self._records.values() if r.date == date
        ]

    def list_all(self) -> list[LearningFeedbackRecord]:
        return list(self._records.values())

    def recommendation_acceptance_rate(self) -> float:
        records = [r for r in self._records.values()
                    if r.recommendation_accepted is not None]
        if not records:
            return 0.0
        return sum(1 for r in records if r.recommendation_accepted) / len(records)

    def usefulness_rate(self) -> dict[str, float]:
        rated = [r for r in self._records.values() if r.answer_usefulness]
        total = len(rated)
        if total == 0:
            return {"useful": 0.0, "partial": 0.0, "not_useful": 0.0}
        return {
            "useful": sum(1 for r in rated if r.answer_usefulness == "useful") / total,
            "partial": sum(1 for r in rated if r.answer_usefulness == "partial") / total,
            "not_useful": sum(1 for r in rated if r.answer_usefulness == "not_useful") / total,
        }

    def gap_detection_rate(self) -> float:
        total = len(self._records)
        if total == 0:
            return 0.0
        return sum(1 for r in self._records.values() if r.learning_gap_detected) / total

    def summary(self) -> dict[str, Any]:
        return {
            "total_records": len(self._records),
            "recommendation_acceptance_rate": self.recommendation_acceptance_rate(),
            "usefulness_distribution": self.usefulness_rate(),
            "gap_detection_rate": self.gap_detection_rate(),
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
    def load_json(cls, path: str) -> LearningUsefulnessTracker:
        tracker = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            record = LearningFeedbackRecord(
                record_id=item["record_id"],
                date=item["date"],
                learning_question=item.get("learning_question", ""),
                answer_usefulness=item.get("answer_usefulness"),
                learning_gap_detected=item.get("learning_gap_detected", False),
                learning_gap_description=item.get("learning_gap_description", ""),
                focus_recommendation=item.get("focus_recommendation", ""),
                recommendation_accepted=item.get("recommendation_accepted"),
                user_feedback=item.get("user_feedback"),
                created_at=item.get("created_at", ""),
            )
            tracker._records[record.record_id] = record
        return tracker

    def clear(self) -> None:
        self._records.clear()
