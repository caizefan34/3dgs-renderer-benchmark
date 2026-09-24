"""DailyReviewFeedback — tracks daily review usefulness.

Users can rate: Today Focus, Evening Review, Tomorrow Plan
as: useful / partially useful / not useful.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from typing import Any


@dataclass
class DailyReviewFeedbackRecord:
    """User feedback on one daily review component."""

    record_id: str
    date: str = field(
        default_factory=lambda: date.today().isoformat()
    )
    component: str = ""  # today_focus, evening_review, tomorrow_plan
    usefulness: str = ""  # useful, partially_useful, not_useful
    user_note: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DailyReviewFeedbackTracker:
    """Tracks daily review usefulness feedback."""

    def __init__(self) -> None:
        self._records: dict[str, DailyReviewFeedbackRecord] = {}

    def record(self, record: DailyReviewFeedbackRecord) -> None:
        self._records[record.record_id] = record

    def get(self, record_id: str) -> DailyReviewFeedbackRecord | None:
        return self._records.get(record_id)

    def list_by_date(self, date_str: str) -> list[DailyReviewFeedbackRecord]:
        return [
            r for r in self._records.values() if r.date == date_str
        ]

    def list_by_component(
        self, component: str
    ) -> list[DailyReviewFeedbackRecord]:
        return [
            r for r in self._records.values() if r.component == component
        ]

    def list_all(self) -> list[DailyReviewFeedbackRecord]:
        return list(self._records.values())

    def usefulness_by_component(
        self, component: str
    ) -> dict[str, float]:
        records = self.list_by_component(component)
        total = len(records)
        if total == 0:
            return {"useful": 0.0, "partially_useful": 0.0, "not_useful": 0.0}
        return {
            "useful": sum(1 for r in records if r.usefulness == "useful") / total,
            "partially_useful": sum(
                1 for r in records if r.usefulness == "partially_useful"
            ) / total,
            "not_useful": sum(1 for r in records if r.usefulness == "not_useful") / total,
        }

    def overall_summary(self) -> dict[str, Any]:
        return {
            "total_records": len(self._records),
            "today_focus": self.usefulness_by_component("today_focus"),
            "evening_review": self.usefulness_by_component("evening_review"),
            "tomorrow_plan": self.usefulness_by_component("tomorrow_plan"),
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
    def load_json(cls, path: str) -> DailyReviewFeedbackTracker:
        tracker = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            record = DailyReviewFeedbackRecord(
                record_id=item["record_id"],
                date=item.get("date", ""),
                component=item.get("component", ""),
                usefulness=item.get("usefulness", ""),
                user_note=item.get("user_note", ""),
                created_at=item.get("created_at", ""),
            )
            tracker._records[record.record_id] = record
        return tracker

    def clear(self) -> None:
        self._records.clear()
