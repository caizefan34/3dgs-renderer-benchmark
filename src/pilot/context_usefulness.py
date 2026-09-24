"""ContextUsefulness — tracks how useful the retrieved context actually was.

Does NOT claim context was useful just because it was loaded.
User feedback on context: HELPFUL, IRRELEVANT, MISSING, CONFLICTING.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class ContextFeedbackRecord:
    """Record of user feedback on context usefulness for one event."""

    event_id: str
    num_context_items: int = 0
    helpful_count: int = 0
    irrelevant_count: int = 0
    missing_context: bool = False
    conflicting_context: bool = False
    feedback_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def helpful_ratio(self) -> float:
        if self.num_context_items == 0:
            return 0.0
        return self.helpful_count / self.num_context_items

    @property
    def irrelevant_ratio(self) -> float:
        if self.num_context_items == 0:
            return 0.0
        return self.irrelevant_count / self.num_context_items

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContextUsefulnessTracker:
    """Tracks context usefulness feedback across all events."""

    def __init__(self) -> None:
        self._records: dict[str, ContextFeedbackRecord] = {}

    def record(self, record: ContextFeedbackRecord) -> None:
        self._records[record.event_id] = record

    def get(self, event_id: str) -> ContextFeedbackRecord | None:
        return self._records.get(event_id)

    def list_all(self) -> list[ContextFeedbackRecord]:
        return list(self._records.values())

    def overall_success_rate(self) -> float:
        records = self.list_all()
        if not records:
            return 0.0
        total_helpful = sum(r.helpful_count for r in records)
        total_items = sum(r.num_context_items for r in records)
        if total_items == 0:
            return 0.0
        return total_helpful / total_items

    def missing_context_rate(self) -> float:
        records = self.list_all()
        if not records:
            return 0.0
        return sum(1 for r in records if r.missing_context) / len(records)

    def conflicting_context_rate(self) -> float:
        records = self.list_all()
        if not records:
            return 0.0
        return sum(1 for r in records if r.conflicting_context) / len(records)

    def summary(self) -> dict[str, Any]:
        records = self.list_all()
        return {
            "total_events": len(records),
            "overall_success_rate": self.overall_success_rate(),
            "missing_context_rate": self.missing_context_rate(),
            "conflicting_context_rate": self.conflicting_context_rate(),
            "avg_helpful_ratio": (
                sum(r.helpful_ratio for r in records) / len(records)
                if records
                else 0.0
            ),
            "avg_irrelevant_ratio": (
                sum(r.irrelevant_ratio for r in records) / len(records)
                if records
                else 0.0
            ),
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
    def load_json(cls, path: str) -> ContextUsefulnessTracker:
        tracker = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            record = ContextFeedbackRecord(
                event_id=item["event_id"],
                num_context_items=item.get("num_context_items", 0),
                helpful_count=item.get("helpful_count", 0),
                irrelevant_count=item.get("irrelevant_count", 0),
                missing_context=item.get("missing_context", False),
                conflicting_context=item.get("conflicting_context", False),
                feedback_at=item.get(
                    "feedback_at",
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            tracker._records[record.event_id] = record
        return tracker

    def clear(self) -> None:
        self._records.clear()
