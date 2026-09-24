"""MemoryFeedback — tracks memory quality feedback.

Supported: Correct memory, Incorrect memory, Missing memory, Outdated memory.
Memory Gate remains the final control layer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class MemoryFeedbackRecord:
    """User feedback on memory quality for one event."""

    record_id: str
    event_id: str = ""
    memory_type: str = ""  # e.g. personal, research, learning, knowledge
    feedback_tag: str = ""  # CORRECT_MEMORY, INCORRECT_MEMORY, MISSING_MEMORY, OUTDATED_MEMORY
    description: str = ""
    memory_id: str | None = None
    user_note: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryFeedbackTracker:
    """Tracks memory quality feedback."""

    def __init__(self) -> None:
        self._records: dict[str, MemoryFeedbackRecord] = {}

    def record(self, record: MemoryFeedbackRecord) -> None:
        self._records[record.record_id] = record

    def get(self, record_id: str) -> MemoryFeedbackRecord | None:
        return self._records.get(record_id)

    def list_all(self) -> list[MemoryFeedbackRecord]:
        return list(self._records.values())

    def count_by_tag(self, tag: str) -> int:
        return sum(
            1 for r in self._records.values() if r.feedback_tag == tag
        )

    def correction_rate(self) -> float:
        total = len(self._records)
        if total == 0:
            return 0.0
        incorrect = self.count_by_tag("INCORRECT_MEMORY")
        missing = self.count_by_tag("MISSING_MEMORY")
        outdated = self.count_by_tag("OUTDATED_MEMORY")
        return (incorrect + missing + outdated) / total

    def summary(self) -> dict[str, Any]:
        return {
            "total_records": len(self._records),
            "correct_count": self.count_by_tag("CORRECT_MEMORY"),
            "incorrect_count": self.count_by_tag("INCORRECT_MEMORY"),
            "missing_count": self.count_by_tag("MISSING_MEMORY"),
            "outdated_count": self.count_by_tag("OUTDATED_MEMORY"),
            "correction_rate": self.correction_rate(),
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
    def load_json(cls, path: str) -> MemoryFeedbackTracker:
        tracker = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            record = MemoryFeedbackRecord(
                record_id=item["record_id"],
                event_id=item.get("event_id", ""),
                memory_type=item.get("memory_type", ""),
                feedback_tag=item.get("feedback_tag", ""),
                description=item.get("description", ""),
                memory_id=item.get("memory_id"),
                user_note=item.get("user_note", ""),
                created_at=item.get("created_at", ""),
            )
            tracker._records[record.record_id] = record
        return tracker

    def clear(self) -> None:
        self._records.clear()
