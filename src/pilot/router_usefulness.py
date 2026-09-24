"""RouterUsefulness — tracks whether PAI-OS Router made the right decision.

Records: selected_route, user_feedback, manual_correction.
Identifies: unnecessary local downgrade, unnecessary cloud, wrong mode,
wrong context, wrong retrieval.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class RouterDecisionRecord:
    """Record of one Router decision and its outcome."""

    record_id: str
    event_id: str = ""
    selected_route: str = ""
    selected_mode: str = ""
    selected_provider: str = ""
    selected_model: str = ""
    retrieval_mode: str = ""
    context_types: list[str] = field(default_factory=list)
    user_feedback: str | None = None  # GOOD, PARTIAL, BAD
    manual_correction: str | None = None

    # Router error types
    unnecessary_local_downgrade: bool = False
    unnecessary_cloud: bool = False
    wrong_mode: bool = False
    wrong_context: bool = False
    wrong_retrieval: bool = False

    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def has_issue(self) -> bool:
        return any([
            self.unnecessary_local_downgrade,
            self.unnecessary_cloud,
            self.wrong_mode,
            self.wrong_context,
            self.wrong_retrieval,
        ])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RouterUsefulnessTracker:
    """Tracks Router decision quality."""

    def __init__(self) -> None:
        self._records: dict[str, RouterDecisionRecord] = {}

    def record(self, record: RouterDecisionRecord) -> None:
        self._records[record.record_id] = record

    def get(self, record_id: str) -> RouterDecisionRecord | None:
        return self._records.get(record_id)

    def list_all(self) -> list[RouterDecisionRecord]:
        return list(self._records.values())

    def correction_rate(self) -> float:
        total = len(self._records)
        if total == 0:
            return 0.0
        return (
            sum(1 for r in self._records.values() if r.manual_correction)
            / total
        )

    def issue_rate(self, issue_type: str) -> float:
        """Rate of a specific issue type."""
        total = len(self._records)
        if total == 0:
            return 0.0
        count = sum(
            1 for r in self._records.values()
            if getattr(r, issue_type, False)
        )
        return count / total

    def good_decision_rate(self) -> float:
        total = len(self._records)
        if total == 0:
            return 0.0
        good = sum(
            1 for r in self._records.values()
            if not r.has_issue
        )
        return good / total

    def summary(self) -> dict[str, Any]:
        return {
            "total_decisions": len(self._records),
            "good_decision_rate": self.good_decision_rate(),
            "correction_rate": self.correction_rate(),
            "issue_rates": {
                "unnecessary_local_downgrade": self.issue_rate(
                    "unnecessary_local_downgrade"
                ),
                "unnecessary_cloud": self.issue_rate("unnecessary_cloud"),
                "wrong_mode": self.issue_rate("wrong_mode"),
                "wrong_context": self.issue_rate("wrong_context"),
                "wrong_retrieval": self.issue_rate("wrong_retrieval"),
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
    def load_json(cls, path: str) -> RouterUsefulnessTracker:
        tracker = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            record = RouterDecisionRecord(
                record_id=item["record_id"],
                event_id=item.get("event_id", ""),
                selected_route=item.get("selected_route", ""),
                selected_mode=item.get("selected_mode", ""),
                selected_provider=item.get("selected_provider", ""),
                selected_model=item.get("selected_model", ""),
                retrieval_mode=item.get("retrieval_mode", ""),
                context_types=item.get("context_types", []),
                user_feedback=item.get("user_feedback"),
                manual_correction=item.get("manual_correction"),
                unnecessary_local_downgrade=item.get(
                    "unnecessary_local_downgrade", False
                ),
                unnecessary_cloud=item.get("unnecessary_cloud", False),
                wrong_mode=item.get("wrong_mode", False),
                wrong_context=item.get("wrong_context", False),
                wrong_retrieval=item.get("wrong_retrieval", False),
                created_at=item.get("created_at", ""),
            )
            tracker._records[record.record_id] = record
        return tracker

    def clear(self) -> None:
        self._records.clear()
