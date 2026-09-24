"""Feedback model — allows the user to rate responses after every answer.

Rating:
  GOOD / PARTIAL / BAD

Reason codes:
  WRONG_MODEL, BAD_RETRIEVAL, MISSING_CONTEXT, WRONG_MEMORY,
  WRONG_LEARNING_RECOMMENDATION, WRONG_RESEARCH_CONTEXT

Context usefulness tags:
  HELPFUL_CONTEXT, IRRELEVANT_CONTEXT, MISSING_CONTEXT, CONFLICTING_CONTEXT
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class FeedbackRating(Enum):
    GOOD = "GOOD"
    PARTIAL = "PARTIAL"
    BAD = "BAD"


class FeedbackReason(Enum):
    WRONG_MODEL = "WRONG_MODEL"
    BAD_RETRIEVAL = "BAD_RETRIEVAL"
    MISSING_CONTEXT = "MISSING_CONTEXT"
    WRONG_MEMORY = "WRONG_MEMORY"
    WRONG_LEARNING_RECOMMENDATION = "WRONG_LEARNING_RECOMMENDATION"
    WRONG_RESEARCH_CONTEXT = "WRONG_RESEARCH_CONTEXT"


class ContextUsefulnessTag(Enum):
    HELPFUL_CONTEXT = "HELPFUL_CONTEXT"
    IRRELEVANT_CONTEXT = "IRRELEVANT_CONTEXT"
    MISSING_CONTEXT = "MISSING_CONTEXT"
    CONFLICTING_CONTEXT = "CONFLICTING_CONTEXT"


class MemoryFeedbackTag(Enum):
    CORRECT_MEMORY = "CORRECT_MEMORY"
    INCORRECT_MEMORY = "INCORRECT_MEMORY"
    MISSING_MEMORY = "MISSING_MEMORY"
    OUTDATED_MEMORY = "OUTDATED_MEMORY"


class DailyReviewUsefulness(Enum):
    USEFUL = "useful"
    PARTIALLY_USEFUL = "partially_useful"
    NOT_USEFUL = "not_useful"


@dataclass
class UserFeedback:
    """Feedback on a single PAI-OS response."""

    feedback_id: str = field(
        default_factory=lambda: f"fb_{uuid.uuid4().hex[:12]}"
    )
    event_id: str = ""
    rating: str = FeedbackRating.GOOD.value
    reason_codes: list[str] = field(default_factory=list)
    context_tags: list[str] = field(default_factory=list)
    free_text: str = ""
    data_source: str = "REAL_USER"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rating"] = self.rating
        return d


@dataclass
class ManualCorrection:
    """A manual user correction recorded as feedback.

    When a user says things like:
    - "这里应该用 Cloud。"
    - "这里不应该查我的研究记录。"
    - "这是我自己的假设，不是事实。"

    We record it without modifying authoritative state.
    """

    correction_id: str = field(
        default_factory=lambda: f"corr_{uuid.uuid4().hex[:12]}"
    )
    event_id: str = ""
    original_text: str = ""
    correction_type: str = ""  # e.g. WRONG_MODEL, BAD_RETRIEVAL, WRONG_MEMORY
    context: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    data_source: str = "REAL_USER"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeedbackStore:
    """Append-only store for user feedback and corrections."""

    def __init__(self) -> None:
        self._feedback: list[UserFeedback] = []
        self._corrections: list[ManualCorrection] = []

    def add_feedback(self, feedback: UserFeedback) -> None:
        self._feedback.append(feedback)

    def add_correction(self, correction: ManualCorrection) -> None:
        self._corrections.append(correction)

    def list_feedback(
        self, data_source: str | None = None
    ) -> list[UserFeedback]:
        if data_source:
            return [f for f in self._feedback if f.data_source == data_source]
        return list(self._feedback)

    def list_corrections(
        self, data_source: str | None = None
    ) -> list[ManualCorrection]:
        if data_source:
            return [
                c for c in self._corrections if c.data_source == data_source
            ]
        return list(self._corrections)

    def count_feedback_by_rating(self, rating: str) -> int:
        return sum(1 for f in self._feedback if f.rating == rating)

    def count_feedback_by_reason(self, reason: str) -> int:
        return sum(
            1 for f in self._feedback if reason in f.reason_codes
        )

    def count_corrections_by_type(self, correction_type: str) -> int:
        return sum(
            1 for c in self._corrections
            if c.correction_type == correction_type
        )

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "feedback": [fb.to_dict() for fb in self._feedback],
                    "corrections": [
                        c.to_dict() for c in self._corrections
                    ],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def load_json(cls, path: str) -> FeedbackStore:
        store = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("feedback", []):
            fb = UserFeedback(
                feedback_id=item.get("feedback_id", ""),
                event_id=item.get("event_id", ""),
                rating=item.get("rating", FeedbackRating.GOOD.value),
                reason_codes=item.get("reason_codes", []),
                context_tags=item.get("context_tags", []),
                free_text=item.get("free_text", ""),
                data_source=item.get("data_source", "REAL_USER"),
                created_at=item.get("created_at", ""),
            )
            store._feedback.append(fb)
        for item in data.get("corrections", []):
            corr = ManualCorrection(
                correction_id=item.get("correction_id", ""),
                event_id=item.get("event_id", ""),
                original_text=item.get("original_text", ""),
                correction_type=item.get("correction_type", ""),
                context=item.get("context", ""),
                created_at=item.get("created_at", ""),
                data_source=item.get("data_source", "REAL_USER"),
            )
            store._corrections.append(corr)
        return store

    def clear(self) -> None:
        """Clear all data (test helper)."""
        self._feedback.clear()
        self._corrections.clear()
