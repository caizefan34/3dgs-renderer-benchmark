"""PersonalizationGain — measures whether PAI-OS personalization improves outcomes.

Evaluates:
  generic answer (no personal context) vs personalized answer (with PAI-OS context)

At least tests: learning, research, knowledge, planning.
Does NOT claim "personalized" just because context was loaded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class PersonalizationScore:
    """Score pair for a single test."""

    domain: str  # learning, research, knowledge, planning
    generic_answer: str = ""
    personalized_answer: str = ""
    generic_baseline_score: float = 0.0
    personalized_score: float = 0.0
    gain: float = 0.0  # personalized_score - generic_baseline_score
    user_feedback: str | None = None
    context_used: list[str] = field(default_factory=list)
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def compute_gain(self) -> None:
        self.gain = self.personalized_score - self.generic_baseline_score

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


class PersonalizationGainTracker:
    """Tracks personalization gain measurements."""

    def __init__(self) -> None:
        self._scores: dict[str, PersonalizationScore] = {}

    def record_score(self, score: PersonalizationScore) -> None:
        score.compute_gain()
        self._scores[score.domain] = score

    def get_score(self, domain: str) -> PersonalizationScore | None:
        return self._scores.get(domain)

    def list_all(self) -> list[PersonalizationScore]:
        return list(self._scores.values())

    def average_gain(self) -> float:
        if not self._scores:
            return 0.0
        return sum(s.gain for s in self._scores.values()) / len(self._scores)

    def average_gain_by_domain(self, domain: str) -> float:
        scores = [s for s in self._scores.values() if s.domain == domain]
        if not scores:
            return 0.0
        return sum(s.gain for s in scores) / len(scores)

    def overall_summary(self) -> dict[str, Any]:
        return {
            "domains_tested": list(self._scores.keys()),
            "average_gain": self.average_gain(),
            "domain_gains": {
                d: self.average_gain_by_domain(d)
                for d in self._scores
            },
            "num_measurements": len(self._scores),
        }

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [s.to_dict() for s in self._scores.values()],
                f,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def load_json(cls, path: str) -> PersonalizationGainTracker:
        tracker = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            score = PersonalizationScore(
                domain=item["domain"],
                generic_answer=item.get("generic_answer", ""),
                personalized_answer=item.get("personalized_answer", ""),
                generic_baseline_score=item.get(
                    "generic_baseline_score", 0.0
                ),
                personalized_score=item.get("personalized_score", 0.0),
                gain=item.get("gain", 0.0),
                user_feedback=item.get("user_feedback"),
                context_used=item.get("context_used", []),
                evaluated_at=item.get(
                    "evaluated_at",
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            tracker._scores[score.domain] = score
        return tracker

    def clear(self) -> None:
        self._scores.clear()
