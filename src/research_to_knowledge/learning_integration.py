"""Learning Integration — feeds research findings into Learning OS.

Research findings are integrated into LearningNodes as LearningEvidence.
Knowledge gaps are generated for unresolved questions.
Focus recommendations are created when new investigation is warranted.

Critical principle:
- Research activity alone does NOT trigger MASTERED state
- Learning state changes require: research activity + explanation + implementation + review
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EvidenceType(Enum):
    EXPERIMENT_RESULT = "EXPERIMENT_RESULT"
    INFERENCE = "INFERENCE"
    RESEARCH = "RESEARCH"


@dataclass
class LearningEvidence:
    """A piece of evidence attached to a LearningNode."""

    evidence_id: str
    node_name: str
    evidence_type: EvidenceType
    statement: str
    source_finding_id: str
    context: dict[str, Any] = field(default_factory=dict)
    uncertainty: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence_type"] = self.evidence_type.value
        return d


@dataclass
class LearningGap:
    """A knowledge gap identified from research findings."""

    gap_id: str
    node_name: str
    question: str
    source_finding_id: str
    reason: str
    priority: str = "medium"  # low, medium, high
    evidence: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved: bool = False
    resolved_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class FocusRecommendation:
    """A recommendation to focus learning on a specific gap."""

    rec_id: str
    gap_id: str
    node_name: str
    title: str
    reason: str
    priority: str
    evidence: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    implemented: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


class LearningIntegration:
    """Integrates research findings into Learning OS components."""

    def __init__(self) -> None:
        self._evidences: dict[str, LearningEvidence] = {}
        self._gaps: dict[str, LearningGap] = {}
        self._focus_recs: dict[str, FocusRecommendation] = {}

    def add_evidence(
        self,
        node_name: str,
        evidence_type: EvidenceType,
        statement: str,
        source_finding_id: str,
        context: dict[str, Any] | None = None,
        uncertainty: str | None = None,
    ) -> LearningEvidence:
        evidence_id = f"lev_{node_name}_{source_finding_id[:8]}"
        evidence = LearningEvidence(
            evidence_id=evidence_id,
            node_name=node_name,
            evidence_type=evidence_type,
            statement=statement,
            source_finding_id=source_finding_id,
            context=context or {},
            uncertainty=uncertainty,
        )
        self._evidences[evidence_id] = evidence
        return evidence

    def identify_gap(
        self,
        node_name: str,
        question: str,
        source_finding_id: str,
        reason: str,
        priority: str = "medium",
        evidence: str | None = None,
    ) -> LearningGap:
        gap_id = f"gap_{node_name}_{source_finding_id[:8]}"
        gap = LearningGap(
            gap_id=gap_id,
            node_name=node_name,
            question=question,
            source_finding_id=source_finding_id,
            reason=reason,
            priority=priority,
            evidence=evidence,
        )
        self._gaps[gap_id] = gap
        return gap

    def create_focus_recommendation(
        self, gap: LearningGap, title: str
    ) -> FocusRecommendation:
        rec_id = f"rec_{gap.gap_id}"
        rec = FocusRecommendation(
            rec_id=rec_id,
            gap_id=gap.gap_id,
            node_name=gap.node_name,
            title=title,
            reason=gap.reason,
            priority=gap.priority,
            evidence=gap.evidence,
        )
        self._focus_recs[rec_id] = rec
        return rec

    def get_evidence(self, evidence_id: str) -> LearningEvidence | None:
        return self._evidences.get(evidence_id)

    def get_gap(self, gap_id: str) -> LearningGap | None:
        return self._gaps.get(gap_id)

    def get_focus_rec(self, rec_id: str) -> FocusRecommendation | None:
        return self._focus_recs.get(rec_id)

    def list_evidence(self, node_name: str | None = None) -> list[LearningEvidence]:
        if node_name:
            return [e for e in self._evidences.values() if e.node_name == node_name]
        return list(self._evidences.values())

    def list_gaps(self, node_name: str | None = None) -> list[LearningGap]:
        if node_name:
            return [g for g in self._gaps.values() if g.node_name == node_name]
        return list(self._gaps.values())

    def list_focus_recs(
        self, node_name: str | None = None
    ) -> list[FocusRecommendation]:
        if node_name:
            return [
                r for r in self._focus_recs.values() if r.node_name == node_name
            ]
        return list(self._focus_recs.values())

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "evidences": [e.to_dict() for e in self._evidences.values()],
                    "gaps": [g.to_dict() for g in self._gaps.values()],
                    "focus_recommendations": [
                        r.to_dict() for r in self._focus_recs.values()
                    ],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def load_json(cls, path: str) -> LearningIntegration:
        obj = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("evidences", []):
            ev = LearningEvidence(
                evidence_id=item["evidence_id"],
                node_name=item["node_name"],
                evidence_type=EvidenceType(item["evidence_type"]),
                statement=item["statement"],
                source_finding_id=item["source_finding_id"],
                context=item.get("context", {}),
                uncertainty=item.get("uncertainty"),
                created_at=item.get("created_at", ""),
            )
            obj._evidences[ev.evidence_id] = ev
        for item in data.get("gaps", []):
            gap = LearningGap(
                gap_id=item["gap_id"],
                node_name=item["node_name"],
                question=item["question"],
                source_finding_id=item["source_finding_id"],
                reason=item.get("reason", ""),
                priority=item.get("priority", "medium"),
                evidence=item.get("evidence"),
                created_at=item.get("created_at", ""),
                resolved=item.get("resolved", False),
                resolved_at=item.get("resolved_at"),
            )
            obj._gaps[gap.gap_id] = gap
        for item in data.get("focus_recommendations", []):
            rec = FocusRecommendation(
                rec_id=item["rec_id"],
                gap_id=item["gap_id"],
                node_name=item["node_name"],
                title=item["title"],
                reason=item["reason"],
                priority=item["priority"],
                evidence=item.get("evidence"),
                created_at=item.get("created_at", ""),
                implemented=item.get("implemented", False),
            )
            obj._focus_recs[rec.rec_id] = rec
        return obj
