"""ResearchFinding model — a contextual, evidence-bound research conclusion.

Every ResearchFinding carries its scope, context, provenance chain,
alternative explanations, contradicting evidence, and reviewer verdict.
No finding can be promoted to FACT without passing the integrity gate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ClaimType(Enum):
    """What kind of claim a ResearchFinding makes."""

    EXPERIMENT_RESULT = "EXPERIMENT_RESULT"
    INFERENCE = "INFERENCE"
    # FACT is intentionally absent — it can only exist in KnowledgeClaim
    # after the integrity gate has been passed.


class ResearchReviewerVerdict(Enum):
    APPROVED = "approved"
    WARNING = "warning"
    INCONCLUSIVE = "inconclusive"


@dataclass
class ScopeMetadata:
    """Immutable record of what this finding covers."""

    project_id: str
    question_id: str | None = None
    hypothesis_id: str | None = None
    experiment_id: str | None = None
    run_ids: list[str] = field(default_factory=list)
    dataset: str | None = None
    hardware: str | None = None
    software: dict[str, str] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    code_commit: str | None = None
    scope: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchFinding:
    """A contextual research finding, not a universal fact.

    Must carry:
    - a qualified claim statement (not overgeneralized)
    - scope metadata (what was actually tested)
    - evidence chain (raw results it came from)
    - alternative explanations
    - contradicting evidence
    - a reviewer verdict before it can be promoted
    """

    finding_id: str
    claim_type: ClaimType
    statement: str  # Qualified, contextual statement
    scope: ScopeMetadata
    evidence_summary: str
    raw_evidence_uris: list[str] = field(default_factory=list)
    alternative_explanations: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    uncertainty: str | None = None  # e.g. "CI95 overlap; insufficient statistical power"
    reviewer_verdict: ResearchReviewerVerdict | None = None
    reviewer_notes: str | None = None
    promoted_to_knowledge: bool = False
    rejected: bool = False
    rejection_reason: str | None = None
    quarantine: bool = False
    quarantine_reason: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["claim_type"] = self.claim_type.value
        d["reviewer_verdict"] = (
            self.reviewer_verdict.value if self.reviewer_verdict else None
        )
        return d


class ResearchFindingStore:
    """In-memory store for ResearchFindings with simple query support."""

    def __init__(self) -> None:
        self._findings: dict[str, ResearchFinding] = {}

    def add(self, finding: ResearchFinding) -> None:
        self._findings[finding.finding_id] = finding

    def get(self, finding_id: str) -> ResearchFinding | None:
        return self._findings.get(finding_id)

    def list_by_project(self, project_id: str) -> list[ResearchFinding]:
        return [
            f
            for f in self._findings.values()
            if f.scope.project_id == project_id
        ]

    def list_all(self) -> list[ResearchFinding]:
        return list(self._findings.values())

    def promote(self, finding_id: str) -> ResearchFinding | None:
        finding = self._findings.get(finding_id)
        if finding and not finding.rejected:
            finding.promoted_to_knowledge = True
        return finding

    def reject(self, finding_id: str, reason: str) -> ResearchFinding | None:
        finding = self._findings.get(finding_id)
        if finding:
            finding.rejected = True
            finding.rejection_reason = reason
        return finding

    def quarantine_record(
        self, finding_id: str, reason: str
    ) -> ResearchFinding | None:
        finding = self._findings.get(finding_id)
        if finding:
            finding.quarantine = True
            finding.quarantine_reason = reason
            finding.rejected = True
        return finding

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [f.to_dict() for f in self._findings.values()],
                f,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def load_json(cls, path: str) -> ResearchFindingStore:
        store = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            scope = ScopeMetadata(**item.pop("scope"))
            finding = ResearchFinding(
                finding_id=item["finding_id"],
                claim_type=ClaimType(item["claim_type"]),
                statement=item["statement"],
                scope=scope,
                evidence_summary=item.get("evidence_summary", ""),
                raw_evidence_uris=item.get("raw_evidence_uris", []),
                alternative_explanations=item.get(
                    "alternative_explanations", []
                ),
                contradicting_evidence=item.get("contradicting_evidence", []),
                uncertainty=item.get("uncertainty"),
                reviewer_verdict=(
                    ResearchReviewerVerdict(item["reviewer_verdict"])
                    if item.get("reviewer_verdict")
                    else None
                ),
                reviewer_notes=item.get("reviewer_notes"),
                promoted_to_knowledge=item.get("promoted_to_knowledge", False),
                rejected=item.get("rejected", False),
                rejection_reason=item.get("rejection_reason"),
                quarantine=item.get("quarantine", False),
                quarantine_reason=item.get("quarantine_reason"),
                created_at=item.get("created_at", ""),
            )
            store.add(finding)
        return store
