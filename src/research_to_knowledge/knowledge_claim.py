"""KnowledgeClaim model — a versioned, contextual knowledge statement.

KnowledgeClaim is the only way knowledge enters the Knowledge OS.
Every claim carries:
- context (scope, hardware, dataset, config)
- provenance chain back to its ResearchFinding
- version history (never deleted, only superseded)
- the ability to query historically, currently, or by context
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class KnowledgeClaimCategory(Enum):
    EXPERIMENT_RESULT = "EXPERIMENT_RESULT"
    INFERENCE = "INFERENCE"
    FACT = "FACT"              # Only after full integrity gate
    CONTEXTUAL_OBSERVATION = "CONTEXTUAL_OBSERVATION"


@dataclass
class KnowledgeVersion:
    """A single versioned snapshot of a KnowledgeClaim."""

    version_number: int
    statement: str
    category: KnowledgeClaimCategory
    context: dict[str, Any] = field(default_factory=dict)
    source_finding_id: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    superseded_at: str | None = None
    change_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d


@dataclass
class KnowledgeClaim:
    """A knowledge claim with full version history and provenance.

    Provenance chain:
        KnowledgeClaim
        → ResearchFinding
        → Conclusion
        → Inference
        → Observation
        → ExperimentResult
        → ExperimentRun
        → Code / Dataset / Hardware
    """

    claim_id: str
    topic: str                    # e.g. "tile-size-performance"
    slug: str                     # Unique stable identifier
    versions: list[KnowledgeVersion] = field(default_factory=list)
    provenance_chain: list[str] = field(default_factory=list)
    active_version: int = 0
    conflicts: list[str] = field(default_factory=list)  # ConflictRecord IDs
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def current(self) -> KnowledgeVersion | None:
        if not self.versions:
            return None
        return self.versions[self.active_version]

    def add_version(
        self,
        statement: str,
        category: KnowledgeClaimCategory,
        context: dict[str, Any] | None = None,
        source_finding_id: str | None = None,
        change_reason: str | None = None,
    ) -> KnowledgeVersion:
        if self.versions:
            self.versions[self.active_version].superseded_at = (
                datetime.now(timezone.utc).isoformat()
            )
        version_number = len(self.versions) + 1
        version = KnowledgeVersion(
            version_number=version_number,
            statement=statement,
            category=category,
            context=context or {},
            source_finding_id=source_finding_id,
            change_reason=change_reason,
        )
        self.versions.append(version)
        self.active_version = len(self.versions) - 1
        return version

    def get_version(self, version_number: int) -> KnowledgeVersion | None:
        for v in self.versions:
            if v.version_number == version_number:
                return v
        return None

    def get_context_matching(
        self, context_filter: dict[str, Any]
    ) -> list[KnowledgeVersion]:
        """Return versions whose context matches all filter keys."""
        results = []
        for v in self.versions:
            matches = all(
                v.context.get(k) == context_filter[k]
                for k in context_filter
            )
            if matches:
                results.append(v)
        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "topic": self.topic,
            "slug": self.slug,
            "versions": [v.to_dict() for v in self.versions],
            "provenance_chain": self.provenance_chain,
            "active_version": self.active_version,
            "conflicts": self.conflicts,
            "created_at": self.created_at,
        }


class KnowledgeClaimStore:
    """Manages all KnowledgeClaims with historical, current, and context queries."""

    def __init__(self) -> None:
        self._claims: dict[str, KnowledgeClaim] = {}

    def add(self, claim: KnowledgeClaim) -> None:
        self._claims[claim.claim_id] = claim

    def get(self, claim_id: str) -> KnowledgeClaim | None:
        return self._claims.get(claim_id)

    def get_by_slug(self, slug: str) -> KnowledgeClaim | None:
        for c in self._claims.values():
            if c.slug == slug:
                return c
        return None

    def list_by_topic(self, topic: str) -> list[KnowledgeClaim]:
        return [c for c in self._claims.values() if c.topic == topic]

    def list_all(self) -> list[KnowledgeClaim]:
        return list(self._claims.values())

    def query_current(self, topic: str) -> str | None:
        """Get the current statement for a topic."""
        claim = self.get_by_slug(topic)
        if claim and claim.current:
            return claim.current.statement
        return None

    def query_historical(self, topic: str) -> list[KnowledgeVersion]:
        """Get all versions (historical) for a topic."""
        claim = self.get_by_slug(topic)
        if claim:
            return claim.versions
        return []

    def query_context_specific(
        self, topic: str, context: dict[str, Any]
    ) -> list[KnowledgeVersion]:
        """Get versions matching the given context for a topic."""
        claim = self.get_by_slug(topic)
        if claim:
            return claim.get_context_matching(context)
        return []

    def add_conflict(
        self, claim_id: str, conflict_id: str
    ) -> KnowledgeClaim | None:
        claim = self._claims.get(claim_id)
        if claim and conflict_id not in claim.conflicts:
            claim.conflicts.append(conflict_id)
        return claim

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [c.to_dict() for c in self._claims.values()],
                f,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def load_json(cls, path: str) -> KnowledgeClaimStore:
        store = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            versions_data = item.pop("versions", [])
            claim = KnowledgeClaim(
                claim_id=item["claim_id"],
                topic=item["topic"],
                slug=item["slug"],
                provenance_chain=item.get("provenance_chain", []),
                active_version=item.get("active_version", 0),
                conflicts=item.get("conflicts", []),
                created_at=item.get("created_at", ""),
            )
            for vd in versions_data:
                version = KnowledgeVersion(
                    version_number=vd["version_number"],
                    statement=vd["statement"],
                    category=KnowledgeClaimCategory(vd["category"]),
                    context=vd.get("context", {}),
                    source_finding_id=vd.get("source_finding_id"),
                    created_at=vd.get("created_at", ""),
                    superseded_at=vd.get("superseded_at"),
                    change_reason=vd.get("change_reason"),
                )
                claim.versions.append(version)
            store.add(claim)
        return store
