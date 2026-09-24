"""Conflict Detection — manages contextual conflicts between knowledge claims.

When a new ResearchFinding contradicts existing knowledge:
1. Old claim is NOT deleted
2. A ConflictRecord is created linking both claims with their scope/context
3. Both old and new claims coexist with versioning
4. Historical queries can trace the evolution
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ConflictResolution(Enum):
    CONTEXTUAL = "contextual"               # Both valid in different contexts
    SUPERSEDED = "superseded"                # New replaces old (rare, needs full evidence)
    INCONCLUSIVE = "inconclusive"            # Cannot resolve
    COMPLEMENTARY = "complementary"          # Both contribute to understanding


@dataclass
class ConflictRecord:
    """A record of conflicting knowledge claims."""

    conflict_id: str
    old_claim_id: str
    old_statement: str
    old_context: dict[str, Any]
    old_version: int
    new_claim_id: str
    new_statement: str
    new_context: dict[str, Any]
    new_version: int
    resolution: ConflictResolution
    resolution_reason: str
    source_finding_id: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["resolution"] = self.resolution.value
        return d


class ConflictDetector:
    """Detects conflicts between old knowledge claims and new research findings."""

    def __init__(self) -> None:
        self._conflicts: dict[str, ConflictRecord] = {}

    def detect(
        self,
        old_claim: "KnowledgeClaim | None",
        new_statement: str,
        new_finding_id: str,
        new_context: dict[str, Any],
    ) -> ConflictRecord | None:
        """Detect if a new finding conflicts with existing knowledge.

        Returns a ConflictRecord if there is a meaningful divergence,
        or None if the statements are compatible.
        """
        if old_claim is None or old_claim.current is None:
            return None

        old = old_claim.current
        if self._are_compatible(old.statement, new_statement):
            return None

        suffix = hashlib.md5(f"{new_finding_id}_{datetime.now(timezone.utc).isoformat()}".encode()).hexdigest()[:8]
        conflict_id = (
            f"conflict_{old_claim.claim_id}_vs_{new_finding_id[:8]}_{suffix}"
        )
        resolution, reason = self._determine_resolution(
            old.statement, old.context, new_statement, new_context
        )

        record = ConflictRecord(
            conflict_id=conflict_id,
            old_claim_id=old_claim.claim_id,
            old_statement=old.statement,
            old_context=old.context,
            old_version=old.version_number,
            new_claim_id=new_finding_id,
            new_statement=new_statement,
            new_context=new_context,
            new_version=len(old_claim.versions) + 1,
            resolution=resolution,
            resolution_reason=reason,
            source_finding_id=new_finding_id,
        )
        self._conflicts[conflict_id] = record
        return record

    def get(self, conflict_id: str) -> ConflictRecord | None:
        return self._conflicts.get(conflict_id)

    def list_all(self) -> list[ConflictRecord]:
        return list(self._conflicts.values())

    def list_by_claim(self, claim_id: str) -> list[ConflictRecord]:
        return [
            c
            for c in self._conflicts.values()
            if c.old_claim_id == claim_id or c.new_claim_id == claim_id
        ]

    def _are_compatible(
        self, old: str, new: str
    ) -> bool:
        """Check if two statements are semantically compatible.

        A new finding is compatible if it:
        - qualifies/refines an old claim (e.g. adds "insufficient evidence")
        - contains context markers that limit its scope
        - is a direct refinement of the old claim
        """
        old_lower = old.lower()
        new_lower = new.lower()

        # "insufficient evidence" is a refinement, not a contradiction
        if "insufficient evidence" in new_lower:
            return True

        # If old claims "X is faster" and new says "not universally faster",
        # they are compatible (contextual refinement)
        if "not " in new_lower and old_lower.split(".")[0] in new_lower:
            return True

        # "not universally" explicitly marks a refinement
        if "not universally" in new_lower:
            return True

        # If old has a general positive statement and new has a qualified one
        # with workload-dependent / context-dependent qualifiers
        if "depend" in new_lower and "faster" in old_lower:
            return True

        return False

    def _determine_resolution(
        self,
        old_statement: str,
        old_context: dict[str, Any],
        new_statement: str,
        new_context: dict[str, Any],
    ) -> tuple[ConflictResolution, str]:
        """Determine how to resolve the conflict."""
        old_lower = old_statement.lower()
        new_lower = new_statement.lower()

        # Contextual conflict — different hardware, dataset, etc.
        if old_context.get("hardware") != new_context.get("hardware"):
            return (
                ConflictResolution.CONTEXTUAL,
                f"Different hardware: {old_context.get('hardware')} vs {new_context.get('hardware')}",
            )
        if old_context.get("dataset") != new_context.get("dataset"):
            return (
                ConflictResolution.CONTEXTUAL,
                f"Different dataset: {old_context.get('dataset')} vs {new_context.get('dataset')}",
            )

        # Inconclusive — same context but different results
        return (
            ConflictResolution.INCONCLUSIVE,
            "Same context yields different results; needs additional experiments",
        )

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [c.to_dict() for c in self._conflicts.values()],
                f,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def load_json(cls, path: str) -> ConflictDetector:
        detector = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            record = ConflictRecord(
                conflict_id=item["conflict_id"],
                old_claim_id=item["old_claim_id"],
                old_statement=item["old_statement"],
                old_context=item.get("old_context", {}),
                old_version=item.get("old_version", 1),
                new_claim_id=item["new_claim_id"],
                new_statement=item["new_statement"],
                new_context=item.get("new_context", {}),
                new_version=item.get("new_version", 1),
                resolution=ConflictResolution(item["resolution"]),
                resolution_reason=item.get("resolution_reason", ""),
                source_finding_id=item.get("source_finding_id"),
                created_at=item.get("created_at", ""),
            )
            detector._conflicts[record.conflict_id] = record
        return detector
