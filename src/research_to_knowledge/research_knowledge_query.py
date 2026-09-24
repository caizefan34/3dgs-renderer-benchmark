"""Research Knowledge Query Engine — answers user questions with evidence.

Handles:
- Current-state queries: "Is tile32 faster?"
- Historical queries: "Why did you think tile32 was faster before?"
- Contradiction queries: "Why do different experiments give different results?"
- Knowledge impact reports

Answers always:
- Cite specific evidence
- Show scope/context
- Report uncertainty
- Preserve contradictions
- Never fabricate facts
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .research_finding import ResearchFindingStore, ResearchFinding
from .knowledge_claim import KnowledgeClaimStore, KnowledgeClaim
from .conflict_detection import ConflictDetector, ConflictRecord
from .learning_integration import LearningIntegration


@dataclass
class QueryResult:
    """Structured answer for a research knowledge query."""

    question: str
    answer: str
    evidence_cited: list[str] = field(default_factory=list)
    scope_context: dict[str, Any] = field(default_factory=dict)
    uncertainty: str | None = None
    contradictions: list[dict[str, Any]] = field(default_factory=list)
    knowledge_versions: list[dict[str, Any]] = field(default_factory=list)
    related_gaps: list[str] = field(default_factory=list)
    raw_findings_cited: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "evidence_cited": self.evidence_cited,
            "scope_context": self.scope_context,
            "uncertainty": self.uncertainty,
            "contradictions": self.contradictions,
            "knowledge_versions": self.knowledge_versions,
            "related_gaps": self.related_gaps,
            "raw_findings_cited": self.raw_findings_cited,
        }


class ResearchKnowledgeQueryEngine:
    """Answers user questions by searching across research findings,
    knowledge claims, conflicts, and learning gaps."""

    def __init__(
        self,
        finding_store: ResearchFindingStore,
        knowledge_store: KnowledgeClaimStore,
        conflict_detector: ConflictDetector,
        learning_integration: LearningIntegration,
    ) -> None:
        self._finding_store = finding_store
        self._knowledge_store = knowledge_store
        self._conflict_detector = conflict_detector
        self._learning = learning_integration

    def query_current_knowledge(
        self, topic: str, context_filter: dict[str, Any] | None = None
    ) -> QueryResult:
        """Answer a current-state question about a topic.

        E.g. "Is tile32 faster?"
        """
        claim = self._knowledge_store.get_by_slug(topic)
        if not claim:
            return QueryResult(
                question=f"Current knowledge about '{topic}'",
                answer=(
                    f"No knowledge claim found for topic '{topic}'. "
                    "No research finding has been promoted yet."
                ),
            )

        current = claim.current
        if current is None:
            return QueryResult(
                question=f"Current knowledge about '{topic}'",
                answer="Knowledge claim exists but has no active version.",
            )

        # Build answer with context and uncertainty
        answer_parts = [f"Current knowledge: {current.statement}"]
        if current.context:
            context_keys = [
                k for k in ("hardware", "dataset", "scope")
                if k in current.context
            ]
            if context_keys:
                ctx_str = "; ".join(
                    f"{k}={current.context[k]}" for k in context_keys
                )
                answer_parts.append(f"Evaluated context: {ctx_str}")

        # Check for conflicting versions
        contradictions = self._build_contradiction_info(claim)

        # Gather related learning gaps
        gaps = self._learning.list_gaps()

        return QueryResult(
            question=f"Current knowledge about '{topic}'",
            answer="\n".join(answer_parts),
            evidence_cited=[current.statement],
            scope_context=current.context,
            uncertainty=(
                "Insufficient evidence for universal claim"
                if current.category.value != "FACT"
                else None
            ),
            contradictions=contradictions,
            knowledge_versions=[
                {"version": v.version_number, "statement": v.statement}
                for v in claim.versions
            ],
            related_gaps=[
                g.question for g in gaps
                if topic.lower() in g.node_name.lower() or topic.lower() in g.question.lower()
            ],
            raw_findings_cited=claim.provenance_chain,
        )

    def query_historical(self, topic: str) -> QueryResult:
        """Answer a historical question about a topic.

        E.g. "Why did you think tile32 was faster before?"
        """
        claim = self._knowledge_store.get_by_slug(topic)
        if not claim:
            return QueryResult(
                question=f"Historical knowledge about '{topic}'",
                answer=(
                    f"No historical knowledge found for topic '{topic}'."
                ),
            )

        if len(claim.versions) <= 1:
            return QueryResult(
                question=f"Historical knowledge about '{topic}'",
                answer=(
                    f"Only one version exists for '{topic}'. "
                    f"Current: {claim.current.statement if claim.current else 'N/A'}"
                ),
                knowledge_versions=[
                    {"version": v.version_number, "statement": v.statement,
                     "created_at": v.created_at, "change_reason": v.change_reason}
                    for v in claim.versions
                ],
                raw_findings_cited=claim.provenance_chain,
            )

        # Build historical narrative
        parts = [
            f"The topic '{topic}' has {len(claim.versions)} versions:"
        ]
        for v in claim.versions:
            reason = v.change_reason or "initial creation"
            parts.append(
                f"  Version {v.version_number}: \"{v.statement}\" "
                f"(reason: {reason})"
            )

        return QueryResult(
            question=f"Historical knowledge about '{topic}'",
            answer="\n".join(parts),
            knowledge_versions=[
                {"version": v.version_number, "statement": v.statement,
                 "created_at": v.created_at,
                 "superseded_at": v.superseded_at,
                 "change_reason": v.change_reason}
                for v in claim.versions
            ],
            raw_findings_cited=claim.provenance_chain,
        )

    def query_contradictions(
        self, topic: str | None = None
    ) -> QueryResult:
        """Answer why different experiments give different results."""
        conflicts = self._conflict_detector.list_all()
        if topic:
            conflicts = [
                c for c in conflicts
                if topic.lower() in c.old_statement.lower()
                or topic.lower() in c.new_statement.lower()
            ]

        if not conflicts:
            return QueryResult(
                question="Contradictions in research",
                answer="No contradictions are currently recorded.",
            )

        parts = [
            f"Found {len(conflicts)} contradiction(s):"
        ]
        for c in conflicts:
            parts.append(
                f"\n  - Old (v{c.old_version}): \"{c.old_statement}\""
                f"\n  - New (v{c.new_version}): \"{c.new_statement}\""
                f"\n  - Resolution: {c.resolution.value}"
                f"\n  - Reason: {c.resolution_reason}"
            )

        return QueryResult(
            question="Contradictions in research",
            answer="\n".join(parts),
            contradictions=[
                {
                    "old_statement": c.old_statement,
                    "old_context": c.old_context,
                    "new_statement": c.new_statement,
                    "new_context": c.new_context,
                    "resolution": c.resolution.value,
                    "reason": c.resolution_reason,
                }
                for c in conflicts
            ],
        )

    def get_knowledge_impact(
        self, finding_id: str
    ) -> dict[str, Any]:
        """Show the full knowledge impact of a research finding."""
        finding = self._finding_store.get(finding_id)
        if not finding:
            return {"error": f"Finding {finding_id} not found"}

        # Which knowledge claims were affected
        affected_claims = [
            c for c in self._knowledge_store.list_all()
            if finding_id in c.provenance_chain
        ]

        # Which conflicts were created
        conflicts = self._conflict_detector.list_all()
        finding_conflicts = [
            c for c in conflicts
            if c.source_finding_id == finding_id
        ]

        # Versions created by this finding
        versions_created = []
        for claim in affected_claims:
            for v in claim.versions:
                if v.source_finding_id == finding_id:
                    versions_created.append({
                        "claim_id": claim.claim_id,
                        "claim_topic": claim.topic,
                        "version": v.version_number,
                        "statement": v.statement,
                        "category": v.category.value,
                    })

        # Learning nodes affected
        learning_evidences = self._learning.list_evidence()

        # Learning gaps created
        gaps = self._learning.list_gaps()
        finding_gaps = [g for g in gaps if g.source_finding_id == finding_id]

        # Focus recommendations
        focus_recs = self._learning.list_focus_recs()

        return {
            "finding_id": finding_id,
            "finding_statement": finding.statement,
            "knowledge_claims_affected": [
                {"claim_id": c.claim_id, "topic": c.topic,
                 "version_count": len(c.versions)}
                for c in affected_claims
            ],
            "conflicts_created": [
                {"conflict_id": c.conflict_id, "resolution": c.resolution.value}
                for c in finding_conflicts
            ],
            "versions_created": versions_created,
            "learning_nodes_affected": [
                {"node_name": e.node_name, "evidence_type": e.evidence_type.value,
                 "statement": e.statement}
                for e in learning_evidences
                if e.source_finding_id == finding_id
            ],
            "learning_gaps_created": [
                {"gap_id": g.gap_id, "node_name": g.node_name,
                 "question": g.question, "priority": g.priority}
                for g in finding_gaps
            ],
            "focus_recommendations": [
                {"rec_id": r.rec_id, "title": r.title,
                 "priority": r.priority}
                for r in focus_recs
            ],
        }

    def _build_contradiction_info(
        self, claim: KnowledgeClaim
    ) -> list[dict[str, Any]]:
        conflicts = []
        for conflict_id in claim.conflicts:
            cr = self._conflict_detector.get(conflict_id)
            if cr:
                conflicts.append({
                    "old_statement": cr.old_statement,
                    "new_statement": cr.new_statement,
                    "resolution": cr.resolution.value,
                    "reason": cr.resolution_reason,
                })
        return conflicts
