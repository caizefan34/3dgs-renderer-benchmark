"""Knowledge Proposal pipeline — from ResearchFinding to KnowledgeClaim.

Pipeline:
    Research Conclusion
    → Research Finding
    → Knowledge Proposal
    → Integrity Gate
    → Conflict Detection
    → Version Creation
    → Knowledge OS

The Integrity Gate enforces:
- Evidence integrity (raw results exist)
- Provenance completeness (chain is traceable)
- Scope preservation (finding is contextual, not universal)
- Reviewer approval (approved or warning only)
- No overgeneralization (statement is qualified)
- No AI hallucination (raw data is never modified)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .research_finding import (
    ClaimType,
    ResearchFinding,
    ResearchReviewerVerdict,
    ScopeMetadata,
)
from .knowledge_claim import (
    KnowledgeClaim,
    KnowledgeClaimCategory,
    KnowledgeClaimStore,
    KnowledgeVersion,
)


class IntegrityGateResult(Enum):
    PASS = "pass"
    QUARANTINE = "quarantine"
    REJECT = "reject"
    WARNING = "warning"


@dataclass
class KnowledgeProposal:
    """A proposal to promote a ResearchFinding into Knowledge OS."""

    proposal_id: str
    finding_id: str
    proposed_category: KnowledgeClaimCategory
    proposed_statement: str
    context: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["proposed_category"] = self.proposed_category.value
        return d


class KnowledgeProposalPipeline:
    """Deterministic rules for promoting research findings to knowledge.

    Rules are deterministic — no AI inference is used for:
    - integrity gate decisions
    - claim promotion
    - version creation
    - contamination gate
    """

    def __init__(self, knowledge_store: KnowledgeClaimStore) -> None:
        self._knowledge_store = knowledge_store
        self._proposals: list[KnowledgeProposal] = []
        self._gate_results: dict[str, IntegrityGateResult] = {}

    def create_proposal(
        self, finding: ResearchFinding
    ) -> KnowledgeProposal | None:
        """Create a KnowledgeProposal from an approved ResearchFinding."""
        if finding.rejected or finding.reviewer_verdict == ResearchReviewerVerdict.INCONCLUSIVE:
            return None

        # Map ClaimType to KnowledgeClaimCategory
        category_map = {
            ClaimType.EXPERIMENT_RESULT: KnowledgeClaimCategory.EXPERIMENT_RESULT,
            ClaimType.INFERENCE: KnowledgeClaimCategory.INFERENCE,
        }
        proposed_cat = category_map.get(finding.claim_type)
        if proposed_cat is None:
            return None

        proposal = KnowledgeProposal(
            proposal_id=f"proposal_{finding.finding_id}",
            finding_id=finding.finding_id,
            proposed_category=proposed_cat,
            proposed_statement=finding.statement,
            context=finding.scope.to_dict(),
            notes=finding.reviewer_notes,
        )
        self._proposals.append(proposal)
        return proposal

    def run_integrity_gate(
        self, proposal: KnowledgeProposal, finding: ResearchFinding
    ) -> IntegrityGateResult:
        """Deterministic integrity check.

        Rules:
        1. Raw evidence URIs must be present
        2. Scope metadata must be complete (project_id, scope text)
        3. Reviewer must have approved or given warning
        4. Statement must not be a universal claim (checked by pattern)
        5. Alternative explanations must be preserved
        6. Contradicting evidence must be preserved
        """
        # Rule 1: Raw evidence required
        if not finding.raw_evidence_uris:
            self._gate_results[proposal.proposal_id] = IntegrityGateResult.REJECT
            return IntegrityGateResult.REJECT

        # Rule 2: Scope completeness
        scope = finding.scope
        if not scope.project_id or not scope.scope:
            self._gate_results[proposal.proposal_id] = IntegrityGateResult.REJECT
            return IntegrityGateResult.REJECT

        # Rule 3: Reviewer verdict
        if finding.reviewer_verdict is None:
            self._gate_results[proposal.proposal_id] = IntegrityGateResult.QUARANTINE
            return IntegrityGateResult.QUARANTINE
        if finding.reviewer_verdict == ResearchReviewerVerdict.INCONCLUSIVE:
            self._gate_results[proposal.proposal_id] = IntegrityGateResult.QUARANTINE
            return IntegrityGateResult.QUARANTINE

        # Rule 4: Overgeneralization check — statement must not claim universality
        # "insufficient evidence that X is universally..." is fine
        # "X is universally faster" is not
        statement_lower = proposal.proposed_statement.lower()
        if "insufficient evidence" in statement_lower:
            pass  # This is a properly qualified statement
        else:
            overgeneral_patterns = [
                " is always ",
                " is never ",
                " always ",
                " never ",
                "under any condition",
                "in all cases",
                "is guaranteed",
                "definitively proves",
            ]
            # Only flag "universally" as overgeneral when NOT preceded by a qualifier
            if " universally " in statement_lower and "not universally " not in statement_lower:
                overgeneral_patterns.append(" universally ")
            for pattern in overgeneral_patterns:
                if pattern in statement_lower:
                    self._gate_results[proposal.proposal_id] = IntegrityGateResult.WARNING
                    return IntegrityGateResult.WARNING

        # Rule 5: Must preserve alternative explanations
        if not finding.alternative_explanations:
            self._gate_results[proposal.proposal_id] = IntegrityGateResult.WARNING
            return IntegrityGateResult.WARNING

        # Rule 6: Must preserve contradicting evidence
        if not finding.contradicting_evidence:
            self._gate_results[proposal.proposal_id] = IntegrityGateResult.WARNING
            return IntegrityGateResult.WARNING

        self._gate_results[proposal.proposal_id] = IntegrityGateResult.PASS
        return IntegrityGateResult.PASS

    def promote_to_knowledge(
        self,
        proposal: KnowledgeProposal,
        finding: ResearchFinding,
        topic: str,
        slug: str,
    ) -> KnowledgeClaim | None:
        """Create or update a KnowledgeClaim from a passed proposal."""
        gate_result = self._gate_results.get(proposal.proposal_id)
        if gate_result not in (IntegrityGateResult.PASS, IntegrityGateResult.WARNING):
            return None

        existing = self._knowledge_store.get_by_slug(slug)
        context_dict = finding.scope.to_dict()

        if existing:
            # Add new version
            existing.add_version(
                statement=proposal.proposed_statement,
                category=proposal.proposed_category,
                context=context_dict,
                source_finding_id=finding.finding_id,
                change_reason=f"Research finding {finding.finding_id} promoted via proposal {proposal.proposal_id}",
            )
            # Add provenance
            if finding.finding_id not in existing.provenance_chain:
                existing.provenance_chain.append(finding.finding_id)
            return existing
        else:
            # Create new KnowledgeClaim
            version = KnowledgeVersion(
                version_number=1,
                statement=proposal.proposed_statement,
                category=proposal.proposed_category,
                context=context_dict,
                source_finding_id=finding.finding_id,
            )
            claim = KnowledgeClaim(
                claim_id=f"kc_{slug}_{finding.finding_id[:8]}",
                topic=topic,
                slug=slug,
                versions=[version],
                provenance_chain=[finding.finding_id],
                active_version=0,
            )
            self._knowledge_store.add(claim)
            return claim

    def get_result(self, proposal_id: str) -> IntegrityGateResult | None:
        return self._gate_results.get(proposal_id)
