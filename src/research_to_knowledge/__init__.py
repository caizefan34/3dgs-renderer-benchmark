"""Research-to-Knowledge Closure (Phase 6.6).

Safely feeds validated research conclusions back into Knowledge OS
and Learning OS without contamination or overgeneralization.
"""
from __future__ import annotations

from .research_finding import (
    ClaimType,
    ResearchFinding,
    ResearchFindingStore,
    ResearchReviewerVerdict,
)
from .knowledge_claim import (
    KnowledgeClaim,
    KnowledgeClaimStore,
    KnowledgeVersion,
)
from .knowledge_proposal import (
    IntegrityGateResult,
    KnowledgeProposal,
    KnowledgeProposalPipeline,
)
from .conflict_detection import (
    ConflictRecord,
    ConflictDetector,
    ConflictResolution,
)
from .learning_integration import (
    FocusRecommendation,
    LearningEvidence,
    LearningGap,
    LearningIntegration,
)
from .research_knowledge_query import (
    ResearchKnowledgeQueryEngine,
    QueryResult,
)

__all__ = [
    "ResearchFinding",
    "ResearchFindingStore",
    "ClaimType",
    "ResearchReviewerVerdict",
    "KnowledgeClaim",
    "KnowledgeClaimStore",
    "KnowledgeVersion",
    "KnowledgeProposal",
    "KnowledgeProposalPipeline",
    "IntegrityGateResult",
    "ConflictRecord",
    "ConflictDetector",
    "ConflictResolution",
    "LearningEvidence",
    "LearningGap",
    "FocusRecommendation",
    "LearningIntegration",
    "ResearchKnowledgeQueryEngine",
    "QueryResult",
]
