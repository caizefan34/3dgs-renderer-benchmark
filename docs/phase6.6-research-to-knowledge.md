# Phase 6.6 — Research-to-Knowledge Closure

## Objective

Close the loop from validated research conclusions into Knowledge OS and Learning OS without contamination or overgeneralization.

## Architecture

```
Research Conclusion
       ↓
 Research Finding (with scope, context, provenance)
       ↓
 Knowledge Proposal
       ↓
 Integrity Gate (deterministic rules)
       ↓
   ┌────┴────┐
   │         │
  PASS    WARNING/REJECT
   │         │
   ↓         └──→ Quarantine
 Conflict Detection
   ↓
 Knowledge Claim (versioned, contextual)
   ↓
 Learning Evidence
   ↓
 Knowledge Gap + Focus Recommendation
```

## New Modules

| Module | Purpose |
|--------|---------|
| `src/research_to_knowledge/__init__.py` | Package init and exports |
| `src/research_to_knowledge/research_finding.py` | ResearchFinding model, ScopeMetadata, ClaimType, ResearchFindingStore |
| `src/research_to_knowledge/knowledge_claim.py` | KnowledgeClaim, KnowledgeVersion, KnowledgeClaimCategory, KnowledgeClaimStore |
| `src/research_to_knowledge/knowledge_proposal.py` | KnowledgeProposal, KnowledgeProposalPipeline, IntegrityGate |
| `src/research_to_knowledge/conflict_detection.py` | ConflictDetector, ConflictRecord, ConflictResolution |
| `src/research_to_knowledge/learning_integration.py` | LearningEvidence, LearningGap, FocusRecommendation, LearningIntegration |
| `src/research_to_knowledge/research_knowledge_query.py` | ResearchKnowledgeQueryEngine — current/historical/contradiction queries |
| `src/research_to_knowledge/orchestrator.py` | ResearchToKnowledgeClosure — full pipeline orchestrator with real 3DGS data |

## Key Design Decisions

1. **No direct FACT promotion**: `ClaimType.FACT` exists only in `KnowledgeClaimCategory`, never in `ResearchFinding.ClaimType`. The Integrity Gate maps `EXPERIMENT_RESULT → EXPERIMENT_RESULT` and `INFERENCE → INFERENCE`. FACT requires explicit Knowledge Integrity Policy approval.

2. **Scope preservation**: Every ResearchFinding carries `ScopeMetadata` with project_id, dataset, hardware, software, configuration, code_commit — the full context. The Integrity Gate rejects findings with empty project_id or scope.

3. **Overgeneralization prevention**: The Integrity Gate scans for patterns like "always", "never", "universally" (when not preceded by "not"), "definitively proves", "in all cases". Matching findings get WARNING status.

4. **Alternative explanations & contradicting evidence preserved**: Both are mandatory fields. Missing either triggers Integrity Gate WARNING.

5. **Conflict detection**: Old claims are never deleted. When new evidence contradicts, a `ConflictRecord` is created linking both claims. Resolution types: CONTEXTUAL, SUPERSEDED, INCONCLUSIVE, COMPLEMENTARY.

6. **Learning state rules**: Research activity alone does not set MASTERED. Gaps and Focus Recommendations are created for unresolved questions.

## API Surface

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/research/{project_id}/findings` | POST | Create a research finding |
| `/research/{project_id}/findings` | GET | List findings for a project |
| `/research/findings/{id}` | GET | Get a specific finding |
| `/research/findings/{id}/promote` | POST | Promote finding to knowledge |
| `/research/findings/{id}/reject` | POST | Reject a finding |
| `/research/findings/{id}/knowledge-impact` | GET | Full knowledge impact report |
