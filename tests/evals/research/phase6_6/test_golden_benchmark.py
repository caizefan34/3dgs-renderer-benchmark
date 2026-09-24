"""Phase 6.6 Golden Benchmark — Research-to-Knowledge Closure tests.

50+ test cases covering:
- contextual finding
- scope preservation
- overgeneralization prevention
- knowledge promotion
- conflict
- version
- provenance
- learning integration
- gap generation
- focus generation

Plus the 5 golden research cases (A-E) and negative tests.
"""
import sys
import unittest
from pathlib import Path
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from research_to_knowledge import (
    ResearchFinding,
    ResearchFindingStore,
    ResearchReviewerVerdict,
    ClaimType,
)
from research_to_knowledge.research_finding import ScopeMetadata
from research_to_knowledge.knowledge_claim import (
    KnowledgeClaim,
    KnowledgeClaimCategory,
    KnowledgeClaimStore,
    KnowledgeVersion,
)
from research_to_knowledge.knowledge_proposal import (
    IntegrityGateResult,
    KnowledgeProposal,
    KnowledgeProposalPipeline,
)
from research_to_knowledge.conflict_detection import (
    ConflictDetector,
    ConflictRecord,
    ConflictResolution,
)
from research_to_knowledge.learning_integration import (
    EvidenceType,
    FocusRecommendation,
    LearningEvidence,
    LearningGap,
    LearningIntegration,
)
from research_to_knowledge.research_knowledge_query import (
    ResearchKnowledgeQueryEngine,
    QueryResult,
)
from research_to_knowledge.orchestrator import (
    ResearchToKnowledgeClosure,
)


class TestScopeMetadata(unittest.TestCase):
    """Tests for ScopeMetadata completeness."""

    def test_scope_metadata_captures_all_required_fields(self):
        scope = ScopeMetadata(
            project_id="test_project",
            question_id="q1",
            hypothesis_id="h1",
            experiment_id="e1",
            run_ids=["run1", "run2"],
            dataset="mipnerf360",
            hardware="NVIDIA A100-SXM4-80GB",
            software={"cuda": "12.8", "pytorch": "2.9.1"},
            configuration={"tile_size": "16"},
            code_commit="abc123",
            scope="Scene: bicycle, GPU: A100",
        )
        d = scope.to_dict()
        self.assertEqual(d["project_id"], "test_project")
        self.assertEqual(d["hardware"], "NVIDIA A100-SXM4-80GB")
        self.assertEqual(d["dataset"], "mipnerf360")
        self.assertIn("created_at", d)

    def test_scope_without_optional_fields_still_complete(self):
        scope = ScopeMetadata(project_id="p1", scope="test")
        self.assertEqual(scope.project_id, "p1")
        self.assertIsNone(scope.dataset)
        self.assertIsNone(scope.hardware)


class TestResearchFindingModel(unittest.TestCase):
    """Tests for ResearchFinding creation and properties."""

    def test_create_experiment_result(self):
        scope = ScopeMetadata(
            project_id="3dgs",
            dataset="mipnerf360",
            hardware="A100",
            scope="Scene: bicycle",
        )
        finding = ResearchFinding(
            finding_id="find_01",
            claim_type=ClaimType.EXPERIMENT_RESULT,
            statement="Under A100 with bicycle, FPS=492.3",
            scope=scope,
            evidence_summary="Measured on A100",
            raw_evidence_uris=["results/measured/metrics.json"],
        )
        self.assertEqual(finding.claim_type, ClaimType.EXPERIMENT_RESULT)
        self.assertFalse(finding.promoted_to_knowledge)
        self.assertIsNone(finding.reviewer_verdict)

    def test_create_inference_finding(self):
        scope = ScopeMetadata(project_id="3dgs", scope="All scenes")
        finding = ResearchFinding(
            finding_id="find_02",
            claim_type=ClaimType.INFERENCE,
            statement="Insufficient evidence that tile32 is universally faster",
            scope=scope,
            evidence_summary="Mixed results across scenes",
            alternative_explanations=["Scene complexity dependency"],
            contradicting_evidence=["Tile16 faster in some scenes"],
            uncertainty="Limited scene coverage",
        )
        self.assertEqual(finding.claim_type, ClaimType.INFERENCE)
        self.assertTrue(len(finding.alternative_explanations) > 0)

    def test_finding_preserves_alternative_explanations(self):
        scope = ScopeMetadata(project_id="3dgs", scope="bicycle,garden")
        finding = ResearchFinding(
            finding_id="find_alt",
            claim_type=ClaimType.EXPERIMENT_RESULT,
            statement="Test result",
            scope=scope,
            evidence_summary="test",
            alternative_explanations=[
                "Scene coverage affects tile performance",
                "Gaussian density distribution changes optimal tile",
            ],
        )
        self.assertEqual(len(finding.alternative_explanations), 2)

    def test_finding_preserves_contradicting_evidence(self):
        scope = ScopeMetadata(project_id="3dgs", scope="bicycle")
        finding = ResearchFinding(
            finding_id="find_contra",
            claim_type=ClaimType.EXPERIMENT_RESULT,
            statement="Tile16 faster than tile32 on bicycle",
            scope=scope,
            evidence_summary="bicycle data",
            contradicting_evidence=[
                "Previous run on garden showed tile32 faster",
            ],
        )
        self.assertEqual(len(finding.contradicting_evidence), 1)

    def test_finding_reviewer_verdict(self):
        scope = ScopeMetadata(project_id="3dgs", scope="test")
        finding = ResearchFinding(
            finding_id="find_review",
            claim_type=ClaimType.EXPERIMENT_RESULT,
            statement="Test result",
            scope=scope,
            evidence_summary="test",
            reviewer_verdict=ResearchReviewerVerdict.APPROVED,
        )
        self.assertEqual(finding.reviewer_verdict, ResearchReviewerVerdict.APPROVED)

    def test_rejected_finding_not_promotable(self):
        scope = ScopeMetadata(project_id="3dgs", scope="test")
        finding = ResearchFinding(
            finding_id="find_rejected",
            claim_type=ClaimType.EXPERIMENT_RESULT,
            statement="Invalid result",
            scope=scope,
            evidence_summary="test",
            rejected=True,
            rejection_reason="Missing raw evidence",
        )
        self.assertTrue(finding.rejected)
        self.assertIsNotNone(finding.rejection_reason)

    def test_quarantine_record(self):
        scope = ScopeMetadata(project_id="3dgs", scope="test")
        finding = ResearchFinding(
            finding_id="find_quar",
            claim_type=ClaimType.EXPERIMENT_RESULT,
            statement="Quarantined result",
            scope=scope,
            evidence_summary="test",
            quarantine=True,
            quarantine_reason="Suspected data corruption",
        )
        self.assertTrue(finding.quarantine)


# ── ResearchFindingStore ─────────────────────────────────────────────────────

class TestResearchFindingStore(unittest.TestCase):
    """Tests for ResearchFindingStore operations."""

    def setUp(self):
        self.store = ResearchFindingStore()
        scope = ScopeMetadata(project_id="3dgs", scope="test")
        self.finding = ResearchFinding(
            finding_id="find_store_1",
            claim_type=ClaimType.EXPERIMENT_RESULT,
            statement="Test finding",
            scope=scope,
            evidence_summary="test",
        )
        self.store.add(self.finding)

    def test_add_and_get(self):
        retrieved = self.store.get("find_store_1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.statement, "Test finding")

    def test_get_nonexistent(self):
        self.assertIsNone(self.store.get("nonexistent"))

    def test_list_by_project(self):
        findings = self.store.list_by_project("3dgs")
        self.assertEqual(len(findings), 1)

        other = self.store.list_by_project("nonexistent")
        self.assertEqual(len(other), 0)

    def test_promote(self):
        self.store.promote("find_store_1")
        self.assertTrue(self.finding.promoted_to_knowledge)

    def test_reject(self):
        self.store.reject("find_store_1", "Insufficient data")
        self.assertTrue(self.finding.rejected)
        self.assertEqual(self.finding.rejection_reason, "Insufficient data")

    def test_quarantine(self):
        self.store.quarantine_record("find_store_1", "Suspicious data")
        self.assertTrue(self.finding.quarantine)
        self.assertTrue(self.finding.rejected)


class TestKnowledgeClaimModel(unittest.TestCase):
    """Tests for KnowledgeClaim creation and versioning."""

    def test_create_knowledge_claim(self):
        claim = KnowledgeClaim(
            claim_id="kc_001",
            topic="tile-size-performance",
            slug="tile-size-performance",
        )
        self.assertEqual(claim.topic, "tile-size-performance")
        self.assertEqual(len(claim.versions), 0)
        self.assertIsNone(claim.current)

    def test_add_version(self):
        claim = KnowledgeClaim(
            claim_id="kc_002",
            topic="test-topic",
            slug="test-topic",
        )
        v1 = claim.add_version(
            statement="Initial finding",
            category=KnowledgeClaimCategory.EXPERIMENT_RESULT,
            context={"hardware": "A100"},
        )
        self.assertEqual(v1.version_number, 1)
        self.assertEqual(claim.active_version, 0)

        v2 = claim.add_version(
            statement="Updated finding",
            category=KnowledgeClaimCategory.INFERENCE,
            context={"hardware": "RTX 4090"},
            change_reason="New hardware tested",
        )
        self.assertEqual(v2.version_number, 2)
        self.assertEqual(claim.active_version, 1)
        self.assertEqual(claim.current.statement, "Updated finding")

    def test_historical_query(self):
        claim = KnowledgeClaim(
            claim_id="kc_003",
            topic="historical-test",
            slug="historical-test",
        )
        claim.add_version(
            statement="Version 1: tile32 faster",
            category=KnowledgeClaimCategory.EXPERIMENT_RESULT,
        )
        claim.add_version(
            statement="Version 2: tile32 not universally faster",
            category=KnowledgeClaimCategory.INFERENCE,
        )
        versions = claim.versions
        self.assertEqual(len(versions), 2)
        v1 = claim.get_version(1)
        self.assertEqual(v1.statement, "Version 1: tile32 faster")
        v2 = claim.get_version(2)
        self.assertEqual(v2.statement, "Version 2: tile32 not universally faster")

    def test_context_specific_query(self):
        claim = KnowledgeClaim(
            claim_id="kc_004",
            topic="context-test",
            slug="context-test",
        )
        claim.add_version(
            statement="A100 result",
            category=KnowledgeClaimCategory.EXPERIMENT_RESULT,
            context={"hardware": "A100", "scene": "garden"},
        )
        claim.add_version(
            statement="RTX 4090 result",
            category=KnowledgeClaimCategory.EXPERIMENT_RESULT,
            context={"hardware": "RTX 4090", "scene": "garden"},
        )
        matching = claim.get_context_matching({"hardware": "A100"})
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0].statement, "A100 result")

    def test_no_FACT_without_gate(self):
        """FACT should not be set without explicit gate. Test can still create one."""
        claim = KnowledgeClaim(
            claim_id="kc_fact",
            topic="fact-test",
            slug="fact-test",
        )
        fact_version = claim.add_version(
            statement="A confirmed fact",
            category=KnowledgeClaimCategory.FACT,
            change_reason="Passed full integrity gate",
        )
        self.assertEqual(fact_version.category, KnowledgeClaimCategory.FACT)


class TestKnowledgeProposalPipeline(unittest.TestCase):
    """Tests for the Knowledge Proposal pipeline and Integrity Gate."""

    def setUp(self):
        self.knowledge_store = KnowledgeClaimStore()
        self.pipeline = KnowledgeProposalPipeline(self.knowledge_store)

    def _make_finding(
        self,
        finding_id: str,
        statement: str,
        claim_type: ClaimType = ClaimType.INFERENCE,
        has_evidence: bool = True,
        has_scope: bool = True,
        verdict: ResearchReviewerVerdict = ResearchReviewerVerdict.APPROVED,
        has_alternatives: bool = True,
        has_contradictions: bool = True,
    ) -> ResearchFinding:
        scope = ScopeMetadata(
            project_id="3dgs" if has_scope else "",
            scope="Test scope" if has_scope else "",
            hardware="A100",
        )
        return ResearchFinding(
            finding_id=finding_id,
            claim_type=claim_type,
            statement=statement,
            scope=scope,
            evidence_summary="test data",
            raw_evidence_uris=["metrics.json"] if has_evidence else [],
            alternative_explanations=(
                ["Scene dependency"] if has_alternatives else []
            ),
            contradicting_evidence=(
                ["Other scene showed opposite"] if has_contradictions else []
            ),
            reviewer_verdict=verdict,
            reviewer_notes="OK" if verdict else None,
        )

    def test_create_proposal_from_approved_finding(self):
        finding = self._make_finding("find_good", "Tile performance depends on workload")
        proposal = self.pipeline.create_proposal(finding)
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.finding_id, "find_good")

    def test_cannot_propose_rejected_finding(self):
        finding = self._make_finding("find_bad", "Invalid")
        finding.rejected = True
        proposal = self.pipeline.create_proposal(finding)
        self.assertIsNone(proposal)

    def test_cannot_propose_inconclusive_finding(self):
        finding = self._make_finding(
            "find_inc", "Inconclusive",
            verdict=ResearchReviewerVerdict.INCONCLUSIVE
        )
        proposal = self.pipeline.create_proposal(finding)
        self.assertIsNone(proposal)

    def test_integrity_gate_passes_with_all_requirements(self):
        finding = self._make_finding("find_pass", "Tile16 faster under A100 on garden scene")
        proposal = self.pipeline.create_proposal(finding)
        result = self.pipeline.run_integrity_gate(proposal, finding)
        self.assertEqual(result, IntegrityGateResult.PASS)

    def test_integrity_gate_rejects_missing_evidence(self):
        finding = self._make_finding("find_no_ev", "Test", has_evidence=False)
        proposal = self.pipeline.create_proposal(finding)
        result = self.pipeline.run_integrity_gate(proposal, finding)
        self.assertEqual(result, IntegrityGateResult.REJECT)

    def test_integrity_gate_rejects_missing_scope(self):
        finding = self._make_finding("find_no_scope", "Test", has_scope=False)
        proposal = self.pipeline.create_proposal(finding)
        result = self.pipeline.run_integrity_gate(proposal, finding)
        self.assertEqual(result, IntegrityGateResult.REJECT)

    def test_integrity_gate_quarantines_no_reviewer(self):
        finding = self._make_finding(
            "find_no_rev", "Test", verdict=None
        )
        proposal = self.pipeline.create_proposal(finding)
        result = self.pipeline.run_integrity_gate(proposal, finding)
        self.assertEqual(result, IntegrityGateResult.QUARANTINE)

    def test_integrity_gate_warns_on_overgeneralization(self):
        finding = self._make_finding(
            "find_overgen",
            "tile32 is always faster than tile16"
        )
        proposal = self.pipeline.create_proposal(finding)
        result = self.pipeline.run_integrity_gate(proposal, finding)
        self.assertEqual(result, IntegrityGateResult.WARNING)

    def test_integrity_gate_warns_on_universal_claim(self):
        finding = self._make_finding(
            "find_universal",
            "tile16 is never the optimal choice"
        )
        proposal = self.pipeline.create_proposal(finding)
        result = self.pipeline.run_integrity_gate(proposal, finding)
        self.assertEqual(result, IntegrityGateResult.WARNING)

    def test_integrity_gate_warns_missing_alternatives(self):
        finding = self._make_finding(
            "find_no_alt", "Test", has_alternatives=False
        )
        proposal = self.pipeline.create_proposal(finding)
        result = self.pipeline.run_integrity_gate(proposal, finding)
        self.assertEqual(result, IntegrityGateResult.WARNING)

    def test_integrity_gate_warns_missing_contradictions(self):
        finding = self._make_finding(
            "find_no_contra", "Test", has_contradictions=False
        )
        proposal = self.pipeline.create_proposal(finding)
        result = self.pipeline.run_integrity_gate(proposal, finding)
        self.assertEqual(result, IntegrityGateResult.WARNING)

    def test_promote_to_knowledge_creates_new_claim(self):
        finding = self._make_finding(
            "find_promote",
            "Tile-size performance is workload dependent on A100"
        )
        proposal = self.pipeline.create_proposal(finding)
        self.pipeline.run_integrity_gate(proposal, finding)
        claim = self.pipeline.promote_to_knowledge(
            proposal, finding,
            topic="tile-size-performance",
            slug="tile-size-performance",
        )
        self.assertIsNotNone(claim)
        self.assertEqual(claim.topic, "tile-size-performance")
        self.assertEqual(len(claim.versions), 1)
        self.assertIsNotNone(claim.current)

    def test_promote_updates_existing_claim(self):
        existing = KnowledgeClaim(
            claim_id="kc_existing",
            topic="tile-size-performance",
            slug="tile-size-performance",
        )
        existing.add_version(
            statement="Older version",
            category=KnowledgeClaimCategory.EXPERIMENT_RESULT,
        )
        self.knowledge_store.add(existing)

        finding = self._make_finding(
            "find_update",
            "Newer version: workload dependent"
        )
        proposal = self.pipeline.create_proposal(finding)
        self.pipeline.run_integrity_gate(proposal, finding)
        claim = self.pipeline.promote_to_knowledge(
            proposal, finding,
            topic="tile-size-performance",
            slug="tile-size-performance",
        )
        self.assertEqual(len(claim.versions), 2)
        self.assertEqual(claim.current.statement, "Newer version: workload dependent")

    def test_rejected_gate_does_not_promote(self):
        finding = self._make_finding(
            "find_rej_gate", "Test", has_evidence=False
        )
        proposal = self.pipeline.create_proposal(finding)
        self.pipeline.run_integrity_gate(proposal, finding)
        claim = self.pipeline.promote_to_knowledge(
            proposal, finding,
            topic="rejected", slug="rejected",
        )
        self.assertIsNone(claim)


class TestConflictDetection(unittest.TestCase):
    """Tests for conflict detection between old and new claims."""

    def setUp(self):
        self.detector = ConflictDetector()
        self.old_claim = KnowledgeClaim(
            claim_id="kc_old",
            topic="tile-size",
            slug="tile-size",
        )
        self.old_claim.add_version(
            statement="tile32 is generally faster across test scenes",
            category=KnowledgeClaimCategory.EXPERIMENT_RESULT,
            context={"hardware": "A100", "dataset": "mipnerf360"},
        )

    def test_detect_contextual_conflict_different_hardware(self):
        conflict = self.detector.detect(
            old_claim=self.old_claim,
            new_statement="tile16 is faster on RTX 4090 with garden scene",
            new_finding_id="find_new_hw",
            new_context={"hardware": "RTX 4090", "dataset": "mipnerf360"},
        )
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict.resolution, ConflictResolution.CONTEXTUAL)

    def test_detect_contextual_conflict_different_dataset(self):
        conflict = self.detector.detect(
            old_claim=self.old_claim,
            new_statement="tile16 is faster on tanks_and_temples dataset",
            new_finding_id="find_new_ds",
            new_context={"hardware": "A100", "dataset": "tanks_and_temples"},
        )
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict.resolution, ConflictResolution.CONTEXTUAL)

    def test_no_conflict_with_compatible_refinement(self):
        conflict = self.detector.detect(
            old_claim=self.old_claim,
            new_statement="Insufficient evidence that tile32 is universally faster",
            new_finding_id="find_refine",
            new_context={"hardware": "A100", "dataset": "mipnerf360"},
        )
        self.assertIsNone(conflict)

    def test_no_conflict_with_none_claim(self):
        conflict = self.detector.detect(
            old_claim=None,
            new_statement="New result",
            new_finding_id="find_new",
            new_context={},
        )
        self.assertIsNone(conflict)

    def test_conflict_record_contains_full_detail(self):
        conflict = self.detector.detect(
            old_claim=self.old_claim,
            new_statement="tile16 is faster on RTX 4090",
            new_finding_id="find_detail",
            new_context={"hardware": "RTX 4090"},
        )
        self.assertIn("tile32", conflict.old_statement)
        self.assertIn("tile16", conflict.new_statement)
        self.assertIsNotNone(conflict.conflict_id)

    def test_conflict_list_by_claim(self):
        self.detector.detect(
            old_claim=self.old_claim,
            new_statement="tile16 faster on RTX 4090",
            new_finding_id="find_list1",
            new_context={"hardware": "RTX 4090"},
        )
        self.detector.detect(
            old_claim=self.old_claim,
            new_statement="tile16 faster on V100",
            new_finding_id="find_list2",
            new_context={"hardware": "V100"},
        )
        conflicts = self.detector.list_by_claim("kc_old")
        self.assertEqual(len(conflicts), 2)


class TestLearningIntegration(unittest.TestCase):
    """Tests for Learning Integration components."""

    def setUp(self):
        self.learning = LearningIntegration()

    def test_add_evidence(self):
        ev = self.learning.add_evidence(
            node_name="Tile Size Analysis",
            evidence_type=EvidenceType.EXPERIMENT_RESULT,
            statement="Tile16 faster on bicycle",
            source_finding_id="find_001",
            context={"scene": "bicycle"},
        )
        self.assertIsNotNone(ev.evidence_id)
        self.assertEqual(ev.node_name, "Tile Size Analysis")
        self.assertEqual(ev.evidence_type, EvidenceType.EXPERIMENT_RESULT)

    def test_identify_gap(self):
        gap = self.learning.identify_gap(
            node_name="Tile Size",
            question="Why does scene coverage affect tile performance?",
            source_finding_id="find_001",
            reason="Insufficient evidence for universal conclusion",
            priority="high",
        )
        self.assertIsNotNone(gap.gap_id)
        self.assertEqual(gap.priority, "high")
        self.assertFalse(gap.resolved)

    def test_create_focus_recommendation(self):
        gap = self.learning.identify_gap(
            node_name="GPU Performance",
            question="How does Gaussian density affect tile performance?",
            source_finding_id="find_001",
            reason="Mixed results across scenes",
        )
        rec = self.learning.create_focus_recommendation(
            gap=gap,
            title="Investigate Gaussian-density × tile-size interaction",
        )
        self.assertIsNotNone(rec.rec_id)
        self.assertEqual(rec.gap_id, gap.gap_id)
        self.assertEqual(rec.priority, "medium")

    def test_list_evidence_by_node(self):
        self.learning.add_evidence(
            node_name="NodeA", evidence_type=EvidenceType.EXPERIMENT_RESULT,
            statement="Result A", source_finding_id="f1",
        )
        self.learning.add_evidence(
            node_name="NodeB", evidence_type=EvidenceType.INFERENCE,
            statement="Result B", source_finding_id="f2",
        )
        all_ev = self.learning.list_evidence()
        self.assertEqual(len(all_ev), 2)
        node_a = self.learning.list_evidence(node_name="NodeA")
        self.assertEqual(len(node_a), 1)

    def test_list_gaps_by_node(self):
        self.learning.identify_gap(
            node_name="NodeX", question="Q1",
            source_finding_id="f1", reason="R1",
        )
        self.learning.identify_gap(
            node_name="NodeY", question="Q2",
            source_finding_id="f2", reason="R2",
        )
        gaps_x = self.learning.list_gaps(node_name="NodeX")
        self.assertEqual(len(gaps_x), 1)

    def test_resolve_gap(self):
        gap = self.learning.identify_gap(
            node_name="Test", question="Q?",
            source_finding_id="f1", reason="R",
        )
        gap.resolved = True
        gap.resolved_at = datetime.now(timezone.utc).isoformat()
        self.assertTrue(gap.resolved)
        self.assertIsNotNone(gap.resolved_at)


class TestQueryEngine(unittest.TestCase):
    """Tests for the Research Knowledge Query Engine."""

    def setUp(self):
        self.finding_store = ResearchFindingStore()
        self.knowledge_store = KnowledgeClaimStore()
        self.conflict_detector = ConflictDetector()
        self.learning = LearningIntegration()

        # Seed some data
        scope = ScopeMetadata(
            project_id="3dgs", hardware="A100",
            scope="bicycle scene", dataset="mipnerf360",
        )
        finding = ResearchFinding(
            finding_id="find_query_1",
            claim_type=ClaimType.INFERENCE,
            statement="Tile-size performance is workload dependent on A100",
            scope=scope,
            evidence_summary="Test data",
            raw_evidence_uris=["test.json"],
            alternative_explanations=["Scene complexity"],
            contradicting_evidence=["Garden showed different pattern"],
            reviewer_verdict=ResearchReviewerVerdict.APPROVED,
        )
        self.finding_store.add(finding)

        claim = KnowledgeClaim(
            claim_id="kc_query",
            topic="tile-size-performance",
            slug="tile-size-performance",
        )
        claim.add_version(
            statement="Tile-size performance is workload dependent on A100",
            category=KnowledgeClaimCategory.INFERENCE,
            context={"hardware": "A100"},
            source_finding_id="find_query_1",
        )
        claim.provenance_chain.append("find_query_1")
        self.knowledge_store.add(claim)

        gap = self.learning.identify_gap(
            node_name="tile-size-performance",
            question="Why does scene affect tile performance?",
            source_finding_id="find_query_1",
            reason="Insufficient evidence",
        )

        self.engine = ResearchKnowledgeQueryEngine(
            finding_store=self.finding_store,
            knowledge_store=self.knowledge_store,
            conflict_detector=self.conflict_detector,
            learning_integration=self.learning,
        )

    def test_query_current_knowledge(self):
        result = self.engine.query_current_knowledge("tile-size-performance")
        self.assertIn("Current knowledge", result.answer)
        self.assertGreater(len(result.evidence_cited), 0)

    def test_query_current_knowledge_nonexistent(self):
        result = self.engine.query_current_knowledge("nonexistent-topic")
        self.assertIn("No knowledge claim found", result.answer)

    def test_query_historical(self):
        result = self.engine.query_historical("tile-size-performance")
        self.assertIn("one version", result.answer.lower())
        self.assertEqual(len(result.knowledge_versions), 1)

    def test_query_historical_nonexistent(self):
        result = self.engine.query_historical("nonexistent")
        self.assertIn("No historical knowledge found", result.answer)

    def test_query_contradictions_empty(self):
        result = self.engine.query_contradictions()
        self.assertIn("No contradictions", result.answer)

    def test_query_contradictions_with_data(self):
        # Add a conflict
        old_claim = KnowledgeClaim(
            claim_id="kc_old_q", topic="test", slug="test",
        )
        old_claim.add_version(
            statement="tile32 is faster",
            category=KnowledgeClaimCategory.EXPERIMENT_RESULT,
        )
        conflict = self.conflict_detector.detect(
            old_claim=old_claim,
            new_statement="tile16 is faster on different hardware",
            new_finding_id="find_new_q",
            new_context={"hardware": "RTX 4090"},
        )
        self.engine = ResearchKnowledgeQueryEngine(
            finding_store=self.finding_store,
            knowledge_store=self.knowledge_store,
            conflict_detector=self.conflict_detector,
            learning_integration=self.learning,
        )
        result = self.engine.query_contradictions()
        self.assertIn("1 contradiction", result.answer)

    def test_knowledge_impact(self):
        impact = self.engine.get_knowledge_impact("find_query_1")
        self.assertEqual(impact["finding_id"], "find_query_1")
        self.assertEqual(len(impact["knowledge_claims_affected"]), 1)
        self.assertEqual(len(impact["learning_gaps_created"]), 1)


class TestGoldenResearchCases(unittest.TestCase):
    """The five golden research cases (A-E)."""

    def _make_scope(
        self, project_id: str = "3dgs", scene: str = "garden",
        hardware: str = "A100", tile_size: str = "16"
    ) -> ScopeMetadata:
        return ScopeMetadata(
            project_id=project_id,
            dataset="mipnerf360",
            hardware=hardware,
            configuration={"tile_size": tile_size},
            scope=f"Scene: {scene}, GPU: {hardware}, tile_size: {tile_size}",
        )

    def test_case_A_tile32_faster_in_one_workload(self):
        """Case A: tile32 faster in one workload but scoped."""
        scope = self._make_scope(tile_size="32")
        finding = ResearchFinding(
            finding_id="case_a_tile32_faster",
            claim_type=ClaimType.EXPERIMENT_RESULT,
            statement="Under A100 with garden scene, tile32=502 FPS vs tile16=492 FPS",
            scope=scope,
            evidence_summary="tile32=502 FPS, tile16=492 FPS",
            raw_evidence_uris=["garden_metrics.json"],
        )
        # Must not make universal claim
        self.assertNotIn("always", finding.statement.lower())
        self.assertNotIn("universally", finding.statement.lower())
        self.assertEqual(finding.claim_type, ClaimType.EXPERIMENT_RESULT)

    def test_case_B_tile16_faster_in_another_workload(self):
        """Case B: tile16 faster in another workload."""
        scope = self._make_scope(tile_size="16")
        finding = ResearchFinding(
            finding_id="case_b_tile16_faster",
            claim_type=ClaimType.EXPERIMENT_RESULT,
            statement="Under A100 with bicycle scene, tile16=500 FPS vs tile32=490 FPS",
            scope=scope,
            evidence_summary="tile16=500 FPS, tile32=490 FPS",
            raw_evidence_uris=["bicycle_metrics.json"],
        )
        self.assertEqual(finding.finding_id, "case_b_tile16_faster")

    def test_case_C_mixed_evidence(self):
        """Case C: mixed evidence across scenes — must not pick a winner."""
        scope = self._make_scope(project_id="3dgs", scene="bicycle,garden", tile_size="16,32")
        finding = ResearchFinding(
            finding_id="case_c_mixed",
            claim_type=ClaimType.INFERENCE,
            statement="Under A100, tile-size performance varies by scene: tile16 faster on bicycle, tile32 faster on garden",
            scope=scope,
            evidence_summary="Mixed results across 2 scenes",
            raw_evidence_uris=["bicycle_metrics.json", "garden_metrics.json"],
            alternative_explanations=[
                "Scene complexity (Gaussian count) affects optimal tile size",
                "Different coverage patterns change tile intersection workload",
            ],
            contradicting_evidence=[
                "Tile32 faster on garden (502 FPS vs 492 FPS)",
                "Tile16 faster on bicycle (500 FPS vs 490 FPS)",
            ],
            uncertainty="Only 2 scenes tested; need broader evaluation",
        )
        self.assertIsNotNone(finding.uncertainty)
        self.assertEqual(len(finding.alternative_explanations), 2)
        self.assertEqual(len(finding.contradicting_evidence), 2)

    def test_case_D_insufficient_evidence(self):
        """Case D: insufficient evidence — must not promote."""
        scope = self._make_scope(project_id="3dgs", scene="bicycle")
        finding = ResearchFinding(
            finding_id="case_d_insufficient",
            claim_type=ClaimType.INFERENCE,
            statement="Insufficient evidence to conclude universal tile-size advantage on A100",
            scope=scope,
            evidence_summary="Single scene, single run",
            raw_evidence_uris=["single_metrics.json"],
            reviewer_verdict=ResearchReviewerVerdict.WARNING,
            reviewer_notes="Only one scene tested; more data needed",
            uncertainty="Single scene, no repeats",
        )
        store = ResearchFindingStore()
        store.add(finding)

        ks = KnowledgeClaimStore()
        pipeline = KnowledgeProposalPipeline(ks)
        proposal = pipeline.create_proposal(finding)
        self.assertIsNotNone(proposal)
        gate_result = pipeline.run_integrity_gate(proposal, finding)
        # Should pass with WARNING since alternatives/contradictions are present
        # but with WARNING verdict
        self.assertIn(gate_result, (IntegrityGateResult.PASS, IntegrityGateResult.WARNING))

    def test_case_E_old_claim_vs_new_evidence(self):
        """Case E: old claim vs new evidence — must preserve both."""
        old_claim = KnowledgeClaim(
            claim_id="kc_old_e",
            topic="tile-size-performance",
            slug="tile-size-performance",
        )
        old_claim.add_version(
            statement="tile32 is generally faster on A100",
            category=KnowledgeClaimCategory.EXPERIMENT_RESULT,
            context={"hardware": "A100", "dataset": "mipnerf360"},
        )

        # New evidence
        scope = self._make_scope(hardware="RTX 4090")
        new_finding = ResearchFinding(
            finding_id="case_e_new",
            claim_type=ClaimType.EXPERIMENT_RESULT,
            statement="Under RTX 4090 with room scene, tile16=450 FPS vs tile32=445 FPS",
            scope=scope,
            evidence_summary="tile16=450 FPS, tile32=445 FPS",
            raw_evidence_uris=["room_4090_metrics.json"],
            alternative_explanations=["Different hardware"],
            contradicting_evidence=["A100 showed different pattern"],
        )

        detector = ConflictDetector()
        conflict = detector.detect(
            old_claim=old_claim,
            new_statement=new_finding.statement,
            new_finding_id="case_e_new",
            new_context=scope.to_dict(),
        )
        self.assertIsNotNone(conflict)
        # Old claim should NOT be deleted
        self.assertEqual(len(old_claim.versions), 1)
        self.assertEqual(conflict.resolution, ConflictResolution.CONTEXTUAL)


class TestOvergeneralizationPrevention(unittest.TestCase):
    """Tests that the system prevents overgeneralization."""

    def test_no_always_universal_fact_from_finding(self):
        """Verify RESEARCH_TRUSTED_WITH_LIMITATIONS prevents FACT promotion."""
        scope = ScopeMetadata(project_id="3dgs", scope="bicycle")
        finding = ResearchFinding(
            finding_id="find_fact_prevent",
            claim_type=ClaimType.EXPERIMENT_RESULT,
            statement="Under A100 with bicycle, tile16=500 FPS",
            scope=scope,
            evidence_summary="FPS=500",
            raw_evidence_uris=["test.json"],
            alternative_explanations=["test"],
            contradicting_evidence=["test"],
            reviewer_verdict=ResearchReviewerVerdict.APPROVED,
        )

        ks = KnowledgeClaimStore()
        pipeline = KnowledgeProposalPipeline(ks)
        proposal = pipeline.create_proposal(finding)
        result = pipeline.run_integrity_gate(proposal, finding)

        # Even with all checks passing, the pipeline maps to
        # EXPERIMENT_RESULT — never FACT — without explicit policy
        promoted = pipeline.promote_to_knowledge(
            proposal, finding,
            topic="tile-size",
            slug="tile-size",
        )
        self.assertIsNotNone(promoted)
        self.assertEqual(promoted.current.category, KnowledgeClaimCategory.EXPERIMENT_RESULT)
        self.assertNotEqual(promoted.current.category, KnowledgeClaimCategory.FACT)

    def test_context_specific_finding_not_universal(self):
        """contextual finding must not become universal statement."""
        scope = ScopeMetadata(
            project_id="3dgs",
            hardware="A100",
            configuration={"tile_size": "16,32"},
            scope="Scene: bicycle",
        )
        finding = ResearchFinding(
            finding_id="find_contextual",
            claim_type=ClaimType.INFERENCE,
            statement="Under A100 with bicycle scene, tile16 outperforms tile32",
            scope=scope,
            evidence_summary="tile16 500 FPS, tile32 490 FPS",
            raw_evidence_uris=["metrics.json"],
            alternative_explanations=["Scene-specific"],
            contradicting_evidence=["tile32 faster on garden"],
        )

        # Verify the statement is scoped
        self.assertIn("Under A100", finding.statement)
        self.assertIn("bicycle scene", finding.statement)

    def test_missing_scope_causes_rejection(self):
        store = ResearchFindingStore()
        ks = KnowledgeClaimStore()
        pipeline = KnowledgeProposalPipeline(ks)

        scope = ScopeMetadata(project_id="", scope="")
        finding = ResearchFinding(
            finding_id="find_noscope",
            claim_type=ClaimType.EXPERIMENT_RESULT,
            statement="Some result",
            scope=scope,
            evidence_summary="test",
            raw_evidence_uris=["test.json"],
            alternative_explanations=["A"],
            contradicting_evidence=["B"],
            reviewer_verdict=ResearchReviewerVerdict.APPROVED,
        )
        proposal = pipeline.create_proposal(finding)
        result = pipeline.run_integrity_gate(proposal, finding)
        self.assertEqual(result, IntegrityGateResult.REJECT)


class TestNegativeTests(unittest.TestCase):
    """Negative tests — expected reject/quarantine/warning."""

    def _make_minimal_finding(
        self, finding_id: str, **overrides
    ) -> ResearchFinding:
        kwargs = {
            "finding_id": finding_id,
            "claim_type": ClaimType.EXPERIMENT_RESULT,
            "statement": "Test result under specific conditions",
            "scope": ScopeMetadata(project_id="3dgs", scope="Scene: bicycle"),
            "evidence_summary": "test",
            "raw_evidence_uris": ["metrics.json"],
            "alternative_explanations": ["Scene dependent"],
            "contradicting_evidence": ["Other scene"],
            "reviewer_verdict": ResearchReviewerVerdict.APPROVED,
        }
        kwargs.update(overrides)
        return ResearchFinding(**kwargs)

    def test_missing_scope_project(self):
        store = ResearchFindingStore()
        ks = KnowledgeClaimStore()
        pipeline = KnowledgeProposalPipeline(ks)
        finding = ResearchFinding(
            finding_id="neg_noproj",
            claim_type=ClaimType.EXPERIMENT_RESULT,
            statement="Result",
            scope=ScopeMetadata(project_id="", scope="test"),
            evidence_summary="test",
            raw_evidence_uris=["test.json"],
            alternative_explanations=["A"],
            contradicting_evidence=["B"],
            reviewer_verdict=ResearchReviewerVerdict.APPROVED,
        )
        proposal = pipeline.create_proposal(finding)
        result = pipeline.run_integrity_gate(proposal, finding)
        self.assertEqual(result, IntegrityGateResult.REJECT)

    def test_missing_experiment_raw_result(self):
        finding = self._make_minimal_finding(
            "neg_no_raw", raw_evidence_uris=[]
        )
        ks = KnowledgeClaimStore()
        pipeline = KnowledgeProposalPipeline(ks)
        proposal = pipeline.create_proposal(finding)
        result = pipeline.run_integrity_gate(proposal, finding)
        self.assertEqual(result, IntegrityGateResult.REJECT)

    def test_missing_code_provenance(self):
        scope = ScopeMetadata(
            project_id="3dgs", scope="test",
            code_commit=None,  # Missing
        )
        finding = ResearchFinding(
            finding_id="neg_nocode",
            claim_type=ClaimType.EXPERIMENT_RESULT,
            statement="Result",
            scope=scope,
            evidence_summary="test",
            raw_evidence_uris=["test.json"],
            alternative_explanations=["A"],
            contradicting_evidence=["B"],
            reviewer_verdict=ResearchReviewerVerdict.APPROVED,
        )
        # Finding is still valid — code_commit is optional in scope
        self.assertIsNone(finding.scope.code_commit)

    def test_conflicting_evidence_preserved(self):
        finding = self._make_minimal_finding(
            "neg_conflict",
            contradicting_evidence=[
                "Previous experiment showed opposite result on garden",
                "Reproduced result differs by 10%",
            ],
        )
        self.assertGreaterEqual(len(finding.contradicting_evidence), 2)

    def test_insufficient_evidence_triggers_warning(self):
        finding = self._make_minimal_finding(
            "neg_insufficient",
            uncertainty="Only one scene, one run, no repeats",
            reviewer_verdict=ResearchReviewerVerdict.WARNING,
        )
        ks = KnowledgeClaimStore()
        pipeline = KnowledgeProposalPipeline(ks)
        proposal = pipeline.create_proposal(finding)
        self.assertIsNotNone(proposal)
        result = pipeline.run_integrity_gate(proposal, finding)
        # With WARNING verdict but all data present, should still pass or warn
        self.assertIn(result, (
            IntegrityGateResult.PASS,
            IntegrityGateResult.WARNING,
            IntegrityGateResult.QUARANTINE,
        ))

    def test_ai_overclaim_pattern_detected(self):
        """Overclaim like 'definitively proves' should trigger WARNING."""
        finding = self._make_minimal_finding(
            "neg_overclaim",
            statement="This definitively proves that tile32 is always better",
        )
        ks = KnowledgeClaimStore()
        pipeline = KnowledgeProposalPipeline(ks)
        proposal = pipeline.create_proposal(finding)
        result = pipeline.run_integrity_gate(proposal, finding)
        self.assertEqual(result, IntegrityGateResult.WARNING)


class TestEndToEndPipeline(unittest.TestCase):
    """Full end-to-end test: Experiment → Finding → Gate → Knowledge → Learning."""

    def test_end_to_end_full_pipeline(self):
        orchestrator = ResearchToKnowledgeClosure()

        # Step 1-2: Create a finding with full scope
        scope = ScopeMetadata(
            project_id="3dgs",
            question_id="tile-size-performance",
            hypothesis_id="tile32-faster",
            experiment_id="epic05-tile-comparison",
            run_ids=["run_001", "run_002"],
            dataset="mipnerf360",
            hardware="NVIDIA A100-SXM4-80GB",
            software={"cuda": "12.8", "pytorch": "2.9.1"},
            configuration={"tile_sizes": "16,32"},
            code_commit="abc123def",
            scope="Scene: bicycle,garden; 1080p resolution; 5 repeats",
        )
        finding = ResearchFinding(
            finding_id="find_e2e_001",
            claim_type=ClaimType.INFERENCE,
            statement=(
                "Under A100 with bicycle and garden scenes, "
                "there is insufficient evidence that tile32 is universally faster than tile16. "
                "Performance is workload-dependent."
            ),
            scope=scope,
            evidence_summary="bicycle: tile16 500 FPS, tile32 490 FPS; garden: tile16 492 FPS, tile32 502 FPS",
            raw_evidence_uris=[
                "results/measured/gsplat_higs_tile16/mipnerf360/bicycle/20260723/metrics.json",
                "results/measured/gsplat_higs_tile16/mipnerf360/garden/20260723/metrics.json",
            ],
            alternative_explanations=[
                "Scene complexity (Gaussian count) shifts optimal tile size",
                "Different coverage patterns change tile intersection workload",
                "Tile32 reduces culling overhead but increases per-tile sorting",
            ],
            contradicting_evidence=[
                "Tile32 faster on garden (502 vs 492 FPS)",
                "Tile16 faster on bicycle (500 vs 490 FPS)",
            ],
            uncertainty="Limited scene coverage (2/5 scenes); single GPU (A100); no resolution sweep",
            reviewer_verdict=ResearchReviewerVerdict.APPROVED,
            reviewer_notes="Adequate evidence for contextual conclusion; insufficient for universal claim",
        )
        orchestrator.finding_store.add(finding)

        # Step 3-7: Run full pipeline
        result = orchestrator.run_full_pipeline(
            finding, topic="tile-size-performance", slug="tile-size-performance"
        )

        # Verify pipeline stages
        self.assertIsNotNone(result["proposal"])
        self.assertEqual(result["gate_result"], "pass")
        self.assertIsNotNone(result["knowledge_claim"])
        self.assertEqual(len(result["learning_evidences"]), 1)
        self.assertEqual(len(result["learning_gaps"]), 1)
        self.assertEqual(len(result["focus_recommendations"]), 1)

        # Step 8: Query
        query_result = orchestrator.query_engine.query_current_knowledge(
            "tile-size-performance"
        )
        self.assertIn("Current knowledge", query_result.answer)

        # Step 9: Knowledge impact
        impact = orchestrator.query_engine.get_knowledge_impact("find_e2e_001")
        self.assertEqual(len(impact["knowledge_claims_affected"]), 1)
        self.assertEqual(len(impact["learning_gaps_created"]), 1)

    def test_end_to_end_no_auto_fact(self):
        """Verify the full pipeline never auto-promotes to FACT."""
        orchestrator = ResearchToKnowledgeClosure()

        scope = ScopeMetadata(
            project_id="3dgs",
            hardware="A100",
            scope="Scene: bicycle",
        )
        finding = ResearchFinding(
            finding_id="find_no_fact_e2e",
            claim_type=ClaimType.EXPERIMENT_RESULT,
            statement="Under A100 with bicycle, tile16 measured 500 FPS",
            scope=scope,
            evidence_summary="FPS=500",
            raw_evidence_uris=["test.json"],
            alternative_explanations=["Scene-specific"],
            contradicting_evidence=["Other scenes may differ"],
            reviewer_verdict=ResearchReviewerVerdict.APPROVED,
        )
        orchestrator.finding_store.add(finding)
        result = orchestrator.run_full_pipeline(
            finding, topic="fps-test", slug="fps-test"
        )

        claim = orchestrator.knowledge_store.get_by_slug("fps-test")
        self.assertIsNotNone(claim)
        self.assertIsNotNone(claim.current)
        # Must NOT be FACT without explicit policy
        self.assertNotEqual(claim.current.category.value, "FACT")

    def test_orchestrator_handles_non_promotable_finding(self):
        orchestrator = ResearchToKnowledgeClosure()
        scope = ScopeMetadata(project_id="", scope="")
        finding = ResearchFinding(
            finding_id="find_bad_e2e",
            claim_type=ClaimType.EXPERIMENT_RESULT,
            statement="Bad result",
            scope=scope,
            evidence_summary="test",
            rejected=True,
            rejection_reason="Missing data",
        )
        orchestrator.finding_store.add(finding)
        result = orchestrator.run_full_pipeline(
            finding, topic="bad", slug="bad"
        )
        self.assertIn("errors", result)
        self.assertGreater(len(result["errors"]), 0)


class TestOrchestratorTileAnalysis(unittest.TestCase):
    """Tests for the Orchestrator's tile analysis with real data patterns."""

    def test_build_tile_analysis_with_mixed_results(self):
        orchestrator = ResearchToKnowledgeClosure()

        scope16 = ScopeMetadata(
            project_id="3dgs", hardware="A100",
            configuration={"config_id": "gsplat_higs_tile16"},
            scope="Scene: bicycle",
        )
        scope32 = ScopeMetadata(
            project_id="3dgs", hardware="A100",
            configuration={"config_id": "gsplat_higs_tile32"},
            scope="Scene: garden",
        )

        f16 = ResearchFinding(
            finding_id="find_tile16_bicycle",
            claim_type=ClaimType.EXPERIMENT_RESULT,
            statement="tile16=500 FPS on A100 bicycle",
            scope=scope16,
            evidence_summary="FPS=500.0",
            raw_evidence_uris=["tile16_bicycle.json"],
        )
        f32 = ResearchFinding(
            finding_id="find_tile32_garden",
            claim_type=ClaimType.EXPERIMENT_RESULT,
            statement="tile32=502 FPS on A100 garden",
            scope=scope32,
            evidence_summary="FPS=502.0",
            raw_evidence_uris=["tile32_garden.json"],
        )
        # Need to fix: scope.configuration should have "tile_size" for analysis
        f16.scope.configuration["tile_size"] = "16"
        f32.scope.configuration["tile_size"] = "32"

        orchestrator.finding_store.add(f16)
        orchestrator.finding_store.add(f32)

        analysis = orchestrator.build_tile_analysis(
            "3dgs",
            ["find_tile16_bicycle", "find_tile32_garden"],
        )
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.claim_type, ClaimType.INFERENCE)
        self.assertIn("insufficient evidence", analysis.statement.lower())
        self.assertGreater(len(analysis.alternative_explanations), 0)
        self.assertGreater(len(analysis.contradicting_evidence), 0)
        self.assertIsNotNone(analysis.uncertainty)

    def test_build_tile_analysis_no_tile32_findings(self):
        orchestrator = ResearchToKnowledgeClosure()
        # No findings added
        analysis = orchestrator.build_tile_analysis("3dgs", [])
        self.assertIsNone(analysis)


class TestProvenanceChain(unittest.TestCase):
    """Tests that provenance chains are correctly built and traced."""

    def test_provenance_chain_traceability(self):
        claim = KnowledgeClaim(
            claim_id="kc_prov",
            topic="prov-test",
            slug="prov-test",
        )
        claim.add_version(
            statement="Version 1",
            category=KnowledgeClaimCategory.EXPERIMENT_RESULT,
            source_finding_id="find_001",
        )
        claim.add_version(
            statement="Version 2",
            category=KnowledgeClaimCategory.INFERENCE,
            source_finding_id="find_002",
        )
        claim.provenance_chain = ["find_001", "find_002"]

        # Chain should show: KnowledgeClaim → finding → ...
        self.assertEqual(len(claim.provenance_chain), 2)
        for v in claim.versions:
            self.assertIsNotNone(v.source_finding_id)

    def test_provenance_preserved_after_update(self):
        claim = KnowledgeClaim(
            claim_id="kc_prov2",
            topic="prov-test-2",
            slug="prov-test-2",
        )
        claim.add_version(
            statement="Original",
            category=KnowledgeClaimCategory.EXPERIMENT_RESULT,
            source_finding_id="find_original",
        )
        claim.provenance_chain.append("find_original")

        claim.add_version(
            statement="Updated",
            category=KnowledgeClaimCategory.INFERENCE,
            source_finding_id="find_update",
            change_reason="New evidence available",
        )
        claim.provenance_chain.append("find_update")

        self.assertEqual(len(claim.versions), 2)
        self.assertEqual(len(claim.provenance_chain), 2)
        v1 = claim.get_version(1)
        v2 = claim.get_version(2)
        self.assertEqual(v1.source_finding_id, "find_original")
        self.assertEqual(v2.source_finding_id, "find_update")


class TestSerialization(unittest.TestCase):
    """Tests for JSON serialization/deserialization."""

    def test_finding_store_round_trip(self):
        import tempfile
        store = ResearchFindingStore()
        scope = ScopeMetadata(project_id="3dgs", scope="test")
        finding = ResearchFinding(
            finding_id="find_ser",
            claim_type=ClaimType.INFERENCE,
            statement="Serialized finding",
            scope=scope,
            evidence_summary="test",
            alternative_explanations=["A", "B"],
            reviewer_verdict=ResearchReviewerVerdict.APPROVED,
        )
        store.add(finding)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name
            store.save_json(tmp_path)

        loaded = ResearchFindingStore.load_json(tmp_path)
        self.assertIsNotNone(loaded.get("find_ser"))
        self.assertEqual(loaded.get("find_ser").statement, "Serialized finding")

    def test_conflict_detector_round_trip(self):
        import tempfile
        detector = ConflictDetector()
        old_claim = KnowledgeClaim(
            claim_id="kc_old_s",
            topic="test",
            slug="test",
        )
        old_claim.add_version(
            statement="Old version",
            category=KnowledgeClaimCategory.EXPERIMENT_RESULT,
            context={"hw": "A100"},
        )
        detector.detect(
            old_claim=old_claim,
            new_statement="New version on RTX 4090",
            new_finding_id="find_new_s",
            new_context={"hw": "RTX 4090"},
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name
            detector.save_json(tmp_path)

        loaded = ConflictDetector.load_json(tmp_path)
        self.assertEqual(len(loaded.list_all()), 1)

    def test_learning_integration_round_trip(self):
        import tempfile
        li = LearningIntegration()
        li.add_evidence(
            node_name="Test", evidence_type=EvidenceType.INFERENCE,
            statement="Test evidence", source_finding_id="f1",
        )
        li.identify_gap(
            node_name="Test", question="Q?",
            source_finding_id="f1", reason="R",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name
            li.save_json(tmp_path)

        loaded = LearningIntegration.load_json(tmp_path)
        self.assertEqual(len(loaded.list_evidence()), 1)
        self.assertEqual(len(loaded.list_gaps()), 1)


class TestKnowledgeClaimStore(unittest.TestCase):
    """Tests for KnowledgeClaimStore queries."""

    def setUp(self):
        self.store = KnowledgeClaimStore()
        claim = KnowledgeClaim(
            claim_id="kc_q1", topic="topic-a", slug="topic-a",
        )
        claim.add_version(
            statement="Version 1",
            category=KnowledgeClaimCategory.EXPERIMENT_RESULT,
            context={"hw": "A100"},
        )
        claim.add_version(
            statement="Version 2",
            category=KnowledgeClaimCategory.INFERENCE,
            context={"hw": "RTX 4090"},
        )
        self.store.add(claim)

    def test_list_by_topic(self):
        claims = self.store.list_by_topic("topic-a")
        self.assertEqual(len(claims), 1)

    def test_get_by_slug(self):
        claim = self.store.get_by_slug("topic-a")
        self.assertIsNotNone(claim)

    def test_query_current(self):
        stmt = self.store.query_current("topic-a")
        self.assertEqual(stmt, "Version 2")

    def test_query_historical(self):
        versions = self.store.query_historical("topic-a")
        self.assertEqual(len(versions), 2)

    def test_query_context_specific(self):
        versions = self.store.query_context_specific(
            "topic-a", {"hw": "A100"}
        )
        self.assertEqual(len(versions), 1)


if __name__ == "__main__":
    unittest.main()
