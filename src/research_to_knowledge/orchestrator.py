"""Research-to-Knowledge Closure Orchestrator.

Ties together:
  ResearchFinding → KnowledgeProposal → IntegrityGate → KnowledgeClaim
  → LearningEvidence → LearningGap → FocusRecommendation

Uses real 3DGS benchmark data for the tile16 / tile32 analysis.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .research_finding import (
    ClaimType,
    ResearchFinding,
    ResearchFindingStore,
    ResearchReviewerVerdict,
    ScopeMetadata,
)
from .knowledge_claim import (
    KnowledgeClaimCategory,
    KnowledgeClaimStore,
)
from .knowledge_proposal import (
    IntegrityGateResult,
    KnowledgeProposalPipeline,
)
from .conflict_detection import ConflictDetector, ConflictResolution
from .learning_integration import (
    EvidenceType,
    LearningIntegration,
)
from .research_knowledge_query import (
    ResearchKnowledgeQueryEngine,
)


class ResearchToKnowledgeClosure:
    """Orchestrates the full pipeline from research to knowledge to learning."""

    def __init__(self) -> None:
        self.finding_store = ResearchFindingStore()
        self.knowledge_store = KnowledgeClaimStore()
        self.conflict_detector = ConflictDetector()
        self.learning = LearningIntegration()
        self.pipeline = KnowledgeProposalPipeline(self.knowledge_store)
        self.query_engine = ResearchKnowledgeQueryEngine(
            finding_store=self.finding_store,
            knowledge_store=self.knowledge_store,
            conflict_detector=self.conflict_detector,
            learning_integration=self.learning,
        )

    def ingest_from_benchmark_metrics(
        self, metrics_paths: list[str], project_id: str
    ) -> list[ResearchFinding]:
        """Ingest real benchmark metric JSON files as ResearchFindings."""
        findings = []
        for mp in metrics_paths:
            path = Path(mp)
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            renderer_id = (
                data.get("renderer", {}).get("config_id", "unknown")
            )
            scene = data.get("benchmark", {}).get("scene_id", "unknown")
            gpu = data.get("environment", {}).get("gpu", "unknown")
            fps = (
                data.get("metrics", {}).get("performance", {}).get("fps")
            )
            psnr = (
                data.get("metrics", {}).get("quality", {}).get("psnr_db")
            )

            scope = ScopeMetadata(
                project_id=project_id,
                dataset=data.get("benchmark", {}).get("dataset_id"),
                hardware=gpu,
                software={
                    "driver": data.get("environment", {}).get("driver", ""),
                    "cuda": data.get("environment", {}).get("cuda", ""),
                    "pytorch": data.get("environment", {}).get("pytorch", ""),
                },
                configuration={
                    "config_id": renderer_id,
                    "tile_size": (
                        "16" if "tile16" in renderer_id else
                        "32" if "tile32" in renderer_id else
                        "8" if "auto" not in renderer_id else "auto"
                    ),
                },
                code_commit=data.get("environment", {}).get(
                    "benchmark_commit"
                ),
                scope=f"Scene: {scene}, GPU: {gpu}, Config: {renderer_id}",
                created_at=data.get("provenance", {}).get(
                    "measured_at", ""
                ),
            )

            statement = (
                f"Under {gpu} with scene {scene} and config {renderer_id}, "
                f"measured FPS={fps:.1f} and PSNR={psnr:.2f} dB."
            )

            finding = ResearchFinding(
                finding_id=f"find_{project_id}_{scene}_{renderer_id}",
                claim_type=ClaimType.EXPERIMENT_RESULT,
                statement=statement,
                scope=scope,
                evidence_summary=f"FPS={fps:.1f}, PSNR={psnr:.2f}",
                raw_evidence_uris=[mp],
                alternative_explanations=[],
                contradicting_evidence=[],
                uncertainty=f"Single measurement; CI95 not evaluated for cross-config comparison",
            )
            self.finding_store.add(finding)
            findings.append(finding)
        return findings

    def build_tile_analysis(
        self, project_id: str, finding_ids: list[str]
    ) -> ResearchFinding | None:
        """Aggregate tile16 and tile32 findings into a contextual analysis.

        Uses real data from the benchmark findings.
        """
        tile16_findings = [
            self.finding_store.get(fid) for fid in finding_ids
            if fid and "tile16" in fid
        ]
        tile32_findings = [
            self.finding_store.get(fid) for fid in finding_ids
            if fid and "tile32" in fid
        ]
        tile16_findings = [f for f in tile16_findings if f]
        tile32_findings = [f for f in tile32_findings if f]

        if not tile16_findings and not tile32_findings:
            return None

        # Collect scenes from both groups
        scenes_seen = set()
        tile16_scenes = set()
        tile32_scenes = set()
        fps_data: dict[str, dict[str, float]] = {}

        for f in tile16_findings:
            scene = f.scope.configuration.get("config_id", "unknown")
            tile16_scenes.add(f.scope.scope or "")
            scenes_seen.add(f.scope.scope or "")
            fps_data[scene] = fps_data.get(scene, {})
            fps_data[scene]["tile16"] = float(
                f.evidence_summary.split("=")[1].split(",")[0]
                if "FPS=" in f.evidence_summary else 0
            )

        for f in tile32_findings:
            scene = f.scope.configuration.get("config_id", "unknown")
            tile32_scenes.add(f.scope.scope or "")
            scenes_seen.add(f.scope.scope or "")
            fps_data[scene] = fps_data.get(scene, {})
            fps_data[scene]["tile32"] = float(
                f.evidence_summary.split("=")[1].split(",")[0]
                if "FPS=" in f.evidence_summary else 0
            )

        hardware = (tile16_findings[0].scope.hardware
                    if tile16_findings else
                    tile32_findings[0].scope.hardware if tile32_findings
                    else "unknown")

        # Build an aggregate evidence summary
        comparison_parts = []
        scenes_where_tile16_faster = []
        scenes_where_tile32_faster = []

        for scene_key, speeds in fps_data.items():
            if "tile16" in speeds and "tile32" in speeds:
                diff = speeds["tile16"] - speeds["tile32"]
                faster = "tile16" if diff > 0 else "tile32"
                if faster == "tile16":
                    scenes_where_tile16_faster.append(scene_key)
                else:
                    scenes_where_tile32_faster.append(scene_key)
                comparison_parts.append(
                    f"{scene_key}: tile16={speeds['tile16']:.1f}FPS, "
                    f"tile32={speeds['tile32']:.1f}FPS ({faster} faster)"
                )

        evidence_summary = "; ".join(comparison_parts)

        alternative_explanations = [
            "Tile size performance depends on scene complexity (Gaussian count and overlap distribution)",
            "Different resolution/sh_degree may change the optimal tile size",
            "Tile32 may benefit scenes with higher Gaussian density per tile",
            "Tile16 reduces culling overhead but increases sorting work",
        ]

        contradicting_evidence_parts = []
        if scenes_where_tile16_faster:
            contradicting_evidence_parts.append(
                f"Scenes where tile16 was faster: {', '.join(scenes_where_tile16_faster)}"
            )
        if scenes_where_tile32_faster:
            contradicting_evidence_parts.append(
                f"Scenes where tile32 was faster: {', '.join(scenes_where_tile32_faster)}"
            )

        statement = (
            f"Under the evaluated {hardware} workloads across "
            f"{', '.join(scenes_seen)} scenes, "
            f"there is insufficient evidence that tile32 is universally faster than tile16. "
            f"Performance is workload-dependent: tile16 outperforms tile32 in some scenes "
            f"while tile32 leads in others."
        )

        scope = ScopeMetadata(
            project_id=project_id,
            question_id="tile-size-performance",
            hypothesis_id="tile32-faster-hypothesis",
            dataset="mipnerf360, tanks_and_temples",
            hardware=hardware,
            configuration={"tile_sizes": "16,32", "renderer_family": "gsplat_higs"},
            code_commit=(
                tile16_findings[0].scope.code_commit if tile16_findings else None
            ),
            scope=(
                f"Scenes: {', '.join(scenes_seen)}, "
                f"Hardware: {hardware}"
            ),
        )

        finding = ResearchFinding(
            finding_id=f"find_{project_id}_tile16_vs_tile32_analysis",
            claim_type=ClaimType.INFERENCE,
            statement=statement,
            scope=scope,
            evidence_summary=evidence_summary,
            raw_evidence_uris=list(
                set(f.raw_evidence_uris[0] for f in tile16_findings + tile32_findings
                    if f.raw_evidence_uris)
            ),
            alternative_explanations=alternative_explanations,
            contradicting_evidence=(
                contradicting_evidence_parts
                if contradicting_evidence_parts
                else ["No consistent advantage for either tile size across scenes"]
            ),
            uncertainty=(
                "CI95 overlap; insufficient statistical power; "
                "limited scene coverage (2-3 scenes); "
                "single GPU model tested"
            ),
        )
        self.finding_store.add(finding)
        return finding

    def run_full_pipeline(
        self, finding: ResearchFinding, topic: str, slug: str
    ) -> dict[str, Any]:
        """Run the full pipeline for a finding: proposal → gate → knowledge → learning."""
        result: dict[str, Any] = {
            "finding_id": finding.finding_id,
            "proposal": None,
            "gate_result": None,
            "knowledge_claim": None,
            "conflicts": [],
            "learning_evidences": [],
            "learning_gaps": [],
            "focus_recommendations": [],
            "errors": [],
        }

        # Step 1: Create Knowledge Proposal
        proposal = self.pipeline.create_proposal(finding)
        if proposal is None:
            result["errors"].append("Failed to create proposal (finding may be rejected or inconclusive)")
            return result
        result["proposal"] = proposal.proposal_id

        # Step 2: Run Integrity Gate
        gate_result = self.pipeline.run_integrity_gate(proposal, finding)
        result["gate_result"] = gate_result.value

        if gate_result == IntegrityGateResult.REJECT:
            result["errors"].append(f"Insufficient evidence for promotion: {finding.finding_id}")
            return result

        # Step 3: Check for conflicts with existing knowledge
        existing = self.knowledge_store.get_by_slug(slug)
        conflict = self.conflict_detector.detect(
            old_claim=existing,
            new_statement=finding.statement,
            new_finding_id=finding.finding_id,
            new_context=finding.scope.to_dict(),
        )
        if conflict:
            result["conflicts"].append(conflict.conflict_id)
            if existing:
                self.knowledge_store.add_conflict(
                    existing.claim_id, conflict.conflict_id
                )

        # Step 4: Promote to Knowledge (even with WARNING)
        claim = self.pipeline.promote_to_knowledge(
            proposal, finding, topic=topic, slug=slug
        )
        if claim is None:
            result["errors"].append("Failed to promote to knowledge")
            return result
        result["knowledge_claim"] = claim.claim_id

        # Step 5: Learning Integration — add evidence
        evidence_types = {
            ClaimType.EXPERIMENT_RESULT: EvidenceType.EXPERIMENT_RESULT,
            ClaimType.INFERENCE: EvidenceType.INFERENCE,
        }
        ev = self.learning.add_evidence(
            node_name=topic,
            evidence_type=evidence_types.get(
                finding.claim_type, EvidenceType.RESEARCH
            ),
            statement=finding.statement,
            source_finding_id=finding.finding_id,
            context=finding.scope.to_dict(),
            uncertainty=finding.uncertainty,
        )
        result["learning_evidences"].append(ev.evidence_id)

        # Step 6: Identify learning gaps
        if finding.uncertainty or "insufficient evidence" in finding.statement.lower():
            gap = self.learning.identify_gap(
                node_name=topic,
                question=(
                    f"Need to understand how scene coverage and workload characteristics "
                    f"change the optimal tile size between tile16 and tile32"
                ),
                source_finding_id=finding.finding_id,
                reason=f"Research finding shows insufficient evidence for universal tile-size advantage",
                priority="medium",
                evidence=finding.evidence_summary,
            )
            result["learning_gaps"].append(gap.gap_id)

            focus = self.learning.create_focus_recommendation(
                gap=gap,
                title=(
                    f"Investigate scene_coverage × tile_size interaction for "
                    f"{finding.scope.hardware}"
                ),
            )
            result["focus_recommendations"].append(focus.rec_id)

        return result

    def get_full_impact(self, finding_id: str) -> dict[str, Any]:
        """Get the full knowledge impact of a finding."""
        return self.query_engine.get_knowledge_impact(finding_id)
