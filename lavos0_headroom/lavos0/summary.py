"""Assemble LAVOS0_SUMMARY.json (PREREG Sec.12) from the phase results.

The frozen analyzer (lavos0_analyze.py) reads exactly these top-level keys:
    source_identity_pass, event_semantics_resolved, structural_headroom_pass,
    topology_exactness_pass, performance_headroom_pass, strong_headroom_pass,
    per_scene, global.
This builder produces that schema so the analyzer can emit the verdict without
any glue code.
"""

from typing import Dict, Optional

from . import config


def build_summary(
    environment: Optional[Dict] = None,
    source_identity: Optional[Dict] = None,
    event_order: Optional[Dict] = None,
    consumer_trace: Optional[Dict] = None,
    headroom: Optional[Dict] = None,
    topology_oracle: Optional[Dict] = None,
    perf_oracle: Optional[Dict] = None,
) -> Dict:
    """Fold every phase result into the LAVOS0_SUMMARY schema.

    Each phase result is the dict emitted by the corresponding module's
    evaluate/result function. Missing phases default to their fail-closed
    value (pass=False) so an incomplete run can never accidentally promote.
    """
    headroom = headroom or {}
    perf_oracle = perf_oracle or {}
    topology_oracle = topology_oracle or {}
    event_order = event_order or {}

    source_identity_pass = bool((source_identity or {}).get("source_identity_pass", False))
    event_semantics_resolved = bool(event_order.get("resolved", False))
    structural_headroom_pass = bool(headroom.get("passed", False))
    topology_exactness_pass = bool(topology_oracle.get("exactness_pass", False))
    performance_headroom_pass = bool(perf_oracle.get("performance_headroom_pass", False))
    strong_headroom_pass = bool(perf_oracle.get("strong_headroom_pass", False))

    per_scene = dict(headroom.get("per_scene", {}))
    # enrich per-scene rows with the perf numbers when available
    perf_per = perf_oracle.get("per_scene", {})
    for scene, perf_row in perf_per.items():
        row = per_scene.setdefault(scene, {})
        row.update(perf_row)

    global_block = {
        "headroom_aggregate": headroom.get("aggregate", {}),
        "perf": {
            k: v for k, v in perf_oracle.items()
            if k not in ("per_scene", "performance_headroom_pass",
                         "strong_headroom_pass")
        },
        "topology_oracle": {
            "exactness_pass": topology_exactness_pass,
            "n_fences": topology_oracle.get("n_fences"),
            "n_mismatches": topology_oracle.get("n_mismatches"),
        },
        "event_order_notes": event_order.get("notes"),
        "source_identity": source_identity,
    }

    return {
        "source_identity_pass": source_identity_pass,
        "event_semantics_resolved": event_semantics_resolved,
        "structural_headroom_pass": structural_headroom_pass,
        "topology_exactness_pass": topology_exactness_pass,
        "performance_headroom_pass": performance_headroom_pass,
        "strong_headroom_pass": strong_headroom_pass,
        "per_scene": per_scene,
        "global": global_block,
    }
