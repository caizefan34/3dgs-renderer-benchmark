"""Mechanical verdict assembly (PREREG Sec.9) + frozen analyzer integration.

The frozen analyzer (lavos0_analyze.py) reads LAVOS0_SUMMARY.json and writes
LAVOS0_VERDICT.json by exact precedence. This module provides the same
precedence fold so the harness can emit a verdict locally for CI, and it is the
single source of truth for how phase results map to pass/fail booleans.
"""

from typing import Dict, Optional

from . import config


def fold_verdict(summary: Dict) -> Dict:
    """Fold a LAVOS0_SUMMARY dict into a LAVOS0_VERDICT dict.

    Mirrors lavos0_analyze.py exactly. Fields read (all optional, default False):
      source_identity_pass, event_semantics_resolved, structural_headroom_pass,
      topology_exactness_pass, performance_headroom_pass, strong_headroom_pass,
      per_scene, global.
    """
    def g(key):
        return bool(summary.get(key, False))

    if not g("source_identity_pass"):
        verdict = "LAVOS0_SOURCE_IDENTITY_FAIL"
    elif not g("event_semantics_resolved"):
        verdict = "LAVOS0_EVENT_SEMANTICS_UNRESOLVED"
    elif not g("structural_headroom_pass"):
        verdict = "LAVOS0_NO_STRUCTURAL_HEADROOM"
    elif not g("topology_exactness_pass"):
        verdict = "LAVOS0_TOPOLOGY_EXACTNESS_FAIL"
    elif not g("performance_headroom_pass"):
        verdict = "LAVOS0_PERFORMANCE_HEADROOM_FAIL"
    else:
        strong = bool(summary.get("strong_headroom_pass", False))
        verdict = ("LAVOS0_STRONG_HEADROOM_PROMOTE" if strong
                   else "LAVOS0_HEADROOM_PROMOTE")

    return {
        "verdict": verdict,
        "source_identity_pass": g("source_identity_pass"),
        "event_semantics_resolved": g("event_semantics_resolved"),
        "structural_headroom_pass": g("structural_headroom_pass"),
        "topology_exactness_pass": g("topology_exactness_pass"),
        "performance_headroom_pass": g("performance_headroom_pass"),
        "strong_headroom_pass": g("strong_headroom_pass"),
        "per_scene": summary.get("per_scene", {}),
        "global": summary.get("global", {}),
    }


def is_promote(verdict: str) -> bool:
    return verdict in ("LAVOS0_HEADROOM_PROMOTE", "LAVOS0_STRONG_HEADROOM_PROMOTE")
