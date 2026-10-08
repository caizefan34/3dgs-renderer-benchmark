#!/usr/bin/env python3
"""DSH-8 APBC0 SUMMARY + MANIFEST.sha256 writer."""
import json
import hashlib
from pathlib import Path

ART = Path("/root/dsh_20261001/artifacts/DSH8_APBC0")

REQUIRED = [
    "APBC0_ENVIRONMENT.json", "APBC0_SOURCE_LOCK.json",
    "APBC0_REACHABILITY_TRACE.json", "APBC0_METADATA_BENCH.json",
    "APBC0_KERNEL_BENCH.json", "APBC0_CORRECTNESS.json",
    "APBC0_SUMMARY.json", "APBC0_VERDICT.json", "APBC0_REPORT.md",
]
ABSENT_BY_STOP_RULE = [
    "APBC0_METADATA_BENCH.json", "APBC0_KERNEL_BENCH.json",
    "APBC0_CORRECTNESS.json",
]


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    verdict = json.loads((ART / "APBC0_VERDICT.json").read_text())
    reach = json.loads((ART / "APBC0_REACHABILITY_TRACE.json").read_text())
    per_scene = verdict["gate_P1"]["per_scene_peak"]

    summary = {
        "p0": "DSH-8 APBC0 Exact Active-Pixel Bucket Compaction",
        "decision": verdict["decision"],
        "evidence_level": verdict["evidence_level"],
        "phase_reached": "A (structural headroom trace)",
        "stop_rule_applied": verdict["stop_rule_applied"],
        "gate_P1": {
            "threshold": 0.25,
            "metric": "peak cost-weighted structurally-unreachable bucket-pixel scan visits across all captured checkpoints",
            "scope": "EVERY scene",
            "per_scene_peak": per_scene,
            "min_scene_peak": verdict["gate_P1"]["min_scene_peak"],
            "pass": verdict["gate_P1"]["every_scene_pass"],
        },
        "predicate_verification": {
            "predicate": "pixel p reachable by bucket b iff n_contrib[p] > 32*b",
            "verified_against": "rasterization/include/kernels_backward.cuh blend_backward_cu (frozen source)",
            "off_by_one_confirmed": True,
        },
        "phase_A_completed_iterations": {
            s: sorted(int(k) for k in d["unreachable_fraction_by_iter"].keys())
            for s, d in reach["scenes"].items()
        },
        "trajectory_note": "non-monotonic; room/garden peak 5K-20K, bicycle rises to 0.205 at 25K; no scene reaches 0.25",
        "files_absent_by_stop_rule": ABSENT_BY_STOP_RULE,
        "absent_reason": "P1 structural headroom gate failed in every scene; "
                         "the spec stop rule forbids Phase B/C without headroom.",
        "claim_scope": verdict["claim_scope"],
        "provenance": {
            "gpu": "EPIC-05 A100 (8x A100-SXM4-80GB)",
            "source_lock": "FasterGSFused ae665af2..., NeRFICG e458ac6...",
            "scenes": ["room", "garden", "bicycle"], "seed": 42,
            "full_schedule_captured": True,
        },
    }
    (ART / "APBC0_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print("wrote APBC0_SUMMARY.json")

    lines = []
    for name in REQUIRED:
        p = ART / name
        if p.exists():
            lines.append("%s  %s" % (sha256_file(p), name))
        else:
            lines.append("ABSENT_BY_STOP_RULE  %s" % name)
    (ART / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    print("wrote MANIFEST.sha256")


if __name__ == "__main__":
    main()
