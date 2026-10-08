#!/usr/bin/env python3
"""DSH-9 CVL0 report writer."""
import json
import hashlib
from pathlib import Path
import platform
import socket

ART = Path("/root/dsh_20261001/artifacts/DSH9_CVL0")


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    verdict = json.loads((ART / "CVL0_VERDICT.json").read_text())
    cvl = json.loads((ART / "CVL0_ACCESS_TRACE.json").read_text())

    L = []
    L.append("# DSH-9 - CVL0: Co-Visibility Locality Layout Audit")
    L.append("")
    L.append("## Verdict")
    L.append("")
    L.append("**%s** (evidence %s)." % (verdict["decision"], verdict["evidence_level"]))
    L.append("")
    L.append("Phase A access trace is recorded. The frozen gates C1-C4 require "
             "Phase B (layout remap) + Phase C (ncu L2/DRAM profiling + replay "
             "timing) + Phase D (amortization), none of which are performed. No "
             "claim that co-visibility layout beats Morton is made without "
             "measured L2/DRAM evidence.")
    L.append("")
    L.append("## Research question")
    L.append("")
    L.append("Faster-GS periodically Morton-sorts Gaussians by 3D position for "
             "spatial locality. But training accesses are camera/tile "
             "conditioned: Gaussians near in 3D are not always the same Gaussians "
             "repeatedly consumed together by sorted tile lists. Can a layout "
             "derived from empirical co-visibility preserve semantics while "
             "reducing DRAM/L2 pressure more than Morton ordering? This P0 does "
             "NOT change rendering order or optimization math - only physical "
             "parameter/state layout and ID remapping.")
    L.append("")
    L.append("## Phase A access trace")
    L.append("")
    L.append("Co-access rule: %s" % cvl["coaccess_rule"])
    L.append("")
    L.append("Per-scene (final captured iteration):")
    L.append("")
    L.append("| scene | iter | unique co-access edges | visible frac>0 | mean tiles/visible | edge weight p50 | p99 | max |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s, d in cvl["scenes"].items():
        fr = d["final_record"]
        L.append("| %s | %s | %d | %.3f | %.1f | %.0f | %.0f | %.0f |" % (
            s, d["iterations"][-1] if d["iterations"] else "?",
            d["n_coaccess_edges_unique"][-1],
            fr["primitive_visible_frac_gt0"], fr["mean_tiles_per_visible"],
            fr["coaccess_edge_weight"]["p50"], fr["coaccess_edge_weight"]["p99"],
            fr["coaccess_edge_weight"]["max"]))
    L.append("")
    L.append("## Structural observations")
    L.append("")
    for obs in verdict["structural_observations"]:
        L.append("- %s" % obs)
    L.append("")
    L.append("## Frozen gates status")
    L.append("")
    L.append("| gate | desc | status |")
    L.append("|---|---|---|")
    for k, g in verdict["gates"].items():
        L.append("| %s | %s | %s |" % (k, g["desc"], g["status"]))
    L.append("")
    L.append("## Why INVESTIGATE")
    L.append("")
    L.append(verdict["claim_scope"])
    L.append("")
    L.append("## Phase B blocker (precise)")
    L.append("")
    L.append(verdict["phase_B_blocker"])
    L.append("")
    L.append("## Provenance")
    L.append("")
    L.append("- GPU: EPIC-05 A100 (%s)" % socket.gethostname())
    L.append("- Python %s" % platform.python_version())
    L.append("- Source lock: FasterGSFused ae665af2..., NeRFICG e458ac6...")
    L.append("- Co-access computed from the frozen forward sorted tile lists "
             "(instance_primitive_indices + keys); no full NxN graph stored.")
    L.append("")
    L.append("## Artifacts")
    L.append("")
    for name in ["CVL0_ENVIRONMENT.json", "CVL0_SOURCE_LOCK.json",
                 "CVL0_ACCESS_TRACE.json", "CVL0_VERDICT.json",
                 "CVL0_SUMMARY.json", "CVL0_REPORT.md"]:
        p = ART / name
        if p.exists():
            L.append("- %s (sha256 %s...)" % (name, sha256_file(p)[:16]))
    L.append("")
    L.append("Absent (Phase B/C/D): CVL0_LAYOUT_DEFINITION.json, "
             "CVL0_PROFILE.json, CVL0_AMORTIZATION.json, CVL0_CORRECTNESS.json.")

    (ART / "CVL0_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote CVL0_REPORT.md (%d lines)" % len(L))


if __name__ == "__main__":
    main()
