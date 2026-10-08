#!/usr/bin/env python3
"""Write MANIFEST.sha256 for DSH-7 and DSH-9 artifacts (DSH-8 done separately)."""
import hashlib
from pathlib import Path

ART = Path("/root/dsh_20261001/artifacts")

JOBS = {
    "DSH7_BARS0": ([
        "BARS0_ENVIRONMENT.json", "BARS0_SOURCE_LOCK.json", "BARS0_TILE_TRACE.json",
        "BARS0_STRUCTURAL_ORACLE.json", "BARS0_ADAPTIVE_ORACLE.json",
        "BARS0_TAIL_PACKING.json", "BARS0_VERDICT.json", "BARS0_SUMMARY.json",
        "BARS0_REPORT.md",
    ], ["BARS0_STATIC_K_BENCH.json"]),
    "DSH9_CVL0": ([
        "CVL0_ENVIRONMENT.json", "CVL0_SOURCE_LOCK.json", "CVL0_ACCESS_TRACE.json",
        "CVL0_VERDICT.json", "CVL0_SUMMARY.json", "CVL0_REPORT.md",
    ], ["CVL0_LAYOUT_DEFINITION.json", "CVL0_PROFILE.json",
        "CVL0_AMORTIZATION.json", "CVL0_CORRECTNESS.json"]),
}


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    for d, (required, absent) in JOBS.items():
        adir = ART / d
        lines = []
        for name in required:
            p = adir / name
            if p.exists():
                lines.append("%s  %s" % (sha256_file(p), name))
            else:
                lines.append("MISSING  %s" % name)
        for name in absent:
            lines.append("ABSENT_PHASE_B_PENDING  %s" % name)
        (adir / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
        print("wrote %s/MANIFEST.sha256" % d)
        print("\n".join(lines))


if __name__ == "__main__":
    main()
