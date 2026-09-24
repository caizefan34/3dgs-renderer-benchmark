#!/usr/bin/env python3
"""Deep-inspect the candidate_c_r3_final audit package."""
import json
from pathlib import Path

AUDIT = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/audit_packages/candidate_c_r3_final")

print("=== FULL TREE ===")
for p in sorted(AUDIT.rglob("*")):
    if p.is_file():
        print(f"  F {p.relative_to(AUDIT).as_posix()}  {p.stat().st_size}")
    else:
        print(f"  D {p.relative_to(AUDIT).as_posix()}/")

print("\n=== CHECKPOINT MANIFEST ===")
cp = AUDIT / "checkpoints/checkpoint_manifest.json"
if cp.exists():
    print(cp.read_text())

print("\n=== AGGREGATED SUMMARY (first 1200 chars) ===")
asum = AUDIT / "analysis/aggregated_summary.json"
if asum.exists():
    print(asum.read_text()[:1200])

print("\n=== FINAL DECISION ===")
fd = AUDIT / "analysis/final_decision.json"
if fd.exists():
    print(fd.read_text())

print("\n=== RAW WINDOW CONTENTS ===")
for w in ["5000", "15000", "29970"]:
    wdir = AUDIT / "raw" / w
    if wdir.is_dir():
        print(f"-- {w}:")
        for fn in sorted(wdir.iterdir()):
            print(f"    {fn.name}  {fn.stat().st_size if fn.is_file() else '<dir>'}")
