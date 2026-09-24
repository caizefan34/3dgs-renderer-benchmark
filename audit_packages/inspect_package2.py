#!/usr/bin/env python3
"""Inspect candidate_c_r3_final package."""
import json
from pathlib import Path

AUDIT = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/audit_packages/candidate_c_r3_final")
print("AUDIT:", AUDIT, AUDIT.exists())

print("\n=== ARCHIVE CONTENTS ===")
arch = AUDIT / "archive"
if arch.exists():
    for p in sorted(arch.iterdir()):
        print(" ", p.name, p.stat().st_size if p.is_file() else "<dir>")
else:
    print("  (no archive dir)")

print("\n=== FINAL DECISION FULL ===")
fd = AUDIT / "analysis" / "final_decision.json"
if fd.exists():
    print(fd.read_text())

print("\n=== AGGREGATED SUMMARY (keys + key fields) ===")
asum = AUDIT / "analysis" / "aggregated_summary.json"
if asum.exists():
    data = json.loads(asum.read_text())
    print("keys:", list(data.keys()))
    for k in ("total_violations", "violations_by_family", "zero_violations", "n_iterations_total", "windows_covered"):
        if k in data:
            print(f"{k}: {data[k]}")
    if "correctness_summary" in data:
        print("correctness_summary:", data["correctness_summary"])
    if "tightness_summary" in data and isinstance(data["tightness_summary"], dict):
        print("tightness_summary keys:", list(data["tightness_summary"].keys())[:12])

print("\n=== README HEAD ===")
rd = AUDIT / "README.md"
if rd.exists():
    print(rd.read_text()[:800])

print("\n=== RAW NPZ REFS ===")
for w in ["5000", "15000", "29970"]:
    wdir = AUDIT / "raw" / w
    if wdir.is_dir():
        for f in wdir.iterdir():
            if f.name.endswith(".npz"):
                print(f"  {w}/{f.name}  {f.stat().st_size}")
