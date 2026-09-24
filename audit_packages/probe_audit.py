#!/usr/bin/env python3
"""Probe the state of the audit package build."""
from pathlib import Path

home = Path.home()
repo = home / "3dgs-renderer-benchmark"
print("home:", home)
print("repo:", repo, repo.exists())

if repo.exists():
    for name in ["candidate_r3_full", "candidate_r3_final", "candidate_c_r3_final"]:
        audit = repo / "audit_packages" / name
        print(f"\n=== {name} ===")
        print("exists:", audit.exists())
        if audit.exists():
            for p in sorted(audit.rglob("*")):
                if p.is_file():
                    print(f"  F {p.relative_to(audit)}  {p.stat().st_size} bytes")
                else:
                    print(f"  D {p.relative_to(audit)}/")
