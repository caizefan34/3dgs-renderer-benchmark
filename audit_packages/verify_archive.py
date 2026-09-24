#!/usr/bin/env python3
"""Verify the audit archive can be opened and list its members."""
import tarfile
from pathlib import Path

ARCHIVE = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/audit_packages/candidate_c_r3_final/archive/candidate_c_r3_final_audit.tar.gz")
print(f"Archive: {ARCHIVE}")
print(f"Exists: {ARCHIVE.exists()}")
print(f"Size: {ARCHIVE.stat().st_size if ARCHIVE.exists() else 0}")

print("\n=== Archive members (first 70) ===")
with tarfile.open(ARCHIVE, "r:gz") as tar:
    names = tar.getnames()
    print(f"Total members: {len(names)}")
    for n in sorted(names)[:70]:
        print(f"  {n}")

print("\n=== SHA256SUMS.txt verification ===")
import hashlib
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

sha_file = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/audit_packages/candidate_c_r3_final/SHA256SUMS.txt")
if sha_file.exists():
    lines = sha_file.read_text().strip().split("\n")
    print(f"SHA256SUMS.txt: {len(lines)} entries")
    # show first 5 and last 5
    for line in lines[:5]:
        print(f"  {line}")
    print("  ...")
    for line in lines[-5:]:
        print(f"  {line}")

print("\n=== manifest.csv head ===")
csv_file = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/audit_packages/candidate_c_r3_final/manifest.csv")
if csv_file.exists():
    lines = csv_file.read_text().strip().split("\n")
    print(f"manifest.csv: {len(lines)} rows")
    for line in lines[:5]:
        print(f"  {line}")
