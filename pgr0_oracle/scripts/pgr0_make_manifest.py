#!/usr/bin/env python3
"""Emit MANIFEST_PGR0.json with SHA256 of every PGR0 evidence file + source."""
import os, sys, json, hashlib, datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

# evidence artifacts + the frozen source they were derived from
EVIDENCE = [
    "artifacts/PGR0_VERDICT.json",
    "artifacts/PGR0_STATS.json",
    "artifacts/PGR0_GRADIENT_SIMILARITY.json",
    "artifacts/PGR0_REPORT.md",
    "artifacts/PGR0_PRIOR_ART_REFRESH.md",
    "artifacts/records/pgr0_records.jsonl",
    "artifacts/records/pgr0_records_fp32.jsonl",
    "scripts/pgr0_projection_ref.py",
    "scripts/pgr0_oracle.py",
    "scripts/pgr0_analyze.py",
    "config/pgr0_oracle.yaml",
    "schemas/pgr0_record.schema.json",
    "FAIL_CLOSED_GATES.md",
    "PGR0_SPEC.md",
    "PRIOR_ART_MATRIX.md",
    "PATCH_ALLOWLIST.md",
    "README.md",
    "DSH_TASK.md",
]
# frozen upstream source (read-only, hashed for provenance)
SOURCE = [
    "public_baselines/gsplat/gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu",
    "public_baselines/gsplat/gsplat/cuda/include/Utils.cuh",
]

def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")

files = []
for relp in EVIDENCE:
    p = os.path.join(ROOT, relp)
    if os.path.exists(p):
        files.append({"path": rel(p), "sha256": sha_file(p), "bytes": os.path.getsize(p)})
    else:
        files.append({"path": rel(p), "sha256": None, "bytes": 0, "missing": True})
source = []
for relp in SOURCE:
    p = os.path.join(REPO, relp)
    if os.path.exists(p):
        source.append({"path": rel(p), "sha256": sha_file(p), "bytes": os.path.getsize(p)})
    else:
        source.append({"path": rel(p), "sha256": None, "bytes": 0, "missing": True})

manifest = {
    "package": "PGR0_PROJECTION_GRADIENT_REDUNDANCY_ORACLE_PACK",
    "freeze_date": "2026-10-01",
    "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "cuda_used_in_p0": False,
    "files": files,
    "frozen_source_provenance": source,
}
out = os.path.join(ROOT, "artifacts", "MANIFEST_PGR0.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print("wrote", rel(out))
for e in files + source:
    print(("MISSING " if e.get("missing") else "        ") + e["path"])
