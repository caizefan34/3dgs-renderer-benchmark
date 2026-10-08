#!/usr/bin/env python3
"""Build MANIFEST_M1_DELIVERABLES.json: sha256 of every local deliverable."""
import hashlib
import json
import os

SKIP_BUNDLES = True  # bundles/snapshots are ~1.7 GB each; hashed on remote, listed in bundles_manifest_*.json


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def walk(root, prefix=""):
    out = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        rel = f"{prefix}{name}"
        if os.path.isdir(p):
            if name == "__pycache__":
                continue
            out.extend(walk(p, rel + "/"))
        else:
            out.append((rel, p))
    return out


def main():
    entries = []
    for rel, p in walk("remote_outputs") + walk("remote_logs") + walk("remote_scripts"):
        entries.append({"path": rel, "bytes": os.path.getsize(p), "sha256": sha256_file(p)})
    for name in ("GRC3D_M1_VERDICT.json", "GRC3D_M1_REPORT.md",
                 "ENVIRONMENT_AND_OCCUPANCY_M1.json", "DSH_EXECUTION.md",
                 "MANIFEST.json", "local_logs_m0_test_core.txt", "make_verdict.py"):
        if os.path.exists(name):
            entries.append({"path": name, "bytes": os.path.getsize(name),
                            "sha256": sha256_file(name)})

    man = {
        "deliverable": "GRC3D M0/M1 execution package",
        "date": "2026-09-27",
        "verdict_file": "GRC3D_M1_VERDICT.json",
        "report_file": "GRC3D_M1_REPORT.md",
        "remote_workdir": "/root/epic05_grc3d (mirrored here under remote_outputs/ remote_logs/ remote_scripts/)",
        "large_inputs_not_copied": {
            "note": "snapshot/bundle .pt files (~1.7 GB each, 48 GB total) stay on EPIC-05; "
                    "their sha256 identity is recorded in remote_outputs/bundles_manifest_*.json",
            "remote_paths": ["/root/epic05_grc3d/inputs/snapshots/", "/root/epic05_grc3d/inputs/bundles/"],
        },
        "count": len(entries),
        "files": entries,
    }
    with open("MANIFEST_M1_DELIVERABLES.json", "w") as f:
        json.dump(man, f, indent=2)
    print(f"[manifest] {len(entries)} files")


if __name__ == "__main__":
    main()
