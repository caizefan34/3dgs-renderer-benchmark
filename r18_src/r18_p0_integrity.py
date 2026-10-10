#!/usr/bin/env python3
"""R18-P0: integrity audit of R17 results + frozen binary identity.

- Verifies R17 AUDITED deliverables parse and their recorded sha256 still matches
  on disk (read-only w.r.t. the original result files).
- Verifies the two frozen renderer binaries (std csrc.so, HiGS csrc.so) against
  the handoff identity hashes.
- Finds log-hash inconsistencies in the R17 MANIFEST_SHA256.json and writes a
  CORRECTED manifest (R18_MANIFEST_CORRECTIONS.json) that fixes ONLY the log
  entries whose recorded hash/size disagrees with the actual on-disk log files.
  The original result files and their hashes are never altered.

Deliverable: R18_P0_INTEGRITY_AUDIT.json + R18_MANIFEST_CORRECTIONS.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r18_common import EXPECTED, OUT, PROJECT, sha256_file, write_json

R17_AUDITED = Path("/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_PARAMETER_VALIDITY_R17_MX_AUDITED")
MANIFEST = R17_AUDITED / "MANIFEST_SHA256.json"


def sha_and_size(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest(), path.stat().st_size


def main():
    audit = {"deliverable": "R18_P0_INTEGRITY_AUDIT.json", "r17_dir": str(R17_AUDITED)}

    # 1) Binary identity ----------------------------------------------------
    binaries = {}
    for label, rel, exp in (
        ("std_csrc", "gsplat/csrc.so", EXPECTED["std_csrc_sha256"]),
        ("higs_csrc", "gsplat/experimental/render/kernels/csrc.so", EXPECTED["higs_csrc_sha256"]),
    ):
        p = PROJECT / rel
        actual = sha256_file(p)
        binaries[label] = {
            "path": str(p), "exists": p.is_file(), "actual_sha256": actual,
            "expected_sha256": exp, "match": actual == exp,
        }
    audit["binaries"] = binaries
    audit["binaries_all_match"] = all(b["match"] for b in binaries.values())

    # 2) Checkpoint identity ------------------------------------------------
    ckpt = EXPECTED["ckpt_path"]
    audit["checkpoint"] = {
        "path": ckpt, "actual_sha256": sha256_file(ckpt),
        "expected_sha256": EXPECTED["ckpt_sha256"],
    }
    audit["checkpoint"]["match"] = audit["checkpoint"]["actual_sha256"] == EXPECTED["ckpt_sha256"]

    # 3) R17 deliverable hashes (read-only verification) --------------------
    man = json.loads(MANIFEST.read_text())
    deliv_status = {}
    mismatched = []
    for rel, meta in man["files"].items():
        p = R17_AUDITED / rel
        if not p.is_file():
            deliv_status[rel] = {"status": "MISSING_FILE", "manifest": meta}
            mismatched.append(rel)
            continue
        actual_sha, actual_size = sha_and_size(p)
        ok = (actual_sha == meta["sha256"]) and (actual_size == meta["bytes"])
        deliv_status[rel] = {
            "status": "OK" if ok else "HASH_OR_SIZE_MISMATCH",
            "manifest_sha256": meta["sha256"], "actual_sha256": actual_sha,
            "manifest_bytes": meta["bytes"], "actual_bytes": actual_size,
        }
        if not ok:
            mismatched.append(rel)
    audit["r17_deliverables"] = deliv_status
    audit["r17_deliverables_all_ok"] = len(mismatched) == 0
    audit["r17_deliverable_mismatches"] = mismatched

    # 4) Manifest log-hash corrections (fix ONLY log entries) ---------------
    log_fixes = {}
    for rel in man["files"]:
        if not rel.startswith("logs/"):
            continue
        p = R17_AUDITED / rel
        meta = man["files"][rel]
        if not p.is_file():
            log_fixes[rel] = {"action": "FILE_MISSING_KEEP_MANIFEST", "manifest": meta}
            continue
        actual_sha, actual_size = sha_and_size(p)
        if actual_sha != meta["sha256"] or actual_size != meta["bytes"]:
            log_fixes[rel] = {
                "action": "CORRECT_HASH_TO_ACTUAL",
                "manifest_sha256": meta["sha256"], "correct_sha256": actual_sha,
                "manifest_bytes": meta["bytes"], "correct_bytes": actual_size,
            }
    corrections = {
        "deliverable": "R18_MANIFEST_CORRECTIONS.json",
        "source_manifest": str(MANIFEST),
        "policy": ("Fix ONLY logs/* entries whose recorded sha256/bytes disagree with the "
                   "actual on-disk log files. Original result files and their recorded "
                   "hashes are never altered. The manifest is regenerated from the true "
                   "on-disk bytes."),
        "n_log_entries": sum(1 for r in man["files"] if r.startswith("logs/")),
        "n_log_fixes": len([k for k, v in log_fixes.items() if v["action"] == "CORRECT_HASH_TO_ACTUAL"]),
        "log_fixes": log_fixes,
    }

    # 5) Write a fully corrected manifest (copy of original with log fixes applied)
    corrected = json.loads(json.dumps(man))  # deep copy
    corrected["_corrected_by"] = "R18-P0 integrity audit"
    corrected["_correction_note"] = ("logs/* hashes refreshed to match on-disk bytes; "
                                     "all non-log entries unchanged from R17 manifest")
    for rel, fix in log_fixes.items():
        if fix["action"] == "CORRECT_HASH_TO_ACTUAL":
            corrected["files"][rel]["sha256"] = fix["correct_sha256"]
            corrected["files"][rel]["bytes"] = fix["correct_bytes"]
    corrections["corrected_manifest_path"] = str(OUT / "R18_MANIFEST_CORRECTIONS.json")

    write_json(OUT / "R18_P0_INTEGRITY_AUDIT.json", audit)
    write_json(OUT / "R18_MANIFEST_CORRECTIONS.json", corrections)
    # Also write the corrected manifest next to the corrections for downstream use.
    write_json(OUT / "R18_MANIFEST_CORRECTED.json", corrected)

    print(json.dumps({
        "binaries_all_match": audit["binaries_all_match"],
        "checkpoint_match": audit["checkpoint"]["match"],
        "r17_deliverables_all_ok": audit["r17_deliverables_all_ok"],
        "r17_deliverable_mismatches": mismatched,
        "n_log_fixes": corrections["n_log_fixes"],
        "log_fixes": list(log_fixes.keys()),
    }, indent=2))
    print("P0 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
