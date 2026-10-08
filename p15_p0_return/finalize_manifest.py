"""Packaging step (local, offline): finalize MANIFEST_P15_P0.json.

The pipeline's p15_p0_report.py emits a manifest mid-run; by construction
that emission (a) hashes logs/report.log before its own last line is
written, and (b) cannot include the pipeline script set (remote tree).
This packaging step rebuilds the same manifest AFTER the run from the
final bytes of every delivered file, with the manifest itself excluded.

It changes no measurement: PARETO/VERDICT/REPORT/CONTAMINATION, the six
summaries, the NPZs and all logs are hashed exactly as delivered.
"""
import glob
import hashlib
import json
import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "artifacts", "MANIFEST_P15_P0.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    man = {
        "phase": "P15-P0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": json.load(open(os.path.join(
            ROOT, "artifacts", "P15_P0_VERDICT.json")))["verdict"],
        "root": "p15_p0_return (local mirror of /root/epic05_p15/p0)",
        "note": ("packaging-time finalization: supersedes the mid-run "
                 "emission of p15_p0_report.py, whose logs/report.log "
                 "entry was hashed before the report's own final line "
                 "was written and which excluded the scripts/ tree; every "
                 "entry below was recomputed from the delivered bytes "
                 "after the run finished; the manifest excludes itself"),
        "artifacts": {}, "reused_certified": {},
        "p14_inputs_from_identity": {}, "packaging_tools": {},
    }
    targets = []
    for pat in ("exports/*.npz", "summaries/*.json", "artifacts/*",
                "logs/*.log", "scripts/*.py", "scripts/*.sh"):
        targets.extend(glob.glob(os.path.join(ROOT, pat)))
    n_verified = 0
    for p in sorted(set(targets)):
        rel = os.path.relpath(p, ROOT).replace(os.sep, "/")
        if os.path.abspath(p) == os.path.abspath(OUT):
            continue
        h = sha256_file(p)
        man["artifacts"][rel] = {"sha256": h,
                                 "bytes": os.path.getsize(p)}
        assert sha256_file(p) == h
        n_verified += 1
    tool = os.path.abspath(__file__)
    man["packaging_tools"][os.path.basename(tool)] = {
        "sha256": sha256_file(tool), "bytes": os.path.getsize(tool)}

    ident_path = os.path.join(ROOT, "artifacts", "P15_SOURCE_IDENTITY.json")
    if os.path.exists(ident_path):
        ident = json.load(open(ident_path))
        for name, rec in (ident.get("fatal_checks") or {}).items():
            if name.startswith("reuse_") or name.startswith("input_"):
                man["reused_certified"][name] = {
                    "path": rec.get("path"), "sha256": rec.get("sha256"),
                    "match": rec.get("match")}
            if name.startswith("ckpt_"):
                man["p14_inputs_from_identity"][name] = {
                    "path": rec.get("path"), "sha256": rec.get("sha256"),
                    "match": rec.get("match")}
    with open(OUT, "w") as f:
        json.dump(man, f, indent=2)
    print(f"[finalize] wrote {OUT} ({n_verified} entries verified)")


if __name__ == "__main__":
    main()
