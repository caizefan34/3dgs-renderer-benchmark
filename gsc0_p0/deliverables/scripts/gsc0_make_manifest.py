#!/usr/bin/env python3
"""Generate GSC0 MANIFEST.sha256 over all deliverables + scripts + raw snapshots."""
import hashlib
import json
import os

OUT = "/root/gsc0_out"
RUN = "/root/gsc0_run"
SCRIPTS = ["/root/gsc0_trace.py", "/root/gsc0_analysis.py", "/root/gsc0_report.py"]

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

lines = []
# 1) the 5 JSON/MD deliverables (not the manifest itself)
deliverables = ["GSC0_VERDICT.json", "GSC0_SUMMARY.json", "GSC0_REPORT.md",
                "GSC0_TRACE_INDEX.json", "GSC0_ERROR_ORACLE.json"]
for d in deliverables:
    p = os.path.join(OUT, d)
    assert os.path.exists(p), p
    lines.append((sha256_file(p), d))

# 2) scripts used for aggregation + harness
for sp in SCRIPTS:
    assert os.path.exists(sp), sp
    lines.append((sha256_file(sp), "scripts/" + os.path.basename(sp)))

# 3) raw snapshots (re-hash for a complete, verifiable manifest)
ti = json.load(open(os.path.join(OUT, "GSC0_TRACE_INDEX.json")))
for scene, entry in ti["scenes"].items():
    for rec in entry.get("trace_records", []):
        rel = os.path.join("raw", scene, "traces", rec["file"])
        p = os.path.join(RUN, scene, "traces", rec["file"])
        assert os.path.exists(p), p
        # verify against trace-index hash
        h = sha256_file(p)
        assert h == rec["sha256"], f"hash mismatch {p}: {h} != {rec['sha256']}"
        lines.append((h, rel))

lines.sort(key=lambda x: x[1])
with open(os.path.join(OUT, "MANIFEST.sha256"), "w", newline="\n") as f:
    for h, name in lines:
        f.write(f"{h}  {name}\n")
print(f"MANIFEST.sha256 written: {len(lines)} entries")
for h, name in lines:
    print(f"  {h[:16]}.. {name}")