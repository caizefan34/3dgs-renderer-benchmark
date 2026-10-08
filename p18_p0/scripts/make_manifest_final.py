#!/usr/bin/env python3
import hashlib, json, os, time
from pathlib import Path
ROOT = Path(r"C:\Users\36570\3dgs-renderer-benchmark\p18_p0")
ART = ROOT / "artifacts"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def entry(rel):
    p = ROOT / rel
    if not p.exists():
        return None
    return {"path": rel.replace("\\", "/"), "size_bytes": p.stat().st_size, "sha256": sha256(p)}

FINAL = [
    "artifacts/P18_FINAL_CLOSEOUT.md",
    "artifacts/P18_EVIDENCE_CHAIN.json",
    "artifacts/P18_VERDICT.json",
]
R1 = [
    "artifacts/r1/P18_R1_SASS_AUDIT.json",
    "artifacts/r1/P18_R1_EVAL_MICROBENCH.json",
    "artifacts/r1/P18_R1_TRACE_MANIFEST.json",
    "artifacts/r1/P18_R1_REPORT.md",
    "artifacts/r1/P18_R1_VERDICT.json",
    "artifacts/r1/P18_R1_NEUTRALITY.json",
    "artifacts/r1/traces/P18_R1_TRACE_room.npz",
    "artifacts/r1/traces/P18_R1_TRACE_bicycle.npz",
    "artifacts/r1/cuda/p18_r1_eval.cu",
    "artifacts/r1/logs/sass_audit_r3.log",
    "artifacts/r1/logs/eval_run_r2.log",
]
SUPERSEDED = [
    "artifacts/P18_EVAL_MICROBENCH.json",
    "artifacts/P18_REFERENCE_VALIDATION.json",
    "artifacts/P18_TRACE_MANIFEST.json",
    "artifacts/traces/P18_TRACE_room.npz",
    "artifacts/traces/P18_TRACE_bicycle.npz",
    "artifacts/P18_REPORT.md",
    "artifacts/P18_VARIANT_TABLE.json",
    "artifacts/MANIFEST_P18.json",
]

def section(paths):
    out = []
    for rel in paths:
        e = entry(rel)
        if e:
            out.append(e)
    return out

manifest = {
    "phase": "P18-P0",
    "artifact": "MANIFEST_P18_FINAL",
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "verdict": "P18_P0_HARD_DROP",
    "closed": "FQ-GS permanently closed; NO P18-P1",
    "root": str(ROOT).replace("\\", "/"),
    "final_closeout_artifacts": section(FINAL),
    "valid_r1_evidence": section(R1),
    "superseded_invalid_pre_r1": [
        {**e, "status": "INVALID_SUPERSEDED_BY_R1"} for e in section(SUPERSEDED)
    ],
    "frozen_baseline_hashes": {
        "trainer": "675b026711b1938b28949c261c4f8e9e04d4a0a2846e2226c11d086b7999d28f50c5",
        "c0_ext": "d26cccf45868386e214650958838c6b6d654f1a55e6003c6a8007409554e637e",
        "core": "935ea6c099355e824d62b2a38b3d57580e578c140e47c48f533a0c6b1c880c7e",
        "fused_ssim": "dd6a4c79a0735d4c43a13b289d08e234270528b29a2b207827d0f0728e250ebb",
        "ckpt_room": "17c1f590110a803f9e594d5671179d4e9f19c4c822c9a57638490219062a0e80",
        "ckpt_bicycle": "0c36f5c7f42997174283d4e378b239875412509f23865e4a455605a761702995",
    },
    "r1_sass_audit_artifact_hashes": {
        "P18_R1_SASS_AUDIT.json": "17bdc4940d9d59af25c2c6237c0448210eff9ae2c131f453f579494fa3140bbf",
        "P18_R1_EVAL_MICROBENCH.json": "1fd16f5c76b69b601bd0e7a4751cc523592092480c85e93cbabd2a0a5e986419",
        "P18_R1_TRACE_MANIFEST.json": "d9d1a604b251c4cc7977aa3742fb52d6931f62f286265cae201ee6cfd7bf08cc",
    },
    "note": "MANIFEST_P18_FINAL.json is listed under itself only via the final section; the final manifest is the authoritative sha256 record. Superseded pre-R1 artifacts are retained as historical record and marked INVALID_SUPERSEDED_BY_R1.",
}
# add self
out = ART / "MANIFEST_P18_FINAL.json"
with open(out, "w") as f:
    json.dump(manifest, f, indent=2, allow_nan=False)

print("wrote", out)
for grp in ("final_closeout_artifacts", "valid_r1_evidence", "superseded_invalid_pre_r1"):
    print(f"\n=== {grp} ===")
    for e in manifest[grp]:
        tag = " [INVALID_SUPERSEDED_BY_R1]" if grp == "superseded_invalid_pre_r1" else ""
        print(f"  {e['sha256'][:16]}  {e['size_bytes']:>11}  {e['path']}{tag}")
