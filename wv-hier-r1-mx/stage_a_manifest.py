#!/usr/bin/env python3
"""Regenerate Stage A manifest to include the corrections note + audit script."""
import hashlib, os, json
OUT = r"C:\Users\36570\3dgs-renderer-benchmark\wv-hier-r1-mx"
mpath = os.path.join(OUT, "SHA256_MANIFEST_STAGE_A.json")
m = json.load(open(mpath, encoding="utf-8"))

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""): h.update(c)
    return h.hexdigest()

# refresh ALL hashes from disk (files may have been edited since last manifest)
missing = []
for t in list(m["files"].keys()):
    p = os.path.join(OUT, t.replace("/", os.sep))
    if os.path.exists(p):
        m["files"][t] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
    else:
        missing.append(t)
# prune entries whose files no longer exist
for t in missing:
    del m["files"][t]

# add new deliverables
for t in (
    "STAGE_A_CORRECTIONS.md",
    "stage_b_revised/_audit_oracle.py",
    # Stage B deliverables
    "STAGE_B_REPORT.md",
    "STAGE_B_SPEC.md",
    "stage_b_min_runner.py",
    "stage_b_variance_check.py",
    "stage_b_n1_n2_audit.py",
    "stage_b_n1_n2_audit.csv",
    "stage_b_n1_n2_verdict.json",
    "variance_identity_check.csv",
    "stage_b_raw_results_TEMPLATE.csv",
    # Round-2 deliverables
    "stage_b_gam_robustness.py",
    "stage_b_n1_n2_gam_robustness.csv",
    "stage_b_n1_n2_gam_robustness.json",
    "_wv0_grad_stage_b.py",
    # Round-3 deliverables
    "stage_b_evidence_provenance.py",
    "stage_b_evidence_provenance.json",
    "stage_b_rho_provenance.csv",
    # Round-4 deliverables
    "stage_b_cost_model_sensitivity.py",
    "stage_b_cost_model_sensitivity.csv",
    "stage_b_cost_model_sensitivity.json",
    # Round-5 deliverables
    "stage_b_mask_decode.py",
    "stage_b_mask_decode.json",
    # Round-6 deliverables
    "STAGE_B_MIN_HANDOFF.md",
    # Round-8 deliverables
    "stage_b_quality_gate.py",
    "stage_b_quality_gate.csv",
    "stage_b_quality_gate.json",
    "apply_corrections.py",
    "fix_verdict_conflicts.py",
):
    p = os.path.join(OUT, t.replace("/", os.sep))
    if os.path.exists(p):
        m["files"][t] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
    else:
        missing.append(t)

m["file_count"] = len(m["files"])
m["note"] = ("SHA256 of WV-HIER-R1-MX Stage A + Stage B deliverables + analysis scripts. "
             "Excludes evidence/ data dir (covered by WV0 SHA256_MANIFEST.json) and this "
             "manifest itself. Updated 20261008: resolved Stage A verdict conflicts (kappa "
             "not count-identifiable; no kappa<=0.5 CUDA GO gate; N1+cost-model conditioning; "
             "N2 independent baseline) and added Stage B-min protocol artifacts. GPU gradient "
             "oracle NOT_EXECUTED (MX unreachable from this host).")
json.dump(m, open(mpath, "w", encoding="utf-8"), indent=2)

# self-verify all
ok = 0
for t, meta in m["files"].items():
    p = os.path.join(OUT, t.replace("/", os.sep))
    if os.path.exists(p) and sha256(p) == meta["sha256"]:
        ok += 1
    else:
        print("VERIFY FAIL/MISSING:", t)
print(f"manifest files: {len(m['files'])}  self-verify: {ok}/{len(m['files'])} MATCH")
if missing:
    print("WARNING missing (removed from disk):", missing)
