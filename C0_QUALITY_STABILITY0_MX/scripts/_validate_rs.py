import json, sys
p = "/mnt/storage_pool/liaoyuanjun/C0_QUALITY_STABILITY0_MX/RUN_STATUS.json"
d = json.load(open(p))
print("PARSE_OK keys=", list(d.keys()))
print("evidence_base=", d["evidence_base"])
print("contam_all_clean=", d["contamination"]["all_clean"])
print("n_new_quality=", len(d["new_run_quality"]))
