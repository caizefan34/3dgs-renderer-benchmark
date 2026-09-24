import json

# Repair local frozen-tree artifacts for runs whose contamination flags were
# lost to scheduler-state resets during the multi-incarnation recovery.
# Evidence (durable mx scheduler.log):
#   [19:22:35] CONTAMINATION c0_counter gpu=3 foreign_pids=[2137836]
#   [19:41:35] CONTAMINATION b1a_bicycle gpu=3 foreign_pids=[2225770]
# Both runs completed while sharing their GPU with a foreign root process.

REPAIRS = [
    ("B1A_ACCUTILE", "bicycle", "root pid 2225770 co-resident on gpu3 from 19:41:35 (mx scheduler.log)"),
    ("C0_V3_FINAL30K", "counter", "root pid 2137836 co-resident on gpu3 from 19:22:35 (mx scheduler.log)"),
]

for cand, scene, evidence in REPAIRS:
    p = f"artifacts/final-30k/{cand}/{scene}/final_status.json"
    fs = json.load(open(p, encoding="utf-8"))
    fs["contaminated"] = True
    fs["timing_grade_effective"] = "FUNCTIONAL_ONLY_CONTAMINATED"
    fs["contamination_evidence"] = evidence
    with open(p, "w", encoding="utf-8") as f:
        json.dump(fs, f, indent=2)
    pv = f"artifacts/final-30k/{cand}/{scene}/provenance.json"
    prov = json.load(open(pv, encoding="utf-8"))
    prov["contaminated"] = True
    prov["contamination_evidence"] = evidence
    with open(pv, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    print(f"repaired {cand}/{scene}: contaminated=True (FUNCTIONAL_ONLY)")
