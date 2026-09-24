import json

# UNDO the earlier (incorrect) contamination repair: forensics proved the
# CURRENT artifacts of b1a/bicycle and c0/counter are CLEAN second attempts
# whose outputs overwrote the contaminated first attempts:
#   b1a/bicycle  clean orphan 20:05:49-20:33:22 gpu0 (results mtime 20:33:22)
#   c0/counter   clean relaunch 20:39:16-20:57:40 gpu0 (results mtime 20:57:40)
UNDO = [
    ("B1A_ACCUTILE", "bicycle"),
    ("C0_V3_FINAL30K", "counter"),
]

for cand, scene in UNDO:
    fs_p = f"artifacts/final-30k/{cand}/{scene}/final_status.json"
    fs = json.load(open(fs_p, encoding="utf-8"))
    fs["contaminated"] = False
    fs.pop("timing_grade_effective", None)
    fs.pop("contamination_evidence", None)
    fs["clean_attempt_note"] = ("current artifacts are the clean second attempt; "
                                "the contaminated first attempt's outputs were "
                                "overwritten (see provenance.contaminated_attempts)")
    with open(fs_p, "w", encoding="utf-8") as f:
        json.dump(fs, f, indent=2)
    pv_p = f"artifacts/final-30k/{cand}/{scene}/provenance.json"
    prov = json.load(open(pv_p, encoding="utf-8"))
    prov["contaminated"] = False
    prov.pop("contamination_evidence", None)
    prov["contaminated_attempts"] = [
        {"arm": "b1a" if cand == "B1A_ACCUTILE" else "c0", "scene": scene, "attempt": 1,
         "note": "contaminated first attempt; outputs overwritten by clean second attempt "
                 "(mx scheduler.log 2026-09-23)"}
    ]
    with open(pv_p, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    print(f"undone: {cand}/{scene} -> contaminated=False (clean attempt2 is current)")
