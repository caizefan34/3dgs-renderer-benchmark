#!/usr/bin/env python3
"""Generate RUN_STATUS.json for R19."""
import json, os, glob, hashlib, time

W = "/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_FULLTRAIN_R19_MX"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def has_results(d):
    return os.path.isfile(os.path.join(d, "train_results.json"))

status = {
    "deliverable": "TRAINABLE_HIGS_FULLTRAIN_R19_MX",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "phases": {
        "P0_trainer_contract": "PASS",
        "P1_correctness_smoke": "PASS",
        "P2_5K_pilot": "PARTIAL",
        "P3_30K_convergence": "PARTIAL_NOT_RUN",
        "P4_time_to_quality": "INCONCLUSIVE",
        "P5_statistics_gates": "PASS",
        "P6_verdict": "COMPLETE",
    },
    "scenes_completed": {},
    "raw_runs": {},
}

for d in sorted(glob.glob(os.path.join(W, "raw", "p2*_*/"))):
    name = os.path.basename(d.rstrip("/"))
    done = has_results(d)
    status["raw_runs"][name] = "DONE" if done else "RUNNING_OR_PARTIAL"
    if done:
        try:
            r = json.load(open(os.path.join(d, "train_results.json")))
            mode = "absgrad" if name.startswith("p2_") else "signed"
            scene = name.split("_")[-1]
            be = "std_ll" if "std_ll" in name else "higs"
            status["scenes_completed"].setdefault(f"{mode}/{scene}", {})[be] = {
                "final_N": r.get("final_N"),
                "final_psnr": round(r["final_eval"]["psnr"], 3) if r.get("final_eval") else None,
                "total_time_s": round(r.get("total_time_s", 0), 1),
            }
        except Exception:
            pass

# deliverable presence
deliverables = [
    "R19_ENVIRONMENT_IDENTITY.json", "R19_TRAINER_CONTRACT.md",
    "R19_TRAINING_CORRECTNESS.json", "R19_5K_PILOT.csv", "R19_30K_TRAINING.csv",
    "R19_QUALITY_METRICS.csv", "R19_TIME_TO_QUALITY.csv", "R19_COMPONENT_TIMINGS.csv",
    "R19_PAIRED_STATISTICS.json", "R19_SCENE_ABLATION.md", "R19_FINAL_REPORT.md",
    "R19_FINAL_VERDICT.json", "RUN_STATUS.json", "MANIFEST_SHA256.json",
]
status["deliverables_present"] = {d: os.path.isfile(os.path.join(W, d)) for d in deliverables}

with open(os.path.join(W, "RUN_STATUS.json"), "w") as f:
    json.dump(status, f, indent=2)
print(json.dumps(status, indent=2))
