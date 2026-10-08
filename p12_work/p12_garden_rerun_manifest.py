#!/usr/bin/env python3
"""P12-B contamination response: preserve the contaminated garden pair attempt
(a1 contaminated by foreign PID 1426795, 2 watchdog samples, max 680 MB; a0
clean but the PAIR must stay back-to-back) and preregister an UNCHANGED
garden pair rerun: same arms, same order ([a1, a0] per the preregistered
alternation), same recipe, fresh back-to-back pair on a verified-clean GPU.
"""
import json
import os
import subprocess
import time

ROOT = "/root/epic05_p12/runs/p12b"
STAMP = "20260927T0119Z"
GPU = int(os.environ.get("RERUN_GPU", "2"))

# 1. preserve the contaminated attempt verbatim
for rid in ("p12b_a1f_s42_garden", "p12b_a0f_s42_garden"):
    src = os.path.join(ROOT, rid)
    dst = os.path.join(ROOT, f"{rid}.contaminated_{STAMP}")
    if os.path.exists(src) and not os.path.exists(dst):
        os.rename(src, dst)
        print(f"PRESERVED {src} -> {dst}")
    else:
        print(f"SKIP (missing or already preserved): {src}")

# 2. verify the target GPU is clean right now
out = subprocess.run(["nvidia-smi", "-i", str(GPU), "--query-compute-apps=pid",
                      "--format=csv,noheader"], capture_output=True,
                     text=True).stdout.strip()
print(f"GPU {GPU} compute apps now: {out or 'NONE (clean)'}")
if out:
    raise SystemExit(f"GPU {GPU} is not clean; set RERUN_GPU to a clean GPU")

# 3. write the UNCHANGED rerun manifest (same arms, same order, same recipe)
man = {
    "campaign_id": "P12B_F9_REVALIDATION_FUSED",
    "purpose": "CONTAMINATION RERUN: p12b_a1f_s42_garden was contaminated by "
               f"foreign PID 1426795 (2 samples, max 680 MB); the pair is "
               "rerun UNCHANGED as a fresh back-to-back pair. Arm order and "
               "recipe identical to the preregistered manifest.",
    "preregistered_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "contamination_event": {
        "run_id": "p12b_a1f_s42_garden",
        "gpu": 3,
        "foreign_pid": 1426795,
        "samples": 2,
        "max_mem_mb": 680,
        "preserved_as": [f"p12b_a1f_s42_garden.contaminated_{STAMP}",
                          f"p12b_a0f_s42_garden.contaminated_{STAMP}"],
        "policy": "contaminated attempts preserved verbatim; rerun unchanged; "
                  "never averaged with clean runs",
    },
    "groups": [{
        "group_id": "p12b_pair_garden_rerun",
        "gpu": GPU,
        "pair_arm_order": ["A1_FUSED", "A0_FUSED"],
        "jobs": [
            {"run_id": "p12b_a1f_s42_garden", "stage": "P12B_F9_REVALIDATION_FUSED",
             "arm": "a1", "arm_label": "A1_FUSED", "scene": "garden",
             "seed": 42, "eps2d": 0.3, "iterations": 30000, "loss": "fused"},
            {"run_id": "p12b_a0f_s42_garden", "stage": "P12B_F9_REVALIDATION_FUSED",
             "arm": "a0", "arm_label": "A0_FUSED", "scene": "garden",
             "seed": 42, "eps2d": 0.3, "iterations": 30000, "loss": "fused"},
        ],
    }],
}
out_path = "/root/epic05_p12/manifests/p12b_garden_rerun_manifest.json"
with open(out_path, "w") as f:
    json.dump(man, f, indent=2)
print("WROTE", out_path)
