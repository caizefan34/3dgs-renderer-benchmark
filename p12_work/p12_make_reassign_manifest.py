#!/usr/bin/env python3
"""P12-B reassignment manifest v2: the GPU-5 pairs (flowers, stump) were
blocked ~45+ min by a persistent foreign co-tenant burst (host PIDs cycling
every few minutes, 0.5-10.7 GB). The clean-gate correctly refused to start
them. Once every other worker is idle, the scheduler is stopped and these
two pairs are reassigned to a verified-clean physical GPU (7, freed at
23:59). Arm order within each pair is UNCHANGED (preregistered alternation);
only the physical GPU changes, which is not part of the alternation.
All completed runs are skipped via EPIC05_SUCCESS (idempotent relaunch).
"""
import json
import subprocess
import time

SRC = "/root/epic05_p12/manifests/p12b_manifest.json"
DST = "/root/epic05_p12/manifests/p12b_manifest_v2_gpu7.json"
NEW_GPU = 7
PAIRS = ("p12b_pair_flowers", "p12b_pair_stump")

m = json.load(open(SRC))
n_moved = 0
for g in m["groups"]:
    if g["group_id"] in PAIRS:
        g["gpu"] = NEW_GPU
        n_moved += 1
assert n_moved == 2, n_moved
m["reassignment_note"] = {
    "reason": "persistent foreign co-tenant burst on GPU 5 (2026-09-26 23:37 .. "
              "00:30+ UTC+8; host PIDs 1426760/1618508/1649708/1740089/1796441/"
              "1918385/2093962 cycling every few minutes, 492 MB - 10.7 GB); "
              "clean-gate refused to start; pairs reassigned to GPU 7",
    "arm_order_unchanged": True,
    "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
m["preregistered_at_utc_v2"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
with open(DST, "w") as f:
    json.dump(m, f, indent=2)
print("WROTE", DST)
for g in m["groups"]:
    print(f"  GPU{g['gpu']} {g['group_id']}")
# verify GPU 7 is clean right now
out = subprocess.run(["nvidia-smi", "-i", "7", "--query-compute-apps=pid",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
print("GPU 7 compute apps now:", out or "NONE (clean)")
