#!/usr/bin/env python3
"""E1: FINAL-30K untouched check + headline byte-stability vs the locally pulled copy."""
import json
import os
import subprocess
import sys

F30K = "/mnt/storage_pool/liaoyuanjun/final30k_runs"
H = "/mnt/storage_pool/liaoyuanjun/pubphase/aggregates/headline_c0_vs_b1a_f30k.json"

# Publication-phase start = first line of the publication scheduler log.
# Invariant: NOTHING under final30k_runs/ is newer than that moment.
r = subprocess.run(["bash", "-c", "head -1 /mnt/storage_pool/liaoyuanjun/pub_runs/pub_scheduler.log"],
                   capture_output=True, text=True)
first = r.stdout.strip()
phase_start = first.split("]")[0].strip("[")  # e.g. 2026-09-24 00:41:12
print(f"publication phase start: {phase_start}")
r = subprocess.run(["bash", "-c",
                    f"find {F30K} -type f -newermt '{phase_start}' | head -3"],
                   capture_output=True, text=True)
newer = r.stdout.strip()
print(f"FINAL-30K files newer than phase start: {newer if newer else 'NONE'}")
ok_untouched = not newer
print(f"[{'PASS' if ok_untouched else 'FAIL'}] E1a FINAL-30K trees untouched since publication phase start")

# canonical run count: 26 (b1a/c0 x 13), excluding *_contaminated_attempt1 forensics dirs
n = sum(1 for d in os.listdir(F30K)
        if os.path.isdir(f"{F30K}/{d}") and not d.endswith("_contaminated_attempt1")
        and os.path.exists(f"{F30K}/{d}/results.json"))
print(f"canonical FINAL-30K runs with results.json: {n}")
print(f"[{'PASS' if n == 26 else 'FAIL'}] E1b 26 canonical FINAL-30K run dirs intact")

# regenerated headline values
h = json.load(open(H))["aggregate"]
vals = (round(h["speedup_geomean"], 4), round(h["mean_d_psnr"], 3))
print(f"regenerated headline: {vals}")
ok_head = vals == (1.0685, 0.07)
print(f"[{'PASS' if ok_head else 'FAIL'}] E1c headline 1.0685 / +0.070 reproduced")

sys.exit(0 if (ok_untouched and n == 26 and ok_head) else 1)
