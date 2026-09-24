#!/usr/bin/env python3
"""Final verification: sections A (run completeness) + B (regeneration) of the
publication close-out checklist. Prints PASS/FAIL per item; exit 0 iff all pass.
"""
import json
import os
import subprocess
import sys

PUB = "/mnt/storage_pool/liaoyuanjun/pub_runs"
F30K = "/mnt/storage_pool/liaoyuanjun/final30k_runs"
PHASE = "/mnt/storage_pool/liaoyuanjun/pubphase"
A = f"{PHASE}/aggregates"
FT = f"{PHASE}/figtables"

fails = []
def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        fails.append(name)

# ---- A1: scheduler final state
st = json.load(open(f"{PUB}/pub_scheduler_state.json"))
done_rids = [d["rid"] for d in st["done"]]
check("A1a scheduler DONE == 91", len(st["done"]) == 91, f"done={len(st['done'])}")
check("A1b failed == 0", len(st["failed"]) == 0, f"failed={len(st['failed'])}")
check("A1c ACTIVE empty", not st.get("active"), str(st.get("active")))
contam = [d["rid"] for d in st["done"] if d.get("contaminated")]
check("A1d no contaminated runs in final done list", not contam, str(contam))
# contaminated attempts preserved as renamed dirs
for rid in ("b0_counter", "b1as43_train", "c0e01_room"):
    p = f"{PUB}/{rid}_contaminated_attempt1/results.json"
    check(f"A1e forensics kept: {rid}_contaminated_attempt1", os.path.exists(p))

# ---- A2: queue-vs-matrix
r = subprocess.run(["python3", f"{PHASE}/queue_audit.py"], capture_output=True, text=True)
out = r.stdout
check("A2 queue audit: no missing, no extra",
      "MISSING from coverage: none" in out and "EXTRA (not in matrix): none" in out,
      out.split("\n")[0] if out else r.stderr[-200:])

# ---- A3: local-facing completeness of canonical run dirs
ALL13 = ["bicycle", "bonsai", "counter", "drjohnson", "flowers", "garden",
         "kitchen", "playroom", "room", "stump", "train", "treehill", "truck"]
missing = []
for s in ALL13:
    for a in ("a0", "a1", "a2", "b1", "b0"):
        if not os.path.exists(f"{PUB}/{a}_{s}/results.json"):
            missing.append(f"{a}_{s}")
for s in ALL13:
    for a in ("b1a", "c0"):
        if not os.path.exists(f"{F30K}/{a}_{s}/results.json"):
            missing.append(f"{a}_{s}")
for s in ("room", "bicycle", "garden"):
    for a in ("c0e01", "b1ae03"):
        if not os.path.exists(f"{PUB}/{a}_{s}/results.json"):
            missing.append(f"{a}_{s}")
for s in ("drjohnson", "train", "bicycle", "room"):
    for a in ("b1as43", "b1as44", "c0s43", "c0s44"):
        if not os.path.exists(f"{PUB}/{a}_{s}/results.json"):
            missing.append(f"{a}_{s}")
for a in ("b1as43", "b1s43", "b1as44", "b1s44"):
    if not os.path.exists(f"{PUB}/{a}_garden/results.json"):
        missing.append(f"{a}_garden")
check("A3 canonical run dirs complete (117 runs)", not missing, str(missing))

# ---- B1: aggregate regenerates + self-check
r = subprocess.run(["python3", f"{PHASE}/aggregate_publication.py"], capture_output=True, text=True)
check("B1 aggregate_publication.py rc=0", r.returncode == 0)
h = json.load(open(f"{A}/headline_c0_vs_b1a_f30k.json"))["aggregate"]
check("B3 headline unchanged (1.0685 / +0.070)",
      abs(h["speedup_geomean"] - 1.0685) < 0.0005 and abs(h["mean_d_psnr"] - 0.070) < 0.005,
      f"{h['speedup_geomean']:.4f} / {h['mean_d_psnr']:+.4f}")

# ---- B2: figtables regenerate
env = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python"
r = subprocess.run([env, f"{PHASE}/make_figtables.py"], capture_output=True, text=True)
check("B2 make_figtables.py rc=0", r.returncode == 0)
pngs = [f for f in os.listdir(FT) if f.endswith(".png")]
tbls = [f for f in os.listdir(FT) if f.endswith(".md")]
check("B2b 6 figures", len(pngs) == 6, str(pngs))
check("B2c 6 tables", len(tbls) == 6, str(tbls))
for f in pngs:
    sz = os.path.getsize(f"{FT}/{f}")
    check(f"B5 figure non-empty: {f}", sz > 20000, f"{sz} bytes")

# ---- B4: p6 analysis clean
r = subprocess.run(["python3", f"{PHASE}/p6_analysis.py"], capture_output=True, text=True)
check("B4 p6_analysis.py rc=0 (contamination guard passes)", r.returncode == 0,
      r.stdout.split("\n")[0] if r.returncode != 0 else "")
p6 = json.load(open(f"{PUB}/p6_results.json"))
geos = p6["per_seed_geomeans"]
check("B4b P6 geomeans 1.056-1.074, all 4/4",
      all(abs(geos[k] - v) < 0.001 for k, v in (("42", 1.0705), ("43", 1.0735), ("44", 1.0561))),
      str(geos))
check("B4c drjohnson reproducible", p6["drjohnson_persistent"] is True)

print()
if fails:
    print(f"RESULT: {len(fails)} FAILURES: {fails}")
    sys.exit(1)
print("RESULT: ALL A+B CHECKS PASS")
