"""Write pub_queue.json (the file-driven queue) with the current wave, then
restart the scheduler into file-queue mode."""
import json
import os
import signal
import subprocess
import time

QF = "/mnt/storage_pool/liaoyuanjun/pubphase/pub_queue.json"
GATE = ["room", "bicycle", "garden"]
ALL13 = ["bicycle", "bonsai", "counter", "drjohnson", "flowers", "garden",
         "kitchen", "playroom", "room", "stump", "train", "treehill", "truck"]

q = []
# W1: B1 gate
for s in GATE:
    q.append({"key": "b1", "arm": "b1", "scene": s, "extra": []})
# W1b: B0 gate
for s in GATE:
    q.append({"key": "b0", "arm": "b0", "scene": s, "extra": []})
# W2: Stage A
for s in GATE:
    for k in ("a0", "a1", "a2"):
        q.append({"key": k, "arm": k, "scene": s, "extra": []})
# W3: P4 corners
for s in GATE:
    q.append({"key": "c0e01", "arm": "c0", "scene": s, "extra": ["--eps2d", "0.1"]})
for s in GATE:
    q.append({"key": "b1ae03", "arm": "b1a", "scene": s, "extra": ["--eps2d", "0.3"]})

with open(QF + ".tmp", "w") as f:
    json.dump(q, f, indent=1)
os.replace(QF + ".tmp", QF)
print(f"WROTE {QF} ({len(q)} entries)")

# restart scheduler
r = subprocess.run(["pgrep", "-f", "publication_scheduler.py"], capture_output=True, text=True)
pids = [int(p) for p in r.stdout.split()]
for p in pids:
    try:
        os.kill(p, signal.SIGTERM)
    except ProcessLookupError:
        pass
if pids:
    time.sleep(2)
print("killed:", pids)
