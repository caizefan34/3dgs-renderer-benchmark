"""Write the next queue wave + repair state (scheduler is DEAD, no clobber risk):
  W-A  b0 gate x3 (fixed trainer, radii [1,N,1]) + c0e01_room clean re-run   (4)
  W-B  garden seed-43 pair: b1a + b1 (resolves the B1-gate garden question)  (2)
  W-C  P2 Stage B: a0/a1/a2 x 10 remaining scenes                           (30)
Then relaunch the scheduler.
"""
import json
import os
import time

QF = "/mnt/storage_pool/liaoyuanjun/pubphase/pub_queue.json"
STATE = "/mnt/storage_pool/liaoyuanjun/pub_runs/pub_scheduler_state.json"

GATE = ["room", "bicycle", "garden"]
ALL13 = ["bicycle", "bonsai", "counter", "drjohnson", "flowers", "garden",
         "kitchen", "playroom", "room", "stump", "train", "treehill", "truck"]
REST10 = [s for s in ALL13 if s not in GATE]

q = []
# keep earlier waves (idempotent: done rids are filtered)
for s in GATE:
    q.append({"key": "b1", "arm": "b1", "scene": s, "extra": []})
for s in GATE:
    q.append({"key": "b0", "arm": "b0", "scene": s, "extra": []})
for s in GATE:
    for k in ("a0", "a1", "a2"):
        q.append({"key": k, "arm": k, "scene": s, "extra": []})
for s in GATE:
    q.append({"key": "c0e01", "arm": "c0", "scene": s, "extra": ["--eps2d", "0.1"]})
for s in GATE:
    q.append({"key": "b1ae03", "arm": "b1a", "scene": s, "extra": ["--eps2d", "0.3"]})
# W-B: garden seed-43 pair (B1-gate garden resolution; doubles as P6-style evidence)
q.append({"key": "b1as43", "arm": "b1a", "scene": "garden", "extra": ["--seed", "43"]})
q.append({"key": "b1s43", "arm": "b1", "scene": "garden", "extra": ["--seed", "43"]})
# W-C: Stage B
for s in REST10:
    for k in ("a0", "a1", "a2"):
        q.append({"key": k, "arm": k, "scene": s, "extra": []})

with open(QF + ".tmp", "w") as f:
    json.dump(q, f, indent=1)
os.replace(QF + ".tmp", QF)
print(f"WROTE {QF} ({len(q)} entries; {3 + 1 + 2 + 30} pending)")

# state repair (safe: scheduler exited)
st = json.load(open(STATE))
st["failed"] = [d for d in st["failed"] if not d["rid"].startswith("b0_")]
st["done"] = [d for d in st["done"] if d["rid"] != "c0e01_room"]
st.setdefault("requeue_history", []).append({
    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "cleared_failed": ["b0_room", "b0_bicycle", "b0_garden"],
    "cleared_failed_reason": "stale failures from pre-fix trainer (1-D radii -> 0-D "
                             "visibility mask); trainer fixed (radii [1,N,1]) and "
                             "verified by 200-iter smoke RC=0",
    "cleared_done": ["c0e01_room"],
    "cleared_done_reason": "only the contaminated first attempt existed; clean re-run needed",
    "b1_gate_garden_note": "b1 gate C3 garden -0.642 dB vs b1a; trajectory shows late "
                           "divergence (b1 ahead at 25K, stalled last 5K) = variance not "
                           "wiring; seed-43 garden pair queued to resolve",
})
tmp = STATE + ".tmp"
with open(tmp, "w") as f:
    json.dump(st, f, indent=2)
os.replace(tmp, STATE)
print("state: done =", len(st["done"]), "failed =", len(st["failed"]),
      "active =", sorted(st["active"]))
