"""Wave 3: extend the file queue with the 13-scene expansions + P6 multiseed.
Full rewrite (idempotent: done/active rids are skipped by rid-key matching).
Priority order = launch order:
  1. gates (b1, b0, a0/a1/a2, c0e01, b1ae03) - done/in-flight, skipped
  2. garden s43 pair (b1s43 in flight)
  3. Stage B a0/a1/a2 x 10  (in progress)
  4. P1 expansion: b1 x 10, b0 x 10
  5. P6 multiseed: b1a/c0 x seeds 43/44 x 4 scenes (16)
"""
import json
import os

QF = "/mnt/storage_pool/liaoyuanjun/pubphase/pub_queue.json"
GATE = ["room", "bicycle", "garden"]
ALL13 = ["bicycle", "bonsai", "counter", "drjohnson", "flowers", "garden",
         "kitchen", "playroom", "room", "stump", "train", "treehill", "truck"]
REST10 = [s for s in ALL13 if s not in GATE]
P6_SCENES = ["drjohnson", "train", "bicycle", "room"]

q = []
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
q.append({"key": "b1as43", "arm": "b1a", "scene": "garden", "extra": ["--seed", "43"]})
q.append({"key": "b1s43", "arm": "b1", "scene": "garden", "extra": ["--seed", "43"]})
# Stage B (P2)
for s in REST10:
    for k in ("a0", "a1", "a2"):
        q.append({"key": k, "arm": k, "scene": s, "extra": []})
# P1 expansions (gate runs already queued above are skipped)
for s in REST10:
    q.append({"key": "b1", "arm": "b1", "scene": s, "extra": []})
for s in REST10:
    q.append({"key": "b0", "arm": "b0", "scene": s, "extra": []})
# P6 multiseed (after expansions; 16 runs)
for s in P6_SCENES:
    for arm, k in (("b1a", "b1a"), ("c0", "c0")):
        for seed in (43, 44):
            q.append({"key": f"{k}s{seed}", "arm": arm, "scene": s,
                      "extra": ["--seed", str(seed)]})

with open(QF + ".tmp", "w") as f:
    json.dump(q, f, indent=1)
os.replace(QF + ".tmp", QF)
print(f"WROTE {QF}: {len(q)} entries total "
      f"(+{10 + 10 + 16} new: b1x10, b0x10, P6x16)")
