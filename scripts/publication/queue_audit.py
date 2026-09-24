"""Queue-vs-matrix audit: verify every planned run is done, in flight, or queued —
and nothing else. The master-plan run matrix:
  P1 gates: b1 x3, b0 x3 (done)
  P2: a0/a1/a2 x 13 scenes (Stage A room/bicycle/garden + Stage B 10)
  P4: c0e01 x3, b1ae03 x3 (done)
  Garden seed pairs: b1as43/b1s43 (done), b1as44/b1s44 (queued)
  P1 expansions: b1 x10, b0 x10
  P6: b1a/c0 x s43/s44 x drjohnson/train/bicycle/room = 16
"""
import json

QF = "/mnt/storage_pool/liaoyuanjun/pubphase/pub_queue.json"
ST = "/mnt/storage_pool/liaoyuanjun/pub_runs/pub_scheduler_state.json"

ALL13 = ["bicycle", "bonsai", "counter", "drjohnson", "flowers", "garden",
         "kitchen", "playroom", "room", "stump", "train", "treehill", "truck"]
GATE = ["room", "bicycle", "garden"]
REST10 = [s for s in ALL13 if s not in GATE]
P6 = ["drjohnson", "train", "bicycle", "room"]

planned = {}
def add(rid):
    planned.setdefault(rid, 0)
    planned[rid] += 1

for s in GATE:
    add(f"b1_{s}")
    add(f"b0_{s}")
for s in ALL13:
    for a in ("a0", "a1", "a2"):
        add(f"{a}_{s}")
for s in GATE:
    add(f"c0e01_{s}")
    add(f"b1ae03_{s}")
add("b1as43_garden"); add("b1s43_garden")
add("b1as44_garden"); add("b1s44_garden")
for s in REST10:
    add(f"b1_{s}")
    add(f"b0_{s}")
for s in P6:
    for arm in ("b1a", "c0"):
        for seed in (43, 44):
            add(f"{arm}s{seed}_{s}")

st = json.load(open(ST))
state_rids = {}
for d in st["done"]:
    state_rids[d["rid"]] = state_rids.get(d["rid"], 0) + 1
for d in st["active"].values():
    state_rids[d["rid"]] = state_rids.get(d["rid"], 0) + 1
q = json.load(open(QF))
q_rids = {}
for e in q:
    rid = f"{e['key']}_{e['scene']}"
    q_rids[rid] = q_rids.get(rid, 0) + 1

covered = {}
for rid in state_rids:
    covered[rid] = covered.get(rid, 0) + 1
for rid in q_rids:
    covered[rid] = covered.get(rid, 0) + 1

print(f"planned={len(planned)} rids; state(done+active)={len(state_rids)}; "
      f"queue-entries={len(q)}")
missing = [r for r in planned if r not in covered]
extra = [r for r in covered if r not in planned]
dupes = {r: c for r, c in covered.items() if planned.get(r, 0) != c}
print("MISSING from coverage:", missing or "none")
print("EXTRA (not in matrix):", extra or "none")
if dupes:
    for r, c in sorted(dupes.items()):
        print(f"  count mismatch {r}: covered={c} planned={planned.get(r)}")
# remaining work
remaining_q = [f"{e['key']}_{e['scene']}" for e in q if f"{e['key']}_{e['scene']}" not in state_rids]
print(f"remaining-to-run (queue minus done/active): {len(remaining_q)}")
from collections import Counter
kinds = Counter(r.split("_")[0].rstrip("0123456789") if not r.startswith(("a0", "a1", "a2")) else r[:2] for r in remaining_q)
print("by arm:", dict(kinds))
