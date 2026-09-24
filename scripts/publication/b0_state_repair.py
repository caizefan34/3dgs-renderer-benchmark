"""Post-B0-fix state repair:
1. clear b0_* from failed (fixed trainer: radii reshape [1,N,1])
2. move contaminated c0e01_room run dir aside (preserve), remove from done
   so the clean re-run actually happens (its old results.json was causing a skip)
"""
import json
import os
import shutil
import time

STATE = "/mnt/storage_pool/liaoyuanjun/pub_runs/pub_scheduler_state.json"
RUNS = "/mnt/storage_pool/liaoyuanjun/pub_runs"

st = json.load(open(STATE))

# 1. clear b0 failures
before = [d["rid"] for d in st["failed"]]
st["failed"] = [d for d in st["failed"] if not d["rid"].startswith("b0_")]

# 2. c0e01_room: preserve contaminated attempt, remove from done
src = os.path.join(RUNS, "c0e01_room")
dst = os.path.join(RUNS, "c0e01_room_contaminated_attempt1")
if os.path.exists(src) and not os.path.exists(dst):
    shutil.move(src, dst)
    print(f"moved {src} -> {dst}")
st["done"] = [d for d in st["done"] if d["rid"] != "c0e01_room"]

st.setdefault("requeue_history", []).append({
    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "cleared_failed": ["b0_room", "b0_bicycle", "b0_garden"],
    "cleared_failed_reason": "b0 radii was returned 1-D [N]; (radii>0).any(dim=-1) "
                             "collapsed to 0-D bool -> boolean-index shape cascade in "
                             "add_densification_stats. Fixed: b0 returns radii as [1,N,1].",
    "cleared_done": ["c0e01_room"],
    "cleared_done_reason": "contaminated first attempt blocked the clean re-run via the "
                           "launch-time results.json skip; moved to c0e01_room_contaminated_attempt1",
})
tmp = STATE + ".tmp"
with open(tmp, "w") as f:
    json.dump(st, f, indent=2)
os.replace(tmp, STATE)
print("cleared failed:", [r for r in before if r.startswith("b0_")])
print("done:", len(st["done"]), "failed:", len(st["failed"]), "active:", sorted(st["active"]))
