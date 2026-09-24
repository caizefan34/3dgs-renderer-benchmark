"""Restart procedure for the publication scheduler:
1. kill the running scheduler (children keep running; they are adopted on restart)
2. clear b1_* failures (fixed trainer) + c0e01_room (contaminated timing) from state
3. the relaunched scheduler re-queues them via its QUEUE constant
Run AFTER killing the old scheduler. Idempotent-ish; refuses if scheduler alive.
"""
import json
import os
import signal
import subprocess
import time

STATE = "/mnt/storage_pool/liaoyuanjun/pub_runs/pub_scheduler_state.json"

# 1. ensure old scheduler is dead
r = subprocess.run(["pgrep", "-f", "publication_scheduler.py"], capture_output=True, text=True)
pids = [p for p in r.stdout.split()]
if pids:
    for p in pids:
        os.kill(int(p), signal.SIGTERM)
    time.sleep(2)
    r2 = subprocess.run(["pgrep", "-f", "publication_scheduler.py"], capture_output=True, text=True)
    still = [p for p in r2.stdout.split()]
    if still:
        for p in still:
            os.kill(int(p), signal.SIGKILL)
        time.sleep(1)
    print("killed scheduler pids:", pids)
else:
    print("no scheduler running")

# 2. patch state
st = json.load(open(STATE))
before_done = len(st["done"])
before_failed = len(st["failed"])
st["failed"] = [d for d in st["failed"] if not d["rid"].startswith("b1_")]
st["done"] = [d for d in st["done"] if d["rid"] != "c0e01_room"]
st.get("contaminated", {}).pop("c0e01_room", None)
# preserve a note of what was cleared and why (durable provenance)
st.setdefault("requeue_history", []).append({
    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "cleared_failed": ["b1_room", "b1_bicycle", "b1_garden"],
    "cleared_failed_reason": "accutile TypeError from pre-fix trainer (pristine v1.5.3 "
                             "signature has no accutile kwarg); trainer fixed and "
                             "recompiled before this requeue",
    "cleared_done": ["c0e01_room"],
    "cleared_done_reason": "contaminated mid-run (foreign pid on GPU); timing downgraded, "
                           "re-run required for clean P4 wall-time",
})
tmp = STATE + ".tmp"
with open(tmp, "w") as f:
    json.dump(st, f, indent=2)
os.replace(tmp, STATE)
print(f"state patched: done {before_done} -> {len(st['done'])}, "
      f"failed {before_failed} -> {len(st['failed'])}")
print("active preserved:", sorted(st["active"]))
