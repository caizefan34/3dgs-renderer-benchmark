"""Fix adopted-run bookkeeping: move b1ae03_room + c0e01_bicycle from failed to
done (they completed and wrote results.json; the old scheduler falsely marked
them failed via the rc-None reaping bug). Kill scheduler first."""
import json
import os
import signal
import subprocess
import time

STATE = "/mnt/storage_pool/liaoyuanjun/pub_runs/pub_scheduler_state.json"

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

st = json.load(open(STATE))

# verify the two "failed" runs really completed
def ok_run(rid):
    d = os.path.join("/mnt/storage_pool/liaoyuanjun/pub_runs", rid)
    rj = os.path.join(d, "results.json")
    if not os.path.exists(rj):
        return False
    try:
        j = json.load(open(rj))
        return "final_eval" in j
    except Exception:
        return False

moved = []
still_failed = []
for d in list(st["failed"]):
    if ok_run(d["rid"]):
        d["note"] = "reap-bug fix: completed with results.json; falsely failed by rc-None adoption path"
        st["done"].append(d)
        moved.append(d["rid"])
    else:
        still_failed.append(d)
st["failed"] = still_failed
st.setdefault("requeue_history", []).append({
    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "reap_bug_fix": "adopted children (rc=None) judged by results.json presence "
                    "instead of (rc==0); two completed runs moved failed->done",
    "moved_failed_to_done": moved,
})
tmp = STATE + ".tmp"
with open(tmp, "w") as f:
    json.dump(st, f, indent=2)
os.replace(tmp, STATE)
print("moved failed->done:", moved)
print("still failed:", [d["rid"] for d in st["failed"]])
print("done:", len(st["done"]), "active:", sorted(st["active"]))
