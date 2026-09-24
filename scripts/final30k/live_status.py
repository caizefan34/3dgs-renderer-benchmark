import re, os, json

dirs = "/mnt/storage_pool/liaoyuanjun/final30k_runs"
log = open(f"{dirs}/scheduler.log").read()
events = []
for line in log.splitlines():
    if re.search(r"LAUNCH|CONTAMINATION|adopt|skip|finished|done|FAIL", line):
        events.append(line)
print("EVENT LOG (last 14):")
for e in events[-14:]:
    print("  ", e)

state = json.load(open(f"{dirs}/scheduler_state.json"))
print("\nstate: done=%d failed=%d active=%s queue=%d contaminated=%s" % (
    len(state["done"]), len(state["failed"]), sorted(state["active"]),
    26 - len(state["done"]) - len(state["failed"]) - len(state["active"]),
    sorted(state.get("contaminated", {}))))
print("done:", [(d["arm"], d["scene"]) for d in state["done"]])
print("contaminated flags:", state.get("contaminated", {}))

# recent GPU state files
import glob
snaps = sorted(glob.glob(f"{dirs}/.snapshots/*.snap.txt"))
if snaps:
    s = open(snaps[-1]).read()
    print("\nlast snapshot:")
    print(s[:400])
