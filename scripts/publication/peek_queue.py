import json

QF = "/mnt/storage_pool/liaoyuanjun/pubphase/pub_queue.json"
q = json.load(open(QF))
print("total entries:", len(q))
print("first entry keys:", list(q[0].keys()))
print(json.dumps(q[0], indent=1))
# find the garden s44 entries (most recent additions) and the c0e01 re-run if present
for e in q:
    if e["key"] in ("b1as44", "c0e01") and e["scene"] in ("garden", "room"):
        print(json.dumps(e, indent=1))
