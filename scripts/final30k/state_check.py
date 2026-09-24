import json

s = json.load(open("/mnt/storage_pool/liaoyuanjun/final30k_runs/scheduler_state.json"))
print("done:", [(d["arm"], d["scene"]) for d in s["done"]])
print("active:", list(s["active"]))
print("failed:", s["failed"])
print("contaminated:", s["contaminated"])
print("queue-size:", 26 - len(s["done"]) - len(s["failed"]) - len(s["active"]))
