import json
d=json.load(open("/dev/shm/wv0_out/WV0_STATE_INVENTORY.json"))
e=d["states"]["room__iter5000"]
print("keys:", list(e.keys()))
print("error:", e.get("error"))
print("tb:", e.get("error_tb"))
