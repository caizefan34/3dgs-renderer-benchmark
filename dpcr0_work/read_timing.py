import json
d=json.load(open("/root/epic05_dpc0/artifacts/DPC0_TIMING.json"))
print("type", type(d), "keys" , list(d.keys())[:10] if isinstance(d,dict) else "")
cells = d.get("cells", d)
for c in ("room@10000","bicycle@10000"):
    v = cells[c]
    print(c, "->", json.dumps(v)[:600])