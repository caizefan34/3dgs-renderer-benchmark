import json

for rid in ("a0_room", "a2_room", "b1_room", "b1a_room"):
    for root in ("/mnt/storage_pool/liaoyuanjun/pub_runs", "/mnt/storage_pool/liaoyuanjun/final30k_runs"):
        try:
            d = json.load(open(f"{root}/{rid}/results.json"))
        except FileNotFoundError:
            continue
        r = d.get("renderer", {})
        print(f"{rid:12s} arm={d.get('arm')} renderer={json.dumps(r)[:220]}")
        break
