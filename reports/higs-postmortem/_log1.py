import json, glob, os
for f in sorted(glob.glob(r"results\training\*.json"))[:1]:
    r = json.load(open(f))
    print(os.path.basename(f))
    print(json.dumps(r.get("config", {}), indent=2)[:800])
    ml = r.get("metrics_log", [])
    print("metrics_log len:", len(ml))
    if ml:
        print(json.dumps(ml[0], indent=2)[:1500])
