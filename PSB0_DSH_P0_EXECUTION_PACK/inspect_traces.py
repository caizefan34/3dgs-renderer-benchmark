import json
for p in ["/root/epic05_mg0/artifacts/OC0_INSTRUMENTATION_PARITY.json",
          "/root/dsh_20261001/artifacts/DSH7_BARS0/BARS0_TILE_TRACE.json",
          "/root/dsh_20261001/artifacts/DSH7_BARS0/BARS0_SUMMARY.json"]:
    print("=====", p)
    try:
        d = json.load(open(p))
        if isinstance(d, dict):
            print("keys:", list(d.keys())[:30])
        else:
            print("type:", type(d), "len:", len(d))
            if isinstance(d, list) and d:
                print("elem0 keys:", list(d[0].keys())[:20] if isinstance(d[0],dict) else d[0])
    except Exception as e:
        print("ERR", e)
