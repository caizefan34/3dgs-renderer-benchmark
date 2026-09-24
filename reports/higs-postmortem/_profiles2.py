import json, glob
files = sorted(glob.glob(r"results/training/*bicycle_t16*"))
for f in files:
    r = json.load(open(f))
    print("=== FILE:", f)
    print("  top-level keys:", list(r.keys()))
    for k in r.keys():
        v = r[k]
        if isinstance(v, dict):
            print("  dict:", k, "->", list(v.keys())[:20])
        elif isinstance(v, list):
            print("  list:", k, "len", len(v))
            if v and isinstance(v[0], dict):
                print("       first item keys:", list(v[0].keys())[:20])
                print("       first:", json.dumps(v[0])[:500])
        else:
            print("  scalar:", k, "=", repr(v)[:100])
    print()
