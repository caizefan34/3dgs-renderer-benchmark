import json, hashlib, os
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
man = json.load(open("CVT0_SHA256_MANIFEST.json"))
allok = True
for fn, meta in man["files"].items():
    if not os.path.isfile(fn):
        print("MISSING", fn); allok = False; continue
    h = sha(fn)
    ok = h == meta["sha256"]
    allok &= ok
    print(("OK  " if ok else "BAD ") + fn)
print("ALL_MATCH:", allok)
v = json.load(open("CVT0_VERDICT.json"))
print("VERDICT:", v["verdict"])
s = json.load(open("CVT0_SUMMARY.json"))
print("cells:", s["aggregates"])
