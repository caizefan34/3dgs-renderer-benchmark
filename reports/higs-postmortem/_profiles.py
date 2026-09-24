import json, glob, os, statistics
files = sorted(glob.glob(r"results/training/*.json"))
print("files:", len(files))
import collections
c = collections.Counter()
for f in files:
    r = json.load(open(f))
    cfg = r.get("config", {}) or {}
    key = (cfg.get("scene"), cfg.get("tile_size"), cfg.get("packed"), cfg.get("method"))
    c[key] += 1
for k, v in c.items():
    print(k, v)
