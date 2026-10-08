import json, glob, os
OUT = "/root/mr0_out"
CELLS = ["room@10000", "room@15000", "bicycle@10000", "bicycle@15000"]
# MR0_GROUPS.json
groups = {}
for c in CELLS:
    groups[c] = json.load(open(os.path.join(OUT, f"{c}_groups.json")))
json.dump(groups, open(os.path.join(OUT, "MR0_GROUPS.json"), "w"))
# MR0_FITS.json (concat all shards per cell)
fits = {}
for c in CELLS:
    recs = []
    for fn in sorted(glob.glob(os.path.join(OUT, f"{c}_fits_*.json"))):
        recs.extend(json.load(open(fn)))
    fits[c] = recs
json.dump(fits, open(os.path.join(OUT, "MR0_FITS.json"), "w"))
# MR0_REPLACEMENT_RAW.json
repl = {}
for c in CELLS:
    repl[c] = json.load(open(os.path.join(OUT, f"{c}_replacement.json")))
json.dump(repl, open(os.path.join(OUT, "MR0_REPLACEMENT_RAW.json"), "w"))
for f in ("MR0_GROUPS.json", "MR0_FITS.json", "MR0_REPLACEMENT_RAW.json"):
    print(f, os.path.getsize(os.path.join(OUT, f)), "bytes")
print("AGG_DONE")
