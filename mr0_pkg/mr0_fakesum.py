import json
OUT = "/root/mr0_out_smoke"
CELLS = ["room@10000","room@15000","bicycle@10000","bicycle@15000"]
rep = json.load(open("%s/room@10000_replacement.json" % OUT))
summary = {"requires_anchor_family_features": False, "cells": {}}
for c in CELLS:
    summary["cells"][c] = {"K": rep["K"], "scene": c.split("@")[0], "it": int(c.split("@")[1]),
                           "N": rep["N"], "ckpt_sha256": rep["ckpt_sha256"], "n_groups": rep["n_groups"]}
json.dump(summary, open("%s/MR0_SUMMARY.json" % OUT, "w"), indent=2)
print("fake 4-cell summary written")
