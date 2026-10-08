#!/usr/bin/env python3
"""SG0-E10 E0 aggregator: combine per-cell E0 JSONs into SG0E10_E0_SUMMARY.json."""
import json, os, sys
CELLS = [("room", 10000), ("bicycle", 10000)]
CELLDIR = sys.argv[1] if len(sys.argv) > 1 else "/root/epic05_sg0e10/cells"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/root/epic05_sg0e10/SG0E10_E0_SUMMARY.json"
cells = {}
for scene, ckpt in CELLS:
    cell = f"{scene}@{ckpt}"
    p = os.path.join(CELLDIR, f"SG0E10_E0_{cell}.json")
    if os.path.isfile(p):
        d = json.load(open(p))
        cells[cell] = {"base_non_descent_K4_count": d["base_non_descent_K4_count"],
                       "base_non_descent_K8_count": d["base_non_descent_K8_count"],
                       "ckpt_sha256": d["ckpt_sha256"]}
    else:
        print("MISSING", p); sys.exit(2)
summary = {"audit_type": "SG0E10_E0_SUMMARY", "schema_version": "SG0E10-E0-v1",
           "stage": "E0", "cells": cells}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(summary, open(OUT, "w"), indent=2)
print(json.dumps(summary, indent=2))