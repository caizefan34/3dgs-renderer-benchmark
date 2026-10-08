import sys, os, json, shutil
sys.path.insert(0, "/root/mr0_pkg")
import mr0_cell as CC
OUT = "/root/mr0_out_smoke"
os.makedirs(OUT, exist_ok=True)
cell = "room@10000"
print("== groups ==", flush=True)
CC.build_groups(cell, OUT)
print("== fit 0-4 (ns=2) ==", flush=True)
CC.fit_shard(cell, OUT, 0, 4, n_steps=2, n_cam_fit=8)
print("== replace ==", flush=True)
CC.replace(cell, OUT, n_isect_sample=16, n_isect_cam=4)
print("== aggregate ==", flush=True)
summary = {"requires_anchor_family_features": False, "cells": {}}
rep = json.load(open("%s/%s_replacement.json" % (OUT, cell)))
summary["cells"][cell] = {"K": rep["K"], "scene": rep["scene"], "it": rep["it"], "N": rep["N"],
                          "ckpt_sha256": rep["ckpt_sha256"], "n_groups": rep["n_groups"]}
json.dump(summary, open("%s/MR0_SUMMARY.json" % OUT, "w"), indent=2)
shutil.copy("/root/mr0_pkg/mr0_analyze.py", "%s/mr0_analyze.py" % OUT)
print("== analyze ==", flush=True)
os.system("/root/miniforge3/envs/gsplat/bin/python %s/mr0_analyze.py %s/MR0_SUMMARY.json" % (OUT, OUT))
print("SMOKE_DONE", flush=True)
