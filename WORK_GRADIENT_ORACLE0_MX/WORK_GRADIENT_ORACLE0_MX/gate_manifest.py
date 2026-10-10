#!/usr/bin/env python3
"""Refresh manifest for the backward-correctness gate deliverables."""
import json, os, hashlib
D="/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp"
OUT=f"{D}/gp0_deliverables"
def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(65536),b""): h.update(b)
    return h.hexdigest()
m={"schema":"BACKWARD_GATE_MANIFEST","files":{}}
for r,_,fs in os.walk(OUT):
    for f in sorted(fs):
        if f in ("SHA256_MANIFEST.json","BACKWARD_GATE_MANIFEST.json"): continue
        p=os.path.join(r,f); rel=os.path.relpath(p,OUT).replace("\\","/")
        m["files"][rel]=sha256(p)
json.dump(m,open(f"{OUT}/BACKWARD_GATE_MANIFEST.json","w"),indent=2)
print("manifest files:",len(m["files"]))
for k in sorted(m["files"]): print(" ",k)
