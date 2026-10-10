#!/usr/bin/env python3
"""Refresh GP0 manifest after supplement (adds SOURCE_IDENTITY, alpha npz)."""
import json, os, hashlib
D="/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp"
OUT=f"{D}/gp0_deliverables"
def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(65536),b""): h.update(b)
    return h.hexdigest()
m={"schema":"GP0_SHA256_MANIFEST","files":{}}
for r,_,fs in os.walk(OUT):
    for f in fs:
        if f=="SHA256_MANIFEST.json": continue
        p=os.path.join(r,f); rel=os.path.relpath(p,OUT).replace("\\","/")
        m["files"][rel]=sha256(p)
json.dump(m,open(f"{OUT}/SHA256_MANIFEST.json","w"),indent=2)
print("manifest files:",len(m["files"]))
for k in sorted(m["files"]): print(" ",k)
