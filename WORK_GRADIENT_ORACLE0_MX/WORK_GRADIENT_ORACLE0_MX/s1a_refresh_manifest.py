#!/usr/bin/env python3
"""Refresh SHA256 manifest after report update."""
import hashlib, os, json
D = "/mnt/storage_pool/liaoyuanjun/work_gradient_oracle0_tmp/s1a_deliverables"
m = {"schema":"S1A_SHA256_MANIFEST","files":{}}
for r,_,fs in os.walk(D):
    for f in fs:
        if f == "SHA256_MANIFEST.json": continue
        p = os.path.join(r,f)
        rel = os.path.relpath(p, D).replace("\\","/")
        m["files"][rel] = hashlib.sha256(open(p,"rb").read()).hexdigest()
json.dump(m, open(D+"/SHA256_MANIFEST.json","w"), indent=2)
print("refreshed", len(m["files"]), "files")
