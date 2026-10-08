#!/usr/bin/env python3
import json, os, sys
REPO = "/root/3dgs-renderer-benchmark"
sys.path.insert(0, os.path.join(REPO, "src"))
cj = json.load(open(os.path.join(REPO, "data", "official", "mipnerf360", "room", "cameras.json")))
print("type:", type(cj))
if isinstance(cj, list):
    print("len:", len(cj))
    print("entry[0] keys:", list(cj[0].keys()) if isinstance(cj[0], dict) else cj[0])
    print("entry[0]:", json.dumps(cj[0])[:400])
elif isinstance(cj, dict):
    print("top keys:", list(cj.keys())[:20])
