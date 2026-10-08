#!/usr/bin/env python3
import hashlib
import json
import os

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

base_dir = "/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_MEMFIX_R5_MX"
manifest = {}

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file == "MANIFEST_SHA256.json":
            continue
        filepath = os.path.join(root, file)
        relpath = os.path.relpath(filepath, base_dir)
        manifest[relpath] = sha256_file(filepath)

with open(os.path.join(base_dir, "MANIFEST_SHA256.json"), 'w') as f:
    json.dump(manifest, f, indent=2, sort_keys=True)

print(f"Generated MANIFEST_SHA256.json with {len(manifest)} files")
