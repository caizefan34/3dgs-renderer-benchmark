#!/usr/bin/env python3
import sys
import hashlib

baseline_ext = "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat/csrc.so"
higs_ext = "/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx/gsplat/experimental/render/kernels/csrc.so"

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

baseline_sha = sha256(baseline_ext)
higs_sha = sha256(higs_ext)

print(f"BASELINE_EXTENSION_PATH={baseline_ext}")
print(f"BASELINE_EXTENSION_SHA256={baseline_sha}")
print(f"HIGS_EXTENSION_PATH={higs_ext}")
print(f"HIGS_EXTENSION_SHA256={higs_sha}")
