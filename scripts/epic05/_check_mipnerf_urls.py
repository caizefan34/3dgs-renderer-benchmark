#!/usr/bin/env python3
"""Check Mip-NeRF 360 dataset URLs and GT image availability."""
import json
import os
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check camera data to understand image requirements
with open(os.path.join(REPO_ROOT, "data", "official", "mipnerf360", "room", "cameras.json")) as f:
    cams = json.load(f)
print(f"Room has {len(cams)} cameras")
print(f"First 3 image names: {[c.get('img_name','?') for c in cams[:3]]}")
print(f"Sample keys: {list(cams[0].keys())}")
print()

# Try known dataset URLs
urls_to_try = [
    "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/360_v2.zip",
    "https://huggingface.co/datasets/graphdeco-inria/3d-gaussian-splatting/resolve/main/360_v2.zip",
    "https://huggingface.co/datasets/graphdeco-inria/3d-gaussian-splatting/resolve/main/input/360_v2.zip",
    "https://jonbarron.info/mipnerf360/",
]

for url in urls_to_try:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"FOUND: {url}")
            print(f"  Status: {resp.status}")
            print(f"  Content-Length: {resp.headers.get('Content-Length', 'unknown')} bytes")
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {url}")
    except urllib.error.URLError as e:
        print(f"URL Error: {url} -> {e.reason}")
    except Exception as e:
        print(f"Error: {url} -> {type(e).__name__}: {str(e)[:80]}")
