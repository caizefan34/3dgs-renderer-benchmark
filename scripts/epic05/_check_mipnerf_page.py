#!/usr/bin/env python3
"""Check Mip-NeRF 360 page for download info and verify scene requirements."""
import json
import os
import sys
import urllib.request
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check Mip-NeRF 360 page
url = "https://jonbarron.info/mipnerf360/"
try:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=20) as resp:
        html = resp.read().decode("utf-8")
        # Find href links
        links = re.findall(r'href=["\x27]([^"\x27]+)["\x27]', html)
        for l in links:
            print(f"Link: {l}")
        # Find download references
        for match in re.finditer(r"(?:download|dataset|image|data|360|zip|tar)[^\<]{0,200}", html, re.I):
            t = match.group().strip()[:150]
            print(f"Text: {t}")
except Exception as e:
    print(f"Error: {e}")

print()

# Check what we need for GT quality
for scene in ["bicycle", "garden", "room"]:
    cam_path = os.path.join(REPO_ROOT, "data", "official", "mipnerf360", scene, "cameras.json")
    with open(cam_path) as f:
        cams = json.load(f)
    print(f"{scene}: {len(cams)} cameras, native res={cams[0]['width']}x{cams[0]['height']}")
    print(f"  Sample image: {cams[0].get('img_name', 'N/A')}")
