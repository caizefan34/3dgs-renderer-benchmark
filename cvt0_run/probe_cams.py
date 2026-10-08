#!/usr/bin/env python3
"""Decision-independent camera probe: canonical counts, matched indices, i%8 holdout."""
import json, os, sys
REPO = "/root/3dgs-renderer-benchmark"
def load_cameras(scene):
    p = os.path.join(REPO, "data", "official", "mipnerf360", scene, "cameras.json")
    return json.load(open(p))

for scene in ("room", "bicycle"):
    cj = load_cameras(scene)
    n = len(cj)
    names = [c.get("image_name") for c in cj]
    # GT image stems present
    gt_dir = os.path.join(REPO, "data", "datasets", "mipnerf360", scene, "images")
    stems = set()
    for f in os.listdir(gt_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            stems.add(os.path.splitext(f)[0])
    matched = [i for i, nm in enumerate(names) if os.path.splitext(str(nm))[0] in stems]
    holdout = [i for i in matched if i % 8 == 0]
    train = [i for i in matched if i % 8 != 0]
    print(f"=== {scene} ===")
    print(f"cameras.json entries: {n}")
    print(f"matched to GT: {len(matched)}  (contiguous 0..n-1? {matched == list(range(len(matched)))})")
    print(f"first 3 image_name: {names[:3]}")
    print(f"i%8==0 holdout count: {len(holdout)}")
    print(f"holdout indices (first 16): {holdout[:16]}")
    print(f"train pool count: {len(train)}")
    print(f"train indices (first 16): {train[:16]}")
