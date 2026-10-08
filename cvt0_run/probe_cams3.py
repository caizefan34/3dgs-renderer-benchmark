#!/usr/bin/env python3
import json, os
REPO = "/root/3dgs-renderer-benchmark"
def probe(scene):
    cj = json.load(open(os.path.join(REPO, "data", "official", "mipnerf360", scene, "cameras.json")))
    n = len(cj)
    ids = [c["id"] for c in cj]
    names = [c["img_name"] for c in cj]
    gt_dir = os.path.join(REPO, "data", "datasets", "mipnerf360", scene, "images")
    stems = set(os.path.splitext(f)[0] for f in os.listdir(gt_dir) if f.lower().endswith((".jpg",".jpeg",".png")))
    # GTDataset matching: position i in cameras.json whose img stem in stems
    matched_pos = [i for i in range(n) if names[i] in stems]
    # canonical dataset index = id field
    id_contiguous = (ids == list(range(n)))
    print(f"=== {scene} ===")
    print(f"n={n}  id_contiguous_0..n-1={id_contiguous}")
    print(f"matched positions={len(matched_pos)}  contiguous={matched_pos==list(range(len(matched_pos)))}")
    # holdout by original dataset index i (position in canonical list == id)
    holdout = [i for i in matched_pos if i % 8 == 0]
    train = [i for i in matched_pos if i % 8 != 0]
    print(f"holdout(i%8==0) count={len(holdout)}  first16={holdout[:16]}")
    print(f"train pool count={len(train)}  first16={train[:16]}")
    return n, matched_pos, holdout, train
for s in ("room","bicycle"):
    probe(s)
