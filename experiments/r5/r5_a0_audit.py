#!/usr/bin/env python3
"""R5-A0: Audit R4 train/truck seed and config."""
import json, os, sys

BASE = "/mnt/storage_pool/liaoyuanjun/r4_13scene_v2"
scenes = ["train", "truck"]
methods = ["baseline", "candidate_c"]

for scene in scenes:
    for method in methods:
        path = f"{BASE}/{scene}/{method}/provenance.json"
        try:
            d = json.load(open(path))
            print(f"{scene}/{method}:")
            print(f"  seed: {d.get('seed')}")
            print(f"  camera_seed: {d.get('camera_seed')}")
            print(f"  git_commit: {d.get('git_commit')}")
            print(f"  git_dirty: {d.get('git_dirty')}")
            print(f"  git_diff_sha256: {d.get('git_diff_sha256')}")
            print(f"  training_script_sha256: {d.get('training_script_sha256')}")
            print(f"  config_sha256: {d.get('config_sha256')}")
            print(f"  gsplat_version: {d.get('gsplat_version')}")
            print(f"  torch_version: {d.get('torch_version')}")
            print(f"  gpu_name: {d.get('gpu_name')}")
            print(f"  initial_gaussian_count: {d.get('initial_gaussian_count')}")
            print(f"  timestamp: {d.get('timestamp')}")
            print()
        except Exception as e:
            print(f"{scene}/{method}: ERROR - {e}")
            print()

# Also check config.json
print("=== Config comparison ===")
for scene in scenes:
    for method in methods:
        path = f"{BASE}/{scene}/{method}/config.json"
        try:
            d = json.load(open(path))
            print(f"{scene}/{method}: seed={d.get('seed')}, iterations={d.get('iterations')}, "
                  f"sh_degree={d.get('sh_degree')}, resolution={d.get('resolution')}, "
                  f"densify_grad_threshold={d.get('densify_grad_threshold')}, "
                  f"checkpoint_iterations={d.get('checkpoint_iterations')}")
        except Exception as e:
            print(f"{scene}/{method} config: ERROR - {e}")

# Check camera_sequence
print("\n=== Camera sequence comparison ===")
for scene in scenes:
    for method in methods:
        path = f"{BASE}/{scene}/{method}/camera_sequence.npy"
        if os.path.exists(path):
            import numpy as np
            seq = np.load(path)
            print(f"{scene}/{method}: len={len(seq)}, first5={seq[:5]}, hash={hash(seq.tobytes())}")
        else:
            print(f"{scene}/{method}: camera_sequence.npy NOT FOUND")
