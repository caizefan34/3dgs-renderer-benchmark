#!/usr/bin/env python3
"""Check camera_sequence.npy and all file timestamps for candidate_c."""
import os, hashlib, json
from pathlib import Path
from datetime import datetime
import numpy as np

R4_BASE = Path("/mnt/storage_pool/liaoyuanjun/r4_13scene_v2")

print("=" * 100)
print("ALL file timestamps in train/truck candidate_c directories")
print("=" * 100)

for scene in ["train", "truck"]:
    run_dir = R4_BASE / scene / "candidate_c"
    print(f"\n--- {scene}/candidate_c ---")
    for fpath in sorted(run_dir.iterdir()):
        if fpath.is_file():
            stat = os.stat(fpath)
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  {fpath.name}: mtime={mtime}, size={stat.st_size}")
    
    # Check camera_sequence
    cs_path = run_dir / "camera_sequence.npy"
    if cs_path.exists():
        seq = np.load(cs_path)
        stat = os.stat(cs_path)
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n  camera_sequence.npy: mtime={mtime}, len={len(seq)}, first5={seq[:5]}, hash={hash(seq.tobytes())}")

    # Also check baseline for comparison
    b_dir = R4_BASE / scene / "baseline"
    cs_b_path = b_dir / "camera_sequence.npy"
    if cs_b_path.exists():
        seq_b = np.load(cs_b_path)
        stat = os.stat(cs_b_path)
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  baseline camera_sequence.npy: mtime={mtime}, len={len(seq_b)}, first5={seq_b[:5]}, hash={hash(seq_b.tobytes())}")

print("\n" + "=" * 100)
print("KEY QUESTION: Does camera_sequence match between baseline and candidate_c?")
print("=" * 100)

for scene in ["train", "truck"]:
    b_seq = np.load(R4_BASE / scene / "baseline" / "camera_sequence.npy")
    c_seq = np.load(R4_BASE / scene / "candidate_c" / "camera_sequence.npy")
    match = np.array_equal(b_seq, c_seq)
    print(f"\n{scene}: baseline vs candidate_c camera_sequence match = {match}")
    if not match:
        diffs = np.where(b_seq != c_seq)[0]
        print(f"  Differences at positions: {diffs[:10]}... ({len(diffs)} total)")
        print(f"  Baseline at diff[0]: {b_seq[diffs[0]]}, Candidate: {c_seq[diffs[0]]}")
