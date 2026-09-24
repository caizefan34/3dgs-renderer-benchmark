#!/usr/bin/env python3
"""Smoke test: run R3 runner with 1 iteration to validate the pipeline."""
import sys, os, json, glob

sys.path.insert(0, "experiments/r3")

# First verify checkpoint file exists
candidates = sorted(glob.glob("results/**/*iter5000*.pt", recursive=True)) + \
             sorted(glob.glob("results/**/*_5000*.pt", recursive=True))
print("Checkpoint candidates:", candidates[:10])

# Verify camera sequence exists
import numpy as np
for p in ["data/camera_sequence.npy"]:
    if os.path.exists(p):
        a = np.load(p)
        print(f"camera_sequence: {p} shape={a.shape} dtype={a.dtype} min={a.min()} max={a.max()}")
    else:
        print(f"camera_sequence MISSING: {p}")
