#!/usr/bin/env python3
"""Check availability of all R3 measurement inputs."""
import os, glob, torch

print("=== CHECKPOINT FILES ===")
ckpt_dir = "results/epic05/phase7/phase7_room_30k_v2_16"
if os.path.isdir(ckpt_dir):
    for f in sorted(os.listdir(ckpt_dir)):
        full = os.path.join(ckpt_dir, f)
        size_mb = os.path.getsize(full) / 1e6
        print(f"  {f}: {size_mb:.1f} MB")
else:
    print(f"  (missing) {ckpt_dir}")

print("\n=== OTHER CANDIDATE CHECKPOINTS ===")
for pat in ["results/**/*.pt", "baseline/**/*.pt", "output/**/*.pt"]:
    for f in glob.glob(pat, recursive=True):
        print(f"  {f} ({os.path.getsize(f)/1e6:.1f} MB)")

print("\n=== CAMERA DATA ===")
for pat in ["data/camera*", "data/videos/*", "data/*.json"]:
    for f in glob.glob(pat):
        print(f"  {f}")

print("\n=== R3 SCRIPT DIR ===")
for f in sorted(os.listdir("experiments/r3")):
    print(f"  {f}")
