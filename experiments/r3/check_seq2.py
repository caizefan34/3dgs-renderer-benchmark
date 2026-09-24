#!/usr/bin/env python3
import numpy as np
s = np.load("data/camera_sequence.npy")
print("len(seq):", len(s), "min:", s.min(), "max:", s.max())
for start in (5000, 15000, 30000):
    idx = start + 1 - 1  # iteration starts at start+1, camera index = iteration-1
    print(f"start={start}: last camera index = {start + 30 - 1} (out of bounds if >= {len(s)})")
    if start + 30 - 1 >= len(s):
        print(f"  -> OUT OF BOUNDS for start={start}")
    else:
        print(f"  -> OK, camera ids: {s[start:start+30]}")
