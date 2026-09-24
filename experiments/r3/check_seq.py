#!/usr/bin/env python3
"""Check camera sequence values at relevant indices."""
import numpy as np
s = np.load("data/camera_sequence.npy")
print(f"shape={s.shape} dtype={s.dtype} min={s.min()} max={s.max()}")
print(f"5000={s[5000]} 15000={s[15000]} 30000={s[29999]}")
