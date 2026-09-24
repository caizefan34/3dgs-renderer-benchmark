#!/usr/bin/env python3
"""Inspect r6_baseline_ckpts room checkpoint and camera_sequence format."""
import json, torch, numpy as np
out = {}
s = torch.load('/mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts/room/checkpoints/iter_30000.pt', map_location='cpu')
if isinstance(s, dict):
    out["keys"] = {k: (tuple(v.shape), str(v.dtype)) if torch.is_tensor(v) else type(v).__name__ for k, v in s.items()}
elif hasattr(s, 'state_dict'):
    out["type"] = type(s).__name__
    sd = s.state_dict()
    out["keys"] = {k: (tuple(v.shape), str(v.dtype)) for k, v in sd.items()}
else:
    out["type"] = type(s).__name__
try:
    a = np.load('/mnt/storage_pool/liaoyuanjun/r6_baseline_ckpts/room/camera_sequence.npy')
    out["camera_sequence_shape"] = list(a.shape)
    out["camera_sequence_dtype"] = str(a.dtype)
    out["camera_0"] = a[0].ravel().tolist()[:16]
except Exception as e:
    out["cam_seq_err"] = f"{type(e).__name__}: {e}"
print(json.dumps(out, sort_keys=True, indent=2, default=str))