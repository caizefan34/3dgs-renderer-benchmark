#!/usr/bin/env python3
import sys
import torch

ckpt_path = sys.argv[1]
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Check for standard gsplat checkpoint format
if 'xyz' in ckpt:
    N = ckpt['xyz'].shape[0]
    print(f"N={N}")
    print(f"Keys: {list(ckpt.keys())}")
    for k in ['xyz', 'shs', 'scaling', 'rotation', 'opacity']:
        if k in ckpt:
            print(f"  {k}: {ckpt[k].shape}")
elif 'splats' in ckpt:
    splats = ckpt['splats']
    print(f"Keys in splats: {list(splats.keys())}")
    if 'means' in splats:
        N = splats['means'].shape[0]
        print(f"N={N}")
else:
    print(f"Keys: {list(ckpt.keys())}")
    for k, v in ckpt.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")
