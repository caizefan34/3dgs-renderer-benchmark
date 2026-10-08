#!/usr/bin/env python3
import sys
import torch

ckpt_path = sys.argv[1]
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

if 'splats' in ckpt:
    means = ckpt['splats']['means']
    print(f"N={means.shape[0]}")
elif 'means' in ckpt:
    print(f"N={ckpt['means'].shape[0]}")
elif 'splats' in ckpt and 'means' in ckpt['splats']:
    print(f"N={ckpt['splats']['means'].shape[0]}")
else:
    print(f"Keys: {list(ckpt.keys())}")
    for k, v in ckpt.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")
