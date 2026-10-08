#!/usr/bin/env python3
import sys
import torch

ckpt_path = sys.argv[1]
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

print(f"Keys: {list(ckpt.keys())}")

# Check for required Gaussian parameters
required = ['means', 'quats', 'scales', 'opacities', 'colors']
optional = ['shs', 'features']

found = []
for k in required + optional:
    if k in ckpt:
        found.append(k)
        print(f"  {k}: {ckpt[k].shape}")

if 'feat' in ckpt:
    print(f"  feat: {ckpt['feat'].shape} (potential SH/features)")

print(f"\nFound: {found}")
