#!/usr/bin/env python3
"""Fix prune_and_reset shape bug in gaussian_model.py.

The bug: load_ply() returns 1D opacity [N], but prune_and_reset creates
2D reset_val [n, 1]. Fix: make reset_val match opacity dimensionality.
"""
import sys
from pathlib import Path

gaussian_model_path = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7/gaussian_model.py")

# Read the file
with open(gaussian_model_path, 'r') as f:
    content = f.read()

# Backup
backup_path = gaussian_model_path.with_suffix('.py.orig_stage4b')
if not backup_path.exists():
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"Backup saved to {backup_path}")

# Fix the reset_val line
old_line = """                reset_val = torch.logit(torch.full((reset_count, 1), opacity_threshold * 2, device=self.device))
                self.opacity.data[near_threshold] = reset_val"""

new_line = """                reset_shape = (reset_count, 1) if self.opacity.dim() > 1 else (reset_count,)
                reset_val = torch.logit(torch.full(reset_shape, opacity_threshold * 2, device=self.device))
                self.opacity.data[near_threshold] = reset_val"""

if old_line in content:
    content = content.replace(old_line, new_line)
    with open(gaussian_model_path, 'w') as f:
        f.write(content)
    print("Fixed prune_and_reset shape bug!")
else:
    print("ERROR: Could not find the target line to replace!")
    # Show the relevant section
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'reset_val' in line or 'near_threshold' in line:
            print(f"  Line {i+1}: {line}")
