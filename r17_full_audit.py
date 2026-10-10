#!/usr/bin/env python3
"""R17: Complete Parameter Validity Audit"""
import subprocess
import json
import os

def ssh_cmd(cmd):
    result = subprocess.run(['ssh', 'mx', cmd], capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

print("=== R17 P0: R16 Arithmetic Audit ===\n")

stdout, _, _ = ssh_cmd('cat /mnt/storage_pool/liaoyuanjun/HIGS_R16_OPACITY_SEMANTICS_MX/R16_REPLAY_COUNTS.csv')
print(stdout)

lines = stdout.strip().split('\n')
data = [line.split(',') for line in lines[1:]]

raw_gpu = int(data[0][1])
sigmoid_gpu = int(data[1][1])
threshold_gpu = int(data[2][1])

print(f"\nRAW: {raw_gpu:,}")
print(f"Sigmoid: {sigmoid_gpu:,}")
print(f"Threshold: {threshold_gpu:,}")
print(f"\nActual diff: {raw_gpu - sigmoid_gpu:,}")
print(f"Claimed: 360,251")
print(f"ERROR: {360251 - (raw_gpu - sigmoid_gpu):,}")

if sigmoid_gpu == threshold_gpu:
    print("\n→ THRESHOLD_BRANCH_NOT_EXERCISED")

print("\n=== P1: Check checkpoint parameters ===")

check_script = '''
import torch
ckpt = torch.load("/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt", map_location="cpu", weights_only=False)
for k in ["opacity", "scaling", "rotation"]:
    if k in ckpt:
        p = ckpt[k]
        print(f"{k}: min={p.min().item():.4f} max={p.max().item():.4f} mean={p.mean().item():.4f}")
        if k == "rotation":
            print(f"  norm_mean={p.norm(dim=-1).mean().item():.4f}")
'''

with open('/tmp/r17_check.py', 'w') as f:
    f.write(check_script)

subprocess.run(['scp', '/tmp/r17_check.py', 'mx:/tmp/'], capture_output=True)
stdout, stderr, rc = ssh_cmd('source /home/liaoyuanjun/miniforge3/bin/activate /mnt/storage_pool/liaoyuanjun/higs-13scene-env && python3 /tmp/r17_check.py')
print(stdout)
