import torch
# Reproduce the Phase 1 worker's subset selection (NO fixed seed) twice
import json, numpy as np
dev=torch.device("cuda:6")
cams=json.load(open("/mnt/storage_pool/liaoyuanjun/benchmark_data/mipnerf360/bicycle/eval_cameras.json"))
# simulate: worker does sub=vis_op.nonzero(); if >200K: sel=randperm; sub=sub[sel].sort()
# Without a fixed seed, two calls give different subsets
for trial in range(3):
    sub=torch.arange(663186,device=dev)  # vis_op count for 5K cam0
    sel=torch.randperm(sub.numel(),device=dev)[:200000]
    sub=sub[sel].sort().values
    print(f"trial{trial}: first5={sub[:5].tolist()} sum={int(sub.sum())}")
print("--- with manual_seed(0) (gate behavior) ---")
for trial in range(3):
    torch.manual_seed(0)
    sub=torch.arange(663186,device=dev)
    sel=torch.randperm(sub.numel(),device=dev)[:200000]
    sub=sub[sel].sort().values
    print(f"trial{trial}: first5={sub[:5].tolist()} sum={int(sub.sum())}")
