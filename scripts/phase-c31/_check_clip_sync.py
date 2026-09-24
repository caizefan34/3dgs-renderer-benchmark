#!/usr/bin/env python3
"""Check if clip_grad_norm_ calls .item() or sync."""
import torch
from torch.nn.utils import clip_grad_norm_

# Create a model like our GaussianModel
p = torch.randn(1000, 3, device="cuda", requires_grad=True)
loss = p.sum()
loss.backward()

# Check if clip_grad_norm has sync
with torch.cuda.profiler.profile(activities=[torch.cuda.profiler.ProfilerActivity.CPU, torch.cuda.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        gn = clip_grad_norm_([p], max_norm=1.0)
        # .item() call on grad norm
        _ = gn.item()
        torch.cuda.synchronize()

prof.export_chrome_trace("/tmp/clip_test_trace.json")
print("Trace saved")

# Also check: does clip_grad_norm return a scalar tensor that requires sync?
print(f"grad_norm type: {type(gn)}, device: {gn.device}, requires item={hasattr(gn, 'item')}")
