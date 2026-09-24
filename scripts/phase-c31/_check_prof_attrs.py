#!/usr/bin/env python3
"""Check FunctionEventAvg attributes."""
import torch

# Create a simple profile to inspect
x = torch.randn(100, 10, device="cuda")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
) as prof:
    y = x @ x.T
    y.sum().backward()

ka = prof.key_averages()
for k in ka:
    print(f"key={k.key[:50]}  attrs={[a for a in dir(k) if not a.startswith('_') and 'time' in a.lower()]}")
    break
