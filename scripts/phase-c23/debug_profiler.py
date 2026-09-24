#!/usr/bin/env python3
"""Debug: print all FunctionEventAvg attributes."""
import torch
torch.manual_seed(0)
from torch.profiler import profile, ProfilerActivity
x=torch.randn(100,100,device='cuda')
with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
 (x@x.t()).sum().backward()
for e in prof.key_averages():
 print(dir(e)); break
