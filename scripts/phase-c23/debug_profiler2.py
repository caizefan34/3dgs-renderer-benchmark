#!/usr/bin/env python3
import torch
torch.manual_seed(0)
from torch.profiler import profile, ProfilerActivity
x=torch.randn(100,100,device='cuda',requires_grad=True)
with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
 (x@x.t()).sum().backward()
e=prof.key_averages()[0]
attrs=[a for a in dir(e) if not a.startswith('_')]
print(attrs)
print(e)
