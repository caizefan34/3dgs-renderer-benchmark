#!/usr/bin/env python3
"""Check checkpoint keys for s22/s23 canonical runs vs room_30k."""
import torch, sys

checkpoints = [
    ("s22/garden/baseline", "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/s22/garden/checkpoints/baseline_iter_30000.pt"),
    ("s22/garden/c42", "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/s22/garden/checkpoints/c42_iter_30000.pt"),
    ("s23/garden/s0.75", "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/s23/garden/checkpoints/s0.750_iter_30000.pt"),
    ("s22/bicycle/baseline", "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/s22/bicycle/checkpoints/baseline_iter_30000.pt"),
    ("room_30k/baseline", "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/room_30k/checkpoints/iter_30000.pt"),
    ("c42_30k/room", "/home/liaoyuanjun/3dgs-renderer-benchmark/results/reference_v1/c42_30k/checkpoints/iter_30000.pt"),
    ("NEW/bicycle_s0.75", "/home/liaoyuanjun/3dgs-renderer-benchmark/results/c42_adaptive/completion_batch/checkpoints/bicycle_s0.75_iter_30000.pt"),
    ("NEW/room_s0.75", "/home/liaoyuanjun/3dgs-renderer-benchmark/results/c42_adaptive/completion_batch/checkpoints/room_s0.75_iter_30000.pt"),
]

for name, path in checkpoints:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        keys = sorted(ckpt.keys())
        print(f"{name}: {keys}")
    except Exception as e:
        print(f"{name}: ERROR - {e}")
