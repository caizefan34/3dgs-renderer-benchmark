#!/usr/bin/env python3
import torch
import subprocess

result = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True)
sm_arch = result.stdout.strip().split('\n')[0]

print(f"PyTorch={torch.__version__}")
print(f"CUDA={torch.version.cuda}")
print(f"SM_ARCH={sm_arch}")
