import subprocess
import json

def ssh_cmd(cmd):
    result = subprocess.run(['ssh', 'mx', cmd], capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

print("=== R17 P1: Parameter Contract Audit ===\n")

# Create a Python script to check parameter ranges
check_script = '''
import torch
import sys

ckpt_path = "/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print("=== Checkpoint Parameter Ranges ===\\n")

for key in ["opacity", "scaling", "rotation"]:
    if key in ckpt:
        param = ckpt[key]
        print(f"{key}:")
        print(f"  shape: {param.shape}")
        print(f"  dtype: {param.dtype}")
        print(f"  min: {param.min().item():.6f}")
        print(f"  max: {param.max().item():.6f}")
        print(f"  mean: {param.mean().item():.6f}")
        print(f"  std: {param.std().item():.6f}")
        
        # Check if already activated
        if key == "opacity":
            # Sigmoid output should be in (0, 1)
            if param.max() <= 1.0 and param.min() >= 0.0:
                print(f"  → Already sigmoid-activated (range in [0,1])")
            else:
                print(f"  → Raw logits (needs sigmoid)")
        elif key == "scaling":
            # Exp output should be positive
            if param.min() > 0:
                print(f"  → Already exp-activated (all positive)")
            else:
                print(f"  → Raw log-scale (needs exp)")
        elif key == "rotation":
            # Should be normalized quaternions
            norms = param.norm(dim=-1)
            print(f"  norm range: [{norms.min().item():.6f}, {norms.max().item():.6f}]")
            if abs(norms.mean().item() - 1.0) < 0.01:
                print(f"  → Already normalized")
            else:
                print(f"  → Needs normalization")
        print()
'''

# Write and execute
with open('/tmp/check_params.py', 'w') as f:
    f.write(check_script)

subprocess.run(['scp', '/tmp/check_params.py', 'mx:/tmp/'], capture_output=True)
stdout, stderr, rc = ssh_cmd('cd /mnt/storage_pool/liaoyuanjun/higs-13scene-env && source activate higs && python3 /tmp/check_params.py')
print(stdout)
if stderr:
    print("STDERR:", stderr)
