import sys, os

# Check what dataset.py contains
f="/home/liaoyuanjun/3dgs-renderer-benchmark/scripts/epic05/phase7/dataset.py"
with open(f) as fh:
    content = fh.read()
print(f"Dataset file length: {len(content)} bytes")
print(f"Contains GTDataset: {'GTDataset' in content}")

# Also check gaussian_model.py imports to see what from src/ it needs
f2="/home/liaoyuanjun/3dgs-renderer-benchmark/baseline/reference_v1/gaussian_model.py"
with open(f2) as fh:
    for line in fh:
        if line.startswith(("import", "from")):
            print(f"gaussian_model import: {line.rstrip()}")
