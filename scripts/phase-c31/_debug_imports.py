#!/usr/bin/env python3
"""Debug path imports."""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent.parent
src_path = repo_root / "src"
phase7_path = repo_root / "scripts" / "epic05" / "phase7"

print(f"repo_root: {repo_root}")
print(f"src exists: {src_path.exists()}")
print(f"phase7 exists: {phase7_path.exists()}")

sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(phase7_path))

print(f"\nsys.path[0:5]:")
for i, p in enumerate(sys.path[:5]):
    print(f"  [{i}] {p}")

# Try imports
try:
    import benchmark_framework
    print(f"\nbenchmark_framework: {benchmark_framework.__file__}")
except ImportError as e:
    print(f"\nbenchmark_framework import FAILED: {e}")

try:
    from gsplat import rasterization
    print(f"gsplat: OK")
except ImportError as e:
    print(f"gsplat import FAILED: {e}")

try:
    from gaussian_model import GaussianModel
    print(f"gaussian_model: OK")
except ImportError as e:
    print(f"gaussian_model import FAILED: {e}")

try:
    from dataset import GTDataset
    print(f"dataset: OK")
except ImportError as e:
    print(f"dataset import FAILED: {e}")
