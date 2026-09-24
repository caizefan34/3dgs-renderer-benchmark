#!/usr/bin/env python3
"""Quick check what fields are in the official PLY files and whether they work with rasterization."""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark_framework import load_ply

for scene in ["bicycle", "garden", "room"]:
    ply_path = REPO_ROOT / "data" / "official" / "mipnerf360" / scene / "point_cloud.ply"
    if not ply_path.exists():
        print(f"{scene}: file not found")
        continue
    size_mb = ply_path.stat().st_size / (1024*1024)
    print(f"\n=== {scene} ({size_mb:.0f} MB) ===")
    
    try:
        data = load_ply(str(ply_path), device="cpu")
        for k, v in data.items():
            if isinstance(v, str):
                print(f"  {k}: {v}")
            elif hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}, min={v.min():.4f}, max={v.max():.4f}")
            else:
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
