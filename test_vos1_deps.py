#!/usr/bin/env python3
import sys
sys.path.insert(0, "src")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
except Exception as e:
    print(f"PyTorch: FAIL - {e}")

try:
    import lpips
    print("LPIPS: OK")
except Exception as e:
    print(f"LPIPS: FAIL - {e}")

try:
    from scene import Scene
    print("Scene: OK")
except Exception as e:
    print(f"Scene: FAIL - {e}")

try:
    from gaussian_renderer import render
    print("Render: OK")
except Exception as e:
    print(f"Render: FAIL - {e}")
