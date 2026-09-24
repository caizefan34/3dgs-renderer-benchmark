"""Check camera resize functionality to 1080p."""
import json, os, sys, math
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo, "src"))

from benchmark_framework import load_cameras_from_json, resize_cameras

cam_path = os.path.join(repo, "data", "official", "mipnerf360", "room", "cameras.json")
cams = load_cameras_from_json(cam_path, device="cpu")
print(f"Native: {cams[0].image_width}x{cams[0].image_height}")
print(f"K: {cams[0].K}")

# Resize to 1080p
resized = resize_cameras(cams, "1080p")
c = resized[0]
print(f"\nResized 1080p: {c.image_width}x{c.image_height}")
print(f"K: {c.K}")

# Check K shape
k = c.K
print(f"K type: {type(k)}, shape: {k.shape if hasattr(k, 'shape') else 'no shape'}")
if hasattr(k, 'shape'):
    print(f"K value:\n{k}")
