"""Check camera structure on A100 to understand format."""
import json, os, sys
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(repo, "src"))

from benchmark_framework import load_cameras_from_json

cam_path = os.path.join(repo, "data", "official", "mipnerf360", "room", "cameras.json")
cams = load_cameras_from_json(cam_path, device="cpu")
c = cams[0]
print(f"  Type: {type(c)}")
print(f"  Attributes: {[a for a in dir(c) if not a.startswith('_')]}")
print(f"  image_height: {c.image_height}")
print(f"  image_width: {c.image_width}")
print(f"  tanfovx: {c.tanfovx}")
print(f"  tanfovy: {c.tanfovy}")
print(f"  camera_center: {c.camera_center}")
print(f"  world_view_transform shape: {c.world_view_transform.shape}")
print(f"  full_proj_transform shape: {c.full_proj_transform.shape}")

# Compute 3x3 intrinsics from fov
fx = c.image_width / (2.0 * c.tanfovx)
fy = c.image_height / (2.0 * c.tanfovy)
print(f"  fx={fx}, fy={fy}")
print(f"  cx={c.image_width / 2}, cy={c.image_height / 2}")

# Check if there's a direct K matrix
if hasattr(c, 'intrinsics'):
    print(f"  intrinsics: {c.intrinsics}")
print(f"  viewmatrix: {c.viewmatrix}")
