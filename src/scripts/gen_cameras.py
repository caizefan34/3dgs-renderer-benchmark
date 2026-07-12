"""Generate a fixed set of camera poses and save as cameras.json."""
import os
import json, math
import torch
import numpy as np

OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cameras.json")

num_cameras = 50
image_width, image_height = 1920, 1080
fov_deg = 60.0
device = "cpu"

fov = math.radians(fov_deg)
aspect = image_width / image_height
fov_y = 2 * math.atan(math.tan(fov * 0.5) / aspect)
tan_fov_x = math.tan(fov * 0.5)
tan_fov_y = math.tan(fov_y * 0.5)

cameras_list = []
for i in range(num_cameras):
    theta = 2 * math.pi * i / num_cameras
    phi = math.radians(15.0 * math.sin(theta * 2))
    radius = 5.0 * 0.8 + 0.4 * math.sin(theta * 3) * 0.5
    
    cam_pos = np.array([
        radius * math.cos(theta) * math.cos(phi),
        radius * math.sin(theta) * math.cos(phi),
        radius * math.sin(phi) + 0.5
    ], dtype=np.float32)
    
    look_at = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    
    z_axis = (cam_pos - look_at) / np.linalg.norm(cam_pos - look_at)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    viewmatrix = np.eye(4, dtype=np.float32)
    viewmatrix[0, :3] = x_axis
    viewmatrix[1, :3] = y_axis
    viewmatrix[2, :3] = z_axis
    viewmatrix[:3, 3] = -viewmatrix[:3, :3] @ cam_pos
    
    fx = 0.5 * image_width / tan_fov_x
    fy = 0.5 * image_height / tan_fov_y
    
    cam_dict = {
        "id": i,
        "image_width": image_width,
        "image_height": image_height,
        "fov_x_rad": round(fov, 6),
        "fov_y_rad": round(fov_y, 6),
        "tanfovx": round(tan_fov_x, 6),
        "tanfovy": round(tan_fov_y, 6),
        "camera_center": cam_pos.tolist(),
        "viewmatrix": viewmatrix.tolist(),
        "fx": round(fx, 4),
        "fy": round(fy, 4),
        "cx": image_width / 2,
        "cy": image_height / 2,
    }
    cameras_list.append(cam_dict)

with open(OUTPUT, "w") as f:
    json.dump({"cameras": cameras_list, "metadata": {
        "description": "Fixed 50-camera orbit for 3DGS renderer benchmarking",
        "scene_center": [0, 0, 0],
        "orbit_radius_variation": [0.8, 5.0, 0.4],
        "num_cameras": num_cameras,
        "image_size": f"{image_width}x{image_height}",
        "fov_degrees": fov_deg,
    }}, f, indent=2)

print(f"Saved {num_cameras} camera poses to {OUTPUT}")
import os
print(f"File size: {os.path.getsize(OUTPUT)/1024:.1f} KB")



