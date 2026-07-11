"""
Camera generation for benchmarking.
"""
import json
import math
import numpy as np
import torch
from dataclasses import dataclass
from typing import List


@dataclass
class Camera:
    """Camera parameters for rendering."""
    image_width: int
    image_height: int
    fov_x: float
    fov_y: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    camera_center: torch.Tensor
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor
    tanfovx: float
    tanfovy: float


def load_cameras_from_json(path: str, device: str = "cuda") -> List[Camera]:
    """Load fixed camera poses from cameras.json.
    
    This ensures reproducible benchmarks across runs and renderers.
    """
    with open(path) as f:
        data = json.load(f)
    
    cameras = []
    for cd in data["cameras"]:
        W, H = cd["image_width"], cd["image_height"]
        tan_fov_x = cd["tanfovx"]
        tan_fov_y = cd["tanfovy"]
        fov_x = cd["fov_x_rad"]
        fov_y = cd["fov_y_rad"]
        
        viewmatrix = torch.tensor(cd["viewmatrix"], dtype=torch.float32, device=device)
        cam_pos = torch.tensor(cd["camera_center"], dtype=torch.float32, device=device)
        
        # Projection matrix
        near, far = 0.01, 100.0
        projmatrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
        projmatrix[0, 0] = 1.0 / tan_fov_x
        projmatrix[1, 1] = 1.0 / tan_fov_y
        projmatrix[2, 2] = far / (far - near)
        projmatrix[2, 3] = -far * near / (far - near)
        projmatrix[3, 2] = 1.0
        
        full_proj = (viewmatrix @ projmatrix).T
        
        cam = Camera(
            image_width=W,
            image_height=H,
            fov_x=fov_x,
            fov_y=fov_y,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            camera_center=cam_pos,
            world_view_transform=viewmatrix.T.contiguous(),
            full_proj_transform=full_proj.contiguous(),
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
        )
        cameras.append(cam)
    
    return cameras


def generate_cameras(
    num_cameras: int,
    image_width: int = 1920,
    image_height: int = 1080,
    fov_deg: float = 60.0,
    scene_radius: float = 5.0,
    device: str = "cuda"
) -> List[Camera]:
    """Generate camera poses orbiting around the origin (for backward compat)."""
    fov = math.radians(fov_deg)
    tan_fov = math.tan(fov * 0.5)
    aspect = image_width / image_height
    fov_y = 2 * math.atan(tan_fov / aspect)
    tan_fov_y = math.tan(fov_y * 0.5)
    tan_fov_x = tan_fov
    
    cameras = []
    for i in range(num_cameras):
        theta = 2 * math.pi * i / num_cameras
        phi = math.radians(15.0 * math.sin(theta * 2))
        radius = scene_radius * 0.8 + 0.4 * math.sin(theta * 3) * 0.5
        
        cam_pos = torch.tensor([
            radius * math.cos(theta) * math.cos(phi),
            radius * math.sin(theta) * math.cos(phi),
            radius * math.sin(phi) + 0.5
        ], dtype=torch.float32, device=device)
        
        look_at = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
        
        z_axis = (cam_pos - look_at) / torch.norm(cam_pos - look_at)
        x_axis = torch.linalg.cross(up, z_axis)
        x_axis = x_axis / torch.norm(x_axis)
        y_axis = torch.linalg.cross(z_axis, x_axis)
        
        viewmatrix = torch.eye(4, dtype=torch.float32, device=device)
        viewmatrix[0, :3] = x_axis
        viewmatrix[1, :3] = y_axis
        viewmatrix[2, :3] = z_axis
        viewmatrix[:3, 3] = -viewmatrix[:3, :3] @ cam_pos
        
        near, far = 0.01, 100.0
        projmatrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
        projmatrix[0, 0] = 1.0 / tan_fov_x
        projmatrix[1, 1] = 1.0 / tan_fov_y
        projmatrix[2, 2] = far / (far - near)
        projmatrix[2, 3] = -far * near / (far - near)
        projmatrix[3, 2] = 1.0
        
        full_proj = (viewmatrix @ projmatrix).T
        
        cam = Camera(
            image_width=image_width,
            image_height=image_height,
            fov_x=fov,
            fov_y=fov_y,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            camera_center=cam_pos,
            world_view_transform=viewmatrix.T.contiguous(),
            full_proj_transform=full_proj.contiguous(),
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
        )
        cameras.append(cam)
    
    return cameras
