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
    """Load fixed camera poses from cameras.json (optimized tensor creation).
    
    This ensures reproducible benchmarks across runs and renderers.
    """
    with open(path) as f:
        data = json.load(f)
    
    near, far = 0.01, 100.0
    cameras = []
    for cd in data["cameras"]:
        W, H = cd["image_width"], cd["image_height"]
        tan_fov_x = cd["tanfovx"]
        tan_fov_y = cd["tanfovy"]
        
        viewmatrix = torch.tensor(cd["viewmatrix"], dtype=torch.float32, device=device)
        cam_pos = torch.tensor(cd["camera_center"], dtype=torch.float32, device=device)
        
        # Build projection matrix inline (avoid torch.zeros + 4 assignments)
        p00, p11 = 1.0 / tan_fov_x, 1.0 / tan_fov_y
        p22, p23 = far / (far - near), -far * near / (far - near)
        projmatrix = torch.tensor([
            [p00, 0, 0, 0],
            [0, p11, 0, 0],
            [0, 0, p22, p23],
            [0, 0, 1, 0],
        ], dtype=torch.float32, device=device)
        
        full_proj = (viewmatrix @ projmatrix).T
        
        cameras.append(Camera(
            image_width=W, image_height=H,
            fov_x=cd["fov_x_rad"], fov_y=cd["fov_y_rad"],
            viewmatrix=viewmatrix, projmatrix=projmatrix,
            camera_center=cam_pos,
            world_view_transform=viewmatrix.T.contiguous(),
            full_proj_transform=full_proj.contiguous(),
            tanfovx=tan_fov_x, tanfovy=tan_fov_y,
        ))
    
    return cameras

def generate_cameras(
    num_cameras: int,
    image_width: int = 1920,
    image_height: int = 1080,
    fov_deg: float = 60.0,
    scene_radius: float = 5.0,
    device: str = "cuda"
) -> List[Camera]:
    """Generate camera poses orbiting around the origin (CPU math, GPU tensors once)."""
    fov = math.radians(fov_deg)
    tan_fov = math.tan(fov * 0.5)
    aspect = image_width / image_height
    fov_y = 2 * math.atan(tan_fov / aspect)
    tan_fov_y = math.tan(fov_y * 0.5)
    tan_fov_x = tan_fov
    near, far = 0.01, 100.0
    proj_00 = 1.0 / tan_fov_x
    proj_11 = 1.0 / tan_fov_y
    proj_22 = far / (far - near)
    proj_23 = -far * near / (far - near)

    cameras = []
    for i in range(num_cameras):
        theta = 2 * math.pi * i / num_cameras
        phi = math.radians(15.0 * math.sin(theta * 2))
        radius = scene_radius * 0.8 + 0.4 * math.sin(theta * 3) * 0.5

        cx = radius * math.cos(theta) * math.cos(phi)
        cy = radius * math.sin(theta) * math.cos(phi)
        cz = radius * math.sin(phi) + 0.5

        dist = math.sqrt(cx*cx + cy*cy + cz*cz)
        zx, zy, zz = cx/dist, cy/dist, cz/dist

        upx, upy, upz = 0.0, 0.0, 1.0
        xx = upy * zz - upz * zy
        xy = upz * zx - upx * zz
        xz = upx * zy - upy * zx
        xnorm = math.sqrt(xx*xx + xy*xy + xz*xz)
        xx /= xnorm; xy /= xnorm; xz /= xnorm

        yx = zy * xz - zz * xy
        yy = zz * xx - zx * xz
        yz = zx * xy - zy * xx

        tx = -(xx * cx + xy * cy + xz * cz)
        ty = -(yx * cx + yy * cy + yz * cz)
        tz = -(zx * cx + zy * cy + zz * cz)

        viewmatrix = torch.tensor([
            [xx, xy, xz, tx],
            [yx, yy, yz, ty],
            [zx, zy, zz, tz],
            [0, 0, 0, 1],
        ], dtype=torch.float32, device=device)

        cam_pos = torch.tensor([cx, cy, cz], dtype=torch.float32, device=device)

        projmatrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
        projmatrix[0, 0] = proj_00
        projmatrix[1, 1] = proj_11
        projmatrix[2, 2] = proj_22
        projmatrix[2, 3] = proj_23
        projmatrix[3, 2] = 1.0

        full_proj = (viewmatrix @ projmatrix).T

        cameras.append(Camera(
            image_width=image_width, image_height=image_height,
            fov_x=fov, fov_y=fov_y,
            viewmatrix=viewmatrix, projmatrix=projmatrix,
            camera_center=cam_pos,
            world_view_transform=viewmatrix.T.contiguous(),
            full_proj_transform=full_proj.contiguous(),
            tanfovx=tan_fov_x, tanfovy=tan_fov_y,
        ))

    return cameras
