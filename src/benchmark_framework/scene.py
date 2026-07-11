"""
Benchmark framework core.

scene.py: Load and manage 3DGS scene data from .ply files.
"""

import numpy as np
import torch
import struct
import os
from typing import Tuple, Optional


def load_ply(path: str, device: str = "cuda") -> dict:
    """Load a 3DGS .ply file and return a dict of tensors.
    
    Expected format: position (x,y,z), opacity, scale_0..2, rot_0..3, f_dc_0..47 (SH degree 3)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PLY file not found: {path}")
    
    with open(path, "rb") as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break
            header_lines.append(line)
        
        # Parse element count
        num_points = 0
        for line in header_lines:
            if line.startswith("element vertex"):
                num_points = int(line.split()[-1])
        
        if num_points == 0:
            raise ValueError("No vertices found in PLY file")
        
        # Detect properties
        props = []
        for line in header_lines:
            if line.startswith("property"):
                parts = line.split()
                dtype_str = parts[1]
                name = parts[2]
                props.append((name, dtype_str))
        
        print(f"  Loading {num_points} Gaussians, {len(props)} properties from {path}")
        
        # Read binary data while file is still open
        data = f.read()
    
    # Calculate stride
    fmt = "<"
    prop_names = []
    for name, dtype_str in props:
        if dtype_str in ("float", "float32"):
            fmt += "f"
        elif dtype_str in ("double", "float64"):
            fmt += "d"
        elif dtype_str in ("int", "int32"):
            fmt += "i"
        elif dtype_str in ("uchar", "uint8"):
            fmt += "B"
        else:
            fmt += "f"
        prop_names.append(name)
    
    stride = struct.calcsize(fmt)
    num_vertices = len(data) // stride
    
    if num_vertices != num_points:
        print(f"  Warning: header says {num_points}, data has {num_vertices}")
    
    # Parse all vertices into dict of arrays
    arrays = {}
    for name in prop_names:
        arrays[name] = np.zeros(num_vertices, dtype=np.float32)
    
    for i in range(num_vertices):
        offset = i * stride
        vals = struct.unpack_from(fmt, data, offset)
        for j, name in enumerate(prop_names):
            arrays[name][i] = vals[j]
    
    # Convert to named tensors
    xyz = np.column_stack([arrays["x"], arrays["y"], arrays["z"]]).astype(np.float32)
    opacity = arrays["opacity"].astype(np.float32)
    
    scales = np.column_stack([arrays["scale_0"], arrays["scale_1"], arrays["scale_2"]]).astype(np.float32)
    rotations = np.column_stack([arrays["rot_0"], arrays["rot_1"], arrays["rot_2"], arrays["rot_3"]]).astype(np.float32)
    
    sh_names = [n for n in prop_names if n.startswith("f_dc_")]
    if sh_names:
        shs = np.column_stack([arrays[n] for n in sh_names]).astype(np.float32)
    else:
        shs = None
    
    # Move to GPU
    result = {
        "xyz": torch.from_numpy(xyz).to(device),
        "opacity": torch.from_numpy(opacity).to(device),
        "scales": torch.from_numpy(scales).to(device),
        "rotations": torch.from_numpy(rotations).to(device),
    }
    if shs is not None:
        result["shs"] = torch.from_numpy(shs).to(device)
        # DC colors
        C0 = 0.28209479177387814
        result["dc_colors"] = torch.from_numpy(shs[:, :3] * C0 + 0.5).clamp(0, 1).to(device)
    
    result["num_points"] = num_points
    
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Loaded {num_points} Gaussians with {shs.shape[1] if shs is not None else 0} SH coefficients")
    
    return result


def compute_cov3d_from_scales_rot(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """Convert scale+rotation to 3D covariance matrix (6 components)."""
    N = scales.shape[0]
    device = scales.device
    
    q = torch.nn.functional.normalize(rotations, dim=-1)
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    R = torch.zeros(N, 3, 3, device=device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    
    S = torch.diag_embed(torch.exp(scales))
    M = R @ S
    cov3d = M @ M.transpose(-2, -1)
    
    idx = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]], device=device)
    cov6 = cov3d[:, idx[:, 0], idx[:, 1]]
    
    return cov6
