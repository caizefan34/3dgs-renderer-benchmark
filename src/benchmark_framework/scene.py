"""
Scene data management for 3D Gaussian Splatting benchmark.

Provides efficient PLY file loading and 3D covariance matrix computation
for 3DGS scenes. Uses vectorized numpy operations for high-throughput
loading of large Gaussian point clouds.

References:
    Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023).
    3D Gaussian Splatting for Real-Time Radiance Field Rendering.
    ACM Transactions on Graphics, 42(4).
"""

import numpy as np
import torch
import os


def load_ply(path: str, device: str = "cuda") -> dict:
    """Load a 3DGS scene from a PLY file and return a dictionary of GPU tensors.

    The PLY format follows the 3DGS convention [Kerbl et al., 2023] with
    the following properties per Gaussian:
      - position: (x, y, z)
      - opacity: scalar
      - scale: (scale_0, scale_1, scale_2)
      - rotation: (rot_0, rot_1, rot_2, rot_3) quaternion
      - spherical harmonics: f_dc_0 through f_dc_{N-1} (N = 3 * (SH_degree + 1)^2)

    Uses vectorized numpy structured array reading for efficiency
    (~1000x faster than per-element Python loops).

    Args:
        path: Path to the .ply file.
        device: Target device for tensors ('cuda' or 'cpu').

    Returns:
        dict with keys: xyz, opacity, scales, rotations, shs, dc_colors, num_points.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PLY file not found: {path}")

    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break
            header_lines.append(line)

        num_points = 0
        for line in header_lines:
            if line.startswith("element vertex"):
                num_points = int(line.split()[-1])

        if num_points == 0:
            raise ValueError("No vertices found in PLY file")

        props = []
        for line in header_lines:
            if line.startswith("property"):
                parts = line.split()
                props.append((parts[2], parts[1]))

        print(f"  Loading {num_points} Gaussians, {len(props)} properties from {path}")
        data = f.read()

    # Build numpy structured dtype from PLY property specifiers
    np_dtype_map = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
                    "int": "i4", "int32": "i4", "uchar": "u1", "uint8": "u1"}
    dt_list = [(name, np.dtype(np_dtype_map.get(dtype, "f4"))) for name, dtype in props]
    col_names = [name for name, _ in props]

    vertex_dtype = np.dtype(dt_list)
    actual_count = len(data) // vertex_dtype.itemsize

    if actual_count != num_points:
        print(f"  Warning: header declares {num_points} vertices, data contains {actual_count}")

    # Vectorized read: single bulk numpy call replaces per-element loop
    records = np.frombuffer(data, dtype=vertex_dtype, count=actual_count)

    xyz = np.column_stack([records["x"], records["y"], records["z"]]).astype(np.float32)
    opacity = records["opacity"].astype(np.float32)
    scales = np.column_stack([records["scale_0"], records["scale_1"], records["scale_2"]]).astype(np.float32)
    rotations = np.column_stack([records["rot_0"], records["rot_1"], records["rot_2"], records["rot_3"]]).astype(np.float32)

    sh_cols = [n for n in col_names if n.startswith("f_dc_")]
    shs = np.column_stack([records[n] for n in sh_cols]).astype(np.float32) if sh_cols else None

    file_size_mb = os.path.getsize(path) / (1024 * 1024)

    result = {
        "xyz": torch.from_numpy(xyz).to(device),
        "opacity": torch.from_numpy(opacity).to(device),
        "scales": torch.from_numpy(scales).to(device),
        "rotations": torch.from_numpy(rotations).to(device),
    }
    if shs is not None:
        result["shs"] = torch.from_numpy(shs).to(device)
        C0 = 0.28209479177387814
        result["dc_colors"] = torch.from_numpy(shs[:, :3] * C0 + 0.5).clamp(0, 1).to(device)
    result["num_points"] = num_points

    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Loaded {num_points} Gaussians with {shs.shape[1] if shs is not None else 0} SH coefficients")

    return result


def compute_cov3d_from_scales_rot(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """Convert scale vectors and rotation quaternions to 3D covariance matrices.

    Following the 3DGS formulation [Kerbl et al., 2023], the covariance matrix
    Σ = R S S^T R^T is computed from the rotation matrix R (derived from the
    quaternion) and the diagonal scaling matrix S = diag(exp(s)). The upper
    triangular elements of the symmetric 3x3 covariance matrix are returned
    as a 6-component vector per Gaussian.

    Args:
        scales: Tensor of shape (N, 3) containing log-scale parameters.
        rotations: Tensor of shape (N, 4) containing rotation quaternions.

    Returns:
        Tensor of shape (N, 6) with upper-triangular covariance elements.
    """
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