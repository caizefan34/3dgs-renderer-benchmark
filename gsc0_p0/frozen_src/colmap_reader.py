"""
COLMAP points3D.bin reader for SfM initialization.

Reads the binary format used by COLMAP sparse reconstruction.
"""

import numpy as np
import struct
import torch
from typing import Dict


def read_points3D_binary(path: str) -> Dict:
    """Read COLMAP points3D.bin file.

    Returns dict with:
        xyz: [N, 3] float32
        rgb: [N, 3] float32 (0-1)
        point_ids: [N] int64
    """
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]

        xyz = np.zeros((num_points, 3), dtype=np.float32)
        rgb = np.zeros((num_points, 3), dtype=np.float32)
        point_ids = np.zeros(num_points, dtype=np.uint64)

        for i in range(num_points):
            point_id = struct.unpack("<Q", f.read(8))[0]
            x, y, z = struct.unpack("<ddd", f.read(24))
            r, g, b = struct.unpack("<BBB", f.read(3))
            error = struct.unpack("<d", f.read(8))

            # Read track
            track_len = struct.unpack("<Q", f.read(8))[0]
            for _ in range(track_len):
                f.read(8)  # image_id (int32) + point2D_idx (int32)

            xyz[i] = [x, y, z]
            rgb[i] = [r / 255.0, g / 255.0, b / 255.0]
            point_ids[i] = point_id

    return {
        "xyz": torch.from_numpy(xyz),
        "rgb": torch.from_numpy(rgb),
        "point_ids": torch.from_numpy(point_ids),
        "num_points": num_points,
    }


def sfm_to_pcd_data(points3d: Dict, sh_degree: int = 3) -> Dict:
    """Convert SfM points to pcd_data dict for GaussianModel.create_from_pcd.

    Follows official 3DGS create_from_pcd:
        - SH: RGB2SH(color), zero for higher orders
        - Opacity: inverse_sigmoid(0.1)
        - Scale: log(sqrt(KNN_distance)) repeated 3x
        - Rotation: identity quaternion
    """
    N = points3d["num_points"]
    xyz = points3d["xyz"]
    rgb = points3d["rgb"]

    # SH: DC coefficient = RGB2SH(color), rest = 0
    C0 = 0.28209479177387814
    shs = torch.zeros(N, (sh_degree + 1) ** 2, 3)
    shs[:, 0, :] = (rgb - 0.5) / C0  # RGB2SH

    # Opacity: inverse_sigmoid(0.1) = logit(0.1)
    opacity = torch.full((N,), float(np.log(0.1 / (1 - 0.1))))

    # Scale and rotation: let GaussianModel handle defaults
    # (it computes KNN distance for scale, identity for rotation)

    return {
        "xyz": xyz,
        "shs": shs,
        "opacity": opacity,
        "scales": None,  # Let model compute KNN-based scale
        "rotations": None,  # Identity quaternion
    }
