"""
Camera management for 3D Gaussian Splatting benchmark.

Provides camera parameter generation and loading for reproducible rendering
evaluation. Camera poses are stored in a standardized JSON format that
encodes view matrices, projection matrices, and field-of-view parameters
compatible with the 3DGS rasterization pipeline [Kerbl et al., 2023].
"""

import json
import math
import numpy as np
import torch
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple


@dataclass(eq=False)
class Camera:
    """Camera parameters for the 3DGS rasterization pipeline.

    Encodes the view and projection transformations required by the CUDA
    rasterizer, following the world-to-NDC convention used in the original
    3DGS implementation [Kerbl et al., 2023].

    Attributes:
        image_width: Output image width in pixels.
        image_height: Output image height in pixels.
        fov_x: Horizontal field of view in radians.
        fov_y: Vertical field of view in radians.
        viewmatrix: 4x4 world-to-view transformation matrix.
        projmatrix: 4x4 perspective projection matrix.
        camera_center: 3D position of the camera in world space.
        world_view_transform: Transposed view matrix (row-major layout).
        full_proj_transform: Transposed view-projection product.
        tanfovx: Tangent of half the horizontal field of view.
        tanfovy: Tangent of half the vertical field of view.
    """
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
    K: torch.Tensor
    image_name: Optional[str] = None
    reference_crop: Optional[Tuple[int, int, int, int]] = None


def resize_cameras(
    cameras: List[Camera], image_width: int, image_height: int
) -> List[Camera]:
    """Resize camera outputs while preserving poses and horizontal/vertical FOV."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Camera dimensions must be positive")
    resized = []
    for camera in cameras:
        K = camera.K.clone()
        K[0, 0] = image_width / (2.0 * camera.tanfovx)
        K[0, 2] = image_width / 2.0
        K[1, 1] = image_height / (2.0 * camera.tanfovy)
        K[1, 2] = image_height / 2.0
        resized.append(
            replace(
                camera,
                image_width=image_width,
                image_height=image_height,
                K=K,
            )
        )
    return resized


def _camera_from_3dgs_json(cd: dict, device: str) -> Camera:
    """Convert one camera exported by graphdeco-inria/gaussian-splatting."""
    W, H = int(cd["width"]), int(cd["height"])
    fx, fy = float(cd["fx"]), float(cd["fy"])
    camera_to_world = np.eye(4, dtype=np.float32)
    camera_to_world[:3, :3] = np.asarray(cd["rotation"], dtype=np.float32)
    camera_to_world[:3, 3] = np.asarray(cd["position"], dtype=np.float32)
    viewmatrix = torch.tensor(
        np.linalg.inv(camera_to_world).astype(np.float32),
        dtype=torch.float32,
        device=device,
    )
    cam_pos = torch.tensor(cd["position"], dtype=torch.float32, device=device)
    tan_fov_x, tan_fov_y = W / (2.0 * fx), H / (2.0 * fy)
    near, far = 0.01, 100.0
    projmatrix = torch.tensor([
        [1.0 / tan_fov_x, 0, 0, 0],
        [0, 1.0 / tan_fov_y, 0, 0],
        [0, 0, far / (far - near), -far * near / (far - near)],
        [0, 0, 1, 0],
    ], dtype=torch.float32, device=device)
    return Camera(
        image_width=W,
        image_height=H,
        fov_x=2.0 * math.atan(tan_fov_x),
        fov_y=2.0 * math.atan(tan_fov_y),
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        camera_center=cam_pos,
        world_view_transform=viewmatrix.T.contiguous(),
        full_proj_transform=(projmatrix @ viewmatrix).T.contiguous(),
        tanfovx=tan_fov_x,
        tanfovy=tan_fov_y,
        K=torch.tensor(
            [[fx, 0, W / 2.0], [0, fy, H / 2.0], [0, 0, 1]],
            dtype=torch.float32,
            device=device,
        ),
        image_name=cd.get("img_name"),
        reference_crop=tuple(cd["reference_crop"]) if cd.get("reference_crop") else None,
    )


def validate_cameras_facing_point(
    cameras: List[Camera], point: torch.Tensor
) -> List[float]:
    """Return camera-space depths and reject paths facing away from a scene point."""
    depths = []
    for camera in cameras:
        homogeneous = torch.cat(
            [point.to(camera.viewmatrix), point.new_ones(1).to(camera.viewmatrix)]
        )
        depths.append(float((camera.viewmatrix @ homogeneous)[2].item()))
    invalid = [depth for depth in depths if depth <= 0.0]
    if invalid:
        raise ValueError(
            f"{len(invalid)}/{len(depths)} cameras place the scene center behind "
            "the camera (z <= 0); regenerate the path with +Z facing forward"
        )
    return depths


def load_cameras_from_json(path: str, device: str = "cuda") -> List[Camera]:
    """Load camera poses from a standardized JSON file.

    This ensures reproducible benchmarks across runs and renderers by
    using a fixed set of camera poses stored in a JSON format.

    Args:
        path: Path to the cameras.json file.
        device: Target device for tensor allocation ('cuda' or 'cpu').

    Returns:
        List of Camera objects, one per viewpoint.
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return [_camera_from_3dgs_json(cd, device) for cd in data]

    near, far = 0.01, 100.0
    cameras = []
    for cd in data["cameras"]:
        W, H = cd["image_width"], cd["image_height"]
        tan_fov_x = cd["tanfovx"]
        tan_fov_y = cd["tanfovy"]

        viewmatrix = torch.tensor(cd["viewmatrix"], dtype=torch.float32, device=device)
        cam_pos = torch.tensor(cd["camera_center"], dtype=torch.float32, device=device)

        # Build perspective projection matrix from intrinsic parameters
        p00, p11 = 1.0 / tan_fov_x, 1.0 / tan_fov_y
        p22, p23 = far / (far - near), -far * near / (far - near)
        projmatrix = torch.tensor([
            [p00, 0, 0, 0],
            [0, p11, 0, 0],
            [0, 0, p22, p23],
            [0, 0, 1, 0],
        ], dtype=torch.float32, device=device)

        full_proj = (projmatrix @ viewmatrix).T
        K = torch.tensor([
            [W / (2.0 * tan_fov_x), 0, W / 2.0],
            [0, H / (2.0 * tan_fov_y), H / 2.0],
            [0, 0, 1],
        ], dtype=torch.float32, device=device)

        cameras.append(Camera(
            image_width=W, image_height=H,
            fov_x=cd["fov_x_rad"], fov_y=cd["fov_y_rad"],
            viewmatrix=viewmatrix, projmatrix=projmatrix,
            camera_center=cam_pos,
            world_view_transform=viewmatrix.T.contiguous(),
            full_proj_transform=full_proj.contiguous(),
            tanfovx=tan_fov_x, tanfovy=tan_fov_y,
            K=K,
            image_name=cd.get("image_name") or cd.get("img_name"),
            reference_crop=tuple(cd["reference_crop"]) if cd.get("reference_crop") else None,
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
    """Generate camera poses orbiting around the scene origin.

    Cameras are placed on a perturbed circular orbit with sinusoidal
    variations in elevation and radius to simulate realistic viewing
    trajectories. All computation is performed on CPU, with tensors
    allocated on the target device.

    Args:
        num_cameras: Number of camera poses to generate.
        image_width: Output image width in pixels.
        image_height: Output image height in pixels.
        fov_deg: Horizontal field of view in degrees.
        scene_radius: Base orbit radius in scene units.
        device: Target device for tensor allocation.

    Returns:
        List of Camera objects.
    """
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
        zx, zy, zz = -cx/dist, -cy/dist, -cz/dist

        upx, upy, upz = 0.0, 0.0, 1.0
        xx = zy * upz - zz * upy
        xy = zz * upx - zx * upz
        xz = zx * upy - zy * upx
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

        full_proj = (projmatrix @ viewmatrix).T
        K = torch.tensor([
            [image_width / (2.0 * tan_fov_x), 0, image_width / 2.0],
            [0, image_height / (2.0 * tan_fov_y), image_height / 2.0],
            [0, 0, 1],
        ], dtype=torch.float32, device=device)

        cameras.append(Camera(
            image_width=image_width, image_height=image_height,
            fov_x=fov, fov_y=fov_y,
            viewmatrix=viewmatrix, projmatrix=projmatrix,
            camera_center=cam_pos,
            world_view_transform=viewmatrix.T.contiguous(),
            full_proj_transform=full_proj.contiguous(),
            tanfovx=tan_fov_x, tanfovy=tan_fov_y,
            K=K,
        ))

    return cameras
