"""
Synthetic 3D Gaussian Splatting scene generator.

Creates a synthetic 3D Gaussian point cloud in PLY format for benchmarking
purposes. The generated scene consists of Gaussian primitives with random
positions, colors, opacities, scales, and rotations, clustered around
randomly placed centers to simulate realistic spatial structure.

Usage:
    python src/scripts/generate_scene.py
    python src/scripts/generate_scene.py --gaussians 200000
    python src/scripts/generate_scene.py --gaussians 500000 --output data/large_scene.ply

References:
    Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023).
    3D Gaussian Splatting for Real-Time Radiance Field Rendering.
    ACM Transactions on Graphics, 42(4).
"""
import numpy as np
import struct
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SH_DEGREE = 3
NUM_SH_COEFFS = (SH_DEGREE + 1) ** 2
SH_CHANNELS = 3
SH_DATA_SIZE = NUM_SH_COEFFS * SH_CHANNELS

def create_scene_ply(path: str, num_gaussians: int):
    """Create a synthetic 3DGS scene and write to a PLY file.

    Generates Gaussians with spherical coordinates perturbed by 20 random
    cluster centers, random RGB colors encoded as SH DC coefficients, and
    randomized opacity, scale, and rotation parameters.

    Args:
        path: Output file path for the PLY file.
        num_gaussians: Number of Gaussian primitives to generate.
    """
    np.random.seed(42)

    # Generate base positions in spherical coordinates
    theta = np.random.uniform(0, np.pi, num_gaussians)
    phi = np.random.uniform(0, 2 * np.pi, num_gaussians)
    r = np.sqrt(np.random.uniform(0.1, 3.0, num_gaussians))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # Add 20 random cluster centers for spatial structure
    for _ in range(20):
        cx, cy, cz = np.random.uniform(-2, 2, 3)
        idx = np.random.choice(num_gaussians, num_gaussians // 20, replace=False)
        x[idx] += np.random.randn(num_gaussians // 20) * 0.3 + cx
        y[idx] += np.random.randn(num_gaussians // 20) * 0.3 + cy
        z[idx] += np.random.randn(num_gaussians // 20) * 0.3 + cz

    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    rgb = np.random.uniform(0.0, 1.0, (num_gaussians, 3)).astype(np.float32)

    # Encode RGB as SH DC coefficients: sh_dc = (color - 0.5) / C0
    shs = np.zeros((num_gaussians, SH_DATA_SIZE), dtype=np.float32)
    C0 = 0.28209479177387814
    shs[:, 0] = (rgb[:, 0] - 0.5) / C0
    shs[:, 1] = (rgb[:, 1] - 0.5) / C0
    shs[:, 2] = (rgb[:, 2] - 0.5) / C0
    shs[:, 3:] = np.random.randn(num_gaussians, SH_DATA_SIZE - 3).astype(np.float32) * 0.01

    opacities = np.random.randn(num_gaussians).astype(np.float32) * 0.5 - 1.0
    scales = np.random.randn(num_gaussians, 3).astype(np.float32) * 0.3 - 1.5
    rotations = np.random.randn(num_gaussians, 4).astype(np.float32)
    rotations = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)

    _write_gsplat_ply(path, xyz, opacities, scales, rotations, shs)
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Created {path} ({num_gaussians} Gaussians, {file_size_mb:.1f} MB)")

def _write_gsplat_ply(path, xyz, opacities, scales, rotations, shs):
    """Write Gaussian data to a binary PLY file in 3DGS format.

    Args:
        path: Output file path.
        xyz: (N, 3) float32 positions.
        opacities: (N,) float32 opacity values.
        scales: (N, 3) float32 scale parameters.
        rotations: (N, 4) float32 rotation quaternions.
        shs: (N, SH_DATA_SIZE) float32 spherical harmonic coefficients.
    """
    N = xyz.shape[0]
    header_lines = ["ply", "format binary_little_endian 1.0"]
    header_lines += [
        f"element vertex {N}",
        "property float x", "property float y", "property float z",
        "property float opacity",
        "property float scale_0", "property float scale_1", "property float scale_2",
        "property float rot_0", "property float rot_1", "property float rot_2", "property float rot_3",
    ]
    for i in range(3):
        header_lines.append(f"property float f_dc_{i}")
    for i in range(SH_DATA_SIZE - 3):
        header_lines.append(f"property float f_rest_{i}")
    header_lines.append("end_header\n")
    header = "\n".join(header_lines)

    buf = bytearray()
    for i in range(N):
        buf += struct.pack("fff", xyz[i, 0], xyz[i, 1], xyz[i, 2])
        buf += struct.pack("f", opacities[i])
        buf += struct.pack("fff", scales[i, 0], scales[i, 1], scales[i, 2])
        buf += struct.pack("ffff", rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3])
        for j in range(SH_DATA_SIZE):
            buf += struct.pack("f", shs[i, j])

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(bytes(buf))

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic 3DGS scene")
    parser.add_argument("--gaussians", type=int, default=None, help="Number of Gaussians")
    parser.add_argument("--output", type=str, default=None, help="Output path for PLY file")
    args, extra = parser.parse_known_args()
    # Support positional argument for backward compatibility
    n = args.gaussians if args.gaussians is not None else (int(extra[0]) if extra else 400000)
    path = args.output if args.output else os.path.join(OUTPUT_DIR, "scene.ply")
    create_scene_ply(path, n)
    print(f"Generated {n} Gaussians -> {path}")
