"""
Scene generator: creates a synthetic 3D Gaussian point cloud for benchmarking.
Produces a .ply file with configurable Gaussian count.
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
    """Create a synthetic 3DGS .ply file."""
    np.random.seed(42)

    theta = np.random.uniform(0, np.pi, num_gaussians)
    phi = np.random.uniform(0, 2 * np.pi, num_gaussians)
    r = np.sqrt(np.random.uniform(0.1, 3.0, num_gaussians))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    for _ in range(20):
        cx, cy, cz = np.random.uniform(-2, 2, 3)
        idx = np.random.choice(num_gaussians, num_gaussians // 20, replace=False)
        x[idx] += np.random.randn(num_gaussians // 20) * 0.3 + cx
        y[idx] += np.random.randn(num_gaussians // 20) * 0.3 + cy
        z[idx] += np.random.randn(num_gaussians // 20) * 0.3 + cz

    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    rgb = np.random.uniform(0.0, 1.0, (num_gaussians, 3)).astype(np.float32)
    
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
    N = xyz.shape[0]
    header_lines = ["ply", "format binary_little_endian 1.0"]
    header_lines += [
        f"element vertex {N}",
        "property float x", "property float y", "property float z",
        "property float opacity",
        "property float scale_0", "property float scale_1", "property float scale_2",
        "property float rot_0", "property float rot_1", "property float rot_2", "property float rot_3",
    ]
    for i in range(SH_DATA_SIZE):
        header_lines.append(f"property float f_dc_{i}")
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
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400000
    path = os.path.join(OUTPUT_DIR, "scene.ply")
    create_scene_ply(path, n)
    print("Done!")
