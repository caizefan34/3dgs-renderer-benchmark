"""
Standard camera path preset generator for 3DGS renderer benchmarking.

Generates canonical camera trajectories (spiral, circular, flythrough,
random walk) and saves them as JSON files for reproducible evaluation.

Usage:
    python generate_camera_path.py --type spiral --num-cameras 60 --radius 10.0
    python generate_camera_path.py --type circle --num-cameras 50 --radius 8.0
    python generate_camera_path.py --type flythrough --num-cameras 30
    python generate_camera_path.py --type random_walk --num-cameras 30 --seed 42

References:
    Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023).
    3D Gaussian Splatting for Real-Time Radiance Field Rendering.
    ACM Transactions on Graphics, 42(4).
"""
import os, sys, json, math, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from benchmark_framework.cameras import Camera

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "camera_presets")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_camera_dict(theta, phi, radius, cx=0.0, cy=0.0, cz=0.0,
                      W=1920, H=1080, fov_deg=60.0):
    """Build a camera dictionary matching the cameras.json schema.

    Args:
        theta: Azimuthal angle in radians.
        phi: Elevation angle in radians.
        radius: Distance from the origin.
        cx, cy, cz: Look-at target in world coordinates.
        W, H: Image dimensions in pixels.
        fov_deg: Horizontal field of view in degrees.

    Returns:
        Dictionary with camera parameters compatible with the benchmarking pipeline.
    """
    fov = math.radians(fov_deg)
    aspect = W / H
    fov_y = 2 * math.atan(math.tan(fov * 0.5) / aspect)
    tan_fov_x = math.tan(fov * 0.5)
    tan_fov_y = math.tan(fov_y * 0.5)

    cam_pos = np.array([
        radius * math.cos(theta) * math.cos(phi) + cx,
        radius * math.sin(theta) * math.cos(phi) + cy,
        radius * math.sin(phi) + cz
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

    return {
        "id": 0,
        "image_width": W,
        "image_height": H,
        "fov_x_rad": round(fov, 6),
        "fov_y_rad": round(fov_y, 6),
        "tanfovx": round(tan_fov_x, 6),
        "tanfovy": round(tan_fov_y, 6),
        "camera_center": cam_pos.tolist(),
        "viewmatrix": viewmatrix.tolist(),
        "fx": round(0.5 * W / tan_fov_x, 4),
        "fy": round(0.5 * H / tan_fov_y, 4),
        "cx": W / 2,
        "cy": H / 2,
    }


def generate_spiral(num_cameras=60, radius=10.0, turns=5, W=1920, H=1080):
    """Generate a spiral camera trajectory with radius oscillation.

    Args:
        num_cameras: Number of camera poses.
        radius: Base orbit radius.
        turns: Number of full spiral turns.
        W, H: Image dimensions.

    Returns:
        List of camera dictionaries.
    """
    cams = []
    for i in range(num_cameras):
        theta = 2 * math.pi * turns * i / num_cameras
        phi = math.radians(15.0 * math.sin(theta * 0.5))
        r = radius + 2.0 * math.sin(theta * 0.5)
        cd = build_camera_dict(theta, phi, r, W=W, H=H)
        cd["id"] = i
        cams.append(cd)
    return cams


def generate_circle(num_cameras=50, radius=8.0, W=1920, H=1080):
    """Generate a circular camera trajectory at fixed elevation.

    Args:
        num_cameras: Number of camera poses.
        radius: Orbit radius.
        W, H: Image dimensions.

    Returns:
        List of camera dictionaries.
    """
    cams = []
    for i in range(num_cameras):
        theta = 2 * math.pi * i / num_cameras
        cd = build_camera_dict(theta, 0, radius, W=W, H=H)
        cd["id"] = i
        cams.append(cd)
    return cams


def generate_flythrough(num_cameras=30, W=1920, H=1080):
    """Generate a linear flythrough camera trajectory.

    Args:
        num_cameras: Number of camera poses.
        W, H: Image dimensions.

    Returns:
        List of camera dictionaries.
    """
    cams = []
    for i in range(num_cameras):
        t = i / max(num_cameras - 1, 1)
        theta = math.radians(5.0 * (1 - t))
        phi = math.radians(10.0 * (1 - t))
        radius = 20.0 - 15.0 * t
        cd = build_camera_dict(theta, phi, radius, W=W, H=H)
        cd["id"] = i
        cams.append(cd)
    return cams


def generate_random_walk(num_cameras=30, radius=7.0, seed=42, W=1920, H=1080):
    """Generate a camera trajectory with random perturbations.

    Args:
        num_cameras: Number of camera poses.
        radius: Base orbit radius.
        seed: Random seed for reproducibility.
        W, H: Image dimensions.

    Returns:
        List of camera dictionaries.
    """
    rng = np.random.RandomState(seed)
    cams = []
    for i in range(num_cameras):
        theta = 2 * math.pi * i / num_cameras + rng.uniform(-0.1, 0.1)
        phi = rng.uniform(-0.15, 0.15)
        r = radius + rng.uniform(-0.5, 0.5)
        cd = build_camera_dict(theta, phi, r, W=W, H=H)
        cd["id"] = i
        cams.append(cd)
    return cams


GENERATORS = {
    "spiral": generate_spiral,
    "circle": generate_circle,
    "flythrough": generate_flythrough,
    "random_walk": generate_random_walk,
}


def main():
    p = argparse.ArgumentParser(description="Generate standard camera path presets")
    p.add_argument("--type", choices=list(GENERATORS.keys()), default="circle",
                   help="Camera path type")
    p.add_argument("--num-cameras", type=int, default=None,
                   help="Number of cameras (default depends on type)")
    p.add_argument("--radius", type=float, default=8.0,
                   help="Orbit radius in scene units")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (random_walk only)")
    p.add_argument("--turns", type=float, default=5.0,
                   help="Number of spiral turns (spiral only)")
    args = p.parse_args()

    defaults = {"spiral": 60, "circle": 50, "flythrough": 30, "random_walk": 30}
    n = args.num_cameras or defaults[args.type]

    if args.type == "spiral":
        cams = generate_spiral(n, args.radius, turns=args.turns, W=args.width, H=args.height)
    elif args.type == "circle":
        cams = generate_circle(n, args.radius, W=args.width, H=args.height)
    elif args.type == "flythrough":
        cams = generate_flythrough(n, W=args.width, H=args.height)
    elif args.type == "random_walk":
        cams = generate_random_walk(n, args.radius, seed=args.seed, W=args.width, H=args.height)
    else:
        print(f"Unknown type: {args.type}")
        sys.exit(1)

    out_path = os.path.join(OUTPUT_DIR, f"{args.type}.json")
    with open(out_path, "w") as f:
        json.dump({
            "cameras": cams,
            "metadata": {
                "name": args.type,
                "num_cameras": len(cams),
                "width": args.width,
                "height": args.height,
                "generator_args": vars(args),
            }
        }, f, indent=2)
    print(f"Generated {len(cams)} cameras -> {out_path}")


if __name__ == "__main__":
    main()