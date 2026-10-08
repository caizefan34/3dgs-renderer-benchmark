#!/usr/bin/env python3
"""BFC0 DSH-3: Trace collector - collect forward/backward state from gsplat.

Uses gsplat's rasterization pipeline to collect:
- isect_offsets, flatten_ids (intersection data)
- means2d, conics, opacities (Gaussian state)
- image dimensions, tile size

NOTE: gsplat 1.5.3 no longer returns last_ids in meta.
We compute it ourselves for O1 oracle.

This is Python-wrapper-only instrumentation; no CUDA math is modified.
"""
import os, sys, json, math, struct
import numpy as np
import torch
from typing import Optional, Dict, Any, Tuple

# Add gsplat to path
import gsplat
from gsplat import rasterization, isect_tiles, fully_fused_projection

def collect_trace(
    scene_dir: str,
    checkpoint_path: Optional[str] = None,
    camera_ids: Optional[list] = None,
    max_cameras: int = 8,
    device: str = "cuda",
    tile_size: int = 16,
) -> Dict[str, Any]:
    """Collect forward/backward trace data from gsplat for a scene."""
    
    # Load camera data - handle both sparse/ and sparse/0/ layouts
    sparse_dir = os.path.join(scene_dir, "sparse")
    if not os.path.exists(os.path.join(sparse_dir, "cameras.bin")):
        # Try sparse/0/ layout
        alt_sparse = os.path.join(sparse_dir, "0")
        if os.path.exists(os.path.join(alt_sparse, "cameras.bin")):
            sparse_dir = alt_sparse
    
    images_dir = os.path.join(scene_dir, "images")
    
    # Parse COLMAP format
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")
    images_bin = os.path.join(sparse_dir, "images.bin")
    points3d_bin = os.path.join(sparse_dir, "points3D.bin")
    
    if not os.path.exists(cameras_bin):
        # Try text format
        cameras_txt = os.path.join(sparse_dir, "cameras.txt")
        images_txt = os.path.join(sparse_dir, "images.txt")
        points3d_txt = os.path.join(sparse_dir, "points3D.txt")
        if os.path.exists(cameras_txt):
            return _collect_from_text_colmap(
                scene_dir, cameras_txt, images_txt, points3d_txt,
                camera_ids, max_cameras, device, tile_size
            )
        raise FileNotFoundError(f"No COLMAP data found in {sparse_dir}")
    
    return _collect_from_binary_colmap(
        scene_dir, cameras_bin, images_bin, points3d_bin,
        camera_ids, max_cameras, device, tile_size
    )

def _read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def _collect_from_binary_colmap(scene_dir, cameras_bin, images_bin, points3d_bin,
                                 camera_ids, max_cameras, device, tile_size):
    """Collect traces from binary COLMAP format."""
    
    # Read cameras
    cameras = {}
    with open(cameras_bin, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q", "<")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read_next_bytes(fid, 24, "iiQQ", "<")
            model_name = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
                         3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE",
                         6: "FULL_OPENCV", 7: "FOV", 8: "SIMPLE_RADIAL_FISHEYE",
                         9: "RADIAL_FISHEYE", 10: "THIN_PRISM_FISHEYE"}.get(model_id, "UNKNOWN")
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5,
                         8: 4, 9: 5, 10: 8}.get(model_id, 0)
            params = _read_next_bytes(fid, 8 * num_params, "d" * num_params, "<")
            cameras[camera_id] = {
                "model": model_name, "width": width, "height": height,
                "params": params
            }
    
    # Read images
    images = {}
    with open(images_bin, "rb") as fid:
        num_images = _read_next_bytes(fid, 8, "Q", "<")[0]
        for _ in range(num_images):
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = _read_next_bytes(fid, 64, "idddddddi", "<")
            image_name = b""
            while True:
                byte = fid.read(1)
                if byte == b"\x00":
                    break
                image_name += byte
            image_name = image_name.decode("utf-8")
            # Read points (2D image observations)
            num_points = _read_next_bytes(fid, 8, "Q", "<")[0]
            fid.read(num_points * 24)  # Skip point data
            
            images[image_id] = {
                "name": image_name, "camera_id": camera_id,
                "q": [qw, qx, qy, qz], "t": [tx, ty, tz]
            }
    
    # Read 3D points
    points_xyz = []
    with open(points3d_bin, "rb") as fid:
        num_points = _read_next_bytes(fid, 8, "Q", "<")[0]
        for _ in range(num_points):
            point_id, x, y, z, r, g, b, error = _read_next_bytes(fid, 43, "QdddBBBd", "<")
            track_len = _read_next_bytes(fid, 8, "Q", "<")[0]
            fid.read(track_len * 8)  # Skip track data
            points_xyz.append([x, y, z])
    
    points_xyz = np.array(points_xyz, dtype=np.float32)
    
    # Select cameras
    sorted_image_ids = sorted(images.keys())
    if camera_ids is None:
        camera_ids = sorted_image_ids[:max_cameras]
    
    # Build camera matrices
    traces = []
    for cam_id in camera_ids:
        if cam_id not in images:
            continue
        img = images[cam_id]
        cam = cameras[img["camera_id"]]
        
        # Build world-to-camera matrix
        q = np.array(img["q"])
        t = np.array(img["t"])
        R = quaternion_to_rotation_matrix(q)
        tvec = t.reshape(3, 1)
        
        # Build projection matrix
        fx, fy, cx, cy = cam["params"][:4]
        if cam["model"] == "SIMPLE_PINHOLE":
            fx = fy = cam["params"][0]
            cx = cy = cam["params"][1]
        elif cam["model"] == "PINHOLE":
            fx, fy, cx, cy = cam["params"][:4]
        
        # Create viewmat (world-to-camera)
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = R.T
        viewmat[:3, 3] = (-R.T @ tvec).flatten()
        
        # Create K matrix (camera intrinsics)
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        traces.append({
            "image_id": cam_id,
            "image_name": img["name"],
            "width": cam["width"],
            "height": cam["height"],
            "viewmat": viewmat,
            "K": K,
        })
    
    return {
        "points_xyz": points_xyz,
        "cameras": traces,
        "num_gaussians": len(points_xyz),
    }

def _collect_from_text_colmap(scene_dir, cameras_txt, images_txt, points3d_txt,
                                camera_ids, max_cameras, device, tile_size):
    """Collect traces from text COLMAP format."""
    # Read cameras
    cameras = {}
    with open(cameras_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[camera_id] = {"model": model, "width": width, "height": height, "params": params}
    
    # Read images
    images = {}
    with open(images_txt) as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        if len(parts) >= 10:
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = int(parts[8])
            image_name = parts[9]
            # Skip next line (point observations)
            i += 2
            images[image_id] = {
                "name": image_name, "camera_id": camera_id,
                "q": [qw, qx, qy, qz], "t": [tx, ty, tz]
            }
        else:
            i += 1
    
    # Read 3D points
    points_xyz = []
    with open(points3d_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            points_xyz.append([x, y, z])
    
    points_xyz = np.array(points_xyz, dtype=np.float32)
    
    # Select cameras
    sorted_image_ids = sorted(images.keys())
    if camera_ids is None:
        camera_ids = sorted_image_ids[:max_cameras]
    
    # Build camera matrices
    traces = []
    for cam_id in camera_ids:
        if cam_id not in images:
            continue
        img = images[cam_id]
        cam = cameras[img["camera_id"]]
        
        q = np.array(img["q"])
        t = np.array(img["t"])
        R = quaternion_to_rotation_matrix(q)
        tvec = t.reshape(3, 1)
        
        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = R.T
        viewmat[:3, 3] = (-R.T @ tvec).flatten()
        
        params = cam["params"]
        if cam["model"] == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx = cy = params[1]
        elif cam["model"] == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        else:
            fx = fy = params[0]
            cx = cy = params[1]
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        traces.append({
            "image_id": cam_id,
            "image_name": img["name"],
            "width": cam["width"],
            "height": cam["height"],
            "viewmat": viewmat,
            "K": K,
        })
    
    return {
        "points_xyz": points_xyz,
        "cameras": traces,
        "num_gaussians": len(points_xyz),
    }

def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float32)
    return R

def compute_last_ids(
    means2d: np.ndarray,
    conics: np.ndarray,
    opacities: np.ndarray,
    flatten_ids: np.ndarray,
    isect_offsets: np.ndarray,
    width: int,
    height: int,
    tile_size: int,
) -> np.ndarray:
    """Compute last_ids from forward pass data.
    
    last_ids[pix_id] = index of the last (furthest back) gaussian that contributes
    to pixel pix_id with alpha > threshold.
    
    This replicates the gsplat forward pass last_ids computation.
    """
    C = isect_offsets.shape[0]
    tile_h = isect_offsets.shape[1]
    tile_w = isect_offsets.shape[2]
    n_isects = len(flatten_ids)
    
    last_ids = np.full((C, height, width), -1, dtype=np.int32)
    
    for cam_id in range(C):
        for tile_idx in range(tile_h * tile_w):
            tile_row = tile_idx // tile_w
            tile_col = tile_idx % tile_w
            
            range_start = int(isect_offsets[cam_id, tile_row, tile_col])
            if tile_col < tile_w - 1:
                range_end = int(isect_offsets[cam_id, tile_row, tile_col + 1])
            elif tile_row < tile_h - 1:
                range_end = int(isect_offsets[cam_id, tile_row + 1, 0])
            else:
                range_end = n_isects
            
            # Process gaussians in reverse order (front to back)
            for d in range(range_end - 1, range_start - 1, -1):
                g_id = int(flatten_ids[d])
                
                mx = float(means2d[cam_id, g_id, 0])
                my = float(means2d[cam_id, g_id, 1])
                cx = float(conics[cam_id, g_id, 0])
                cy = float(conics[cam_id, g_id, 1])
                cz = float(conics[cam_id, g_id, 2])
                opac = float(opacities[g_id])
                
                # Check which pixels this gaussian contributes to
                tile_y_start = tile_row * tile_size
                tile_x_start = tile_col * tile_size
                
                for ty in range(tile_size):
                    py = tile_y_start + ty
                    if py >= height:
                        continue
                    for tx in range(tile_size):
                        px = tile_x_start + tx
                        if px >= width:
                            continue
                        
                        dx = px + 0.5 - mx
                        dy = py + 0.5 - my
                        sigma = 0.5 * (cx * dx * dx + cz * dy * dy) + cy * dx * dy
                        
                        if sigma < 0:
                            vis = math.exp(-sigma)
                            alpha = min(0.999, opac * vis)
                            if alpha > 1.0 / 255.0:
                                if last_ids[cam_id, py, px] == -1:
                                    last_ids[cam_id, py, px] = g_id
    
    return last_ids

def run_forward_and_collect(
    scene_data: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
    max_gaussians: int = 100000,
    device: str = "cuda",
    tile_size: int = 16,
) -> Dict[str, Any]:
    """Run gsplat forward and collect all trace data needed for BFC0 oracle."""
    
    points_xyz = torch.from_numpy(scene_data["points_xyz"]).to(device)
    
    # Limit gaussians if needed
    if len(points_xyz) > max_gaussians:
        indices = torch.randperm(len(points_xyz))[:max_gaussians]
        points_xyz = points_xyz[indices]
    
    N = len(points_xyz)
    
    # Initialize Gaussian parameters (random if no checkpoint)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        means = ckpt.get("means", points_xyz).to(device)
        quats = ckpt.get("quats", torch.randn(N, 4, device=device)).to(device)
        scales = ckpt.get("scales", torch.rand(N, 3, device=device) * 0.1).to(device)
        opacities = ckpt.get("opacities", torch.rand(N, device=device)).to(device)
        colors = ckpt.get("colors", torch.rand(N, 3, device=device)).to(device)
    else:
        # Use default/initial parameters
        means = points_xyz
        quats = torch.nn.functional.normalize(torch.randn(N, 4, device=device), dim=-1)
        scales = torch.rand(N, 3, device=device) * 0.05 + 0.01
        opacities = torch.sigmoid(torch.randn(N, device=device) * 0.5)
        colors = torch.rand(N, 3, device=device)
    
    # Ensure float32
    means = means.float()
    quats = quats.float()
    scales = scales.float()
    opacities = opacities.float()
    colors = colors.float()
    
    # Enable gradients
    for p in [means, quats, scales, opacities, colors]:
        p.requires_grad_(True)
    
    cam_data = scene_data["cameras"][0]  # Use first camera
    viewmats = torch.from_numpy(cam_data["viewmat"]).to(device).unsqueeze(0)  # [1, 4, 4]
    Ks = torch.from_numpy(cam_data["K"]).to(device).unsqueeze(0)  # [1, 3, 3]
    width = cam_data["width"]
    height = cam_data["height"]
    
    # Run forward pass
    render_colors, render_alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        tile_size=tile_size,
        render_mode="RGB",
        packed=False,
    )
    
    # Extract metadata (gsplat 1.5.3 API)
    isect_offsets = meta["isect_offsets"].detach().cpu().numpy()  # [C, tile_h, tile_w]
    flatten_ids = meta["flatten_ids"].detach().cpu().numpy()  # [n_isects]
    means2d = meta["means2d"].detach().cpu().numpy()  # [C, N, 2]
    conics = meta["conics"].detach().cpu().numpy()  # [C, N, 3]
    
    # Compute last_ids (not available in gsplat 1.5.3)
    print("Computing last_ids from forward pass...")
    opacities_cpu = opacities.detach().cpu().numpy()  # [N]
    last_ids = compute_last_ids(
        means2d, conics, opacities_cpu,
        flatten_ids, isect_offsets, width, height, tile_size
    )
    
    # Get colors in packed form
    colors_cpu = colors.detach().cpu().numpy()  # [N, 3]
    
    return {
        "isect_offsets": isect_offsets,
        "flatten_ids": flatten_ids,
        "last_ids": last_ids,
        "means2d": means2d,
        "conics": conics,
        "opacities": opacities_cpu,
        "colors": colors_cpu,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_isects": len(flatten_ids),
        "n_gaussians": N,
        "meta_keys": list(meta.keys()),
    }

if __name__ == "__main__":
    out_dir = os.environ.get("BFC0_OUT_DIR", "C:\\Users\\36570\\3dgs-renderer-benchmark\\bfc0_p0\\output")
    scene_dir = sys.argv[1] if len(sys.argv) > 1 else "C:\\Users\\36570\\3dgs-renderer-benchmark\\data\\datasets\\mipnerf360\\room"
    
    os.makedirs(os.path.join(out_dir, "trace"), exist_ok=True)
    
    print(f"Collecting trace from {scene_dir}...")
    scene_data = collect_trace(scene_dir, max_cameras=8)
    
    print(f"Loaded {scene_data['num_gaussians']} gaussians, {len(scene_data['cameras'])} cameras")
    
    trace = run_forward_and_collect(scene_data)
    
    print(f"Trace collected: {trace['n_isects']} intersections, {trace['n_gaussians']} gaussians")
    print(f"Image: {trace['width']}x{trace['height']}, tile_size: {trace['tile_size']}")
    
    # Save trace
    trace_path = os.path.join(out_dir, "trace", "room_trace.npz")
    np.savez_compressed(
        trace_path,
        isect_offsets=trace["isect_offsets"],
        flatten_ids=trace["flatten_ids"],
        last_ids=trace["last_ids"],
        means2d=trace["means2d"],
        conics=trace["conics"],
        opacities=trace["opacities"],
        colors=trace["colors"],
        width=trace["width"],
        height=trace["height"],
        tile_size=trace["tile_size"],
    )
    
    # Save metadata
    meta = {k: v for k, v in trace.items() if not isinstance(v, np.ndarray)}
    meta_path = os.path.join(out_dir, "trace", "room_trace_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Trace saved to {trace_path}")
    print(f"Metadata saved to {meta_path}")
