"""
Real GT dataset loader for 3DGS training.

Loads Mip-NeRF 360 images, cameras, and optional initial PLY checkpoints.
Supports multiple scenes with consistent camera-to-image mapping.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from benchmark_framework import load_ply, load_cameras_from_json, resize_cameras


class GTDataset:
    """Dataset loading real GT images with matched cameras.

    Provides:
      - Pre-loaded camera list with image_name mapping
      - Lazy GT image loading (loads to CPU, transfers to GPU on demand)
      - Resolution control (resize cameras)
      - Camera iteration across all views

    The dataset expects:
      data/
        official/mipnerf360/{scene}/           ← PLY + cameras (trained checkpoint)
            point_cloud.ply
            cameras.json
        datasets/mipnerf360/{scene}/images/    ← Real GT photographs
    """

    def __init__(
        self,
        scene: str,
        repo_root: str | Path,
        resolution: str = "1080p",
        device: str = "cuda",
        background: str = "black",
    ):
        self.scene = scene
        self.repo_root = Path(repo_root).resolve()
        self.device = device
        self.background = background

        RESOLUTIONS = {"720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160)}
        target_res = RESOLUTIONS.get(resolution, (1920, 1080))
        self.resolution = resolution
        self.target_width, self.target_height = target_res

        # Paths
        self.official_dir = self.repo_root / "data" / "official" / "mipnerf360" / scene
        self.gt_dir = self.repo_root / "data" / "datasets" / "mipnerf360" / scene / "images"

        # Load cameras
        camera_path = self.official_dir / "cameras.json"
        if not camera_path.exists():
            raise FileNotFoundError(f"Camera file not found: {camera_path}")
        print(f"  [GTDataset] Loading cameras from {camera_path}")
        self._cameras = load_cameras_from_json(str(camera_path), device="cpu")
        self._indices = list(range(len(self._cameras)))

        # Resize to target resolution
        self._cameras = resize_cameras(self._cameras, self.target_width, self.target_height)

        # Verify GT images
        if not self.gt_dir.exists():
            raise FileNotFoundError(f"GT image directory not found: {self.gt_dir}")

        # Build image name -> path mapping
        self._image_index: Dict[str, Path] = {}
        for fpath in self.gt_dir.iterdir():
            if fpath.is_file() and fpath.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                self._image_index[fpath.stem] = fpath

        # Match cameras to GT images
        self._valid_indices: List[int] = []
        self._valid_cameras: List = []
        self._valid_image_paths: List[Path] = []
        for i, cam in enumerate(self._cameras):
            img_name = getattr(cam, "image_name", None)
            if img_name is None:
                img_name = getattr(cam, "img_name", None)
            if img_name and Path(img_name).stem in self._image_index:
                self._valid_indices.append(i)
                self._valid_cameras.append(cam)
                self._valid_image_paths.append(self._image_index[Path(img_name).stem])

        print(f"  [GTDataset] {len(self._valid_indices)}/{len(self._cameras)} cameras matched to GT images")
        if not self._valid_indices:
            raise RuntimeError(f"No cameras matched GT images for scene '{scene}'")

        # Pre-cache all GT images as CPU uint8 tensors (avoids PIL malloc fragmentation)
        print("  [GTDataset] Pre-caching GT images (CPU uint8)...")
        self._gt_cache: List[torch.Tensor] = [None] * len(self._valid_indices)
        from PIL import Image
        import numpy as np
        for i, path in enumerate(self._valid_image_paths):
            if i % 100 == 0:
                print(f"    Caching image {i+1}/{len(self._valid_indices)}...")
            with Image.open(path) as source:
                if source.width != self.target_width or source.height != self.target_height:
                    source = source.resize((self.target_width, self.target_height), Image.LANCZOS)
                rgba_np = np.array(source.convert("RGBA"), dtype=np.uint8)
            self._gt_cache[i] = torch.from_numpy(rgba_np)  # [H, W, 4] uint8 on CPU
        total_mb = sum(t.numel() for t in self._gt_cache) / (1024**2)
        print(f"  [GTDataset] All {len(self._valid_indices)} images cached (CPU, {total_mb:.0f} MB)")

    def __len__(self) -> int:
        return len(self._valid_indices)

    def get_camera(self, index: int):
        """Return camera at index (on self.device)."""
        cam = self._valid_cameras[index]
        # Move tensors to device
        for attr in ["viewmatrix", "projmatrix", "camera_center", "world_view_transform",
                      "full_proj_transform", "K"]:
            t = getattr(cam, attr)
            if isinstance(t, torch.Tensor):
                setattr(cam, attr, t.to(self.device))
        return cam

    def get_gt_image(self, index: int) -> torch.Tensor:
        """Return [H, W, 3] float32 GT image on self.device (from CPU uint8 cache, divided by 255)."""
        rgba = self._gt_cache[index]  # [H, W, 4] uint8 on CPU
        # Scale uint8 [0,255] to float32 [0,1]
        return rgba.to(self.device, non_blocking=True, dtype=torch.float32)[..., :3].div_(255.0).contiguous()

    def get_item(self, index: int) -> Tuple:
        """Return (camera, gt_image) pair at index."""
        return self.get_camera(index), self.get_gt_image(index)

    def iterate_all(self, shuffle: bool = True) -> List[Tuple]:
        """Return all (camera, gt_image) pairs in order or shuffled."""
        indices = self._indices[:]
        if shuffle:
            import random
            random.shuffle(indices)
        return [self.get_item(i) for i in indices]


def load_initial_checkpoint(
    scene: str,
    repo_root: str | Path,
    device: str = "cuda",
) -> dict:
    """Load the official trained PLY checkpoint for initialization."""
    ply_path = Path(repo_root) / "data" / "official" / "mipnerf360" / scene / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY checkpoint not found: {ply_path}")
    print(f"  Loading SfM checkpoint: {ply_path}")
    return load_ply(str(ply_path), device=device)
