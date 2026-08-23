#!/usr/bin/env python3
"""
Download Mip-NeRF 360 GT images for the three benchmark scenes.

The Mip-NeRF 360 dataset is available from:
https://jonbarron.info/mipnerf360/

For 3DGS, the preprocessed images come from the 360_v2 dataset.
This script downloads the required images for GT quality evaluation.

Usage:
    python scripts/epic05/_download_mipnerf_gt.py
    python scripts/epic05/_download_mipnerf_gt.py --scene room
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def download_file(url, dest_path):
    """Download a file with progress."""
    print(f"  Downloading: {url}")
    print(f"  -> {dest_path}")
    
    def reporthook(block, block_size, total_size):
        downloaded = block * block_size / (1024 * 1024)
        total = total_size / (1024 * 1024) if total_size > 0 else 0
        if total > 0:
            pct = min(100, downloaded / total * 100)
            print(f"\r    {downloaded:.1f}/{total:.1f} MB ({pct:.0f}%)", end="", flush=True)
        else:
            print(f"\r    {downloaded:.1f} MB", end="", flush=True)
    
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print()
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def try_download_mipnerf_scene(scene_id, output_dir):
    """Try to download Mip-NeRF 360 individual scene images."""
    # The Mip-NeRF 360 dataset is not directly available per-scene.
    # Options:
    # 1. Download full 360_v2.zip (15 GB) and extract
    # 2. Use the individual scene images from the original dataset
    
    # From the original Mip-NeRF 360 dataset page:
    # Each scene's images are available via the NeRF On-the-go work or
    # from the Mip-NeRF 360 original release.
    
    # For 3DGS, the correct dataset is 360_v2 from:
    # https://drive.google.com/drive/folders/1jfIRvu0Rsv7B8lypI-FXy5NLY7HTHnIF
    # This includes undistorted images + COLMAP SfM data.
    
    print(f"Download of Mip-NeRF 360 '{scene_id}' GT images not yet automated.")
    print(f"Manual download required from: https://jonbarron.info/mipnerf360/")
    print(f"Or from the 3DGS dataset release.")
    print()
    print(f"After download, place images in:")
    print(f"  {output_dir / 'images'}/")
    print(f"with filenames matching the camera image_name references.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Download Mip-NeRF 360 GT images")
    parser.add_argument("--scene", choices=["bicycle", "garden", "room", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    scenes = ["bicycle", "garden", "room"] if args.scene == "all" else [args.scene]

    all_available = True
    for scene in scenes:
        gt_dir = args.output_dir or (REPO_ROOT / "data" / "official" / "mipnerf360" / scene)
        gt_images_dir = gt_dir / "images"
        
        if gt_images_dir.exists() and any(gt_images_dir.iterdir()):
            n_images = len(list(gt_images_dir.iterdir()))
            print(f"[OK] {scene}: {n_images} GT images already at {gt_images_dir}")
            continue
        
        print(f"\n[MISSING] {scene}: GT images not found at {gt_images_dir}")
        print(f"  Scene data available at: {gt_dir}")
        print(f"  Cameras: {gt_dir / 'cameras.json'}")
        print(f"  Point cloud: {gt_dir / 'point_cloud.ply'}")
        print()
        all_available = False
        
        # Try to offer download
        try_download_mipnerf_scene(scene, gt_images_dir)

    if all_available:
        print(f"\nAll {len(scenes)} scenes have GT images! Ready for quality evaluation.")
        print(f"Run: python scripts/epic05/evaluate_official_quality.py --all --resolution 1080p --tile-sizes 16 32")
    else:
        print(f"\nGT images missing for some scenes.")
        print(f"Required: Mip-NeRF 360 official images at native resolution.")
        print(f"Download from: https://jonbarron.info/mipnerf360/")
        print(f"Then convert/move to the appropriate data/official/mipnerf360/<scene>/images/ directory.")
        print()
        print(f"Alternatively, download the full 360_v2 dataset and extract needed scenes.")


if __name__ == "__main__":
    main()
