#!/usr/bin/env python
"""Create factor-4 downsampled images_4/ dirs for scenes that lack them.

Mirrors gsplat's Parser._resize_image_folder exactly: imageio raw read (RGB,
no EXIF orientation), PIL BICUBIC resize, JPG output with matching relative
paths, so the zip() filename mapping in the Parser stays aligned with images/.
"""
from __future__ import annotations

import os
from pathlib import Path

import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path("/mnt/workspace/codex-3dgs-epic05/datasets/raw")
SCENES = [
    "tanks_and_temples/train",
    "tanks_and_temples/truck",
    "deep_blending/drjohnson",
    "deep_blending/playroom",
]
FACTOR = 4
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def main() -> int:
    for rel in SCENES:
        scene = ROOT / rel
        src = scene / "images"
        dst = scene / f"images_{FACTOR}"
        if not src.is_dir():
            print(f"skip {rel}: no images/")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        names = sorted(p for p in os.listdir(src) if Path(p).suffix.lower() in IMAGE_EXTS)
        for name in tqdm(names, desc=rel):
            out_name = Path(name).stem + ".jpg"
            if (dst / out_name).is_file():
                continue
            image = imageio.imread(src / name)[..., :3]
            size = (
                int(round(image.shape[1] / FACTOR)),
                int(round(image.shape[0] / FACTOR)),
            )
            resized = np.array(Image.fromarray(image).resize(size, Image.BICUBIC))
            imageio.imwrite(dst / out_name, resized)
        print(f"{rel}: {len(list(dst.iterdir()))} files -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
