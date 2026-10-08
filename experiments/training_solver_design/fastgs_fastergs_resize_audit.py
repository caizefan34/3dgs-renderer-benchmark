"""Compare the pinned FastGS and NeRFICG target-image resize paths."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.io import read_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    args = parser.parse_args()
    images = sorted(args.images.glob("*"))
    selected = [images[i] for i in (0, len(images) // 2, len(images) - 1)]
    result = []
    for path in selected:
        fastgs = torch.from_numpy(np.asarray(Image.open(path).convert("RGB").resize((1600, 1066))).copy()).float() / 255
        source = read_image(str(path)).float() / 255
        fastergs = F.interpolate(source[None], size=(1066, 1600), mode="area")[0].permute(1, 2, 0)
        diff = fastgs - fastergs
        mse = diff.square().mean().item()
        result.append({
            "image": path.name,
            "fastgs_vs_fastergs_gt_psnr_db": -10 * np.log10(mse),
            "mean_absolute_difference": diff.abs().mean().item(),
            "max_absolute_difference": diff.abs().max().item(),
        })
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
