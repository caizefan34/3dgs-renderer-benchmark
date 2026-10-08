"""Probe FastGS's unchanged Pillow-to-tensor image preparation.

Run from the pinned FastGS checkout so `utils.general_utils` is the author code.
This measures CPU image preparation only, not camera construction or training.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image

from utils.general_utils import PILtoTorch


def prepare(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        width, height = image.size
        resolution = (1600, int(height / (width / 1600))) if width > 1600 else (width, height)
        return PILtoTorch(image, resolution)


def run(paths: list[Path], workers: int) -> tuple[float, list[torch.Tensor]]:
    start = time.perf_counter()
    if workers == 1:
        images = [prepare(path) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            images = list(pool.map(prepare, paths))
    return time.perf_counter() - start, images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4, 8])
    args = parser.parse_args()
    paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})[: args.limit]
    if not paths:
        raise RuntimeError(f"No images under {args.images}")

    serial_seconds = 0.0
    parallel_seconds = {workers: 0.0 for workers in args.workers}
    for start in range(0, len(paths), args.batch_size):
        batch = paths[start:start + args.batch_size]
        elapsed, serial = run(batch, 1)
        serial_seconds += elapsed
        for workers in args.workers:
            elapsed, images = run(batch, workers)
            if len(images) != len(serial) or not all(torch.equal(a, b) for a, b in zip(serial, images)):
                raise AssertionError(f"workers={workers} changed resized image tensors in batch {start}")
            parallel_seconds[workers] += elapsed
            del images
        del serial
    results = [
        {"workers": workers, "seconds": seconds, "speedup_vs_serial": serial_seconds / seconds}
        for workers, seconds in parallel_seconds.items()
    ]
    print(json.dumps({"images": len(paths), "batch_size": args.batch_size, "torch_cpu_threads": torch.get_num_threads(), "serial_seconds": serial_seconds, "parallel": results}, indent=2))


if __name__ == "__main__":
    main()
