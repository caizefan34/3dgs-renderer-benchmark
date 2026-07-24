#!/usr/bin/env python
"""Run one-frame same-checkpoint differential checks for candidate renderers."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_framework import load_cameras_from_json, load_ply  # noqa: E402
from renderers import get_renderer  # noqa: E402
from scripts.validate_quality import compute_psnr, compute_ssim  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(scene_path: Path, cameras_path: Path, renderer_name: str, device: str) -> dict:
    scene = load_ply(str(scene_path), device=device)
    cameras = load_cameras_from_json(str(cameras_path), device=device)
    camera = cameras[0]
    reference_renderer = get_renderer("gsplat", device=device)
    candidate_renderer = get_renderer(renderer_name, device=device)
    if reference_renderer is None or candidate_renderer is None:
        raise RuntimeError(f"reference or candidate renderer is unavailable: {renderer_name}")
    reference_scene = reference_renderer.prepare_scene(scene)
    candidate_scene = candidate_renderer.prepare_scene(scene)
    with torch.no_grad():
        reference = reference_renderer.render(reference_scene, camera)
        candidate = candidate_renderer.render(candidate_scene, camera)
    torch.cuda.synchronize(device)
    psnr = compute_psnr(candidate, reference)
    return {
        "renderer": renderer_name,
        "scene_sha256": _sha256(scene_path),
        "camera_sha256": _sha256(cameras_path),
        "shape": list(candidate.shape),
        "psnr_vs_gsplat_db": psnr if math.isfinite(psnr) else 999.0,
        "ssim_vs_gsplat": compute_ssim(candidate, reference),
        "max_abs_error": float((candidate - reference).abs().max().item()),
        "mean_abs_error": float((candidate - reference).abs().mean().item()),
        "candidate_metadata": candidate_renderer.metadata(),
        "reference_metadata": reference_renderer.metadata(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--renderer", choices=["flashgs", "local_gs", "gemm_gs"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    result = run(args.scene.resolve(), args.cameras.resolve(), args.renderer, args.device)
    result["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
