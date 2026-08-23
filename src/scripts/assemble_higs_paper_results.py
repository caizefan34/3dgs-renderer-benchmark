#!/usr/bin/env python
"""Assemble protocol-valid result JSONs from one audited gsplat/HiGS run dir.

The trainer writes stats/val_step*.json (quality at eval steps) and
stats/train_step*_rank0.json (accumulated wall time and peak memory at save
steps). This assembler turns those artifacts plus paper-run-metadata.json into
the result contract validated by validate_higs_paper_results.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from higs_paper_protocol import build_experiment_plan  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _linear_ttq(curve, threshold_psnr: float) -> float:
    """First wall time at which PSNR reaches threshold, linear interpolation."""
    prev_i, prev_w, prev_p = None, None, None
    for point in curve:
        i, w, p = point["iteration"], point["wall_time_seconds"], point["psnr_db"]
        if p >= threshold_psnr:
            if prev_p is not None and prev_p < threshold_psnr and w > prev_w:
                frac = (threshold_psnr - prev_p) / max(p - prev_p, 1e-9)
                return prev_w + frac * (w - prev_w)
            return w
        prev_i, prev_w, prev_p = i, w, p
    return curve[-1]["wall_time_seconds"]


def assemble(run_dir: Path, job: dict, *, gpu_index: int, energy_joules: float) -> dict:
    metadata = _load(run_dir / "paper-run-metadata.json")
    iterations = job["iterations"]
    curve_steps = sorted({min(s, iterations) for s in (7_000, 15_000, 30_000)})
    if curve_steps[-1] != iterations:
        raise ValueError(f"final curve step {curve_steps[-1]} != budget {iterations}")

    curve = []
    final_wall = float(metadata["wall_time_seconds"])
    for index, step in enumerate(curve_steps):
        val = _load(run_dir / "stats" / f"val_step{step - 1:04d}.json")
        if index == len(curve_steps) - 1:
            wall = final_wall
        else:
            train = _load(
                run_dir / "stats" / f"train_step{step - 1:04d}_rank0.json"
            )
            wall = float(train["ellipse_time"])
        curve.append(
            {
                "iteration": step,
                "wall_time_seconds": wall,
                "psnr_db": float(val["psnr"]),
                "ssim": float(val["ssim"]),
                "lpips": float(val["lpips"]),
                "num_GS": int(val["num_GS"]),
            }
        )
    for prev, current in zip(curve, curve[1:]):
        if current["wall_time_seconds"] <= prev["wall_time_seconds"]:
            raise ValueError(
                f"quality curve wall times not strictly ordered at {current['iteration']}"
            )

    final = curve[-1]
    threshold = final["psnr_db"] - 0.05
    time_to_quality = _linear_ttq(curve, threshold)

    train_final = _load(run_dir / "stats" / f"train_step{iterations - 1:04d}_rank0.json")
    peak_mib = float(train_final["mem"]) * 1024.0
    gaussian_count = int(final["num_GS"]) if "num_GS" in final else int(train_final["num_GS"])

    ckpts = sorted(
        (run_dir / "ckpts").glob("ckpt_*_rank0.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not ckpts:
        raise ValueError("no final checkpoint under ckpts/")
    final_ckpt = ckpts[-1]
    import hashlib
    digest = hashlib.sha256()
    with final_ckpt.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    result = {
        "schema_version": "1.0",
        "job_id": job["job_id"],
        "method": job["method"],
        "scene": job["scene"],
        "hardware": job["hardware"],
        "seed": job["seed"],
        "status": "complete",
        "training": {
            "initialization": job["initialization"],
            "iterations": iterations,
        },
        "performance": {
            "wall_time_seconds": final_wall,
            "time_to_quality_seconds": time_to_quality,
        },
        "quality": {
            "psnr_db": final["psnr_db"],
            "ssim": final["ssim"],
            "lpips": final["lpips"],
        },
        "resources": {
            "peak_gpu_memory_mib": peak_mib,
            "energy_joules": float(energy_joules),
            "final_gaussian_count": gaussian_count,
        },
        "quality_curve": curve,
        "artifact": {
            "path": final_ckpt.relative_to(run_dir).as_posix(),
            "sha256": digest.hexdigest(),
            "size_bytes": final_ckpt.stat().st_size,
        },
        "provenance": {
            "timing_boundary": "dataset_ready_to_final_checkpoint",
            "clean_process": True,
            "gpu_index": gpu_index,
            "started_at_utc": metadata.get("started_at_utc"),
            "source": metadata.get("source"),
            "dataset_inventory_sha256": metadata.get("dataset", {}).get("inventory_sha256"),
            "gpu_name": metadata.get("hardware", {}).get("gpu_name"),
        },
        "run_dir": str(run_dir.resolve()),
    }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--energy-joules", type=float, default=0.0)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "benchmark" / "higs-paper-protocol.json"
    )
    args = parser.parse_args(argv)

    protocol = _load(args.protocol)
    metadata = _load(args.run_dir / "paper-run-metadata.json")
    method, scene, seed = metadata["method"], metadata["scene"], metadata["seed"]
    jobs = [
        job
        for job in build_experiment_plan(protocol)
        if job["method"] == method
        and job["scene"] == scene
        and job["seed"] == seed
        and job["hardware"] == "a100"
    ]
    if not jobs:
        raise SystemExit(f"no planned job matches method={method} scene={scene} seed={seed}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        result = assemble(args.run_dir, job, gpu_index=args.gpu_index, energy_joules=args.energy_joules)
        out = args.output_dir / f"{job['job_id']}.json"
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
