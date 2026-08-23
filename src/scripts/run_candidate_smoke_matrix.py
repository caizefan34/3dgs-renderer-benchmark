#!/usr/bin/env python
"""Run isolated candidate-renderer differential smoke checks on one idle GPU."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from run_linux_tier_a_matrix import wait_for_idle_gpu


ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = (
    ("flashgs", "flashgs"),
    ("local_gs", "localgs"),
    ("gemm_gs", "gemmgs"),
)


def build_plan(root: Path, env_root: Path, output: Path) -> list[dict]:
    scene = root / "datasets" / "processed" / "mipnerf360" / "bonsai" / "point_cloud.ply"
    cameras = root / "datasets" / "processed" / "mipnerf360" / "bonsai" / "eval_cameras.json"
    return [
        {
            "renderer": renderer,
            "environment": environment,
            "python": str(env_root / environment / "bin" / "python"),
            "scene": str(scene),
            "cameras": str(cameras),
            "output": str(output / f"{renderer}.json"),
        }
        for renderer, environment in CANDIDATES
    ]


def run(args) -> dict:
    root, env_root, output = args.root.resolve(), args.env_root.resolve(), args.output.resolve()
    plan = build_plan(root, env_root, output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for row in plan:
        python = Path(row["python"])
        if not python.is_file():
            raise FileNotFoundError(f"missing candidate environment: {python}")
        wait_for_idle_gpu(
            args.wait_gpu, args.idle_max_memory_mib, args.idle_max_utilization,
            args.idle_samples, args.idle_poll_seconds,
        )
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.wait_gpu), "PYTHONNOUSERSITE": "1"}
        subprocess.run([
            str(python), "src/scripts/smoke_candidate_renderers.py",
            "--scene", row["scene"], "--cameras", row["cameras"],
            "--renderer", row["renderer"], "--output", row["output"],
        ], cwd=root, env=environment, check=True)
        result = json.loads(Path(row["output"]).read_text(encoding="utf-8"))
        result["adapter_smoke_pass"] = (
            result["psnr_vs_gsplat_db"] > 20.0
            and result["ssim_vs_gsplat"] > 0.8
            and result["max_abs_error"] <= 1.0
        )
        if not result["adapter_smoke_pass"]:
            raise RuntimeError(f"candidate differential smoke failed: {row['renderer']}")
        results.append(result)
    summary = {
        "schema_version": "1.0", "status": "complete",
        "scope": "one-frame adapter correctness; not Tier A performance",
        "gpu_index": args.wait_gpu, "results": results,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--env-root", type=Path, default=Path.home() / "miniforge3" / "envs")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "candidate-smoke")
    parser.add_argument("--wait-gpu", type=int, default=7)
    parser.add_argument("--idle-max-memory-mib", type=float, default=1024.0)
    parser.add_argument("--idle-max-utilization", type=float, default=5.0)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-poll-seconds", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(build_plan(args.root.resolve(), args.env_root.resolve(), args.output.resolve()), indent=2))
        return 0
    summary = run(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
