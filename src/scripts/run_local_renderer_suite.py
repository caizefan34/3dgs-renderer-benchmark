#!/usr/bin/env python
"""Run local speed and optional GT-quality checks across renderer adapters."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
REPO_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from benchmark_suite import BENCHMARK_SUITE_VERSION  # noqa: E402
from renderers import get_renderer_class, list_available, list_renderers  # noqa: E402


def _resolve_renderers(requested: list[str] | None) -> list[str]:
    if not requested or requested == ["all"]:
        return list_renderers()
    return requested


def _run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout.splitlines()[-20:],
        "stderr_tail": completed.stderr.splitlines()[-20:],
    }


def _availability(renderers: list[str], device: str) -> list[dict[str, Any]]:
    available = set(list_available(device=device))
    rows: list[dict[str, Any]] = []
    for renderer in renderers:
        cls = get_renderer_class(renderer)
        rows.append(
            {
                "renderer": renderer,
                "registered": cls is not None,
                "available": renderer in available,
                "implementation": getattr(cls, "implementation", None) if cls else None,
                "source_url": getattr(cls, "source_url", None) if cls else None,
            }
        )
    return rows


def _speed_command(
    renderer: str,
    scene: Path,
    cameras: Path | None,
    output_dir: Path,
    frames: int,
    warmup: int,
    repeats: int,
    benchmark_type: str,
) -> list[str]:
    command = [
        sys.executable,
        "src/run_benchmark.py",
        "--scene",
        str(scene),
        "--renderers",
        renderer,
        "--frames",
        str(frames),
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--benchmark-type",
        benchmark_type,
        "--output",
        str(output_dir / renderer / "speed"),
    ]
    if cameras is not None:
        command.extend(["--cameras", str(cameras)])
    return command


def _quality_command(
    renderer: str,
    scene: Path,
    cameras: Path,
    ground_truth_dir: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "src/scripts/validate_quality.py",
        "--renderers",
        renderer,
        "--scene",
        str(scene),
        "--cameras",
        str(cameras),
        "--ground-truth-dir",
        str(ground_truth_dir),
        "--split-label",
        "official dataset evaluation split",
        "--output",
        str(output_dir / renderer / "quality" / "quality_gt.json"),
    ]


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    renderers = _resolve_renderers(args.renderers)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = args.scene.resolve() if args.scene else None
    cameras = args.cameras.resolve() if args.cameras else None
    ground_truth_dir = args.ground_truth_dir.resolve() if args.ground_truth_dir else None

    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_suite_version": BENCHMARK_SUITE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Official datasets are required for quality-bearing comparisons; generated scenes are speed stress only.",
        "availability": _availability(renderers, args.device),
        "speed_runs": [],
        "quality_runs": [],
    }

    available = {row["renderer"] for row in report["availability"] if row["available"]}
    if scene is None or not scene.exists():
        report["speed_skip_reason"] = "missing_scene"
    else:
        for renderer in renderers:
            if renderer not in available:
                report["speed_runs"].append({"renderer": renderer, "status": "skipped_unavailable"})
                continue
            command = _speed_command(
                renderer,
                scene,
                cameras,
                output_dir,
                args.frames,
                args.warmup,
                args.repeats,
                args.benchmark_type,
            )
            result = _run_command(command, REPO_ROOT)
            report["speed_runs"].append({"renderer": renderer, "status": "ok" if result["returncode"] == 0 else "failed", **result})

    if cameras is None or ground_truth_dir is None or not ground_truth_dir.exists() or scene is None:
        report["quality_skip_reason"] = "missing_scene_cameras_or_ground_truth"
    else:
        for renderer in renderers:
            if renderer not in available:
                report["quality_runs"].append({"renderer": renderer, "status": "skipped_unavailable"})
                continue
            result = _run_command(
                _quality_command(renderer, scene, cameras, ground_truth_dir, output_dir),
                REPO_ROOT,
            )
            report["quality_runs"].append({"renderer": renderer, "status": "ok" if result["returncode"] == 0 else "failed", **result})
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument("--cameras", type=Path, default=None)
    parser.add_argument("--ground-truth-dir", type=Path, default=None)
    parser.add_argument("--renderers", nargs="+", default=["all"])
    parser.add_argument("--output-dir", type=Path, default=Path("results/local_renderer_suite"))
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--benchmark-type",
        choices=["synthetic_stress", "real_scene_speed"],
        default="real_scene_speed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_suite(args)
    output_path = args.output_dir / "local_renderer_suite_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
