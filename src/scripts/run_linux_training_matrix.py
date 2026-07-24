#!/usr/bin/env python
"""Run fixed-budget native 3DGS training rows on an idle EPIC-05 GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from schema_validation import validate_schema  # noqa: E402
from scripts.run_linux_tier_a_matrix import wait_for_idle_gpu  # noqa: E402


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ply_vertices(path: Path) -> int:
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[1])
            if line == "end_header":
                break
    raise ValueError(f"PLY vertex count is missing: {path}")


def build_plan(root: Path, repository_root: Path, env_root: Path, output_root: Path,
               backend_filter: set[str] | None = None,
               case_filter: set[str] | None = None) -> list[dict]:
    config = _load(root / "benchmark" / "training.json")
    suite = _load(root / "benchmark" / "suite.json")
    cases = {row["case_id"]: row for row in suite["cases"]}
    rows = []
    for backend in config["backends"]:
        if backend_filter and backend["id"] not in backend_filter:
            continue
        for item in config["cases"]:
            if case_filter and item["case_id"] not in case_filter:
                continue
            case = cases[item["case_id"]]
            destination = output_root / backend["id"] / item["case_id"]
            command = [
                str(env_root / backend["environment"] / "bin" / "python"), "train.py",
                "-s", str(root / item["source_path"]), "-m", str(destination),
                "--eval", "--iterations", str(config["iterations"]),
                "--test_iterations", str(config["iterations"]),
                "--save_iterations", str(config["iterations"]),
                *backend["extra_args"],
            ]
            rows.append({
                "step": len(rows) + 1, "backend": backend, "case": case,
                "source": str(root / item["source_path"]),
                "repository": str(repository_root / backend["repository"]),
                "environment_python": command[0], "output": str(destination),
                "command": command, "iterations": config["iterations"],
            })
    return rows


def _gpu_memory_mib(gpu: int, process) -> float:
    try:
        import psutil
        pids = {process.pid}
        pids.update(child.pid for child in psutil.Process(process.pid).children(recursive=True))
        output = subprocess.check_output([
            "nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ], text=True, stderr=subprocess.DEVNULL)
        return sum(float(line.split(",")[1].strip()) for line in output.splitlines()
                   if line.strip() and int(line.split(",")[0].strip()) in pids)
    except Exception:
        return 0.0


def _run_command(row: dict, gpu: int, log: Path) -> tuple[int, float, float]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    peak = 0.0
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(row["command"], cwd=row["repository"], env=env,
                                   stdout=handle, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            peak = max(peak, _gpu_memory_mib(gpu, process))
            time.sleep(1.0)
    return process.returncode, time.monotonic() - started, peak


def _evaluate(row: dict, args, ply: Path, destination: Path) -> dict:
    case = row["case"]
    quality_path = destination / "quality.json"
    command = [
        str(args.evaluator_python.resolve()), "src/scripts/validate_quality.py",
        "--renderers", "gsplat", "--scene", str(ply),
        "--cameras", str(args.root.resolve() / case["camera_path"]),
        "--ground-truth-dir", str(args.root.resolve() / case["ground_truth_path"]),
        "--frames", str(args.quality_frames), "--width", "1920", "--height", "1080",
        "--split-label", "official_eval", "--output", str(quality_path),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.wait_gpu)
    subprocess.run(command, cwd=args.root.resolve(), env=env, check=True)
    report = _load(quality_path)
    return report["results"][0]["quality"]


def _result(row: dict, args, returncode: int, wall: float, peak: float) -> dict:
    output = Path(row["output"])
    ply = output / "point_cloud" / f"iteration_{row['iterations']}" / "point_cloud.ply"
    complete = returncode == 0 and ply.is_file()
    artifact = None
    quality = None
    if complete:
        artifact = {"path": str(ply), "sha256": _sha256(ply),
                    "size_bytes": ply.stat().st_size, "gaussian_count": _ply_vertices(ply)}
        if row["backend"]["id"] != "gemm_gs_train":
            quality = _evaluate(row, args, ply, output)
    backend = {"id": row["backend"]["id"], "commit": row["backend"]["commit"]}
    if row["backend"].get("patches"):
        backend["patches"] = row["backend"]["patches"]
    return {
        "schema_version": "1.0", "status": "complete" if complete else "failed",
        "evidence_tier": "measured", "track": "native_training",
        "backend": backend,
        "case": {"case_id": row["case"]["case_id"], "dataset_id": row["case"]["dataset_id"],
                 "scene_id": row["case"]["scene_id"]},
        "budget": {"iterations": row["iterations"], "eval_split": True},
        "performance": {"wall_time_seconds": wall,
                        "iterations_per_second": row["iterations"] / wall,
                        "peak_process_gpu_memory_mib": peak},
        "artifact": artifact, "quality": quality,
        "provenance": {"gpu_index": args.wait_gpu, "command": row["command"],
                       "repository": row["repository"],
                       "cohort_mode": os.environ.get("EPIC05_TRAINING_COHORT_MODE", "isolated_single_gpu"),
                       "measured_at_utc": datetime.now(timezone.utc).isoformat()},
    }


def run(args) -> dict:
    root = args.root.resolve()
    plan = build_plan(root, args.repository_root.resolve(), args.env_root.resolve(),
                      args.output_root.resolve(), set(args.backend or []), set(args.case or []))
    session_path = args.session.resolve()
    if session_path.is_file():
        if not args.resume:
            raise RuntimeError(f"session already exists; use --resume: {session_path}")
        session = _load(session_path)
    else:
        session = {"schema_version": 1, "status": "running", "completed": [], "failed": []}
        _write(session_path, session)
    done = {(row["backend"], row["case_id"]) for row in session["completed"] + session["failed"]}
    schema = _load(root / "benchmark" / "schemas" / "training-result.schema.json")
    for index, row in enumerate(plan, start=1):
        key = (row["backend"]["id"], row["case"]["case_id"])
        if key in done:
            continue
        wait_for_idle_gpu(args.wait_gpu, args.idle_max_memory_mib,
                          args.idle_max_utilization, args.idle_samples, args.idle_poll_seconds)
        print(f"[{index:02d}/{len(plan)}] {key[0]} :: {key[1]}", flush=True)
        destination = Path(row["output"])
        destination.mkdir(parents=True, exist_ok=True)
        returncode, wall, peak = _run_command(row, args.wait_gpu, destination / "training.log")
        result = _result(row, args, returncode, wall, peak)
        validate_schema(result, schema)
        metrics = destination / "metrics.json"
        _write(metrics, result)
        bucket = "completed" if result["status"] == "complete" else "failed"
        session[bucket].append({"backend": key[0], "case_id": key[1],
                                "metrics_path": str(metrics.relative_to(root)).replace("\\", "/")})
        _write(session_path, session)
    session["status"] = "complete" if not session["failed"] else "complete_with_failures"
    _write(session_path, session)
    return session


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--repository-root", type=Path, default=Path.home() / "renderer-candidates")
    parser.add_argument("--env-root", type=Path, default=Path.home() / "miniforge3" / "envs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "training")
    parser.add_argument("--session", type=Path, default=ROOT / "artifacts" / "run-logs" / "linux-training-session.json")
    parser.add_argument("--evaluator-python", type=Path, default=Path.home() / "miniforge3" / "envs" / "gsplat" / "bin" / "python")
    parser.add_argument("--backend", action="append")
    parser.add_argument("--case", action="append")
    parser.add_argument("--quality-frames", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-gpu", type=int, default=7)
    parser.add_argument("--idle-max-memory-mib", type=float, default=1024.0)
    parser.add_argument("--idle-max-utilization", type=float, default=5.0)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-poll-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(build_plan(args.root.resolve(), args.repository_root.resolve(),
                                    args.env_root.resolve(), args.output_root.resolve(),
                                    set(args.backend or []), set(args.case or [])), indent=2))
        return 0
    session = run(args)
    print(json.dumps({"status": session["status"], "completed": len(session["completed"]),
                      "failed": len(session["failed"])}))
    return 0 if not session["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
