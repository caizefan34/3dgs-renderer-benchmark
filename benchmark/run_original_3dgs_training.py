#!/usr/bin/env python
"""Run one audited from-SfM official 3DGS job for the HiGS paper protocol.

The official graphdeco-inria/gaussian-splatting train.py runs as a subprocess
at the protocol's fixed commit (plus the audited seed patch). Wall-clock
markers are captured from training.log at the 7k/15k/30k eval+save points, and
each checkpoint is scored on the official test split (every 8th image) with the
official PSNR/SSIM implementations and the official VGG LPIPS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from higs_training_commands import build_original_3dgs_invocation  # noqa: E402

EVAL_SCRIPT = ROOT / "src" / "scripts" / "eval_original_3dgs_checkpoint.py"
EVAL_STEPS = (7_000, 15_000, 30_000)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("original_3dgs",), required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, choices=(30_000,), default=30_000)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--smoke-steps",
        type=int,
        help="Run a non-paper smoke test; its metadata is never paper-eligible.",
    )
    return parser.parse_args()


class Sampler:
    """Sample nvidia-smi power and memory while the training subprocess runs."""

    def __init__(self, gpu: int, poll_seconds: float = 2.0):
        self.gpu = gpu
        self.poll_seconds = poll_seconds
        self.energy_joules = 0.0
        self.peak_memory_mib = 0.0
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def _run(self) -> None:
        last = time.monotonic()
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.gpu}",
                        "--query-gpu=power.draw,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout.strip()
                power, mem = (float(x) for x in out.splitlines()[0].split(","))
            except Exception:
                power = 0.0
                mem = 0.0
            now = time.monotonic()
            self.energy_joules += power * max(now - last, 0.0)
            self.peak_memory_mib = max(self.peak_memory_mib, mem)
            last = now
            self._stop.wait(self.poll_seconds)


class LogWatcher:
    """Record monotonic wall times when the official trainer saves checkpoints.

    The official train.py evaluates (and prints) before it prints
    "[ITER N] Saving Gaussians", so the save marker time is the wall clock at
    which the eval plus save for that checkpoint is complete.
    """

    def __init__(self, log_path: Path, steps):
        self.log_path = log_path
        self.steps = set(steps)
        self.save_times = {}
        self._stop = threading.Event()
        self._thread = None
        self._save_re = re.compile(r"\[ITER (\d+)\] Saving Gaussians")

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def _run(self) -> None:
        pos = 0
        while not self._stop.is_set():
            try:
                with self.log_path.open("r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(pos)
                    data = fh.read()
                    pos = fh.tell()
            except OSError:
                data = ""
            now = time.monotonic()
            for match in self._save_re.finditer(data):
                step = int(match.group(1))
                if step in self.steps:
                    self.save_times.setdefault(step, now)
            self._stop.wait(1.0)


def main() -> int:
    args = _parse_args()
    if args.data_dir.suffix.lower() in {".ckpt", ".ply", ".pt", ".pth"}:
        raise SystemExit("--data-dir must be a COLMAP dataset directory")
    if not args.data_dir.is_dir():
        raise SystemExit(f"dataset directory does not exist: {args.data_dir}")
    protocol = json.loads(
        (ROOT / "benchmark" / "higs-paper-protocol.json").read_text(encoding="utf-8")
    )
    invocation = build_original_3dgs_invocation(
        protocol=protocol,
        scene=args.scene,
        seed=args.seed,
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        source_dir=args.source_dir,
        python_executable=sys.executable,
        repository_root=ROOT,
        smoke_steps=args.smoke_steps,
    )
    args.result_dir.mkdir(parents=True, exist_ok=True)
    iterations = invocation["iterations"]
    eval_steps = invocation["eval_steps"]
    log_path = args.result_dir / "training.log"
    env = dict(os.environ)
    started_at = datetime.now(timezone.utc).isoformat()
    sampler = Sampler(args.gpu_id)
    watcher = LogWatcher(log_path, eval_steps)
    wall_started = time.monotonic()
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            invocation["command"],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(Path(args.source_dir).resolve()),
        )
        sampler.start()
        watcher.start()
        returncode = process.wait()
    wall = time.monotonic() - wall_started
    sampler.stop()
    watcher.stop()
    if returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise SystemExit(f"official 3DGS training failed rc={returncode}\n{tail}")

    stats_dir = args.result_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    for step in eval_steps:
        save_wall = watcher.save_times.get(step)
        if save_wall is None:
            raise SystemExit(f"missing eval/save marker for step {step} in {log_path}")
        val_path = stats_dir / f"val_step{step - 1:04d}.json"
        eval_command = [
            sys.executable,
            str(EVAL_SCRIPT),
            "--source-dir",
            str(Path(args.source_dir).resolve()),
            "-m",
            str(args.result_dir.resolve()),
            "--iteration",
            str(step),
            "--output",
            str(val_path),
        ]
        subprocess.run(eval_command, check=True, env=env, cwd=str(ROOT))
        val = json.loads(val_path.read_text(encoding="utf-8"))
        stats_path = stats_dir / f"train_step{step - 1:04d}_rank0.json"
        stats_path.write_text(
            json.dumps(
                {
                    "mem": sampler.peak_memory_mib / 1024.0,
                    "ellipse_time": save_wall - wall_started,
                    "num_GS": int(val["num_GS"]),
                }
            )
            + "\n",
            encoding="utf-8",
        )

    import torch

    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    artifact_files = sorted(
        [p for p in (args.result_dir / "point_cloud").rglob("point_cloud.ply")]
        + list(stats_dir.glob("*.json"))
        + [args.result_dir / "cfg_args", log_path]
    )
    artifacts = [
        {
            "path": path.relative_to(args.result_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in artifact_files
        if path.is_file()
    ]
    dataset_inventory = args.data_dir / "dataset_inventory.json"
    metadata = {
        "schema_version": "1.0",
        "method": "original_3dgs",
        "scene": args.scene,
        "run_kind": "smoke" if args.smoke_steps is not None else "paper",
        "paper_eligible": args.smoke_steps is None,
        "initialization": "from_scratch_sfm",
        "iterations": iterations,
        "seed": args.seed,
        "timing_boundary": "dataset_ready_to_final_checkpoint",
        "started_at_utc": started_at,
        "wall_time_seconds": wall,
        "source_dir": str(Path(args.source_dir).resolve()),
        "source": invocation["source"],
        "source_audit": invocation["source_audit"],
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "hardware": {
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
        },
        "dataset": {
            "path": str(args.data_dir.resolve()),
            "inventory_sha256": (
                _sha256(dataset_inventory) if dataset_inventory.is_file() else None
            ),
        },
        "artifacts": artifacts,
        "resources": {
            "energy_joules": sampler.energy_joules,
            "peak_gpu_memory_mib": sampler.peak_memory_mib,
        },
        "clean_process": True,
    }
    (args.result_dir / "paper-run-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
