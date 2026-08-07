#!/usr/bin/env python
"""Run the executable A100 subset of the HiGS paper protocol across 8 GPUs.

Each GPU runs one audited from-SfM 30k job at a time. Per-job artifacts land
under --run-root/<method>/<scene>/s<seed>; protocol-valid result JSONs are
assembled under --result-root. A session JSON records per-job status and
supports --resume. GPU power draw is sampled every few seconds so each job
carries an attributable energy estimate (one job per GPU).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from higs_paper_protocol import build_experiment_plan  # noqa: E402
from higs_training_commands import build_training_invocation  # noqa: E402
from scripts.assemble_higs_paper_results import assemble  # noqa: E402



def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


class PowerSampler:
    """Accumulate nvidia-smi power.draw for one GPU while a process runs."""

    def __init__(self, gpu: int, poll_seconds: float = 5.0):
        self.gpu = gpu
        self.poll_seconds = poll_seconds
        self.energy_joules = 0.0
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
                        "--query-gpu=power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout.strip()
                power = float(out.splitlines()[0])
            except Exception:
                power = 0.0
            now = time.monotonic()
            self.energy_joules += power * max(now - last, 0.0)
            last = now
            self._stop.wait(self.poll_seconds)


def _run_job(job: dict, args, gpu: int, protocol: dict) -> dict:
    scene = job["scene"]
    data_dir = (args.data_root / scene).resolve()
    result_dir = (args.run_root / job["method"] / scene / f"s{job['seed']}").resolve()
    method_spec = protocol["methods"][job["method"]]
    use_higs_source = (method_spec.get("algorithm") or {}).get(
        "renderer"
    ) == "higs_dynamic_native_backward"
    source_dir = (
        args.source_higs.resolve()
        if use_higs_source
        else args.source_gsplat.resolve()
    )
    invocation = build_training_invocation(
        protocol=protocol,
        method=job["method"],
        scene=scene,
        seed=job["seed"],
        data_dir=data_dir,
        result_dir=result_dir,
        source_dir=source_dir,
        python_executable=args.python,
        repository_root=args.root,
        protocol_path=args.protocol,
    )
    command = invocation["command"]
    if args.smoke_steps is not None:
        command = command + ["--smoke-steps", str(args.smoke_steps)]
    result_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started_at = datetime.now(timezone.utc).isoformat()
    sampler = PowerSampler(gpu)
    log_path = result_dir / "training.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env
        )
        sampler.start()
        wall_started = time.perf_counter()
        returncode = process.wait()
        wall = time.perf_counter() - wall_started
        sampler.stop()
    record = {
        "job_id": job["job_id"],
        "status": "failed" if returncode != 0 else "running",
        "gpu": gpu,
        "returncode": returncode,
        "wall_time_seconds": wall,
        "energy_joules": sampler.energy_joules,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        record["error"] = tail
        return record
    if args.smoke_steps is not None:
        record["status"] = "smoke_complete"
        record["smoke_steps"] = args.smoke_steps
        return record
    out = args.result_root / f"{job['job_id']}.json"
    if out.is_file():
        record["status"] = "complete"
        record["result"] = str(out)
        return record
    try:
        result = assemble(
            result_dir, job, gpu_index=gpu, energy_joules=sampler.energy_joules
        )
    except Exception as exc:  # pragma: no cover - defensive
        record["status"] = "needs_assembly"
        record["error"] = f"result assembly failed: {exc}"
        return record
    _write(out, result)
    record["status"] = "complete"
    record["result"] = str(out)
    return record


def _assembly_only(job: dict, args, energy_joules: float = 0.0):
    """Assemble a result from an existing run dir without retraining."""
    result_dir = (args.run_root / job["method"] / job["scene"] / f"s{job['seed']}").resolve()
    out = args.result_root / f"{job['job_id']}.json"
    if out.is_file():
        return {"job_id": job["job_id"], "status": "complete", "result": str(out)}
    if not (result_dir / "paper-run-metadata.json").is_file():
        return None
    try:
        result = assemble(
            result_dir, job, gpu_index=-1, energy_joules=energy_joules
        )
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "job_id": job["job_id"],
            "status": "needs_assembly",
            "error": f"result assembly failed: {exc}",
        }
    _write(out, result)
    return {"job_id": job["job_id"], "status": "complete", "result": str(out)}


def _done(job: dict, session: dict, result_root: Path) -> bool:
    record = session["jobs"].get(job["job_id"])
    if record is None or record.get("status") != "complete":
        return False
    return (result_root / f"{job['job_id']}.json").is_file()


def _plan_jobs(protocol: dict, methods: set[str]) -> list[dict]:
    jobs = [
        job
        for job in build_experiment_plan(protocol)
        if job["executable"]
        and job["hardware"] == "a100"
        and job["method"] in methods
    ]
    # deterministic interleave: scene-major keeps one scene's methods near each
    # other while seeds rotate across GPUs
    jobs.sort(key=lambda job: (job["scene"], job["method"], job["seed"]))
    return jobs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "benchmark" / "higs-paper-protocol.json"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-higs", type=Path, required=True)
    parser.add_argument("--source-gsplat", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=ROOT / "artifacts" / "training-paper" / "runs")
    parser.add_argument("--result-root", type=Path, default=ROOT / "artifacts" / "training-paper" / "results")
    parser.add_argument("--session", type=Path, default=ROOT / "artifacts" / "training-paper" / "session.json")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--methods", default="gsplat,higs_full,higs_proposed")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--smoke-steps", type=int, default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)

    if args.session.is_file() and not args.resume:
        raise SystemExit(
            f"refusing to clobber existing session {args.session}: pass --resume "
            "to continue/assemble, or move the session aside to start a fresh run"
        )

    protocol = _load(args.protocol)
    methods = set(part.strip() for part in args.methods.split(",") if part.strip())
    unknown = methods - set(protocol["methods"])
    if unknown:
        raise SystemExit(f"unknown methods: {sorted(unknown)}")
    gpus = [int(part) for part in args.gpus.split(",") if part.strip()]
    if not gpus:
        raise SystemExit("--gpus must list at least one GPU")
    jobs = _plan_jobs(protocol, methods)
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]

    session = {"schema_version": "1.0", "started_at_utc": None, "jobs": {}}
    if args.resume and args.session.is_file():
        session = _load(args.session)
    session["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    known = {job["job_id"]: job for job in jobs}
    session["jobs"] = {
        job_id: record
        for job_id, record in session.get("jobs", {}).items()
        if job_id in known
    }
    _write(args.session, session)

    lock = threading.Lock()
    for job in jobs:
        record = session["jobs"].get(job["job_id"])
        if record is None or record.get("status") not in ("needs_assembly", "complete", "failed"):
            continue
        if _done(job, session, args.result_root):
            continue
        repaired = _assembly_only(
            job, args, energy_joules=float(record.get("energy_joules", 0.0))
        )
        if repaired is not None:
            session["jobs"][job["job_id"]] = repaired
            _write(args.session, session)
            print(
                f"[scheduler] assembled existing run {job['job_id']} -> {repaired['status']}",
                flush=True,
            )
    remaining = [job for job in jobs if not _done(job, session, args.result_root)]
    complete = len(jobs) - len(remaining)
    print(
        f"[scheduler] planned={len(jobs)} completed={complete} remaining={len(remaining)} "
        f"gpus={gpus} methods={sorted(methods)}",
        flush=True,
    )
    if not remaining:
        print("[scheduler] nothing to do", flush=True)
        return 0

    def worker(gpu: int) -> None:
        while True:
            with lock:
                if not remaining:
                    return
                job = remaining.pop(0)
            record = _run_job(job, args, gpu, protocol)
            with lock:
                session["jobs"][job["job_id"]] = record
                _write(args.session, session)
            print(
                f"[scheduler] {job['job_id']} -> {record['status']} "
                f"wall={record['wall_time_seconds']:.1f}s energy={record['energy_joules']:.0f}J gpu={gpu}",
                flush=True,
            )

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failed = [
        job["job_id"]
        for job in jobs
        if session["jobs"].get(job["job_id"], {}).get("status") in ("failed", "needs_assembly")
    ]
    print(
        f"[scheduler] done: complete={len(jobs) - len(failed)} failed={len(failed)} "
        f"session={args.session}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
