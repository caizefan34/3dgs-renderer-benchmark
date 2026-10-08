#!/usr/bin/env python3
"""Run one isolated, foreground BES0 stage on an idle MX GPU. No implicit SSH."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from mx_guard import GuardError, ROOT, assert_mx_host, inspect_host, wait_idle_window


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--stage", required=True, choices=["e0", "e1", "e2", "e3"])
    p.add_argument("--entry", required=True, help="Python script inside this package; DSH may add audited stage scripts")
    p.add_argument("--max-minutes", type=int, default=60)
    args = p.parse_args()
    try:
        cfg = assert_mx_host()  # FIRST: no scientific imports or subprocess before this check.
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id):
            raise GuardError("Invalid run id")
        if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", args.gpu_uuid):
            raise GuardError("Invalid GPU UUID")
        if not 1 <= args.max_minutes <= 720:
            raise GuardError("Stage budget must be 1..720 minutes")
        if Path(sys.executable).resolve() != Path(cfg["python"]).resolve():
            raise GuardError("STOP_REMOTE_GUARD: invoke this launcher with the specified nerficg Python")
        entry = (ROOT / args.entry).resolve()
        if not entry.is_relative_to(ROOT) or not entry.is_file() or entry.suffix != ".py":
            raise GuardError("Entry must be an existing Python script inside this package")
        base = Path(cfg["work_root"]).resolve()
        base.mkdir(parents=True, exist_ok=True)
        lock_dir = base / "locks"
        lock_dir.mkdir(exist_ok=True)
        task_lock = (lock_dir / "BES0_TASK.lock").open("a+")
        try:
            fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GuardError("BLOCKED_GPU_OCCUPIED: another BES0 stage is active; this task permits one GPU at a time") from exc
        lock = (lock_dir / (args.gpu_uuid + ".lock")).open("a+")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GuardError("BLOCKED_GPU_OCCUPIED: another BES0 stage holds the task GPU lock") from exc
        # Charge completed wall time; incomplete launches consume their full declared limit.
        charged = 0.0
        for command_path in base.glob("*/*/COMMAND.json"):
            status_path = command_path.parent / "RUN_STATUS.json"
            if status_path.exists():
                charged += float(json.loads(status_path.read_text())["elapsed_wall_seconds"])
            else:
                charged += float(json.loads(command_path.read_text())["timeout_seconds"])
        if charged + args.max_minutes * 60 > 12 * 3600:
            raise GuardError("INCONCLUSIVE_BUDGET: BES0 cumulative 12-hour launch budget exhausted; do not erase history")
        evidence = inspect_host(check_checkpoint=args.stage != "e0")
        evidence["idle_samples"] = wait_idle_window(cfg, args.gpu_uuid)
        stage_dir = base / args.run_id / args.stage
        stage_dir.mkdir(parents=True, exist_ok=False)
        evidence.update({"runner_pid": os.getpid(), "gpu_uuid": args.gpu_uuid, "run_id": args.run_id,
                         "stage": args.stage, "entry": str(entry), "stage_dir": str(stage_dir),
                         "note": "Task-local flock is not a cluster reservation. DSH must also use the site's reservation procedure and prove no interfering processes during authoritative timing."})
        proof = stage_dir / "GUARD.json"
        proof.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": args.gpu_uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_HOME": cfg["cuda_home"], "BES0_GUARD_PROOF": str(proof),
                    "BES0_OUTPUT_DIR": str(stage_dir), "BES0_PACKAGE_ROOT": str(ROOT)})
        env["PATH"] = cfg["cuda_home"] + "/bin:" + str(Path(cfg["python"]).parent) + ":" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = cfg["cuda_home"] + "/lib:" + str(Path(cfg["python"]).parent.parent / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
        (stage_dir / "COMMAND.json").write_text(json.dumps({"argv": [cfg["python"], str(entry)], "timeout_seconds": args.max_minutes * 60}, indent=2))
        began = time.monotonic()
        with (stage_dir / "console.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen([cfg["python"], str(entry)], cwd=stage_dir, env=env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            def interrupt_own_stage(signum, frame):
                raise KeyboardInterrupt
            for sig in (signal.SIGTERM, signal.SIGHUP):
                signal.signal(sig, interrupt_own_stage)
            try:
                rc = child.wait(timeout=args.max_minutes * 60)
                status = "COMPLETED" if rc == 0 else "STAGE_ERROR"
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                # Terminate ONLY this launcher's own process group, never another user's jobs.
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                rc, status = (124, "INCONCLUSIVE_BUDGET") if isinstance(exc, subprocess.TimeoutExpired) else (130, "INTERRUPTED")
        (stage_dir / "RUN_STATUS.json").write_text(json.dumps({"status": status, "exit_code": rc,
            "elapsed_wall_seconds": time.monotonic() - began, "scientific_gate_pass": False,
            "note": "A zero process exit is not a scientific PASS. Read stage evidence and reduce_verdict.py."}, indent=2))
        print(json.dumps({"status": status, "stage_dir": str(stage_dir), "exit_code": rc}))
        return rc
    except (GuardError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
