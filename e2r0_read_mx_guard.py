#!/usr/bin/env python3
"""Fail-closed MX host and GPU guard. Standard library only; no CUDA import."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = ROOT / "config" / "mx_target.json"


class GuardError(RuntimeError):
    pass


def load_target():
    return json.loads(TARGET_PATH.read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assert_mx_host():
    cfg = load_target()
    actual = socket.gethostname().split(".", 1)[0]
    if platform.system() != "Linux" or actual != cfg["expected_hostname"]:
        raise GuardError(f"STOP_REMOTE_GUARD: target=MX expected={cfg['expected_hostname']} actual={actual}")
    return cfg


def smi(args):
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise GuardError("STOP_REMOTE_GUARD: nvidia-smi unavailable")
    p = subprocess.run([executable, *args], text=True, capture_output=True, timeout=20)
    if p.returncode:
        raise GuardError("STOP_REMOTE_GUARD: GPU inventory could not be verified")
    return list(csv.reader(p.stdout.strip().splitlines(), skipinitialspace=True))


def inventory():
    rows = smi(["--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
    out = []
    try:
        for row in rows:
            if len(row) != 6:
                raise ValueError("unexpected GPU record")
            out.append({"index": int(row[0]), "uuid": row[1].strip(), "name": row[2].strip(),
                        "memory_total_mib": int(row[3]), "memory_used_mib": int(row[4]), "utilization_percent": int(row[5])})
    except (ValueError, IndexError) as exc:
        raise GuardError("STOP_REMOTE_GUARD: incomplete GPU metrics") from exc
    return out


def compute_processes():
    rows = smi(["--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"])
    out = []
    for row in rows:
        if not row:
            continue
        if len(row) < 3:
            raise GuardError("STOP_REMOTE_GUARD: GPU process inventory incomplete")
        try:
            out.append({"uuid": row[0].strip(), "pid": int(row[1]), "process": ",".join(row[2:]).strip()})
        except ValueError as exc:
            raise GuardError("STOP_REMOTE_GUARD: GPU process PID unavailable") from exc
    return out


def checked_inventory(cfg):
    rows = inventory()
    if len(rows) != cfg["expected_gpu_count"] or any(x["name"] != cfg["expected_gpu_name"] for x in rows):
        raise GuardError("STOP_REMOTE_GUARD: MX GPU inventory differs from the frozen target")
    return rows


def inspect_host(check_checkpoint=True):
    cfg = assert_mx_host()
    gpus = checked_inventory(cfg)
    if not Path(cfg["python"]).is_file() or not Path(cfg["cuda_home"]).is_dir():
        raise GuardError("STOP_REMOTE_GUARD: expected MX environment paths missing")
    evidence = {"schema": "mss0.host_evidence.v1", "hostname": socket.gethostname(),
                "target_config_sha256": sha256(TARGET_PATH), "unix_time": time.time(),
                "gpu_inventory": gpus, "compute_processes": compute_processes(),
                "checkpoint_sha256": None, "ssh_identity_verification": "DSH must attach its pre-existing trusted MX SSH-profile evidence; this script does not verify the SSH endpoint."}
    if check_checkpoint:
        p = Path(cfg["checkpoint"])
        if not p.is_file() or sha256(p) != cfg["checkpoint_sha256"]:
            raise GuardError("BLOCKED_INPUTS: source checkpoint missing or SHA256 differs")
        evidence["checkpoint_sha256"] = cfg["checkpoint_sha256"]
    return evidence


def assert_idle(cfg, gpu_uuid):
    rows = checked_inventory(cfg)
    found = [x for x in rows if x["uuid"] == gpu_uuid]
    if len(found) != 1:
        raise GuardError("STOP_REMOTE_GUARD: GPU UUID not found uniquely")
    g = found[0]
    if g["index"] not in cfg["allowed_gpu_indices"]:
        raise GuardError("STOP_REMOTE_GUARD: GPU is outside this task's permitted set")
    processes = [x for x in compute_processes() if x["uuid"] == gpu_uuid]
    if processes or g["memory_used_mib"] > cfg["idle_memory_mib_max"] or g["utilization_percent"] > cfg["idle_utilization_percent_max"]:
        raise GuardError("BLOCKED_GPU_OCCUPIED: GPU has resident work or is not idle; do not kill, reset or evict it")
    return g


def wait_idle_window(cfg, gpu_uuid):
    samples = []
    for i in range(cfg["idle_observation_seconds"] + 1):
        samples.append({"unix_time": time.time(), **assert_idle(cfg, gpu_uuid)})
        if i < cfg["idle_observation_seconds"]:
            time.sleep(1)
    return samples


def require_guarded_process():
    cfg = assert_mx_host()
    proof_name = os.environ.get("MSS0_GUARD_PROOF")
    if not proof_name:
        raise GuardError("STOP_REMOTE_GUARD: launch through scripts/run_on_mx.py")
    proof_path = Path(proof_name).resolve()
    if not proof_path.is_relative_to(Path(cfg["work_root"]).resolve()):
        raise GuardError("STOP_REMOTE_GUARD: proof is outside MX task workspace")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if proof.get("target_config_sha256") != sha256(TARGET_PATH) or proof.get("hostname", "").split(".", 1)[0] != cfg["expected_hostname"]:
        raise GuardError("STOP_REMOTE_GUARD: proof does not match this target")
    if os.getppid() != proof.get("runner_pid") or os.environ.get("CUDA_VISIBLE_DEVICES") != proof.get("gpu_uuid"):
        raise GuardError("STOP_REMOTE_GUARD: proof is stale or GPU mapping differs")
    if not Path(sys.executable).resolve() == Path(cfg["python"]).resolve():
        raise GuardError("STOP_REMOTE_GUARD: incorrect Python environment")
    return cfg, proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(inspect_host(), indent=2, ensure_ascii=False))
    except GuardError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
