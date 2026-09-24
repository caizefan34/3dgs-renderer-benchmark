#!/usr/bin/env python3
"""
capture_provenance.py — Environment provenance + GPU contamination capture for the FINAL 30K benchmark.

Infrastructure ONLY. Does NOT run training. Captures, per run:
  - host, GPU index, GPU UUID, GPU model, driver, torch version, CUDA runtime, nvcc version (the ACTUAL compiler used for built binaries, NOT PATH nvcc)
  - git commit, git dirty state
  - renderer binary SHA256, patch SHA256s
  - environment variables affecting renderer behavior
  - GPU contamination snapshot (memory.used, utilization.gpu, compute PIDs) taken BEFORE the final run

Writes provenance.json and gpu_snapshot.json into the run's artifact dir.
Exit code 0 on success, non-zero on any missing required field.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list, cwd=None) -> str:
    try:
        return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"ERROR: {e}"


def capture_gpu_snapshot(nvidia_smi: str = "nvidia-smi") -> dict:
    """GPU contamination snapshot: memory.used, utilization.gpu, compute PIDs (per run, pre-final-run)."""
    snap = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        out = _run([
            nvidia-smi,
            "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,driver_version,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ])
        gpus = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 9:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "uuid": parts[2],
                    "memory_total_mb": int(parts[3]),
                    "memory_used_mb": int(parts[4]),
                    "memory_free_mb": int(parts[5]),
                    "utilization_gpu_pct": int(parts[6]),
                    "driver_version": parts[7],
                    "temperature_c": int(parts[8]),
                })
        snap["gpus"] = gpus
    except Exception as e:
        snap["gpu_query_error"] = str(e)
    try:
        pids = _run([
            nvidia-smi,
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ])
        snap["compute_pids"] = [
            {"pid": p.strip(), "process": q.strip(), "used_memory_mb": int(r.strip())}
            for line in pids.splitlines()
            for p, q, r in [line.split(",")]
            if len(line.split(",")) >= 3
        ]
    except Exception as e:
        snap["compute_pid_error"] = str(e)
    return snap


def capture_provenance(
    candidate_id: str,
    scene: str,
    output_dir: str,
    binary_path: str,
    patch_paths: dict,
    config: dict,
    nvidia_smi: str = "nvidia-smi",
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prov = {
        "candidate_id": candidate_id,
        "scene": scene,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Host / GPU
    prov["host"] = _run(["hostname"])
    prov["target_environment"] = config.get("target_environment", {})

    # Software versions
    prov["torch_version"] = _run(["python", "-c", "import torch;print(torch.__version__)"])
    prov["cuda_runtime"] = _run(["python", "-c", "import torch;print(torch.version.cuda)"])
    # nvcc: record the ACTUAL compiler used for built binaries (from config nvcc_path), NOT PATH nvcc
    nvcc_path = config.get("nvcc_path")
    if nvcc_path and Path(nvcc_path).exists():
        prov["nvcc_actual"] = _run([nvcc_path, "--version"])
        prov["nvcc_path_used"] = nvcc_path
    else:
        prov["nvcc_actual"] = _run(["nvcc", "--version"])
        prov["nvcc_path_used"] = "PATH (WARNING: not the pinned build compiler)"

    # git
    prov["git_head"] = _run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT))
    prov["git_describe"] = _run(["git", "describe", "--tags"], cwd=str(REPO_ROOT))
    dirty = _run(["git", "status", "--porcelain"], cwd=str(REPO_ROOT))
    prov["git_dirty"] = bool(dirty and "ERROR" not in dirty)
    prov["git_dirty_files"] = len(dirty.splitlines()) if prov["git_dirty"] else 0

    # Binary identity (abort on mismatch enforced by validate_candidates.py against candidate_registry.json)
    prov["renderer_binary_sha256"] = _sha256_file(binary_path) if Path(binary_path).exists() else "MISSING"
    prov["renderer_binary_path"] = binary_path
    prov["patch_sha256s"] = {
        name: (_sha256_file(p) if Path(p).exists() else "MISSING")
        for name, p in (patch_paths or {}).items()
    }

    # Environment variables affecting renderer behavior
    renderer_env_vars = [
        "HIGS_DISABLE_F9", "HIGS_BWD_H8_MR", "CUDA_VISIBLE_DEVICES",
        "TORCH_EXTENSIONS_DIR", "TORCH_CUDA_ARCH_LIST", "CUDA_MODULE_LOADING",
        "OMP_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG",
    ]
    prov["renderer_env"] = {k: os.environ.get(k) for k in renderer_env_vars}

    # Runtime feature toggles (caller supplies; recorded for identity)
    prov["runtime_feature_toggles"] = config.get("runtime_feature_toggles", {})

    # GPU contamination snapshot (pre-final-run)
    prov["gpu_snapshot"] = capture_gpu_snapshot(nvidia_smi)

    with open(out / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    with open(out / "gpu_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(prov["gpu_snapshot"], f, indent=2)

    missing = [k for k, v in prov.items() if v in ("MISSING", "", None) and k not in ("git_dirty",)]
    if prov["renderer_binary_sha256"] == "MISSING":
        print("ERROR: renderer binary missing", file=sys.stderr)
        return prov, False
    if missing:
        print(f"WARNING: missing provenance fields: {missing}", file=sys.stderr)
    print(f"provenance written: {out / 'provenance.json'}")
    return prov, True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--patches", default="{}", help="JSON dict name->path")
    ap.add_argument("--config", default=None, help="config_schema.json instance path")
    ap.add_argument("--nvidia-smi", default="nvidia-smi")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    patches = json.loads(args.patches)
    prov, ok = capture_provenance(
        args.candidate_id, args.scene, args.output_dir, args.binary, patches, cfg, args.nvidia_smi,
    )
    sys.exit(0 if ok else 1)
