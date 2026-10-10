#!/usr/bin/env python3
"""R17 collector: RUN_STATUS.json + MANIFEST_SHA256.json.

Scans the R17 output directory, states per-deliverable status (EXECUTED with a
parseable payload / EMPTY / NOT_EXECUTED), merges the per-step exit codes written
by run_on_mx.sh, and hashes every produced file.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CKPT, OUT, PROJECT, R15_CAPTURE, R16_DIR, ensure_out_dirs,
                    gpu_mem_snapshot, sha256_file, write_json)

REQUIRED = [
    "R17_R16_ARITHMETIC_AUDIT.json",
    "R17_PARAMETER_CONTRACT.md",
    "R17_PARAMETER_DISTRIBUTIONS.json",
    "R17_CAMERA_IDENTITY.json",
    "R17_580K_CONTROLLED_COMPARISON.csv",
    "R17_INTERSECTION_COVERAGE.csv",
    "R17_VALIDATED_BENCHMARK.csv",
    "R17_4P4M_SAFETY_GATE.json",
    "R17_FINAL_REPORT.md",
    "R17_FINAL_VERDICT.json",
]


def check_deliverable(p: Path):
    info = {"path": str(p), "exists": p.is_file()}
    if not info["exists"]:
        info["status"] = "NOT_EXECUTED"
        return info
    info["bytes"] = p.stat().st_size
    info["sha256"] = sha256_file(p)
    if info["bytes"] == 0:
        info["status"] = "EMPTY"
        return info
    try:
        if p.suffix == ".json":
            json.loads(p.read_text())
        elif p.suffix == ".csv":
            with p.open(newline="") as f:
                rows = list(csv.reader(f))
            info["csv_rows"] = max(0, len(rows) - 1)
        info["status"] = "EXECUTED"
    except Exception as e:  # noqa: BLE001
        info["status"] = "PARSE_ERROR"
        info["error"] = repr(e)
    return info


def main():
    ensure_out_dirs()
    steps = {"status": "NOT_WRITTEN"}
    txt = OUT / "logs" / "step_exit_codes.txt"
    if txt.is_file():
        steps = {}
        for line in txt.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                steps[parts[0]] = int(parts[1])
    status = {
        "deliverable": "RUN_STATUS.json",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(OUT),
        "project": str(PROJECT),
        "checkpoint": str(CKPT),
        "r15_capture": str(R15_CAPTURE),
        "r16_dir": str(R16_DIR),
        "step_exit_codes": steps,
        "deliverables": {},
        "extra_outputs": {},
        "cuda_final": gpu_mem_snapshot("final"),
        "env": {},
    }
    import os
    status["env"] = {k: os.environ.get(k) for k in
                     ("CUDA_VISIBLE_DEVICES", "HIGS_PROJECT", "R17_OUT", "R17_CKPT")}
    try:
        import torch
        status["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception as e:  # noqa: BLE001
        status["torch"] = {"error": repr(e)}

    executed = 0
    for rel in REQUIRED:
        info = check_deliverable(OUT / rel)
        status["deliverables"][rel] = info
        executed += info["status"] == "EXECUTED"
    status["summary"] = {"required": len(REQUIRED), "executed": executed,
                         "not_executed": [k for k, v in status["deliverables"].items()
                                          if v["status"] != "EXECUTED"]}
    for sub in ("raw", "logs", "oracle", "captures"):
        d = OUT / sub
        if d.is_dir():
            status["extra_outputs"][sub] = sorted(p.name for p in d.rglob("*") if p.is_file())
    write_json(OUT / "RUN_STATUS.json", status)

    manifest = {"deliverable": "MANIFEST_SHA256.json",
                "generated_utc": status["generated_utc"], "files": {}}
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in ("MANIFEST_SHA256.json",):
            manifest["files"][str(p.relative_to(OUT))] = {
                "bytes": p.stat().st_size, "sha256": sha256_file(p)}
    write_json(OUT / "MANIFEST_SHA256.json", manifest)
    print(f"collector: {executed}/{len(REQUIRED)} required deliverables executed; "
          f"{len(manifest['files'])} files hashed", flush=True)


if __name__ == "__main__":
    main()
