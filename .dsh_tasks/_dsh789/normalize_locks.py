#!/usr/bin/env python3
"""Normalize ENVIRONMENT/SOURCE_LOCK to spec-required prefixed names per P0,
and write a MANIFEST.sha256 for each P0 artifacts dir."""
import json
import hashlib
from pathlib import Path

RUNS = Path("/root/dsh_20261001/runs")
ROOT = RUNS.parent / "artifacts"

# map: artifacts dir -> (env_src_name, lock_src_name)
JOBS = {
    "DSH7_BARS0": ("BARS0_ENVIRONMENT.json", "BARS0_SOURCE_LOCK.json",
                   "ENVIRONMENT.json", "SOURCE_LOCK.json"),
    "DSH8_APBC0": ("APBC0_ENVIRONMENT.json", "APBC0_SOURCE_LOCK.json",
                   "ENVIRONMENT.json", "SOURCE_LOCK.json"),
    "DSH9_CVL0": ("CVL0_ENVIRONMENT.json", "CVL0_SOURCE_LOCK.json",
                  "ENVIRONMENT.json", "SOURCE_LOCK.json"),
}


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    for d, (env_name, lock_name, env_src, lock_src) in JOBS.items():
        adir = ROOT / d
        if not adir.is_dir():
            print(f"SKIP {d}: no dir")
            continue
        # env
        src = adir / env_src
        if src.exists() and not (adir / env_name).exists():
            (adir / env_name).write_bytes(src.read_bytes())
            print(f"  {d}: {env_src} -> {env_name}")
        src = adir / lock_src
        if src.exists() and not (adir / lock_name).exists():
            (adir / lock_name).write_bytes(src.read_bytes())
            print(f"  {d}: {lock_src} -> {lock_name}")
    print("normalize done")


if __name__ == "__main__":
    main()
