#!/usr/bin/env python
"""Export reproducibility metadata for a benchmark environment."""
import argparse
import json
import os
import platform
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from benchmark_suite import BENCHMARK_SUITE_VERSION  # noqa: E402


def _command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def collect_environment():
    try:
        import torch
    except Exception:
        torch = None
    cuda_available = bool(torch and torch.cuda.is_available())
    return {
        "schema_version": 1,
        "benchmark_suite_version": BENCHMARK_SUITE_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "nvidia_smi": _command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "torch": getattr(torch, "__version__", None) if torch else None,
        "torch_cuda": getattr(getattr(torch, "version", None), "cuda", None) if torch else None,
        "cuda_available": cuda_available,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(collect_environment(), handle, indent=2, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    main()

