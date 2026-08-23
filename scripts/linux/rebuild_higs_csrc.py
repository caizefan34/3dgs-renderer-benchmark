#!/usr/bin/env python3
"""Incremental rebuild of the gsplat CUDA extensions (main + experimental HiGS).

The old EPIC-05 flow used a one-off /root/rebuild_higs_csrc.py that was lost
when the container was re-provisioned. This is its tracked replacement:
it runs ``setup.py build_ext --inplace`` (ninja-based, incremental) so a
single .cu edit only recompiles that file, then verifies the built .so files
and clears the stale torch JIT cache entry.

Usage (on EPIC-05):
    cd /root/3dgs-roadmap-matrix && \\
    PATH=/root/miniforge3/envs/gsplat/bin:/usr/local/cuda-12.9/bin:/usr/bin:/bin \\
    CUDA_HOME=/usr/local/cuda-12.9 CC=gcc CXX=g++ CUDA_VISIBLE_DEVICES=0 \\
    TORCH_DONT_CHECK_COMPILER_ABI=1 MAX_JOBS=6 \\
    python scripts/linux/rebuild_higs_csrc.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GSPLAT = ROOT / "artifacts" / "renderer-sources" / "gsplat"

EXPECTED_SO = [
    GSPLAT / "gsplat" / "csrc" / "csrc.so",
    GSPLAT / "gsplat" / "experimental" / "render" / "kernels" / "csrc.so",
]

def main() -> int:
    if not GSPLAT.exists():
        print(f"ERROR: gsplat tree not found at {GSPLAT}", file=sys.stderr)
        return 1
    env = dict(os.environ)
    env.setdefault("MAX_JOBS", "6")
    env.setdefault("BUILD_EXPERIMENTAL", "1")
    cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
    print(f"=== building in {GSPLAT} ===")
    ret = subprocess.call(cmd, cwd=str(GSPLAT), env=env)
    if ret != 0:
        print(f"=== build failed (exit {ret}); tail of setup.py output above ===")
        return ret
    missing = [p for p in EXPECTED_SO if not p.exists()]
    if missing:
        print("ERROR: expected .so files missing:")
        for p in missing:
            print("  ", p)
        return 1
    for p in EXPECTED_SO:
        print(f"OK {p} ({p.stat().st_size} bytes)")
    # Clear the torch JIT cache so a stale inplace build is never reused.
    cache = Path.home() / ".cache" / "torch_extensions"
    if cache.exists():
        stale = [d for d in cache.iterdir() if "higs_csrc_inplace" in d.name]
        for d in stale:
            shutil.rmtree(d, ignore_errors=True)
            print(f"cleared stale JIT cache {d}")
    print("=== rebuild OK ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
