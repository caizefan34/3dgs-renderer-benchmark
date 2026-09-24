#!/usr/bin/env python3
"""C20/C7 local RTX 5070 execution of exactly the C2 coupled tile/CTA protocol."""
from __future__ import annotations
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
# Delegate to shared harness; metadata identifies this hardware result as C7.
cmd=[sys.executable,str(ROOT/"scripts/phase-c20/c20_render_bench.py"),"--candidate","C7","--out",str(ROOT/"results/phase-c20/c7_rtx5070.json"),"--tiles","8","12","16","20","24","32","--camera-count","8","--warmup","5","--repeats","20"]
raise SystemExit(subprocess.call(cmd))
