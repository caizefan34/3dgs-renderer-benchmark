#!/usr/bin/env python
"""Repository-local wrapper for the measured-first benchmark CLI."""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from benchmark_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
