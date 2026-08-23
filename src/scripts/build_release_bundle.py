#!/usr/bin/env python
"""Build the deterministic evidence bundle for a release."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from release_bundle import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())