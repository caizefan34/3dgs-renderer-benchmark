"""Shared utilities for Phase C0 audit scripts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "a100" / "phase-c0"
OFFICIAL_SHA = "54c035f7834b564019656c3e3fcc3646292f727d"


def write_json(name: str, payload: dict) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment() -> dict:
    import torch

    try:
        import gsplat
        gsplat_version = getattr(gsplat, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - diagnostic path
        gsplat_version = f"unavailable: {exc}"
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gsplat": gsplat_version,
    }
