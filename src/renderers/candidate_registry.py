"""Renderer candidate registry helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def load_renderer_candidates(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def validate_renderer_candidates(registry: Mapping) -> dict:
    if registry.get("schema_version") != 1:
        raise ValueError("renderer candidate registry schema_version must be 1")
    seen = set()
    for candidate in registry.get("candidates", []):
        renderer_id = candidate.get("renderer_id")
        if not renderer_id:
            raise ValueError("renderer candidate is missing renderer_id")
        if renderer_id in seen:
            raise ValueError(f"duplicate renderer_id: {renderer_id}")
        seen.add(renderer_id)
        if not candidate.get("official_url"):
            raise ValueError(f"{renderer_id} is missing official_url")
        if candidate.get("quality_mode") is None:
            raise ValueError(f"{renderer_id} is missing quality_mode")
    return {
        "schema_version": 1,
        "status": "ok",
        "candidate_count": len(seen),
        "measured_or_tracked": sorted(seen),
    }

