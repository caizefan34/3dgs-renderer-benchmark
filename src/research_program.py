"""Validation for the repository's independent research-paper tracks."""
from __future__ import annotations

import json
from pathlib import Path

from paper_evidence import PaperEvidenceError, validate_paper_evidence


class ResearchProgramError(ValueError):
    """Raised when research-track manifests are incomplete or ambiguous."""


def validate_research_program(
    manifest_paths: list[str | Path],
    repository_root: str | Path | None = None,
) -> dict[str, dict[str, int]]:
    if not manifest_paths:
        raise ResearchProgramError("at least one research-track manifest is required")

    root = Path(repository_root).resolve() if repository_root else None
    summaries: dict[str, dict[str, int]] = {}
    for manifest_path in manifest_paths:
        path = Path(manifest_path).resolve()
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResearchProgramError(f"cannot read {path.name}: {exc}") from exc
        track_id = manifest.get("track_id")
        if not isinstance(track_id, str) or not track_id.strip():
            raise ResearchProgramError(f"{path.name}: track_id is required")
        if track_id in summaries:
            raise ResearchProgramError(f"duplicate track_id: {track_id}")
        try:
            summaries[track_id] = validate_paper_evidence(path, repository_root=root)
        except PaperEvidenceError as exc:
            raise ResearchProgramError(f"{track_id}: {exc}") from exc
    return summaries
