#!/usr/bin/env python
"""Create an auditable non-ranking index of pre-Matrix-v2 result artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reasons(document) -> list[str]:
    found = ["legacy_schema_without_evidence_tier"]
    suite = document.get("benchmark_suite", {}) if isinstance(document, dict) else {}
    if not isinstance(suite, dict) or not suite.get("validated"):
        found.append("not_hash_validated_matrix_case")
    serialized = json.dumps(document, allow_nan=True)
    if '"renderer_commit_hash": null' in serialized:
        found.append("missing_renderer_commit")
    has_speed = any(token in serialized for token in ('"mean_fps"', '"fps"', '"wall_fps"'))
    has_quality = all(token in serialized for token in ('"psnr"', '"ssim"', '"lpips"'))
    if not has_speed:
        found.append("missing_speed")
    if not has_quality:
        found.append("missing_coupled_gt_quality")
    if "synthetic" in serialized.lower():
        found.append("synthetic_diagnostic_not_gt_ranking")
    return found


def main() -> int:
    rows = []
    for path in sorted((ROOT / "data" / "results").glob("**/*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            exclusion_reasons = reasons(document)
        except json.JSONDecodeError:
            exclusion_reasons = ["invalid_json"]
        rows.append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(path),
            "ranking_eligible": False,
            "reasons": exclusion_reasons,
        })
    output = ROOT / "results" / "quarantine" / "legacy-index.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema_version": 1,
        "policy": "Historical values are preserved verbatim but are not promoted into Matrix v2.",
        "artifact_count": len(rows),
        "artifacts": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
