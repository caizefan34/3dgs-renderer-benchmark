"""Validation for paper claims bound to immutable repository evidence."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path


class PaperEvidenceError(ValueError):
    """Raised when a paper claim is unsupported or its evidence has drifted."""


def _pointer(document, pointer: str):
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise PaperEvidenceError(f"invalid JSON pointer: {pointer!r}")
    value = document
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        try:
            value = value[int(part)] if isinstance(value, list) else value[part]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise PaperEvidenceError(f"JSON pointer not found: {pointer!r}") from exc
    return value


def _matches(item, where: dict) -> bool:
    return all(_pointer(item, pointer) == expected for pointer, expected in where.items())


def _assert_evidence(document, assertion: dict, label: str) -> None:
    kind = assertion["type"]
    if kind == "value_equals":
        actual = _pointer(document, assertion["pointer"])
        expected = assertion["equals"]
        tolerance = float(assertion.get("tolerance", 0.0))
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            passed = abs(actual - expected) <= tolerance
        else:
            passed = actual == expected
        if not passed:
            raise PaperEvidenceError(f"{label}: expected {expected!r}, got {actual!r}")
        return

    rows = _pointer(document, assertion["pointer"])
    if not isinstance(rows, list):
        raise PaperEvidenceError(f"{label}: assertion pointer must resolve to a list")
    selected = [item for item in rows if _matches(item, assertion.get("where", {}))]
    if kind == "select_count":
        actual = len(selected)
    else:
        if not selected:
            raise PaperEvidenceError(f"{label}: assertion selected no rows")
        field = assertion["field"]
        values = [_pointer(item, field) for item in selected]
        if kind == "select_all":
            if not all(value == assertion["equals"] for value in values):
                raise PaperEvidenceError(f"{label}: not every selected value equals {assertion['equals']!r}")
            return
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise PaperEvidenceError(f"{label}: numeric assertion received a non-finite value")
        actual = {
            "select_min": min,
            "select_max": max,
            "select_max_abs": lambda items: max(abs(value) for value in items),
        }.get(kind, lambda _items: None)(values)
        if actual is None:
            raise PaperEvidenceError(f"{label}: unsupported assertion type {kind!r}")

    if "equals" in assertion:
        tolerance = float(assertion.get("tolerance", 0.0))
        if isinstance(actual, (int, float)) and isinstance(assertion["equals"], (int, float)):
            passed = abs(actual - assertion["equals"]) <= tolerance
        else:
            passed = actual == assertion["equals"]
        if not passed:
            raise PaperEvidenceError(
                f"{label}: expected {assertion['equals']!r}, got {actual!r}"
            )
    elif "less_than" in assertion:
        if not actual < assertion["less_than"]:
            raise PaperEvidenceError(
                f"{label}: expected {actual!r} < {assertion['less_than']!r}"
            )
    else:
        raise PaperEvidenceError(f"{label}: assertion has no comparison")


def _require_tracked(path: Path, repository_root: Path) -> None:
    relative = path.relative_to(repository_root).as_posix()
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=repository_root,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise PaperEvidenceError(f"evidence is not Git-tracked: {relative}")


def validate_paper_evidence(
    manifest_path: str | Path, repository_root: str | Path | None = None
) -> dict[str, int]:
    manifest_path = Path(manifest_path).resolve()
    root = Path(repository_root).resolve() if repository_root else manifest_path.parent
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperEvidenceError(f"cannot read claim manifest: {exc}") from exc

    if manifest.get("schema_version") != "1.0" or not manifest.get("paper_scope"):
        raise PaperEvidenceError("claim manifest requires schema_version 1.0 and paper_scope")
    contributions = manifest.get("contributions", [])
    if not 1 <= len(contributions) <= 3:
        raise PaperEvidenceError("a paper must declare one to at most three contributions")
    contribution_ids = [item.get("id") for item in contributions]
    if len(set(contribution_ids)) != len(contribution_ids) or any(
        not item.get("statement") for item in contributions
    ):
        raise PaperEvidenceError("contributions require unique IDs and statements")

    summary = {"supported": 0, "blocked": 0, "out_of_scope": 0}
    claim_ids = set()
    for claim in manifest.get("claims", []):
        claim_id = claim.get("id")
        if not claim_id or claim_id in claim_ids or not claim.get("statement"):
            raise PaperEvidenceError("claims require unique IDs and statements")
        claim_ids.add(claim_id)
        if claim.get("contribution_id") not in contribution_ids:
            raise PaperEvidenceError(f"{claim_id}: unknown contribution_id")
        status = claim.get("status")
        if status not in summary:
            raise PaperEvidenceError(f"{claim_id}: invalid status {status!r}")
        summary[status] += 1
        if status == "blocked":
            if not claim.get("blocking_gate"):
                raise PaperEvidenceError(f"{claim_id}: blocked claim requires blocking_gate")
            continue
        if status == "out_of_scope":
            if not claim.get("rationale"):
                raise PaperEvidenceError(f"{claim_id}: out-of-scope claim requires rationale")
            continue

        evidence_items = claim.get("evidence", [])
        if not evidence_items:
            raise PaperEvidenceError(f"{claim_id}: supported claim requires evidence")
        for index, evidence in enumerate(evidence_items):
            label = f"{claim_id}.evidence[{index}]"
            path = (root / evidence["path"]).resolve()
            if root not in path.parents:
                raise PaperEvidenceError(f"{label}: evidence escapes repository root")
            if not path.is_file():
                raise PaperEvidenceError(f"{label}: missing evidence {evidence['path']}")
            if repository_root:
                _require_tracked(path, root)
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != evidence.get("sha256", "").lower():
                raise PaperEvidenceError(f"{label}: SHA-256 mismatch for {evidence['path']}")
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise PaperEvidenceError(f"{label}: evidence must be JSON") from exc
            for assertion_index, assertion in enumerate(evidence.get("assertions", [])):
                _assert_evidence(document, assertion, f"{label}.assertions[{assertion_index}]")
    return summary
