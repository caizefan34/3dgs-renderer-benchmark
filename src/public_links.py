"""Validate local links exposed by public repository entry points."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote


_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "data:", "#")


class PublicLinkError(ValueError):
    """Raised when a public local link cannot work from GitHub."""


def _target_text(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1:target.index(">")]
    elif " " in target:
        target = target.split(" ", 1)[0]
    return unquote(target)


def _is_tracked(path: Path, repository_root: Path) -> bool:
    relative = path.relative_to(repository_root).as_posix()
    result = subprocess.run(
        ["git", "ls-files", "--", relative],
        cwd=repository_root,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def validate_public_links(
    repository_root: str | Path,
    entry_points: list[str],
    require_tracked: bool = False,
) -> dict[str, int]:
    root = Path(repository_root).resolve()
    relative_links = 0
    checked_targets = set()
    for entry_point in entry_points:
        source = (root / entry_point).resolve()
        if not source.is_file():
            raise PublicLinkError(f"missing entry point: {entry_point}")
        if require_tracked and not _is_tracked(source, root):
            raise PublicLinkError(f"entry point is not Git-tracked: {entry_point}")
        content = source.read_text(encoding="utf-8")
        for match in _MARKDOWN_LINK.finditer(content):
            target = _target_text(match.group(1))
            if not target or target.startswith(_EXTERNAL_PREFIXES):
                continue
            path_text = target.split("#", 1)[0].split("?", 1)[0]
            if not path_text:
                continue
            resolved = (source.parent / path_text).resolve()
            if resolved != root and root not in resolved.parents:
                raise PublicLinkError(f"link escapes repository: {entry_point} -> {target}")
            relative_links += 1
            identity = resolved.as_posix()
            if identity in checked_targets:
                continue
            checked_targets.add(identity)
            if not resolved.exists():
                raise PublicLinkError(f"missing target: {entry_point} -> {target}")
            if require_tracked and not _is_tracked(resolved, root):
                relative = resolved.relative_to(root).as_posix()
                raise PublicLinkError(f"target is not Git-tracked: {relative}")
    return {"entry_points": len(entry_points), "relative_links": relative_links}
