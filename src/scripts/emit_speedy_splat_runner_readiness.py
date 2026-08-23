#!/usr/bin/env python3
"""Emit the Speedy-Splat runner smoke evidence for the frozen protocol.

The runner writes a full ``paper-run-metadata.json`` for every job (smoke and
paper alike). Smoke runs are paper-ineligible, but before ``speedy_splat`` may
flip to ``runner_status: ready`` the protocol requires a curated smoke evidence
file at ``paper/higs/speedy-splat-runner-readiness.json``. This helper derives
that file from a smoke run and cross-checks every field the protocol validator
relies on.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from higs_training_commands import (  # noqa: E402
    HigsTrainingCommandError,
    audit_speedy_splat_source,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper" / "higs" / "speedy-splat-runner-readiness.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "benchmark" / "higs-paper-protocol.json",
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    meta = _load(args.metadata)
    protocol = _load(args.protocol)
    spec = protocol["methods"]["speedy_splat"]

    errors = []
    if meta.get("method") != "speedy_splat":
        errors.append("metadata method is not speedy_splat")
    if meta.get("run_kind") != "smoke":
        errors.append("metadata run_kind is not smoke")
    if meta.get("paper_eligible") is not False:
        errors.append("metadata paper_eligible is not false")
    if meta.get("initialization") != "from_scratch_sfm":
        errors.append("metadata initialization is not from_scratch_sfm")
    if meta.get("clean_process") is not True:
        errors.append("metadata clean_process is not true")
    if len(meta.get("dataset", {}).get("inventory_sha256", "")) != 64:
        errors.append("dataset inventory hash missing")
    source = meta.get("source", {})
    if source.get("commit") != spec.get("commit"):
        errors.append("source commit does not match the frozen protocol")
    for key in ("trainer_sha256", "source_diff_sha256", "source_state_sha256"):
        if source.get(key) != spec.get(key):
            errors.append(f"source {key} does not match the frozen protocol")

    audit = audit_speedy_splat_source(args.source_dir)
    if audit["source_diff_sha256"] != spec.get("source_diff_sha256"):
        errors.append("source audit diff hash does not match the frozen protocol")
    if audit["source_state_sha256"] != spec.get("source_state_sha256"):
        errors.append("source audit state hash does not match the frozen protocol")
    if not (
        audit["has_seed_argument"]
        and audit["uses_seeded_safe_state"]
        and audit["safe_state_seeded"]
        and audit["has_network_gui_init"]
        and audit["has_speedy_rasterizer_submodule"]
    ):
        errors.append("source audit flags are not all satisfied")

    artifacts = meta.get("artifacts", [])
    if not any(
        item.get("path", "").startswith("point_cloud/")
        and len(item.get("sha256", "")) == 64
        for item in artifacts
    ):
        errors.append("no hashed point_cloud checkpoint in artifacts")
    for prefix in ("stats/train_", "stats/val_"):
        if not any(
            item.get("path", "").startswith(prefix)
            and len(item.get("sha256", "")) == 64
            for item in artifacts
        ):
            errors.append(f"no hashed {prefix} output in artifacts")

    if errors:
        raise HigsTrainingCommandError(
            "smoke evidence is not ready: " + "; ".join(errors)
        )

    readiness = {
        "schema_version": "1.0",
        "method": meta["method"],
        "scene": meta["scene"],
        "run_kind": "smoke",
        "paper_eligible": False,
        "initialization": meta["initialization"],
        "iterations": meta["iterations"],
        "seed": meta["seed"],
        "timing_boundary": meta.get("timing_boundary"),
        "started_at_utc": meta.get("started_at_utc"),
        "wall_time_seconds": meta.get("wall_time_seconds"),
        "source_dir": meta.get("source_dir"),
        "source": source,
        "source_audit": audit,
        "software": meta.get("software"),
        "hardware": meta.get("hardware"),
        "dataset": meta.get("dataset"),
        "artifacts": artifacts,
        "clean_process": True,
    }
    _write(args.output, readiness)
    print(f"wrote {args.output} with {len(artifacts)} hashed artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
