"""Official-dataset training policy validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def load_official_dataset_manifest(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def validate_training_manifest(manifest: Mapping) -> dict:
    if manifest.get("schema_version") != 1:
        raise ValueError("official training dataset manifest schema_version must be 1")
    policy = manifest.get("policy", {})
    if "Official datasets only" not in policy.get("training_data_rule", ""):
        raise ValueError("training_data_rule must require official datasets")
    official_families = {
        source["dataset_family"]
        for source in manifest.get("official_sources", [])
        if source.get("used_by_official_3dgs")
    }
    if not official_families:
        raise ValueError("at least one official 3DGS dataset family is required")

    jobs = []
    for job in manifest.get("training_jobs", []):
        if job.get("dataset_family") not in official_families:
            raise ValueError(f"training job {job.get('job_id')} uses a non-official dataset family")
        command = job.get("train_command", "")
        if "--eval" not in command.split():
            raise ValueError(f"training job {job.get('job_id')} must use --eval")
        jobs.append(job["job_id"])
    return {
        "schema_version": 1,
        "status": "ok",
        "official_dataset_families": sorted(official_families),
        "validated_training_jobs": jobs,
    }

