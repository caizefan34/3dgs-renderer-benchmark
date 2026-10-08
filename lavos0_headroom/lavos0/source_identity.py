"""Source-identity freeze + verification (PREREG Sec.0 / Phase A lock).

LAVOS0 must build on the frozen C0 trainer, the VOS0 settlement primitive, and
the ReproTopo0 deterministic topology state machine WITHOUT modifying them. This
module records the SHA256 of the frozen source files it depends on and verifies
them at run time, so a silent source change fails closed with
LAVOS0_SOURCE_IDENTITY_FAIL before any science runs.

The frozen-source roots are resolved relative to the repo root (two levels up
from this file: lavos0_headroom/lavos0/ -> repo). Override with the
LAVOS0_FROZEN_ROOT env var if the sources live elsewhere on the A100.
"""

import hashlib
import os
from typing import Dict, List, Optional

# Frozen source files LAVOS0 depends on (relative to the frozen root). These are
# the OBSERVED-semantics sources; the harness never writes to them.
FROZEN_FILES = {
    "c0_gaussian_model":
        "vos/sources/c0_gaussian_inference.py",
    "c0_gaussian_model_params":
        ".c0-official-reference/scene/gaussian_model.py",
    "c0_publication_trainer":
        "vos/sources/publication_trainer.py",
    "vos_train_driver":
        "vos/scripts/vos_train_driver.py",
    "reprotopo_statemachine":
        "reprotopo0_src/rt0_statemachine.py",
    "reprotopo_event":
        "reprotopo0_src/rt0_event.py",
    "reprotopo_common":
        "reprotopo0_src/rt0_common.py",
}


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def frozen_root():
    """Resolve the frozen-source root (env override or repo root)."""
    return os.environ.get("LAVOS0_FROZEN_ROOT", _repo_root())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_source_identity(root: Optional[str] = None) -> Dict:
    """Snapshot the SHA256 of every frozen source file.

    Returns a dict suitable for LAVOS0_SOURCE_IDENTITY.json with
    source_identity_pass=True iff all files are present.
    """
    root = root or frozen_root()
    files = {}
    missing = []
    for name, rel in FROZEN_FILES.items():
        path = os.path.join(root, rel.replace("/", os.sep))
        if os.path.isfile(path):
            files[name] = {"path": rel, "sha256": sha256_file(path)}
        else:
            missing.append(rel)
    return {
        "source_identity_pass": len(missing) == 0,
        "frozen_root": root,
        "files": files,
        "missing": missing,
        "n_files": len(files),
        "n_expected": len(FROZEN_FILES),
    }


def verify_source_identity(expected: Dict, root: Optional[str] = None) -> Dict:
    """Verify current sources match a previously captured identity snapshot.

    Returns a dict with source_identity_pass=True only if every file's current
    SHA256 matches the snapshot exactly (no drift, no missing files).
    """
    root = root or frozen_root()
    current = freeze_source_identity(root)
    drift = []
    for name, meta in expected.get("files", {}).items():
        cur = current["files"].get(name)
        if cur is None:
            drift.append({"name": name, "reason": "missing"})
        elif cur["sha256"] != meta.get("sha256"):
            drift.append({"name": name, "reason": "sha256_mismatch",
                          "expected": meta.get("sha256"),
                          "actual": cur["sha256"]})
    return {
        "source_identity_pass": len(drift) == 0 and current["source_identity_pass"],
        "n_drift": len(drift),
        "drift": drift,
        "current": current,
    }


def phase_a_source_gate(root: Optional[str] = None):
    """Phase A entry gate: freeze + verify the frozen sources are present.

    On the A100 this is called before any run; locally it proves the sources
    exist and records their hashes (source_identity_pass reflects presence, not
    a prior snapshot, on the first freeze).
    """
    snap = freeze_source_identity(root)
    return snap
