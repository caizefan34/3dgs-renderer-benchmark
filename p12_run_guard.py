"""Fail-closed P12 preflight. Call guard_run before renderer import or training."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


SCENE_ORDER = tuple("bicycle bonsai counter flowers garden kitchen room stump treehill train truck drjohnson playroom".split())
SCENES = frozenset(SCENE_ORDER)
GMAIN_COMMIT = "512d366b67073d77ca099ede742683c165dfc23b"
GMAIN_SO = "8ca8ec1e0c8618f058638b93010061a5bafb7f6c92005b54d07dc6c22658c811"
C0_SO = "9baf8655f859f3e0f456e99a2d8e5b3ddfe414fe17a1a3e9d8072e0afed95b91"
FUSED_COMMIT = "a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38"
FUSED_SO = "dd6a4c799b98ff9fa82b4203acce3b36b1529c97155c0ef4cd50fd0f583d8ebb"
ARMS = ("GMAIN_FUSED", "C0_FUSED", "A0_FUSED", "A1_FUSED")
SWITCHES = ("HIGS_DISABLE_F9", "HIGS_BWD_SCALAR_ADJOINT", "HIGS_BWD_H8_MR", "HIGS_BWD_ABSGRAD", "HIGS_PX_RUNTIME")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _actual_hash(path, errors, label):
    try:
        return sha256_file(path)
    except (OSError, TypeError) as exc:
        errors.append(f"{label}: cannot hash {path!r}: {exc}")
        return None


def validate(config, environ=None):
    """Config is a trainer-supplied runtime attestation; binaries and git are measured here."""
    environ = os.environ if environ is None else environ
    errors = []
    observed = {}

    def require(label, actual, expected):
        observed[label] = actual
        if actual != expected:
            errors.append(f"{label}: expected {expected!r}, got {actual!r}")

    arm = config.get("arm")
    observed["expected renderer arm"] = arm
    if arm not in ARMS:
        errors.append(f"renderer arm: expected one of {ARMS!r}, got {arm!r}")
    require("renderer arm", config.get("runtime_arm"), arm)
    require("scene", config.get("scene"), config.get("expected_scene"))
    if config.get("scene") not in SCENES:
        errors.append(f"scene: unknown P12 scene {config.get('scene')!r}")
    require("padding", config.get("padding"), "same")
    require("eps2d", config.get("eps2d"), 0.3)
    require("seed", config.get("seed"), 42)
    require("loss weights", config.get("loss_weights"), {"l1": 0.8, "ssim": 0.2})
    require("absgrad", config.get("absgrad"), True)
    require("absgrad threshold", config.get("absgrad_threshold"), 0.0008)
    require("rasterization mode", config.get("rasterization_mode"), "classic" if arm == "GMAIN_FUSED" else "higs_dynamic")
    if arm != "GMAIN_FUSED":
        require("backward mode", config.get("backward_mode"), "higs_native")
    require("iterations", config.get("iterations"), 30000)

    repo = config.get("repo_dir")
    expected_repo_sha = config.get("expected_repo_sha")
    observed["expected git SHA"] = expected_repo_sha
    if not expected_repo_sha:
        errors.append("expected git SHA: missing expected_repo_sha")
    try:
        repo_sha = git_value(repo, "rev-parse", "HEAD") if repo else None
    except (OSError, subprocess.CalledProcessError) as exc:
        repo_sha = None
        errors.append(f"git SHA: cannot inspect repo {repo!r}: {exc}")
    require("git SHA", repo_sha, expected_repo_sha)

    renderer_repo = config.get("renderer_repo")
    try:
        renderer_commit = git_value(renderer_repo, "rev-parse", "HEAD") if renderer_repo else None
        renderer_dirty = bool(git_value(renderer_repo, "status", "--porcelain", "--untracked-files=all")) if renderer_repo else None
    except (OSError, subprocess.CalledProcessError) as exc:
        renderer_commit, renderer_dirty = None, None
        errors.append(f"renderer git: cannot inspect {renderer_repo!r}: {exc}")
    expected_renderer_commit = GMAIN_COMMIT if arm == "GMAIN_FUSED" else config.get("expected_renderer_commit")
    if not expected_renderer_commit:
        errors.append("frozen renderer commit: missing expected_renderer_commit")
    require("renderer commit", renderer_commit, expected_renderer_commit)
    require("renderer source dirty", renderer_dirty, False)

    expected_so = GMAIN_SO if arm == "GMAIN_FUSED" else C0_SO
    require("renderer .so SHA256", _actual_hash(config.get("renderer_so"), errors, "renderer .so"), expected_so)
    require("fused-SSIM .so SHA256", _actual_hash(config.get("fused_ssim_so"), errors, "fused-SSIM .so"), FUSED_SO)
    fused_repo = config.get("fused_ssim_repo")
    if fused_repo:
        try:
            fused_commit = git_value(fused_repo, "rev-parse", "HEAD")
            fused_tree = git_value(fused_repo, "rev-parse", "HEAD^{tree}")
            fused_dirty = bool(git_value(fused_repo, "status", "--porcelain", "--untracked-files=all"))
        except (OSError, subprocess.CalledProcessError) as exc:
            fused_commit, fused_tree, fused_dirty = None, None, None
            errors.append(f"fused-SSIM source: cannot inspect {fused_repo!r}: {exc}")
        require("fused-SSIM source commit", fused_commit, FUSED_COMMIT)
        observed["fused-SSIM source tree hash"] = fused_tree
        require("fused-SSIM source dirty", fused_dirty, False)
    else:
        observed["fused-SSIM source commit"] = "unavailable locally"

    # A frozen source snapshot hash is mandatory for the C0 lineage, since a
    # binary hash alone cannot establish that the current source is unchanged.
    if arm != "GMAIN_FUSED":
        expected_source = config.get("expected_renderer_source_sha256")
        source_manifest = config.get("renderer_source_manifest")
        if not expected_source or not source_manifest:
            errors.append("frozen C0 source identity: expected_renderer_source_sha256 and renderer_source_manifest required")
        else:
            require("renderer source manifest SHA256", _actual_hash(source_manifest, errors, "renderer source manifest"), expected_source)

    expected_switches = {
        "GMAIN_FUSED": {"HIGS_DISABLE_F9": None, "HIGS_BWD_SCALAR_ADJOINT": None, "HIGS_BWD_H8_MR": None, "HIGS_BWD_ABSGRAD": None, "HIGS_PX_RUNTIME": None},
        "C0_FUSED": {"HIGS_DISABLE_F9": None, "HIGS_BWD_SCALAR_ADJOINT": "scalar_adjoint", "HIGS_BWD_H8_MR": "1", "HIGS_BWD_ABSGRAD": "1", "HIGS_PX_RUNTIME": "2"},
        "A0_FUSED": {"HIGS_DISABLE_F9": "1", "HIGS_BWD_SCALAR_ADJOINT": None, "HIGS_BWD_H8_MR": None, "HIGS_BWD_ABSGRAD": "1", "HIGS_PX_RUNTIME": "2"},
        "A1_FUSED": {"HIGS_DISABLE_F9": None, "HIGS_BWD_SCALAR_ADJOINT": None, "HIGS_BWD_H8_MR": None, "HIGS_BWD_ABSGRAD": "1", "HIGS_PX_RUNTIME": "2"},
    }.get(arm, {})
    for name, expected in expected_switches.items():
        require(f"switch {name}", environ.get(name), expected)
        require(f"attested switch {name}", config.get("switches", {}).get(name), expected)
    require("C0 custom backward", config.get("c0_custom_backward"), False if arm == "GMAIN_FUSED" else config.get("c0_custom_backward"))
    return observed, errors


def guard_run(config, output_dir, environ=None):
    observed, errors = validate(config, environ)
    for key, value in observed.items():
        print(f"{key}: {value}")
    if errors:
        failure = {"status": "FAIL", "violations": errors, "observed": observed}
        path = Path(output_dir) / "RUN_GUARD_FAIL.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        for error in errors:
            print(f"INVARIANT VIOLATION: {error}", file=sys.stderr)
        print(f"wrote {path}", file=sys.stderr)
        return False
    print("P12 run guard: PASS")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON attestation captured by trainer at process start")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"config: {exc}", file=sys.stderr)
        return 2
    return 0 if guard_run(config, args.output_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
