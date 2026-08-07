"""Fail-closed command construction for the HiGS full-training paper track."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


class HigsTrainingCommandError(ValueError):
    """Raised when a command would violate the frozen paper protocol."""


_CHECKPOINT_SUFFIXES = {".ckpt", ".ply", ".pt", ".pth"}
def _higs_method_ids(protocol: dict) -> set[str]:
    """Protocol-driven HiGS method identification via the renderer contract."""
    return {
        method_id
        for method_id, spec in protocol.get("methods", {}).items()
        if (spec.get("algorithm") or {}).get("renderer")
        == "higs_dynamic_native_backward"
    }


_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
def trainer_cfg_kwargs(method_spec: dict) -> dict:
    """Protocol-driven trainer kwargs; HiGS methods force packed=False, sparse_grad=False."""
    algorithm = method_spec.get("algorithm") or {}
    trainer_cfg = algorithm.get("trainer_cfg", {})
    cfg_kwargs = {}
    if algorithm.get("renderer") == "higs_dynamic_native_backward":
        cfg_kwargs.update(packed=False, sparse_grad=False)
        cfg_kwargs.update(trainer_cfg)
    return cfg_kwargs




def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_gsplat_source(source_dir: Path) -> dict:
    """Inspect the pinned source tree without importing CUDA modules."""
    source_dir = Path(source_dir).resolve()
    trainer = source_dir / "examples" / "simple_trainer.py"
    higs_api = (
        source_dir
        / "gsplat"
        / "experimental"
        / "render"
        / "functional"
        / "gaussian_inference.py"
    )
    trainer_text = trainer.read_text(encoding="utf-8") if trainer.is_file() else ""
    higs_text = higs_api.read_text(encoding="utf-8") if higs_api.is_file() else ""
    git_root = None
    head_commit = None
    diff_sha256 = None
    state_sha256 = None
    untracked_files = []
    try:
        git_root_text = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        candidate_root = Path(git_root_text).resolve()
        if candidate_root == source_dir:
            git_root = str(candidate_root)
            head_commit = subprocess.run(
                ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            diff = subprocess.run(
                ["git", "-C", str(source_dir), "diff", "--binary", "HEAD"],
                check=True,
                capture_output=True,
            ).stdout
            diff_sha256 = hashlib.sha256(diff).hexdigest()
            untracked_files = subprocess.run(
                [
                    "git",
                    "-C",
                    str(source_dir),
                    "ls-files",
                    "--others",
                    "--exclude-standard",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            state_digest = hashlib.sha256()
            state_digest.update(b"tracked-diff\0")
            state_digest.update(diff)
            for relative in sorted(untracked_files):
                state_digest.update(b"untracked\0")
                state_digest.update(relative.encode("utf-8"))
                state_digest.update(b"\0")
                state_digest.update((source_dir / relative).read_bytes())
            state_sha256 = state_digest.hexdigest()
    except (OSError, subprocess.CalledProcessError):
        pass
    required_density_tokens = (
        '"densification_info"',
        '"means2d"',
        '"radii"',
        '"gaussian_ids"',
    )
    return {
        "source_dir": str(source_dir),
        "git_root_verified": git_root is not None,
        "head_commit": head_commit,
        "source_diff_sha256": diff_sha256,
        "source_state_sha256": state_sha256,
        "untracked_files": untracked_files,
        "has_simple_trainer": trainer.is_file(),
        "trainer_sha256": _sha256(trainer) if trainer.is_file() else None,
        "has_sfm_initialization": 'init_type: str = "sfm"' in trainer_text,
        "loads_pretrained_ply": "point_cloud.ply" in trainer_text,
        "has_higs_dynamic_api": "def rasterize_gaussian_higs_dynamic(" in higs_text,
        "has_higs_trainer_adapter": all(
            token in trainer_text
            for token in (
                "rasterize_gaussian_higs_dynamic(",
                'higs_result["densification_info"]',
                "_HIGS_DYNAMIC_SCENE.mark_dirty()",
            )
        ),
        "has_higs_densification_info": all(
            token in higs_text for token in required_density_tokens
        ),
    }


def _validate_selection(protocol: dict, method: str, scene: str, seed: int) -> None:
    if method not in protocol.get("methods", {}):
        raise HigsTrainingCommandError(f"method is outside the frozen protocol: {method}")
    scene_ids = {item["id"] for item in protocol.get("scenes", [])}
    if scene not in scene_ids:
        raise HigsTrainingCommandError(f"scene is outside the frozen protocol: {scene}")
    if seed not in protocol.get("training", {}).get("seeds", []):
        raise HigsTrainingCommandError(f"seed is outside the frozen protocol: {seed}")


def build_training_invocation(
    *,
    protocol: dict,
    method: str,
    scene: str,
    seed: int,
    data_dir: Path,
    result_dir: Path,
    source_dir: Path,
    python_executable: str | None = None,
    repository_root: Path | None = None,
    protocol_path: Path | None = None,
) -> dict:
    """Build an auditable single-GPU invocation without executing training."""
    _validate_selection(protocol, method, scene, seed)
    higs_methods = _higs_method_ids(protocol)
    training = protocol.get("training", {})
    if training.get("initialization") != "from_scratch_sfm":
        raise HigsTrainingCommandError("paper training must initialize from SfM")
    if training.get("iterations") != 30_000:
        raise HigsTrainingCommandError("paper training budget must be exactly 30000")

    data_dir = Path(data_dir)
    if data_dir.suffix.lower() in _CHECKPOINT_SUFFIXES:
        raise HigsTrainingCommandError(
            "data_dir must be a dataset directory, not a PLY or checkpoint"
        )
    source_dir = Path(source_dir).resolve()
    audit = audit_gsplat_source(source_dir)
    if not audit["has_simple_trainer"] or not audit["has_sfm_initialization"]:
        raise HigsTrainingCommandError("source tree lacks the audited SfM trainer")
    if audit["loads_pretrained_ply"]:
        raise HigsTrainingCommandError("source trainer loads point_cloud.ply")
    if method in higs_methods:
        if not audit["has_higs_dynamic_api"]:
            raise HigsTrainingCommandError("source tree lacks the dynamic HiGS API")
        if not audit["has_higs_densification_info"]:
            raise HigsTrainingCommandError(
                "HiGS densification contract is incomplete: native backward must "
                "expose screen-space gradients, radii, and Gaussian ids"
            )
        if not audit["has_higs_trainer_adapter"]:
            raise HigsTrainingCommandError(
                "HiGS source trainer does not use the dynamic renderer and "
                "topology-dirty callback"
            )

    root = Path(repository_root or Path(__file__).resolve().parents[1]).resolve()
    launcher = root / "benchmark" / "run_higs_full_training.py"
    protocol_path = Path(
        protocol_path or root / "benchmark" / "higs-paper-protocol.json"
    ).resolve()
    executable = python_executable or sys.executable
    iterations = training["iterations"]
    command = [
        executable,
        str(launcher),
        "--protocol",
        str(protocol_path),
        "--method",
        method,
        "--scene",
        scene,
        "--source-dir",
        str(source_dir),
        "--data-dir",
        str(data_dir.resolve()),
        "--result-dir",
        str(Path(result_dir).resolve()),
        "--seed",
        str(seed),
        "--init-type",
        "sfm",
        "--max-steps",
        str(iterations),
    ]
    method_spec = protocol["methods"][method]
    expected_commit = method_spec.get("commit")
    if not audit["git_root_verified"] or audit["head_commit"] != expected_commit:
        raise HigsTrainingCommandError(
            "source identity is not verified: source_dir must be its own Git "
            f"checkout at {expected_commit}"
        )
    if method == "gsplat" and audit["untracked_files"]:
        raise HigsTrainingCommandError(
            "source checkout contains untracked files and is not auditable"
        )
    if method == "gsplat" and audit["source_diff_sha256"] != _EMPTY_SHA256:
        raise HigsTrainingCommandError(
            "official gsplat requires a clean checkout; use a separate patched "
            "source tree for HiGS"
        )
    if method in higs_methods and audit["source_state_sha256"] != method_spec.get(
        "source_state_sha256"
    ):
        raise HigsTrainingCommandError(
            "patched HiGS source state does not match the frozen protocol"
        )
    if method in higs_methods and audit["trainer_sha256"] != method_spec.get(
        "trainer_sha256"
    ):
        raise HigsTrainingCommandError(
            "patched HiGS trainer does not match the frozen protocol"
        )
    patch_path = root / "patches" / "higs-differentiable.patch"
    extension_cache = (
        root
        / "artifacts"
        / "cuda-build"
        / f"{source_dir.name}-py{sys.version_info.major}{sys.version_info.minor}"
    )
    source = {
        "repository": method_spec.get("repository"),
        "commit": method_spec.get("commit"),
        "trainer_sha256": audit["trainer_sha256"],
        "source_diff_sha256": audit["source_diff_sha256"],
        "source_state_sha256": audit["source_state_sha256"],
    }
    if method in higs_methods and patch_path.is_file():
        patch_sha256 = _sha256(patch_path)
        if patch_sha256 != method_spec.get("patch_sha256"):
            raise HigsTrainingCommandError(
                "HiGS patch SHA-256 does not match the frozen protocol"
            )
        source["patch_sha256"] = patch_sha256
    return {
        "schema_version": "1.0",
        "method": method,
        "scene": scene,
        "seed": seed,
        "initialization": "from_scratch_sfm",
        "iterations": iterations,
        "command": command,
        "environment": {
            "PYTHONPATH": str(source_dir),
            "TORCH_EXTENSIONS_DIR": str(extension_cache),
        },
        "source": source,
        "source_audit": audit,
    }


def audit_original_3dgs_source(source_dir: Path) -> dict:
    """Inspect the pinned official 3DGS tree for the audited seed patch."""
    source_dir = Path(source_dir).resolve()
    train_py = source_dir / "train.py"
    general_utils = source_dir / "utils" / "general_utils.py"
    train_text = train_py.read_text(encoding="utf-8") if train_py.is_file() else ""
    gu_text = (
        general_utils.read_text(encoding="utf-8")
        if general_utils.is_file()
        else ""
    )
    git_root = None
    head_commit = None
    diff_sha256 = None
    state_sha256 = None
    untracked_files = []
    try:
        git_root_text = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        candidate_root = Path(git_root_text).resolve()
        if candidate_root == source_dir:
            git_root = str(candidate_root)
            head_commit = subprocess.run(
                ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            diff = subprocess.run(
                ["git", "-C", str(source_dir), "diff", "--binary", "HEAD"],
                check=True,
                capture_output=True,
            ).stdout
            diff_sha256 = hashlib.sha256(diff).hexdigest()
            untracked_files = subprocess.run(
                [
                    "git",
                    "-C",
                    str(source_dir),
                    "ls-files",
                    "--others",
                    "--exclude-standard",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            state_digest = hashlib.sha256()
            state_digest.update(b"tracked-diff\0")
            state_digest.update(diff)
            for relative in sorted(untracked_files):
                state_digest.update(b"untracked\0")
                state_digest.update(relative.encode("utf-8"))
                state_digest.update(b"\0")
                state_digest.update((source_dir / relative).read_bytes())
            state_sha256 = state_digest.hexdigest()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "source_dir": str(source_dir),
        "git_root_verified": git_root is not None,
        "head_commit": head_commit,
        "source_diff_sha256": diff_sha256,
        "source_state_sha256": state_sha256,
        "untracked_files": untracked_files,
        "has_train_py": train_py.is_file(),
        "trainer_sha256": _sha256(train_py) if train_py.is_file() else None,
        "has_seed_argument": (
            "parser.add_argument('--seed', type=int, default=0)" in train_text
        ),
        "uses_seeded_safe_state": "safe_state(args.quiet, args.seed)" in train_text,
        "safe_state_seeded": "def safe_state(silent, seed=0):" in gu_text,
    }


def build_original_3dgs_invocation(
    *,
    protocol: dict,
    scene: str,
    seed: int,
    data_dir: Path,
    result_dir: Path,
    source_dir: Path,
    python_executable: str | None = None,
    repository_root: Path | None = None,
    smoke_steps: int | None = None,
    images_dir: str = "images_4",
) -> dict:
    """Build an auditable single-GPU official 3DGS invocation."""
    _validate_selection(protocol, "original_3dgs", scene, seed)
    training = protocol.get("training", {})
    if training.get("initialization") != "from_scratch_sfm":
        raise HigsTrainingCommandError("paper training must initialize from SfM")
    if training.get("iterations") != 30_000:
        raise HigsTrainingCommandError("paper training budget must be exactly 30000")
    if smoke_steps is not None and smoke_steps < 1:
        raise HigsTrainingCommandError("smoke_steps must be positive")
    iterations = training["iterations"]
    effective_steps = smoke_steps or iterations

    data_dir = Path(data_dir)
    if data_dir.suffix.lower() in _CHECKPOINT_SUFFIXES:
        raise HigsTrainingCommandError(
            "data_dir must be a dataset directory, not a PLY or checkpoint"
        )
    result_dir = Path(result_dir)
    source_dir = Path(source_dir).resolve()
    audit = audit_original_3dgs_source(source_dir)
    method_spec = protocol["methods"]["original_3dgs"]
    if not audit["git_root_verified"] or audit["head_commit"] != method_spec.get(
        "commit"
    ):
        raise HigsTrainingCommandError(
            "official source identity is not verified: source_dir must be its own "
            f"Git checkout at {method_spec.get('commit')}"
        )
    if not (
        audit["has_train_py"]
        and audit["has_seed_argument"]
        and audit["uses_seeded_safe_state"]
        and audit["safe_state_seeded"]
    ):
        raise HigsTrainingCommandError(
            "official trainer is missing the audited seed patch"
        )
    for key in ("trainer_sha256", "source_diff_sha256", "source_state_sha256"):
        expected = method_spec.get(key)
        if expected and audit[key] != expected:
            raise HigsTrainingCommandError(
                f"official source {key} does not match the frozen protocol"
            )

    root = Path(repository_root or Path(__file__).resolve().parents[1]).resolve()
    patch_path = root / "patches" / "original-3dgs-seed.patch"
    if patch_path.is_file():
        patch_sha256 = _sha256(patch_path)
        expected_patch = method_spec.get("patch_sha256")
        if expected_patch and patch_sha256 != expected_patch:
            raise HigsTrainingCommandError(
                "original_3dgs seed patch SHA-256 does not match the frozen protocol"
            )

    eval_steps = sorted({min(s, effective_steps) for s in (7_000, 15_000, 30_000)})
    executable = python_executable or sys.executable
    command = [
        executable,
        # -u: stdout must flush as soon as each "[ITER N] Saving Gaussians"
        # marker is printed; a buffered non-tty stdout only flushes at exit,
        # so the runner could not timestamp the three checkpoints in time.
        "-u",
        str(source_dir / "train.py"),
        "-s",
        str(data_dir.resolve()),
        "-m",
        str(result_dir.resolve()),
        "-i",
        images_dir,
        "--resolution=-1",
        "--eval",
        "--iterations",
        str(effective_steps),
        "--test_iterations",
        *[str(step) for step in eval_steps],
        "--save_iterations",
        *[str(step) for step in eval_steps],
        "--seed",
        str(seed),
        "--disable_viewer",
        # NOTE: no --quiet: official safe_state(silent=True) swallows stdout,
        # which would hide the "[ITER N] Saving Gaussians" markers the runner
        # needs to timestamp eval+save checkpoints.
    ]
    source = {
        "repository": method_spec.get("repository"),
        "commit": method_spec.get("commit"),
        "trainer_sha256": audit["trainer_sha256"],
        "source_diff_sha256": audit["source_diff_sha256"],
        "source_state_sha256": audit["source_state_sha256"],
    }
    if patch_path.is_file():
        source["patch_sha256"] = _sha256(patch_path)
    return {
        "schema_version": "1.0",
        "method": "original_3dgs",
        "scene": scene,
        "seed": seed,
        "initialization": "from_scratch_sfm",
        "iterations": effective_steps,
        "eval_steps": eval_steps,
        "images_dir": images_dir,
        "command": command,
        "environment": {"PYTHONPATH": str(source_dir)},
        "source": source,
        "source_audit": audit,
    }


def audit_speedy_splat_source(source_dir: Path) -> dict:
    """Inspect the pinned official Speedy-Splat tree for the audited seed patch."""
    source_dir = Path(source_dir).resolve()
    train_py = source_dir / "train.py"
    general_utils = source_dir / "utils" / "general_utils.py"
    renderer = source_dir / "gaussian_renderer" / "__init__.py"
    rasterizer_setup = (
        source_dir / "submodules" / "diff-gaussian-rasterization" / "setup.py"
    )
    train_text = train_py.read_text(encoding="utf-8") if train_py.is_file() else ""
    gu_text = (
        general_utils.read_text(encoding="utf-8")
        if general_utils.is_file()
        else ""
    )
    render_text = renderer.read_text(encoding="utf-8") if renderer.is_file() else ""
    git_root = None
    head_commit = None
    diff_sha256 = None
    state_sha256 = None
    untracked_files = []
    try:
        git_root_text = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        candidate_root = Path(git_root_text).resolve()
        if candidate_root == source_dir:
            git_root = str(candidate_root)
            head_commit = subprocess.run(
                ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            diff = subprocess.run(
                ["git", "-C", str(source_dir), "diff", "--binary", "HEAD"],
                check=True,
                capture_output=True,
            ).stdout
            diff_sha256 = hashlib.sha256(diff).hexdigest()
            untracked_files = subprocess.run(
                [
                    "git",
                    "-C",
                    str(source_dir),
                    "ls-files",
                    "--others",
                    "--exclude-standard",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            state_digest = hashlib.sha256()
            state_digest.update(b"tracked-diff\0")
            state_digest.update(diff)
            for relative in sorted(untracked_files):
                state_digest.update(b"untracked\0")
                state_digest.update(relative.encode("utf-8"))
                state_digest.update(b"\0")
                state_digest.update((source_dir / relative).read_bytes())
            state_sha256 = state_digest.hexdigest()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "source_dir": str(source_dir),
        "git_root_verified": git_root is not None,
        "head_commit": head_commit,
        "source_diff_sha256": diff_sha256,
        "source_state_sha256": state_sha256,
        "untracked_files": untracked_files,
        "has_train_py": train_py.is_file(),
        "trainer_sha256": _sha256(train_py) if train_py.is_file() else None,
        "has_seed_argument": (
            "parser.add_argument('--seed', type=int, default=0)" in train_text
            or 'parser.add_argument("--seed", type=int, default=0)' in train_text
        ),
        "uses_seeded_safe_state": "safe_state(args.quiet, args.seed)" in train_text,
        "safe_state_seeded": "def safe_state(silent, seed=0):" in gu_text,
        "has_network_gui_init": "network_gui.init(args.ip, args.port)" in train_text,
        "has_speedy_render_scores": "scores = None" in render_text,
        "has_speedy_rasterizer_submodule": rasterizer_setup.is_file(),
    }


def build_speedy_splat_invocation(
    *,
    protocol: dict,
    scene: str,
    seed: int,
    data_dir: Path,
    result_dir: Path,
    source_dir: Path,
    python_executable: str | None = None,
    repository_root: Path | None = None,
    smoke_steps: int | None = None,
    images_dir: str = "images_4",
    gui_port: int | None = None,
) -> dict:
    """Build an auditable single-GPU official Speedy-Splat invocation.

    ``gui_port`` is the port for the official trainer's inert localhost
    network_gui server. Concurrent jobs must use distinct ports because the
    official train.py binds it unconditionally (there is no --disable_viewer).
    """
    _validate_selection(protocol, "speedy_splat", scene, seed)
    training = protocol.get("training", {})
    if training.get("initialization") != "from_scratch_sfm":
        raise HigsTrainingCommandError("paper training must initialize from SfM")
    if training.get("iterations") != 30_000:
        raise HigsTrainingCommandError("paper training budget must be exactly 30000")
    if smoke_steps is not None and smoke_steps < 1:
        raise HigsTrainingCommandError("smoke_steps must be positive")
    iterations = training["iterations"]
    effective_steps = smoke_steps or iterations

    data_dir = Path(data_dir)
    if data_dir.suffix.lower() in _CHECKPOINT_SUFFIXES:
        raise HigsTrainingCommandError(
            "data_dir must be a dataset directory, not a PLY or checkpoint"
        )
    result_dir = Path(result_dir)
    source_dir = Path(source_dir).resolve()
    audit = audit_speedy_splat_source(source_dir)
    method_spec = protocol["methods"]["speedy_splat"]
    if not audit["git_root_verified"] or audit["head_commit"] != method_spec.get(
        "commit"
    ):
        raise HigsTrainingCommandError(
            "official source identity is not verified: source_dir must be its own "
            f"Git checkout at {method_spec.get('commit')}"
        )
    if not (
        audit["has_train_py"]
        and audit["has_seed_argument"]
        and audit["uses_seeded_safe_state"]
        and audit["safe_state_seeded"]
        and audit["has_network_gui_init"]
        and audit["has_speedy_rasterizer_submodule"]
    ):
        raise HigsTrainingCommandError(
            "official Speedy-Splat trainer is missing the audited seed patch "
            "or a required fork component"
        )
    for key in ("trainer_sha256", "source_diff_sha256", "source_state_sha256"):
        expected = method_spec.get(key)
        if expected and audit[key] != expected:
            raise HigsTrainingCommandError(
                f"official Speedy-Splat source {key} does not match the frozen protocol"
            )

    root = Path(repository_root or Path(__file__).resolve().parents[1]).resolve()
    patch_path = root / "patches" / "speedy-splat-seed.patch"
    if patch_path.is_file():
        patch_sha256 = _sha256(patch_path)
        expected_patch = method_spec.get("patch_sha256")
        if expected_patch and patch_sha256 != expected_patch:
            raise HigsTrainingCommandError(
                "Speedy-Splat seed patch SHA-256 does not match the frozen protocol"
            )

    cfloat_patch = root / "patches" / "speedy-splat-cfloat.patch"
    if cfloat_patch.is_file():
        cfloat_patch_sha256 = _sha256(cfloat_patch)
        expected_cfloat = method_spec.get("cfloat_patch_sha256")
        if expected_cfloat and cfloat_patch_sha256 != expected_cfloat:
            raise HigsTrainingCommandError(
                "Speedy-Splat cfloat compatibility patch SHA-256 does not match "
                "the frozen protocol"
            )

    eval_steps = sorted({min(s, effective_steps) for s in (7_000, 15_000, 30_000)})
    executable = python_executable or sys.executable
    command = [
        executable,
        # -u: stdout must flush as soon as each "[ITER N] Saving Gaussians"
        # marker is printed; a buffered non-tty stdout only flushes at exit,
        # so the runner could not timestamp the three checkpoints in time.
        "-u",
        str(source_dir / "train.py"),
        "-s",
        str(data_dir.resolve()),
        "-m",
        str(result_dir.resolve()),
        "-i",
        images_dir,
        "--resolution=-1",
        "--eval",
        "--iterations",
        str(effective_steps),
        "--test_iterations",
        *[str(step) for step in eval_steps],
        "--save_iterations",
        *[str(step) for step in eval_steps],
        "--seed",
        str(seed),
        # NOTE: the official Speedy-Splat train.py has no --disable_viewer; its
        # network_gui.init() binds a localhost server unconditionally, so each
        # concurrent job must pass a unique --port (6009 + gpu index).
        # NOTE: no --quiet: official safe_state(silent=True) swallows stdout,
        # which would hide the "[ITER N] Saving Gaussians" markers the runner
        # needs to timestamp eval+save checkpoints.
    ]
    if gui_port is not None:
        command += ["--port", str(gui_port)]
    source = {
        "repository": method_spec.get("repository"),
        "commit": method_spec.get("commit"),
        "trainer_sha256": audit["trainer_sha256"],
        "source_diff_sha256": audit["source_diff_sha256"],
        "source_state_sha256": audit["source_state_sha256"],
    }
    if patch_path.is_file():
        source["patch_sha256"] = _sha256(patch_path)
    return {
        "schema_version": "1.0",
        "method": "speedy_splat",
        "scene": scene,
        "seed": seed,
        "initialization": "from_scratch_sfm",
        "iterations": effective_steps,
        "eval_steps": eval_steps,
        "images_dir": images_dir,
        "command": command,
        "environment": {"PYTHONPATH": str(source_dir)},
        "source": source,
        "source_audit": audit,
    }
