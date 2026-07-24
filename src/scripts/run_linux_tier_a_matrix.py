#!/usr/bin/env python3
"""Run the complete Linux Tier A matrix with isolated renderer environments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_matrix import (  # noqa: E402
    generate_matrix_report,
    load_results,
    validate_result,
    write_report,
)


ROOT = Path(__file__).resolve().parents[2]

CASE_ORDERS = (
    (
        "small-garden-1080p",
        ("original_3dgs", "gsplat", "gsplat_higs", "speedy_splat", "tcgs"),
    ),
    (
        "medium-truck-1080p",
        ("gsplat", "gsplat_higs", "speedy_splat", "tcgs", "original_3dgs"),
    ),
    (
        "medium-train-1080p",
        ("gsplat_higs", "speedy_splat", "tcgs", "original_3dgs", "gsplat"),
    ),
    (
        "large-bicycle-1080p",
        ("speedy_splat", "tcgs", "original_3dgs", "gsplat", "gsplat_higs"),
    ),
    (
        "large-bonsai-1080p",
        ("tcgs", "original_3dgs", "gsplat", "gsplat_higs", "speedy_splat"),
    ),
)
MATRIX_ORDER = tuple(
    (case_id, renderer)
    for case_id, renderers in CASE_ORDERS
    for renderer in renderers
)
ALL_CONFIGS = (
    "original_3dgs",
    "gsplat",
    "gsplat_dense",
    "gsplat_higs",
    "gsplat_higs_tile16",
    "gsplat_higs_sh32",
    "gsplat_higs_sh16",
    "gsplat_higs_tile16_sh32",
    "gsplat_higs_tile16_sh16",
    "gsplat_higs_auto",
    "speedy_splat",
    "tcgs",
)
HIGS_ABLATION_CONFIGS = tuple(
    config for config in ALL_CONFIGS if config.startswith("gsplat_higs")
)
CANDIDATE_CONFIGS = ("flashgs", "local_gs", "gemm_gs")
PROFILE_RENDERERS = {
    "primary": tuple(dict.fromkeys(renderer for _, renderer in MATRIX_ORDER)),
    "all-configs": ALL_CONFIGS,
    "higs-ablation": HIGS_ABLATION_CONFIGS,
    "candidate-renderers": CANDIDATE_CONFIGS,
}
ENV_BY_RENDERER = {
    "original_3dgs": "original3dgs",
    "gsplat": "gsplat",
    "gsplat_higs": "gsplat",
    "gsplat_dense": "gsplat",
    "gsplat_higs_tile16": "gsplat",
    "gsplat_higs_sh32": "gsplat",
    "gsplat_higs_sh16": "gsplat",
    "gsplat_higs_tile16_sh32": "gsplat",
    "gsplat_higs_tile16_sh16": "gsplat",
    "gsplat_higs_auto": "gsplat",
    "speedy_splat": "speedy",
    "tcgs": "tcgs",
    "flashgs": "flashgs",
    "local_gs": "localgs",
    "gemm_gs": "gemmgs",
}
COHORT_FIELDS = ("gpu_uuid", "driver", "cuda", "pytorch", "benchmark_commit", "os")


def matrix_order(profile: str = "primary") -> tuple[tuple[str, str], ...]:
    if profile == "primary":
        return MATRIX_ORDER
    renderers = PROFILE_RENDERERS[profile]
    cases = tuple(case_id for case_id, _ in CASE_ORDERS)
    return tuple(
        (case_id, renderer)
        for case_index, case_id in enumerate(cases)
        for renderer in renderers[case_index % len(renderers):]
        + renderers[:case_index % len(renderers)]
    )


def build_plan(
    root: Path = ROOT, env_root: Path | None = None, profile: str = "primary"
) -> list[dict]:
    env_root = env_root or Path.home() / "miniforge3" / "envs"
    plan = []
    for index, (case_id, renderer) in enumerate(matrix_order(profile), start=1):
        python = env_root / ENV_BY_RENDERER[renderer] / "bin" / "python"
        plan.append({
            "step": index,
            "case_id": case_id,
            "renderer": renderer,
            "environment": ENV_BY_RENDERER[renderer],
            "command": [
                str(python),
                str(root / "benchmark.py"),
                "run",
                renderer,
                "--dataset",
                case_id,
            ],
        })
    return plan


def parse_nvml_process_memory(output: str, pid: int) -> float:
    found_pid = False
    for row in csv.reader(output.splitlines()):
        if not row:
            continue
        try:
            row_pid = int(row[0].strip())
        except (ValueError, IndexError):
            continue
        if row_pid != pid:
            continue
        found_pid = True
        memory = row[-1].strip()
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(?:MiB)?", memory)
        if match and float(match.group(1)) > 0:
            return float(match.group(1))
        raise RuntimeError(f"PID {pid} does not expose numeric NVML process memory: {memory}")
    if found_pid:
        raise RuntimeError(f"PID {pid} does not expose numeric NVML process memory")
    raise RuntimeError(f"PID {pid} was not reported by NVML")


def query_gpu_state(gpu_index: int) -> tuple[float, float]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()[0]
    memory_mib, utilization = (float(value.strip()) for value in output.split(","))
    return memory_mib, utilization


def wait_for_idle_gpu(
    gpu_index: int,
    max_memory_mib: float = 1024.0,
    max_utilization: float = 5.0,
    required_samples: int = 3,
    poll_seconds: float = 30.0,
) -> dict:
    consecutive = 0
    while consecutive < required_samples:
        memory_mib, utilization = query_gpu_state(gpu_index)
        if memory_mib <= max_memory_mib and utilization <= max_utilization:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive < required_samples:
            time.sleep(poll_seconds)
    return {
        "gpu_index": gpu_index,
        "memory_used_mib": memory_mib,
        "utilization_percent": utilization,
        "required_samples": required_samples,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_case_assets(root: Path, case: dict) -> dict:
    canonical = case.get("canonical_assets", {})
    if canonical.get("status") != "pinned":
        raise ValueError(f"{case.get('case_id')}: canonical assets are not pinned")
    scene_path = root / case["scene_path"]
    camera_path = root / case["camera_path"]
    ground_truth_dir = root / case["ground_truth_path"]
    preparation_path = root / case["preparation_path"]
    for path in (scene_path, camera_path, preparation_path):
        if not path.is_file():
            raise ValueError(f"{case['case_id']}: missing canonical asset {path}")
    if not ground_truth_dir.is_dir():
        raise ValueError(f"{case['case_id']}: missing ground-truth directory")

    actual = {
        "checkpoint_sha256": _sha256(scene_path),
        "camera_sha256": _sha256(camera_path),
    }
    preparation = _load(preparation_path)
    if preparation.get("status") != "canonical":
        raise ValueError(f"{case['case_id']}: preparation is not canonical")
    gt_entries = preparation.get("ground_truth_files", [])
    verified_entries = []
    for entry in gt_entries:
        image_path = ground_truth_dir / entry["image"]
        if not image_path.is_file():
            raise ValueError(f"{case['case_id']}: missing GT image {entry['image']}")
        image_sha = _sha256(image_path)
        if image_sha != entry.get("sha256"):
            raise ValueError(f"{case['case_id']}: GT image hash mismatch {entry['image']}")
        verified_entries.append({"image": entry["image"], "sha256": image_sha})
    gt_payload = json.dumps(
        verified_entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    actual["ground_truth_manifest_sha256"] = hashlib.sha256(gt_payload).hexdigest()
    for key, value in actual.items():
        if canonical.get(key) != value:
            raise ValueError(f"{case['case_id']}: canonical asset mismatch {key}")
    if preparation.get("ground_truth_file_manifest_sha256") != actual["ground_truth_manifest_sha256"]:
        raise ValueError(f"{case['case_id']}: preparation GT manifest mismatch")
    return {
        "case_id": case["case_id"],
        **actual,
        "ground_truth_file_count": len(verified_entries),
    }


def validate_canonical_assets(root: Path) -> list[dict]:
    return [validate_case_assets(root, case) for case in _suite_cases(root).values()]


def _validate_metric(
    path: Path, root: Path = ROOT, expected_commit: str | None = None
) -> tuple[tuple[str, str], tuple]:
    document = _load(path)
    validate_result(document)
    renderer_document = document.get("renderer", {})
    renderer = renderer_document.get("id")
    config_id = renderer_document.get("config_id")
    case_id = document.get("benchmark", {}).get("case_id")
    if document.get("status") != "complete" or not renderer or not config_id or not case_id:
        raise ValueError(f"{path}: incomplete Tier A metric")
    if document.get("evidence_tier") != "measured":
        raise ValueError(f"{path}: Tier A metric must use measured evidence")

    renderers = {
        row["id"]: row for row in _load(root / "benchmark" / "renderers.json")["renderers"]
    }
    if renderer not in renderers:
        raise ValueError(f"{path}: renderer is not registered: {renderer}")
    if document["renderer"]["source_commit"] != renderers[renderer]["source_commit"]:
        raise ValueError(f"{path}: renderer source commit does not match registry")
    adapter_ids = renderers[renderer].get("adapter_ids", [renderer])
    if config_id not in adapter_ids:
        raise ValueError(f"{path}: renderer config is not registered for family {renderer}: {config_id}")

    suite = _load(root / "benchmark" / "suite.json")
    protocol_path = root / suite["protocol_path"]
    protocol = _load(protocol_path)
    if _sha256(protocol_path) != suite["protocol_sha256"]:
        raise ValueError("suite protocol hash does not match benchmark/protocol.json")
    cases = {case["case_id"]: case for case in suite["cases"]}
    if case_id not in cases:
        raise ValueError(f"{path}: case is not registered: {case_id}")
    case = cases[case_id]
    benchmark = document["benchmark"]
    expected_benchmark = {
        "suite_id": suite["suite_id"],
        "suite_version": suite["version"],
        "track_id": suite["primary_track"],
        "dataset_id": case["dataset_id"],
        "scene_id": case["scene_id"],
        "checkpoint_sha256": case["canonical_assets"]["checkpoint_sha256"],
        "camera_trajectory_sha256": case["canonical_assets"]["camera_sha256"],
        "quality_reference_sha256": case["canonical_assets"]["ground_truth_manifest_sha256"],
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": suite["protocol_sha256"],
    }
    for key, expected in expected_benchmark.items():
        if benchmark.get(key) != expected:
            raise ValueError(f"{path}: benchmark {key} does not match suite")
    try:
        peak = float(document["metrics"]["performance"]["peak_vram_mb"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: missing NVML peak metric") from exc
    if peak <= 0:
        raise ValueError(f"{path}: expected positive NVML peak memory")

    metrics_raw = document["metrics"]["raw_samples"]
    provenance = document["provenance"]
    raw_uri = metrics_raw.get("uri")
    raw_sha = metrics_raw.get("sha256")
    if raw_uri != provenance.get("raw_samples_uri"):
        raise ValueError(f"{path}: raw-sample paths disagree")
    if raw_sha != provenance.get("raw_samples_sha256"):
        raise ValueError(f"{path}: raw-sample hashes disagree")
    raw_path = (root / raw_uri).resolve() if raw_uri else path.parent / "raw_samples.json"
    if raw_path != (path.parent / "raw_samples.json").resolve():
        raise ValueError(f"{path}: raw-sample path does not identify adjacent raw_samples.json")
    if not raw_path.is_file():
        raise ValueError(f"{path}: missing raw_samples.json")
    if _sha256(raw_path) != raw_sha:
        raise ValueError(f"{path}: raw-sample file hash mismatch")
    samples = _load(raw_path).get("nvml_process_memory_samples", [])
    positive_samples = [
        float(sample.get("used_gpu_memory_mib", 0.0))
        for sample in samples
        if sample.get("used_gpu_memory_mib") is not None
    ]
    if not positive_samples or max(positive_samples) <= 0:
        raise ValueError(f"{raw_path}: expected positive NVML process-memory samples")

    environment = document.get("environment", {})
    if expected_commit and environment.get("benchmark_commit") != expected_commit:
        raise ValueError(f"{path}: benchmark commit does not match checkout")
    cohort = tuple(environment.get(field) for field in COHORT_FIELDS)
    if any(value in (None, "") for value in cohort):
        raise ValueError(f"{path}: incomplete hardware cohort metadata")
    return (case_id, config_id), cohort


def validate_session_metrics(
    paths: list[Path], root: Path = ROOT, expected_commit: str | None = None,
    expected_order: tuple[tuple[str, str], ...] = MATRIX_ORDER,
) -> dict:
    if len(paths) != len(expected_order):
        raise ValueError(f"expected {len(expected_order)} metrics, found {len(paths)}")
    pairs = []
    cohorts = set()
    for path in paths:
        pair, cohort = _validate_metric(Path(path), root, expected_commit)
        pairs.append(pair)
        cohorts.add(cohort)
    if len(set(pairs)) != len(pairs) or set(pairs) != set(expected_order):
        raise ValueError("metrics do not contain exactly one row for every renderer/case pair")
    if len(cohorts) != 1:
        raise ValueError("Tier A metrics must belong to a single hardware cohort")
    return {
        "metric_count": len(paths),
        "renderer_count": len({renderer for _, renderer in pairs}),
        "case_count": len({case_id for case_id, _ in pairs}),
        "cohort": dict(zip(COHORT_FIELDS, next(iter(cohorts)))),
    }


def _suite_cases(root: Path) -> dict[str, dict]:
    suite = _load(root / "benchmark" / "suite.json")
    return {case["case_id"]: case for case in suite["cases"]}


def _metric_paths(
    root: Path, renderer: str, case_id: str, cases: dict[str, dict],
    benchmark_commit: str | None = None,
) -> set[Path]:
    case = cases[case_id]
    directory = (
        root / "results" / "measured" / renderer /
        case["dataset_id"] / case["scene_id"]
    )
    paths = set(directory.glob("*/metrics.json")) if directory.is_dir() else set()
    if benchmark_commit is None:
        return paths
    matching = set()
    for path in paths:
        try:
            if _load(path).get("environment", {}).get("benchmark_commit") == benchmark_commit:
                matching.add(path)
        except (OSError, json.JSONDecodeError):
            continue
    return matching


def preflight_nvml(gsplat_python: Path, holder_seconds: float = 60.0) -> float:
    holder_code = (
        "import os,time,torch; "
        "x=torch.empty(512*1024*1024,dtype=torch.uint8,device='cuda'); "
        "torch.cuda.synchronize(); print(os.getpid(),flush=True); "
        f"time.sleep({holder_seconds!r})"
    )
    holder = subprocess.Popen(
        [str(gsplat_python), "-c", holder_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        if holder.stdout is None:
            raise RuntimeError("CUDA holder did not expose stdout")
        line = holder.stdout.readline().strip()
        if not line:
            stderr = holder.stderr.read() if holder.stderr else ""
            raise RuntimeError(f"CUDA allocation preflight failed: {stderr.strip()}")
        pid = int(line)
        deadline = time.monotonic() + min(holder_seconds, 10.0)
        memory_mb = None
        while holder.poll() is None and time.monotonic() < deadline:
            query = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_gpu_memory",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            try:
                memory_mb = parse_nvml_process_memory(query.stdout, pid)
                break
            except RuntimeError:
                time.sleep(0.1)
        if memory_mb is None:
            raise RuntimeError("CUDA holder did not expose positive NVML process memory")
        holder.wait(timeout=holder_seconds + 5.0)
        return memory_mb
    finally:
        if holder.poll() is None:
            holder.terminate()
            try:
                holder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                holder.kill()
                holder.wait(timeout=5)


def _git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _clean_checkout_commit(root: Path) -> str:
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("formal Tier A runs require a clean tracked checkout")
    return _git_commit(root)


def _write_session(path: Path, session: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")


def _validate_report(report_path: Path, expected_order=MATRIX_ORDER) -> dict:
    report = _load(report_path)
    if report.get("rejected_files"):
        raise RuntimeError("benchmark report rejected one or more result files")
    overall = report.get("tiers", {}).get("measured", {}).get("overall", [])
    renderers = {row.get("competitor_id") for row in overall}
    expected = {renderer for _, renderer in expected_order}
    if len(overall) != len(expected) or renderers != expected:
        raise RuntimeError("measured overall ranking must contain every requested configuration")
    return report


def generate_session_report(metric_paths: list[Path], root: Path, output_dir: Path) -> dict:
    documents, rejected = load_results(metric_paths)
    report = generate_matrix_report(documents, _load(root / "benchmark" / "suite.json"))
    report["rejected_files"] = rejected
    write_report(report, output_dir)
    return report


def run_matrix(
    root: Path,
    env_root: Path,
    session_path: Path,
    resume: bool = False,
    max_steps: int | None = None,
    profile: str = "primary",
    report_output: Path | None = None,
    wait_gpu: int | None = None,
    idle_max_memory_mib: float = 1024.0,
    idle_max_utilization: float = 5.0,
    idle_samples: int = 3,
    idle_poll_seconds: float = 30.0,
) -> dict:
    if platform.system() != "Linux":
        raise RuntimeError("the Tier A matrix runner must execute on native Linux")
    checkout_commit = _clean_checkout_commit(root)
    expected_order = matrix_order(profile)
    plan = build_plan(root, env_root, profile)
    for row in plan:
        python = Path(row["command"][0])
        if not python.is_file():
            raise FileNotFoundError(f"missing renderer interpreter: {python}")

    cases = _suite_cases(root)
    validate_canonical_assets(root)
    if session_path.exists():
        if not resume:
            raise RuntimeError(f"session already exists; use --resume: {session_path}")
        session = _load(session_path)
        if session.get("benchmark_commit") != checkout_commit:
            raise RuntimeError("session benchmark commit does not match the checkout")
        if session.get("profile", "primary") != profile:
            raise RuntimeError("session matrix profile does not match the requested profile")
    else:
        existing = {
            path
            for row in plan
            for path in _metric_paths(
                root, row["renderer"], row["case_id"], cases, checkout_commit
            )
        }
        if existing:
            raise RuntimeError(
                "existing metrics would create duplicate renderer/case rows; archive them first"
            )
        session = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "benchmark_commit": checkout_commit,
            "profile": profile,
            "completed": [],
        }
        _write_session(session_path, session)

    completed = session.setdefault("completed", [])
    expected_cohort = None
    for index, item in enumerate(completed):
        expected = plan[index]
        if (item.get("case_id"), item.get("renderer")) != (
            expected["case_id"], expected["renderer"]
        ):
            raise RuntimeError("session completion order does not match the canonical plan")
        _, cohort = _validate_metric(
            root / item["metrics_path"], root, checkout_commit
        )
        if expected_cohort is not None and cohort != expected_cohort:
            raise RuntimeError("registered session metrics use mixed hardware cohorts")
        expected_cohort = cohort

    for row in plan[len(completed):]:
        existing = _metric_paths(
            root, row["renderer"], row["case_id"], cases, checkout_commit
        )
        if len(existing) > 1:
            raise RuntimeError(
                f"multiple unregistered metrics for {row['renderer']}/{row['case_id']}"
            )

    if not completed:
        if wait_gpu is not None:
            session.setdefault("idle_checks", []).append(wait_for_idle_gpu(
                wait_gpu, idle_max_memory_mib, idle_max_utilization,
                idle_samples, idle_poll_seconds,
            ))
            _write_session(session_path, session)
        memory_mb = preflight_nvml(env_root / "gsplat" / "bin" / "python")
        session["nvml_preflight_mib"] = memory_mb
        _write_session(session_path, session)

    for row in plan[len(completed):]:
        if max_steps is not None and len(session["completed"]) >= max_steps:
            break
        before = _metric_paths(
            root, row["renderer"], row["case_id"], cases, checkout_commit
        )
        if len(before) == 1:
            metric_path = next(iter(before))
            _, cohort = _validate_metric(metric_path, root, checkout_commit)
            if expected_cohort is not None and cohort != expected_cohort:
                raise RuntimeError(
                    f"orphan metric has a different cohort: {row['renderer']}/{row['case_id']}"
                )
            expected_cohort = cohort
            session["completed"].append({
                "step": row["step"],
                "case_id": row["case_id"],
                "renderer": row["renderer"],
                "metrics_path": str(metric_path.relative_to(root)),
            })
            _write_session(session_path, session)
            if max_steps is not None and len(session["completed"]) >= max_steps:
                break
            continue
        if wait_gpu is not None:
            session.setdefault("idle_checks", []).append(wait_for_idle_gpu(
                wait_gpu, idle_max_memory_mib, idle_max_utilization,
                idle_samples, idle_poll_seconds,
            ))
            _write_session(session_path, session)
        print(f"[{row['step']:02d}/{len(plan)}] {row['case_id']} :: {row['renderer']}", flush=True)
        subprocess.run(row["command"], cwd=root, check=True, env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            "PYTHONNOUSERSITE": "1",
        })
        after = _metric_paths(
            root, row["renderer"], row["case_id"], cases, checkout_commit
        )
        created = after - before
        if len(created) != 1:
            raise RuntimeError(
                f"expected one new metrics.json for {row['renderer']}/{row['case_id']}, "
                f"found {len(created)}"
            )
        metric_path = created.pop()
        _, cohort = _validate_metric(metric_path, root, checkout_commit)
        if expected_cohort is not None and cohort != expected_cohort:
            raise RuntimeError(
                f"new metric has a different cohort: {row['renderer']}/{row['case_id']}"
            )
        expected_cohort = cohort
        session["completed"].append({
            "step": row["step"],
            "case_id": row["case_id"],
            "renderer": row["renderer"],
            "metrics_path": str(metric_path.relative_to(root)),
        })
        _write_session(session_path, session)
        if max_steps is not None and len(session["completed"]) >= max_steps:
            break

    if len(session["completed"]) < len(plan):
        session["status"] = "partial"
        session["completed_count"] = len(session["completed"])
        _write_session(session_path, session)
        return session

    metric_paths = [root / item["metrics_path"] for item in session["completed"]]
    summary = validate_session_metrics(metric_paths, root, checkout_commit, expected_order)
    report_output = report_output or (
        root / "docs" / "leaderboard" if profile == "primary"
        else root / "reports" / "generated" / profile
    )
    generate_session_report(metric_paths, root, report_output)
    _validate_report(report_output / "ranking.json", expected_order)
    session["status"] = "complete"
    session["summary"] = summary
    session["report_output"] = str(report_output.relative_to(root))
    _write_session(session_path, session)
    return session


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--env-root", type=Path, default=Path.home() / "miniforge3" / "envs"
    )
    parser.add_argument(
        "--session",
        type=Path,
        default=ROOT / "artifacts" / "run-logs" / "linux-tier-a-session.json",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile", choices=sorted(PROFILE_RENDERERS), default="primary")
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--wait-gpu", type=int)
    parser.add_argument("--idle-max-memory-mib", type=float, default=1024.0)
    parser.add_argument("--idle-max-utilization", type=float, default=5.0)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    env_root = args.env_root.expanduser().resolve()
    requested_order = matrix_order(args.profile)
    if args.max_steps is not None and not 1 <= args.max_steps <= len(requested_order):
        parser.error(f"--max-steps must be between 1 and {len(requested_order)}")
    if args.dry_run:
        print(json.dumps(build_plan(root, env_root, args.profile), indent=2))
        return 0
    session = run_matrix(
        root, env_root, args.session.resolve(), args.resume, args.max_steps,
        args.profile,
        args.report_output.resolve() if args.report_output else None,
        args.wait_gpu,
        args.idle_max_memory_mib,
        args.idle_max_utilization,
        args.idle_samples,
        args.idle_poll_seconds,
    )
    print(json.dumps(session.get("summary", {
        "status": session["status"],
        "completed_count": len(session["completed"]),
    }), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
