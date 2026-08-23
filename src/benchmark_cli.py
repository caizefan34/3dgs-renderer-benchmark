"""Measured-first command line entry point."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _suite_and_protocol():
    suite_path = ROOT / "benchmark" / "suite.json"
    protocol_path = ROOT / "benchmark" / "protocol.json"
    suite, protocol = _load(suite_path), _load(protocol_path)
    actual = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if actual != suite["protocol_sha256"]:
        raise SystemExit(f"protocol hash mismatch: suite has {suite['protocol_sha256']}, file is {actual}")
    return suite, protocol


def _renderers(registry):
    return {row["id"]: row for row in registry["renderers"]}


def _selected_cases(suite, dataset):
    if not dataset or dataset == "all":
        return suite["cases"]
    rows = [row for row in suite["cases"] if dataset in {row["dataset_id"], row["scene_id"], row["case_id"]}]
    if not rows:
        raise SystemExit(f"unknown dataset, scene, or case: {dataset}")
    return rows


def _selected_renderers(registry, requested):
    by_id = _renderers(registry)
    if requested == "all":
        return [
            {**row, "family_id": row["id"], "id": row["default_adapter_id"]}
            for row in registry["renderers"] if row["execution_status"] == "automatic_ready"
        ]
    family = by_id.get(requested)
    if family is None:
        family = next((row for row in registry["renderers"] if requested in row["adapter_ids"]), None)
    if family is None:
        raise SystemExit(f"unknown renderer: {requested}")
    if family["execution_status"] != "automatic_ready":
        raise SystemExit(f"{requested} is not automatic-ready: {family['gap']}")
    adapter_id = requested if requested in family["adapter_ids"] else family["default_adapter_id"]
    if adapter_id in family.get("primary_excluded_adapter_ids", []):
        raise SystemExit(f"{adapter_id} is a diagnostic alias/config and cannot enter the primary track")
    return [{**family, "family_id": family["id"], "id": adapter_id}]


def _run(args) -> int:
    suite, protocol = _suite_and_protocol()
    registry = _load(ROOT / "benchmark" / "renderers.json")
    renderers = _selected_renderers(registry, args.renderer)
    cases = _selected_cases(suite, args.dataset)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plans = []
    for case_index, case in enumerate(cases):
        offset = case_index % len(renderers)
        ordered_renderers = renderers[offset:] + renderers[:offset]
        for renderer in ordered_renderers:
            output = ROOT / "results" / "measured" / renderer["id"] / case["dataset_id"] / case["scene_id"] / timestamp
            command = [
                sys.executable, "src/scripts/run_local_renderer_suite.py",
                "--scene", case["scene_path"], "--cameras", case["camera_path"],
                "--ground-truth-dir", case["ground_truth_path"], "--renderers", renderer["id"],
                "--output-dir", str(output),
                "--frames", str(protocol["timing"]["measured_frames_per_repeat"]),
                "--warmup", str(protocol["timing"]["warmup_frames"]),
                "--repeats", str(protocol["timing"]["repeats"]),
                "--width", str(protocol["resolution"][0]),
                "--height", str(protocol["resolution"][1]),
                "--benchmark-type", "real_scene_speed",
            ]
            plans.append({"renderer": renderer["id"], "case": case["case_id"], "output": str(output), "command": command})

    if args.dry_run:
        print(json.dumps(plans, indent=2))
        return 0

    failures = 0
    for plan in plans:
        for required in (Path(plan["command"][3]), Path(plan["command"][5]), Path(plan["command"][7])):
            if not (ROOT / required).exists() and not required.exists():
                print(f"missing prepared asset: {required}", file=sys.stderr)
                failures += 1
                break
        else:
            completed = subprocess.run(plan["command"], cwd=ROOT, check=False)
            if completed.returncode != 0:
                failures += 1
                continue
            collector = [
                sys.executable, "src/scripts/collect_matrix_result.py",
                "--run-dir", plan["output"], "--renderer", plan["renderer"],
                "--case-id", plan["case"], "--output", str(Path(plan["output"]) / "metrics.json"),
            ]
            failures += int(subprocess.run(collector, cwd=ROOT, check=False).returncode != 0)
    return 1 if failures else 0


def _report(args) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from benchmark_matrix import generate_matrix_report, load_results, write_report
    paths = []
    for tier in ("measured", "reproduced", "paper"):
        paths.extend((ROOT / "results" / tier).glob("**/metrics.json"))
    documents, rejected = load_results(paths)
    suite, _ = _suite_and_protocol()
    report = generate_matrix_report(documents, suite)
    report["rejected_files"] = rejected
    write_report(report, Path(args.output_dir))
    print(Path(args.output_dir) / "ranking.md")
    return 1 if rejected else 0


def _list(args) -> int:
    if args.kind == "renderers":
        for row in _load(ROOT / "benchmark" / "renderers.json")["renderers"]:
            for adapter_id in row["adapter_ids"]:
                suffix = " (diagnostic)" if adapter_id in row.get("primary_excluded_adapter_ids", []) else ""
                print(f"{adapter_id:<28} {row['execution_status']:<26} {row['name']}{suffix}")
    else:
        for row in _load(ROOT / "benchmark" / "suite.json")["cases"]:
            print(f"{row['case_id']:<28} {row['dataset_id']}/{row['scene_id']}")
    return 0


def _prepare(args) -> int:
    command = [sys.executable, str(ROOT / "src" / "scripts" / "prepare_datasets.py"), args.dataset]
    if args.archive:
        command.extend(["--archive", args.archive])
    if args.scene:
        command.extend(["--scene", args.scene])
    if args.data_root:
        command.extend(["--data-root", args.data_root])
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def _prepare_case(args) -> int:
    command = [
        sys.executable, str(ROOT / "src" / "scripts" / "prepare_suite_case.py"), args.case_id,
    ]
    if args.model_archive:
        command.extend(["--model-archive", args.model_archive])
    if args.candidate_output:
        command.extend(["--candidate-output", args.candidate_output])
    if args.audit_only:
        command.append("--audit-only")
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def _stage_case(args) -> int:
    command = [
        sys.executable, str(ROOT / "src" / "scripts" / "stage_dataset_case.py"), args.case_id,
        "--checkpoint", args.checkpoint, "--cameras", args.cameras,
        "--ground-truth-dir", args.ground_truth_dir,
        "--derivation-manifest", args.derivation_manifest,
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="benchmark", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run local Tier A measurements")
    run.add_argument("renderer", nargs="?", default="all")
    run.add_argument("--dataset", default="all")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=_run)
    listing = subparsers.add_parser("list")
    listing.add_argument("kind", choices=["renderers", "datasets"])
    listing.set_defaults(func=_list)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("dataset")
    prepare.add_argument("--archive")
    prepare.add_argument("--scene")
    prepare.add_argument("--data-root")
    prepare.set_defaults(func=_prepare)
    prepare_case = subparsers.add_parser("prepare-case", help="prepare one deterministic official suite case")
    prepare_case.add_argument("case_id")
    prepare_case.add_argument("--model-archive")
    prepare_case.add_argument("--candidate-output")
    prepare_case.add_argument("--audit-only", action="store_true")
    prepare_case.set_defaults(func=_prepare_case)
    stage_case = subparsers.add_parser("stage-case", help="manually stage a reviewed custom case")
    stage_case.add_argument("case_id")
    stage_case.add_argument("--checkpoint", required=True)
    stage_case.add_argument("--cameras", required=True)
    stage_case.add_argument("--ground-truth-dir", required=True)
    stage_case.add_argument("--derivation-manifest", required=True)
    stage_case.set_defaults(func=_stage_case)
    report = subparsers.add_parser("report", help="generate measured-first rankings and Pareto charts")
    report.add_argument("--output-dir", default=str(ROOT / "reports" / "generated"))
    report.set_defaults(func=_report)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
