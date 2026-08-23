#!/usr/bin/env python
"""Run the EPIC-05 common-compatible compression matrix."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scripts.collect_compression_result import collect  # noqa: E402
from scripts.run_linux_tier_a_matrix import wait_for_idle_gpu  # noqa: E402


CODECS = (
    "reference-ply", "xz-lossless", "block-float", "tile-codebook",
    "playcanvas-compressed-ply", "playcanvas-sog", "spz", "spz-6-6", "spz-5-4",
)

CODEC_CONFIGS = {
    "reference-ply": {"encoder": None, "suffix": ".ply", "options": []},
    "xz-lossless": {"encoder": "xz-lossless", "suffix": ".ply.xz", "options": []},
    "block-float": {"encoder": "block-float", "suffix": ".zip", "options": []},
    "tile-codebook": {"encoder": "tile-codebook", "suffix": ".zip", "options": []},
    "playcanvas-compressed-ply": {
        "encoder": "playcanvas-compressed-ply", "suffix": ".compressed.ply", "options": [],
    },
    "playcanvas-sog": {"encoder": "playcanvas-sog", "suffix": ".sog", "options": []},
    "spz": {
        "encoder": "spz", "suffix": ".spz",
        "options": ["--sh1-bits", "8", "--sh-rest-bits", "8"],
    },
    "spz-6-6": {
        "encoder": "spz", "suffix": ".spz",
        "options": ["--sh1-bits", "6", "--sh-rest-bits", "6"],
    },
    "spz-5-4": {
        "encoder": "spz", "suffix": ".spz",
        "options": ["--sh1-bits", "5", "--sh-rest-bits", "4"],
    },
}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_plan(
    root: Path, case_ids: set[str] | None = None,
    codecs: tuple[str, ...] = CODECS,
    run_root: Path | None = None,
) -> list[dict]:
    run_root = run_root or root / "artifacts" / "compression" / "runs"
    suite = _load(root / "benchmark" / "suite.json")
    rows = []
    for case in suite["cases"]:
        if case_ids and case["case_id"] not in case_ids:
            continue
        for codec in codecs:
            config = CODEC_CONFIGS[codec]
            stem = f"{case['case_id']}.{codec}"
            archive_name = f"{case['case_id']}{config['suffix']}" if codec == "spz" else f"{stem}{config['suffix']}"
            rows.append({
                "step": len(rows) + 1,
                "case_id": case["case_id"],
                "dataset_id": case["dataset_id"],
                "scene_id": case["scene_id"],
                "codec": codec,
                "encoder": config["encoder"],
                "encode_options": config["options"],
                "source": str(root / case["scene_path"]),
                "cameras": str(root / case["camera_path"]),
                "ground_truth": str(root / case["ground_truth_path"]),
                "archive": str(root / "artifacts" / "compression" / archive_name),
                "manifest": str(root / "artifacts" / "compression" / f"{stem}.json"),
                "decoded": str(root / "artifacts" / "compression" / "decoded" / f"{stem}.ply"),
                "run_dir": str(run_root / case["case_id"] / codec),
            })
    return rows


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _encode(row: dict, python: Path, root: Path) -> None:
    if row["codec"] == "reference-ply" or Path(row["manifest"]).is_file():
        return
    subprocess.run([
        str(python), "src/scripts/compress_ply.py", "encode",
        "--input", row["source"], "--output", row["archive"],
        "--manifest", row["manifest"], "--codec", row["encoder"],
        *row["encode_options"],
    ], cwd=root, check=True)


def _decode(row: dict, python: Path, root: Path) -> Path:
    if row["codec"] == "reference-ply":
        return Path(row["source"])
    decoded = Path(row["decoded"])
    if not decoded.is_file():
        decoded.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            str(python), "src/scripts/compress_ply.py", "decode",
            "--input", row["archive"], "--output", str(decoded),
        ], cwd=root, check=True)
    return decoded


def _completed_renderer_run(run_dir: Path, document: dict) -> bool:
    speed_runs = document.get("speed_runs", [])
    quality_runs = document.get("quality_runs", [])
    outputs = (
        run_dir / "gsplat" / "speed" / "benchmark_results.json",
        run_dir / "gsplat" / "speed" / "nvml_samples.json",
        run_dir / "gsplat" / "quality" / "quality_gt.json",
    )
    return bool(speed_runs) and bool(quality_runs) and all(
        item.get("status") == "ok" for item in (*speed_runs, *quality_runs)
    ) and all(path.is_file() for path in outputs)


def _run_renderer(row: dict, scene: Path, python: Path, root: Path, protocol: dict) -> Path:
    run_dir = Path(row["run_dir"])
    report = run_dir / "local_renderer_suite_report.json"
    if report.is_file():
        document = _load(report)
        if _completed_renderer_run(run_dir, document):
            return run_dir
    width, height = protocol["resolution"]
    timing = protocol["timing"]
    subprocess.run([
        str(python), "src/scripts/run_local_renderer_suite.py",
        "--scene", str(scene), "--cameras", row["cameras"],
        "--ground-truth-dir", row["ground_truth"], "--renderers", "gsplat",
        "--output-dir", str(run_dir),
        "--frames", str(timing["measured_frames_per_repeat"]),
        "--warmup", str(timing["warmup_frames"]),
        "--repeats", str(timing["repeats"]),
        "--width", str(width), "--height", str(height),
    ], cwd=root, check=True)
    return run_dir


def _reference_path(session: dict, case_id: str, root: Path) -> Path:
    for item in session["completed"]:
        if item["case_id"] == case_id and item["codec"] == "reference-ply":
            return root / item["metrics_path"]
    raise ValueError(f"reference-ply must complete first for {case_id}")


def _collect_row(row: dict, scene: Path, run_dir: Path, session: dict, root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        root / "results" / "measured-compression" / row["codec"] /
        row["dataset_id"] / row["scene_id"] / timestamp
    )
    reference = None if row["codec"] == "reference-ply" else _reference_path(session, row["case_id"], root)
    result, raw = collect(
        run_dir, scene, row["case_id"], row["codec"],
        None if row["codec"] == "reference-ply" else Path(row["manifest"]),
        reference,
        "not_applicable" if row["codec"] == "reference-ply" else "pending",
    )
    output.mkdir(parents=True, exist_ok=False)
    raw_path = output / "raw_samples.json"
    _write_json(raw_path, raw)
    from scripts.collect_matrix_result import _sha256
    result["raw_samples"] = {
        "uri": str(raw_path.relative_to(root)).replace("\\", "/"),
        "sha256": _sha256(raw_path),
    }
    metrics_path = output / "metrics.json"
    _write_json(metrics_path, result)
    return metrics_path


def _generate_report(session: dict, root: Path, output: Path) -> None:
    results = [_load(root / item["metrics_path"]) for item in session["completed"]]
    pending_count = sum(
        result["metrics"]["near_lossless_gate"]["visual_audit"] == "pending"
        for result in results
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "compression-results.json", {
        "schema_version": "1.0", "evidence_tier": "measured", "results": results,
    })
    lines = [
        "# EPIC-05 compression matrix", "",
        "All rows use gsplat packed on decoded standard PLY files.",
        f"Visual audits pending: {pending_count}.", "",
        "| Case | Codec | Ratio | Decode ms | FPS | PSNR | PSNR delta | LPIPS delta | Numeric gate | Visual audit |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for result in results:
        artifact = result["codec"]["artifact"]
        metrics = result["metrics"]
        lines.append(
            f"| {result['case']['case_id']} | {result['codec']['id']} | "
            f"{artifact['compression_ratio']:.3f}x | {artifact['decode_ms']:.1f} | "
            f"{metrics['performance']['fps']:.2f} | {metrics['quality']['psnr_db']:.3f} | "
            f"{metrics['quality_delta']['psnr_db']:+.3f} | {metrics['quality_delta']['lpips']:+.4f} | "
            f"{'pass' if metrics['near_lossless_gate']['numeric_pass'] else 'fail'} | "
            f"{metrics['near_lossless_gate']['visual_audit']} |"
        )
    (output / "compression-results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> dict:
    root, python = args.root.resolve(), args.python.resolve()
    session_path = args.session.resolve()
    run_root = (
        args.run_root.resolve() if args.run_root else
        root / "artifacts" / "compression" / "runs" / session_path.stem
    )
    selected_case_ids = set(args.case_id) if args.case_id else None
    selected_codecs = tuple(args.codec) if args.codec else CODECS
    plan = build_plan(
        root, selected_case_ids, selected_codecs, run_root,
    )
    selection = {
        "case_ids": sorted(selected_case_ids) if selected_case_ids else None,
        "codecs": list(selected_codecs),
        "run_root": str(run_root),
    }
    if session_path.is_file():
        if not args.resume:
            raise RuntimeError(f"session already exists; use --resume: {session_path}")
        session = _load(session_path)
        if session.get("benchmark_commit") != subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip():
            raise RuntimeError("session benchmark commit does not match the checkout")
        if session.get("selection") != selection:
            raise RuntimeError("session selection does not match the requested matrix")
    else:
        session = {
            "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "benchmark_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "selection": selection, "completed": [], "encoded": [],
        }
        _write_json(session_path, session)
    completed = {(item["case_id"], item["codec"]) for item in session["completed"]}
    encoded = {(item["case_id"], item["codec"]) for item in session["encoded"]}
    for row in plan:
        key = (row["case_id"], row["codec"])
        if row["codec"] != "reference-ply" and key not in encoded:
            print(f"encode {row['case_id']} :: {row['codec']}", flush=True)
            _encode(row, python, root)
            session["encoded"].append({"case_id": row["case_id"], "codec": row["codec"], "manifest": row["manifest"]})
            _write_json(session_path, session)
        if args.encode_only or key in completed:
            continue
        scene = _decode(row, python, root)
        wait_for_idle_gpu(
            args.wait_gpu, args.idle_max_memory_mib, args.idle_max_utilization,
            args.idle_samples, args.idle_poll_seconds,
        )
        print(f"measure {row['step']:02d}/{len(plan)} {row['case_id']} :: {row['codec']}", flush=True)
        run_dir = _run_renderer(row, scene, python, root, _load(root / "benchmark" / "protocol.json"))
        metrics = _collect_row(row, scene, run_dir, session, root)
        session["completed"].append({
            "case_id": row["case_id"], "codec": row["codec"],
            "metrics_path": str(metrics.relative_to(root)).replace("\\", "/"),
        })
        _write_json(session_path, session)
        if row["codec"] != "reference-ply":
            scene.unlink()
    session["status"] = "encoded" if args.encode_only else "complete"
    if not args.encode_only:
        _generate_report(session, root, args.report_output.resolve())
    _write_json(session_path, session)
    return session


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--python", type=Path, default=Path.home() / "miniforge3" / "envs" / "gsplat" / "bin" / "python")
    parser.add_argument("--session", type=Path, default=ROOT / "artifacts" / "run-logs" / "linux-compression-session.json")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--report-output", type=Path, default=ROOT / "reports" / "generated" / "compression")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-gpu", type=int, default=7)
    parser.add_argument("--idle-max-memory-mib", type=float, default=1024.0)
    parser.add_argument("--idle-max-utilization", type=float, default=5.0)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-poll-seconds", type=float, default=30.0)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--codec", action="append", choices=CODECS)
    args = parser.parse_args(argv)
    if args.dry_run:
        root = args.root.resolve()
        run_root = (
            args.run_root.resolve() if args.run_root else
            root / "artifacts" / "compression" / "runs" / args.session.resolve().stem
        )
        print(json.dumps(build_plan(
            root, set(args.case_id) if args.case_id else None,
            tuple(args.codec) if args.codec else CODECS,
            run_root,
        ), indent=2))
        return 0
    session = run(args)
    print(json.dumps({"status": session["status"], "encoded": len(session["encoded"]), "completed": len(session["completed"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
