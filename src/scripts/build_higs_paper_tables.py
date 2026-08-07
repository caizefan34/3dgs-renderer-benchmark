#!/usr/bin/env python
"""Build paper tables and convergence data from validated HiGS result JSONs.

Aggregates the protocol-valid result JSONs (one per job) into:
  summary.md        per-scene method table (mean +/- std over seeds)
  aggregate.md      geometric-mean speed and quality deltas
  convergence.csv   quality-vs-wall-clock points per job for curves
  aggregate.json    machine-readable aggregation for plotting

All inputs are validated against the frozen protocol before any table is
written, so a table can never be built from non-paper-eligible runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from higs_paper_protocol import build_experiment_plan  # noqa: E402
from higs_paper_results import HigsPaperResultError, validate_result_set  # noqa: E402

METHOD_ORDER = [
    "original_3dgs",
    "gsplat",
    "speedy_splat",
    "higs_full",
    "higs_proposed",
]
FAMILY = {
    "mipnerf360/bicycle": "Mip-NeRF 360",
    "mipnerf360/bonsai": "Mip-NeRF 360",
    "mipnerf360/counter": "Mip-NeRF 360",
    "mipnerf360/garden": "Mip-NeRF 360",
    "mipnerf360/kitchen": "Mip-NeRF 360",
    "mipnerf360/room": "Mip-NeRF 360",
    "mipnerf360/stump": "Mip-NeRF 360",
    "tanks_and_temples/train": "Tanks and Temples",
    "tanks_and_temples/truck": "Tanks and Temples",
    "deep_blending/drjohnson": "Deep Blending",
    "deep_blending/playroom": "Deep Blending",
}


def _mean_std(values):
    if not values:
        return None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def _fmt(mean, std, digits=2, scale=1.0):
    return f"{mean * scale:.{digits}f} +/- {std * scale:.{digits}f}"


def _geomean(values):
    return math.exp(statistics.fmean(math.log(v) for v in values if v > 0))


def _load_all(results_dir: Path) -> list[dict]:
    results = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"malformed result JSON {path}: {exc}")
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmark" / "higs-paper-protocol.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper" / "higs" / "tables")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--methods",
        help="Comma-separated method filter (e.g. original_3dgs,gsplat,higs_full,higs_proposed).",
    )
    parser.add_argument("--hardware", help="Comma-separated hardware filter (e.g. a100).")
    args = parser.parse_args(argv)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    results = _load_all(args.results_dir)
    methods = (
        set(part.strip() for part in args.methods.split(",") if part.strip())
        if args.methods
        else None
    )
    hardware = (
        set(part.strip() for part in args.hardware.split(",") if part.strip())
        if args.hardware
        else None
    )
    try:
        report = validate_result_set(
            results,
            protocol,
            require_complete=args.require_complete,
            methods=methods,
            hardware=hardware,
        )
    except HigsPaperResultError as exc:
        raise SystemExit(f"cannot build tables from invalid results: {exc}")
    if args.require_complete and report["missing"]:
        raise SystemExit(f"matrix incomplete: {report}")

    plan = {job["job_id"]: job for job in build_experiment_plan(protocol)}
    rows = []
    for result in results:
        job = plan.get(result["job_id"])
        if not job:
            continue
        rows.append(result)

    scenes = sorted({row["scene"] for row in rows}, key=lambda s: (FAMILY.get(s, ""), s))
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]

    keyed = {}
    for row in rows:
        keyed.setdefault((row["scene"], row["method"]), []).append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        "# HiGS full-training results (A100, 30k steps, 3 seeds)",
        "",
        f"Protocol: `benchmark/higs-paper-protocol.json`; {report['complete']} complete "
        f"jobs, {report['failed']} failed, {report['missing']} missing.",
        "",
        "## Per-scene table",
        "",
        "| Scene | Method | PSNR (dB) | SSIM | LPIPS | Wall (s) | TTQ (s) | Mem (GiB) | Energy (kJ) | Gaussians |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scene in scenes:
        family = FAMILY.get(scene, "")
        first = True
        for method in methods:
            cells = keyed.get((scene, method), [])
            if not cells:
                continue
            psnr = _mean_std([c["quality"]["psnr_db"] for c in cells])
            ssim = _mean_std([c["quality"]["ssim"] for c in cells])
            lpips = _mean_std([c["quality"]["lpips"] for c in cells])
            wall = _mean_std([c["performance"]["wall_time_seconds"] for c in cells])
            ttq = _mean_std([c["performance"]["time_to_quality_seconds"] for c in cells])
            mem = _mean_std([c["resources"]["peak_gpu_memory_mib"] / 1024.0 for c in cells])
            energy = _mean_std([c["resources"]["energy_joules"] / 1000.0 for c in cells])
            count = _mean_std([float(c["resources"]["final_gaussian_count"]) for c in cells])
            scene_cell = f"{scene} ({family})" if first else ""
            summary_lines.append(
                f"| {scene_cell} | {method} | {_fmt(*psnr)} | {_fmt(*ssim, 3)} | "
                f"{_fmt(*lpips, 3)} | {_fmt(*wall, 0)} | {_fmt(*ttq, 0)} | "
                f"{_fmt(*mem, 1)} | {_fmt(*energy, 0)} | {_fmt(*count, 0)} |"
            )
            first = False
        summary_lines.append("")
    (args.output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    convergence = []
    for row in rows:
        for point in row["quality_curve"]:
            convergence.append({
                "job_id": row["job_id"],
                "method": row["method"],
                "scene": row["scene"],
                "seed": row["seed"],
                "iteration": point["iteration"],
                "wall_time_seconds": point["wall_time_seconds"],
                "psnr_db": point["psnr_db"],
                "ssim": point["ssim"],
                "lpips": point["lpips"],
                "num_GS": point["num_GS"],
            })
    conv_path = args.output_dir / "convergence.csv"
    with conv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(convergence[0].keys()))
        writer.writeheader()
        writer.writerows(convergence)

    aggregate = {"schema_version": "1.0", "scenes": len(scenes), "methods": methods}
    baseline_pairs = [
        ("gsplat", "higs_proposed"),
        ("higs_full", "higs_proposed"),
        # Official accelerated baseline sanity pair: how the frozen 30k
        # Speedy-Splat runs compare to the same-backend gsplat control.
        ("gsplat", "speedy_splat"),
    ]
    for baseline, proposed in baseline_pairs:
        speed_ratios = []
        psnr_deltas = []
        for scene in scenes:
            base_cells = keyed.get((scene, baseline), [])
            prop_cells = keyed.get((scene, proposed), [])
            if not base_cells or not prop_cells:
                continue
            base_wall = _mean_std([c["performance"]["wall_time_seconds"] for c in base_cells])
            prop_wall = _mean_std([c["performance"]["wall_time_seconds"] for c in prop_cells])
            base_psnr = _mean_std([c["quality"]["psnr_db"] for c in base_cells])
            prop_psnr = _mean_std([c["quality"]["psnr_db"] for c in prop_cells])
            if base_wall and prop_wall and base_psnr and prop_psnr:
                speed_ratios.append(base_wall[0] / prop_wall[0])
                psnr_deltas.append(prop_psnr[0] - base_psnr[0])
        if speed_ratios:
            aggregate[f"{baseline}_to_{proposed}"] = {
                "wall_speedup_geomean": _geomean(speed_ratios),
                "psnr_delta_mean_db": statistics.fmean(psnr_deltas),
            }
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )

    per_method = {}
    for method in methods:
        cells = [row for row in rows if row["method"] == method]
        if not cells:
            continue
        per_method[method] = {
            "jobs": len(cells),
            "psnr_db_mean": statistics.fmean(c["quality"]["psnr_db"] for c in cells),
            "wall_time_seconds_mean": statistics.fmean(
                c["performance"]["wall_time_seconds"] for c in cells
            ),
            "time_to_quality_seconds_mean": statistics.fmean(
                c["performance"]["time_to_quality_seconds"] for c in cells
            ),
            "peak_gpu_memory_mib_mean": statistics.fmean(
                c["resources"]["peak_gpu_memory_mib"] for c in cells
            ),
        }
    matrix_summary = {
        "schema_version": "1.0",
        "report": report,
        "scenes": len(scenes),
        "methods": methods,
        "per_method": per_method,
    }
    (args.output_dir / "matrix-summary.json").write_text(
        json.dumps(matrix_summary, indent=2) + "\n", encoding="utf-8"
    )

    aggregate_lines = [
        "# Aggregate speed and quality (A100, 30k, 3 seeds)",
        "",
        "| Comparison | Wall-time speedup (geomean) | PSNR delta (mean, dB) |",
        "| --- | --- | --- |",
    ]
    for key, value in aggregate.items():
        if key.startswith("schema") or key in ("scenes", "methods"):
            continue
        label = key.replace("_to_", " vs ").replace("_", " ")
        aggregate_lines.append(
            f"| {label} | {value['wall_speedup_geomean']:.2f}x | {value['psnr_delta_mean_db']:+.3f} |"
        )
    (args.output_dir / "aggregate.md").write_text("\n".join(aggregate_lines) + "\n", encoding="utf-8")

    print(json.dumps({"complete": report["complete"], "scenes": len(scenes), "methods": methods, "output": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
