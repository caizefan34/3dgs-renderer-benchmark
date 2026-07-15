#!/usr/bin/env python
"""Aggregate Mip-NeRF 360 resolution runs and render a scaling plot."""
import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
SCHEMA_DIR = os.path.join(PROJECT_ROOT, "schemas")
sys.path.insert(0, PROJECT_ROOT)

from benchmark_suite import BENCHMARK_SUITE_VERSION  # noqa: E402
from schema_validation import validate_json_file  # noqa: E402


RESOLUTION_ORDER = {"720p": 0, "1080p": 1, "4k": 2}
SCENE_ORDER = {"garden": 0, "bicycle": 1, "room": 2}


def records_from_document(document, source_path):
    scene_id = document["scene"]["scene_id"]
    resolution_name = document["protocol"]["resolution_name"]
    width, height = document["protocol"]["resolution"]
    return [
        {
            "scene_id": scene_id,
            "renderer": row["renderer"],
            "resolution_name": resolution_name,
            "width": width,
            "height": height,
            "mean_fps": row["mean_fps"],
            "mean_latency_ms": row["mean_latency_ms"],
            "p99_latency_ms": row["p99_latency_ms"],
            "peak_vram_mb": row["peak_vram_mb"],
            "stability_score": row["stability_score"],
            "source_json": os.path.relpath(source_path, REPO_ROOT),
        }
        for row in document["results"].values()
    ]


def load_records(paths):
    records = []
    first_document = None
    schema = os.path.join(SCHEMA_DIR, "scene_speed.schema.json")
    for path in paths:
        validate_json_file(path, schema)
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        first_document = first_document or document
        records.extend(records_from_document(document, path))
    records.sort(
        key=lambda row: (
            SCENE_ORDER[row["scene_id"]],
            row["renderer"],
            RESOLUTION_ORDER[row["resolution_name"]],
        )
    )
    return records, first_document


def render_plot(records, output_path):
    renderer_style = {
        "gsplat_higs_auto": ("HiGS Auto", "#0f766e", "o"),
        "speedy_splat": ("Speedy-Splat", "#b45309", "s"),
    }
    labels = ["720p", "1080p", "4K"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=False)
    for axis, scene_id in zip(axes, ("garden", "bicycle", "room")):
        for renderer, (label, color, marker) in renderer_style.items():
            rows = [
                row for row in records
                if row["scene_id"] == scene_id and row["renderer"] == renderer
            ]
            rows.sort(key=lambda row: RESOLUTION_ORDER[row["resolution_name"]])
            axis.plot(
                labels,
                [row["mean_fps"] for row in rows],
                color=color,
                marker=marker,
                linewidth=2,
                markersize=6,
                label=label,
            )
        axis.set_title(scene_id.title(), fontsize=12)
        axis.set_xlabel("Output resolution")
        axis.set_ylabel("Mean FPS")
        axis.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Mip-NeRF 360 Resolution Scaling", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026-07-15")
    parser.add_argument("--results-root", default=os.path.join(REPO_ROOT, "data", "results"))
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-plot", default=None)
    args = parser.parse_args()

    pattern = os.path.join(
        args.results_root,
        f"mipnerf360_*_*_{args.date}",
        "benchmark_results.json",
    )
    paths = sorted(glob.glob(pattern))
    records, first_document = load_records(paths)
    if len(paths) != 9 or len(records) != 18:
        raise SystemExit(f"Expected 9 source JSONs and 18 records, got {len(paths)} and {len(records)}")

    output_json = args.output_json or os.path.join(
        args.results_root, f"mipnerf360_resolution_scaling_{args.date}.json"
    )
    output_plot = args.output_plot or os.path.join(
        args.results_root, f"mipnerf360_resolution_scaling_{args.date}.png"
    )
    document = {
        "schema_version": 1,
        "benchmark_suite_version": BENCHMARK_SUITE_VERSION,
        "environment": first_document["environment"],
        "protocol": {
            "scenes": ["garden", "bicycle", "room"],
            "resolutions": ["720p", "1080p", "4k"],
            "warmup_frames": 10,
            "measured_frames_per_repeat": 30,
            "repeats": 3,
            "source_documents": len(paths),
        },
        "records": records,
    }
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False, allow_nan=False)
    validate_json_file(
        output_json,
        os.path.join(SCHEMA_DIR, "resolution_scaling.schema.json"),
    )
    render_plot(records, output_plot)
    print(f"validated {output_json}")
    print(f"wrote {output_plot}")


if __name__ == "__main__":
    main()
