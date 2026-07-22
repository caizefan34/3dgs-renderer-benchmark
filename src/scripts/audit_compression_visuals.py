#!/usr/bin/env python
"""Create an auditable worst-frame visual comparison for compression rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_index(metrics_path: Path) -> tuple[dict, Path]:
    metrics = _load(metrics_path)
    raw_record = metrics["raw_samples"]
    raw_path = Path(raw_record["uri"])
    if not raw_path.is_absolute():
        root = next(
            (parent for parent in metrics_path.parents if (parent / "benchmark" / "suite.json").is_file()),
            None,
        )
        if root is None:
            raise ValueError("cannot resolve repository-relative raw sample path")
        raw_path = root / raw_path
    if _sha256(raw_path) != raw_record["sha256"]:
        raise ValueError(f"raw sample hash mismatch: {raw_path}")
    raw = _load(raw_path)
    root = Path(raw["render_output_root"])
    rows = {row["image"]: row for row in raw["render_outputs"]}
    return rows, root


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def compare(reference_metrics: Path, candidate_metrics: Path) -> tuple[list[dict], dict]:
    reference_rows, reference_root = _render_index(reference_metrics)
    candidate_rows, candidate_root = _render_index(candidate_metrics)
    if reference_rows.keys() != candidate_rows.keys():
        raise ValueError("reference and candidate render-output image sets differ")
    frames = []
    for image_name in reference_rows:
        reference = _read_rgb(reference_root / reference_rows[image_name]["path"])
        candidate = _read_rgb(candidate_root / candidate_rows[image_name]["path"])
        if reference.shape != candidate.shape:
            raise ValueError(f"render shape mismatch: {image_name}")
        error = np.abs(candidate - reference)
        mse = float(np.mean((candidate - reference) ** 2))
        frames.append({
            "image": image_name,
            "reference_path": str(reference_root / reference_rows[image_name]["path"]),
            "candidate_path": str(candidate_root / candidate_rows[image_name]["path"]),
            "mean_abs_error": float(error.mean()),
            "max_abs_error": float(error.max()),
            "psnr_vs_reference_db": "inf" if mse == 0 else 10.0 * math.log10(1.0 / mse),
            "pixel_fraction_over_1_255": float(np.mean(np.max(error, axis=2) > 1.0 / 255.0)),
            "pixel_fraction_over_3_255": float(np.mean(np.max(error, axis=2) > 3.0 / 255.0)),
        })
    ordered = sorted(frames, key=lambda row: row["mean_abs_error"], reverse=True)
    summary = {
        "frame_count": len(frames),
        "mean_frame_abs_error": float(np.mean([row["mean_abs_error"] for row in frames])),
        "worst_frame_abs_error": ordered[0]["mean_abs_error"],
        "worst_frame": ordered[0]["image"],
    }
    return ordered, summary


def make_contact_sheet(frames: list[dict], output: Path, count: int = 6) -> None:
    selected = frames[:count]
    panels = []
    target_width = 480
    for row in selected:
        reference = Image.open(row["reference_path"]).convert("RGB")
        candidate = Image.open(row["candidate_path"]).convert("RGB")
        height = round(reference.height * target_width / reference.width)
        reference = reference.resize((target_width, height), Image.Resampling.LANCZOS)
        candidate = candidate.resize((target_width, height), Image.Resampling.LANCZOS)
        ref_np = np.asarray(reference, dtype=np.int16)
        cand_np = np.asarray(candidate, dtype=np.int16)
        heat = np.clip(np.abs(cand_np - ref_np) * 12, 0, 255).astype(np.uint8)
        heat = Image.fromarray(heat, "RGB")
        label_height = 42
        panel = Image.new("RGB", (target_width * 3, height + label_height), "white")
        panel.paste(reference, (0, label_height))
        panel.paste(candidate, (target_width, label_height))
        panel.paste(heat, (target_width * 2, label_height))
        draw = ImageDraw.Draw(panel)
        draw.text((8, 5), f"{row['image']}  reference", fill="black")
        draw.text((target_width + 8, 5), "decoded candidate", fill="black")
        draw.text(
            (target_width * 2 + 8, 5),
            f"12x abs diff  MAE={row['mean_abs_error']:.6f}",
            fill="black",
        )
        panels.append(panel)
    sheet = Image.new(
        "RGB", (target_width * 3, sum(panel.height for panel in panels)), "white"
    )
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-metrics", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worst-frames", type=int, default=6)
    args = parser.parse_args(argv)
    frames, summary = compare(args.reference_metrics.resolve(), args.candidate_metrics.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet = args.output_dir / "worst-frames.png"
    make_contact_sheet(frames, contact_sheet, args.worst_frames)
    audit = {
        "schema_version": "1.0",
        "status": "review_required",
        "reference_metrics": str(args.reference_metrics.resolve()),
        "candidate_metrics": str(args.candidate_metrics.resolve()),
        "summary": summary,
        "frames": frames,
        "contact_sheet": {"path": str(contact_sheet), "sha256": _sha256(contact_sheet)},
        "decision": None,
    }
    output = args.output_dir / "visual-audit.json"
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
