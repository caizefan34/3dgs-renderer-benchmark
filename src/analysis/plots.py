"""Publication-oriented plot generation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plot generation") from exc
    return plt


def _eligible(records: Iterable[Mapping], *keys):
    return [record for record in records if all(record.get(key) is not None for key in keys)]


def _scatter(records, x_key, y_key, path_base, title, xlabel, ylabel):
    plt = _require_matplotlib()
    rows = _eligible(records, x_key, y_key)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=160)
    if rows:
        ax.scatter([row[x_key] for row in rows], [row[y_key] for row in rows])
        for row in rows:
            ax.annotate(row["renderer"], (row[x_key], row[y_key]), fontsize=7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{path_base}.png")
    fig.savefig(f"{path_base}.pdf")
    plt.close(fig)


def generate_plots(records: Iterable[Mapping], output_dir: str) -> list[str]:
    records = list(records)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    outputs = []
    specs = [
        ("speed_quality", "fps", "psnr", "FPS vs Quality", "FPS", "PSNR vs GT"),
        ("memory_tradeoff", "peak_vram_mb", "fps", "FPS vs VRAM", "Peak VRAM (MB)", "FPS"),
        ("pareto", "fps", "psnr", "Pareto Frontier", "FPS", "PSNR vs GT"),
        ("fps_gaussian_count", "gaussians", "fps", "FPS vs Gaussian Count", "Gaussians", "FPS"),
        ("latency_distribution", "p99_latency_ms", "latency_ms", "Latency Distribution", "P99 latency (ms)", "Mean latency (ms)"),
    ]
    for stem, x_key, y_key, title, xlabel, ylabel in specs:
        path_base = os.path.join(output_dir, stem)
        _scatter(records, x_key, y_key, path_base, title, xlabel, ylabel)
        outputs.extend([f"{path_base}.png", f"{path_base}.pdf"])
    return outputs

