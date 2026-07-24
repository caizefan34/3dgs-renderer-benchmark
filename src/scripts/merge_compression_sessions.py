#!/usr/bin/env python
"""Merge split EPIC-05 compression sessions into one comparison report."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def merge(session_paths: list[Path], root: Path) -> dict:
    sessions = [_load(path) for path in session_paths]
    commits = {session["benchmark_commit"] for session in sessions}
    if len(commits) != 1:
        raise ValueError("compression sessions use different benchmark commits")
    results = []
    identities = set()
    for session in sessions:
        if session.get("status") != "complete":
            raise ValueError("compression session is incomplete")
        for item in session["completed"]:
            identity = (item["case_id"], item["codec"])
            if identity in identities:
                raise ValueError(f"duplicate compression row: {identity}")
            identities.add(identity)
            results.append(_load(root / item["metrics_path"]))
    results.sort(key=lambda row: (row["case"]["case_id"], row["codec"]["id"]))
    return {
        "schema_version": "1.0",
        "evidence_tier": "measured",
        "benchmark_commit": commits.pop(),
        "results": results,
    }


def summarize(document: dict) -> list[dict]:
    grouped = defaultdict(list)
    for result in document["results"]:
        grouped[result["codec"]["id"]].append(result)
    rows = []
    for codec, results in sorted(grouped.items()):
        source_bytes = sum(row["codec"]["artifact"]["source_bytes"] for row in results)
        compressed_bytes = sum(row["codec"]["artifact"]["compressed_bytes"] for row in results)
        deltas = [row["metrics"]["quality_delta"] for row in results]
        gates = [row["metrics"]["near_lossless_gate"] for row in results]
        rows.append({
            "codec": codec,
            "cases": len(results),
            "source_bytes": source_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": source_bytes / compressed_bytes,
            "worst_psnr_delta_db": min(delta["psnr_db"] for delta in deltas),
            "worst_ssim_delta": min(delta["ssim"] for delta in deltas),
            "worst_lpips_delta": max(delta["lpips"] for delta in deltas),
            "numeric_passes": sum(gate["numeric_pass"] for gate in gates),
            "overall_passes": sum(gate["overall_pass"] for gate in gates),
        })
    return rows


def write_report(document: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "compression-results.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# EPIC-05 expanded compression comparison", "",
        f"Benchmark commit: `{document['benchmark_commit']}`", "",
        "| Codec | Cases | Aggregate ratio | Worst PSNR delta | Worst SSIM delta | Worst LPIPS delta | Numeric passes | Overall passes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summarize(document):
        lines.append(
            f"| {row['codec']} | {row['cases']} | {row['compression_ratio']:.3f}x | "
            f"{row['worst_psnr_delta_db']:+.6f} dB | {row['worst_ssim_delta']:+.6f} | "
            f"{row['worst_lpips_delta']:+.6f} | {row['numeric_passes']}/{row['cases']} | "
            f"{row['overall_passes']}/{row['cases']} |"
        )
    (output_dir / "compression-results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    document = merge([path.resolve() for path in args.sessions], args.root.resolve())
    write_report(document, args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
