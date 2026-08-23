"""Build immutable per-seed tables + markdown renderings for the confirmatory matrices.

Reads results/confirmatory-*/summary.json (collector output) and
paper/tables/confirmatory-*-<metric>-bootstrap.json, and emits:
  paper/tables/<prefix>-per-seed.json   (machine-readable; every markdown cell traces here)
  paper/tables/<prefix>-table.md        (human-readable paper tables)

Run: python src/scripts/build_confirmatory_tables.py --family confirmatory-matrix
     python src/scripts/build_confirmatory_tables.py --family confirmatory-db
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "paper" / "tables"

METRIC_ORDER = [
    ("train_ms", "train_ms (ms/step)"),
    ("total_wall_s", "total wall (s)"),
    ("psnr", "PSNR (dB)"),
    ("ssim", "SSIM"),
    ("lpips", "LPIPS"),
]

TITLES = {
    "confirmatory-matrix": "canonical five scenes (A100, 1080p)",
    "confirmatory-db": "Deep Blending held-out family (A100, 1080p)",
    "confirmatory-consumer-720p": "720p resolution leg (EPIC-05 A100, 960x540)",
}

# Quality guardrail from paper/confirmatory-protocol.md section 5.
PSNR_GUARDRAIL_DB = -0.05
LPIPS_GUARDRAIL = 0.005


def load_summary(family: str) -> dict:
    path = ROOT / "results" / family / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_bootstrap(family: str, metric: str) -> dict:
    path = TABLES / f"{family}-{metric}-bootstrap.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def per_seed_table(summary: dict) -> dict:
    rows = []
    for run_id, run in sorted(summary["runs"].items()):
        rows.append({
            "run_id": run_id,
            "scene": run["scene"],
            "arm": run["arm"],
            "seed": int(run["seed"]),
            "train_ms": run.get("train_ms"),
            "total_wall_s": run.get("total_wall_s"),
            "psnr": run.get("psnr"),
            "ssim": run.get("ssim"),
            "lpips": run.get("lpips"),
            "final_n": run.get("final_n"),
            "peak_vram_gb": run.get("peak_vram_gb"),
            "eval_points": run.get("eval_points"),
        })
    return {"runs": rows}


def time_to_target_summary(summary: dict) -> dict:
    """Per-scene median wall_s/step at which pd first reaches the paired ctrl final PSNR."""
    by_scene: dict[str, list] = {}
    for _run_id, ttt in summary.get("time_to_target", {}).items():
        reached = ttt.get("reached")
        if not reached:
            continue
        by_scene.setdefault(ttt["scene"], []).append((reached["step"], reached["wall_s"]))
    out = {}
    for scene, hits in sorted(by_scene.items()):
        steps = sorted(h[0] for h in hits)
        walls = sorted(h[1] for h in hits)
        out[scene] = {
            "n_pd_runs": len(hits),
            "min_step": steps[0],
            "max_step": steps[-1],
            "median_wall_s": walls[len(walls) // 2],
            "all_reached_step_300": all(s == 300 for s in steps),
        }
    return out


def _fmt(value, digits=3) -> str:
    return f"{value:.{digits}f}" if value is not None else "-"


def _dominance(paired: dict[str, list[float]]) -> bool:
    """Strict per-scene dominance: negative train_ms delta for all seeds AND quality guardrail."""
    train = paired.get("train_ms", [])
    if len(train) < 1 or not all(d < 0 for d in train):
        return False
    psnr = paired.get("psnr", [])
    lpips = paired.get("lpips", [])
    if psnr and mean(psnr) < PSNR_GUARDRAIL_DB:
        return False
    if lpips and mean(lpips) > LPIPS_GUARDRAIL:
        return False
    return True


def build_md(family: str, per_seed: dict, ttt: dict) -> str:
    lines = [
        f"# Confirmatory tables: {TITLES[family]}",
        "",
        "- Protocol: `paper/confirmatory-protocol.md` (2026-08-06).",
        f"- Raw per-run values: `paper/tables/{family}-per-seed.json`; bootstrap"
        f" artifacts: `paper/tables/{family}-<metric>-bootstrap.json`.",
        "- Arm `ctrl` = full-resolution frozen baseline; arm `pd` ="
        " progressive-resolution + masked-Adam union-decay cell.",
        "- Every cell below traces to the JSON artifacts above.",
        "",
    ]
    scenes = sorted({r["scene"] for r in per_seed["runs"]})

    # Per-scene mean table (mean over seeds).
    metric_keys = [m for m, _ in METRIC_ORDER]
    means: dict[tuple[str, str], dict] = {}
    for r in per_seed["runs"]:
        bucket = means.setdefault((r["scene"], r["arm"]), {m: [] for m in metric_keys})
        for metric in metric_keys + ["final_n", "peak_vram_gb"]:
            if r.get(metric) is not None:
                bucket.setdefault(metric, []).append(r[metric])

    lines.append("## Per-scene means (3 seeds)")
    lines.append("")
    header = ["scene", "arm"] + [label for _, label in METRIC_ORDER] + ["final_n", "peak_vram_gb"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for scene in scenes:
        for arm in ("ctrl", "pd"):
            bucket = means.get((scene, arm), {})
            row = [scene, arm]
            for metric, _ in METRIC_ORDER:
                vals = bucket.get(metric, [])
                row.append(_fmt(mean(vals)) if vals else "-")
            fn = bucket.get("final_n", [])
            row.append(f"{mean(fn):.0f}" if fn else "-")
            vr = bucket.get("peak_vram_gb", [])
            row.append(_fmt(mean(vr), 2) if vr else "-")
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Paired deltas: per-scene means (no CI) + aggregate row with bootstrap CI.
    def paired_deltas(scene: str, metric: str) -> list[float]:
        out = []
        for r in per_seed["runs"]:
            if r["arm"] != "ctrl" or r["scene"] != scene or r.get(metric) is None:
                continue
            pd_run = next(
                (x for x in per_seed["runs"]
                 if x["scene"] == scene and x["arm"] == "pd" and x["seed"] == r["seed"]),
                None,
            )
            if pd_run and pd_run.get(metric) is not None:
                out.append(pd_run[metric] - r[metric])
        return out

    lines.append("## Paired deltas (pd - ctrl)")
    lines.append("")
    lines.append("Lower is better for `train_ms`, `total_wall_s`, `lpips`; higher is better for `psnr`, `ssim`.")
    lines.append("Per-scene rows are the mean of the three paired deltas for that scene; the `all` row is the")
    lines.append("scene-level block-bootstrap mean with a 95% percentile interval over the whole matrix.")
    lines.append("Strict dominance (per protocol section 5) requires a negative paired train_ms delta on every")
    lines.append("seed for that scene and the quality guardrail (mean PSNR delta >= -0.05 dB, mean LPIPS delta <= 0.005).")
    lines.append("")
    header2 = ["scene"] + [f"delta {label}" for _, label in METRIC_ORDER] + ["strict dominance (train_ms)"]
    lines.append("| " + " | ".join(header2) + " |")
    lines.append("|" + "---|" * len(header2))

    dominant_scenes = 0
    for scene in scenes:
        paired = {metric: paired_deltas(scene, metric) for metric in metric_keys}
        row = [scene]
        for metric, _ in METRIC_ORDER:
            deltas = paired[metric]
            row.append(_fmt(mean(deltas)) if deltas else "-")
        dominant = _dominance(paired)
        dominant_scenes += int(dominant)
        row.append("yes" if dominant else "no")
        lines.append("| " + " | ".join(row) + " |")

    agg = ["all"]
    for metric, _ in METRIC_ORDER:
        doc = load_bootstrap(family, metric)
        if not doc:
            agg.append("-")
            continue
        ci = doc.get("percentile_ci", [])
        mean_delta = doc.get("observed_mean_delta")
        agg.append(f"{mean_delta:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    agg.append(f"{dominant_scenes} of {len(scenes)}")
    lines.append("| " + " | ".join(agg) + " |")
    lines.append("")

    # Time-to-target table.
    lines.append("## Time to target quality (pd vs paired ctrl final PSNR)")
    lines.append("")
    lines.append("Pre-registered rule: pd wall_s of the first eval point whose PSNR reaches the paired ctrl arm's final (30k-step) PSNR.")
    lines.append("")
    lines.append("| scene | pd runs | min step | max step | median wall_s |")
    lines.append("|---|--:|--:|--:|--:|")
    for scene, info in sorted(ttt.items()):
        lines.append(f"| {scene} | {info['n_pd_runs']} | {info['min_step']} | {info['max_step']} | {info['median_wall_s']:.2f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=list(TITLES))
    args = parser.parse_args()

    summary = load_summary(args.family)
    per_seed = per_seed_table(summary)
    ttt = time_to_target_summary(summary)
    doc = {
        "tool": "build_confirmatory_tables",
        "family": args.family,
        "n_runs": len(per_seed["runs"]),
        "scenes": sorted({r["scene"] for r in per_seed["runs"]}),
        "time_to_target": ttt,
        "runs": per_seed["runs"],
    }
    out_json = TABLES / f"{args.family}-per-seed.json"
    out_json.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md = TABLES / f"{args.family}-table.md"
    out_md.write_text(build_md(args.family, per_seed, ttt), encoding="utf-8")
    print(f"wrote {out_json} ({len(doc['runs'])} runs) and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())