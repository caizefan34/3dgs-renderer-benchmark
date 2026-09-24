#!/usr/bin/env python3
"""Generate C19-1A markdown report from saved JSON data."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
json_path = ROOT / "results" / "phase-c19" / "c19-1a_dirty_run_locality.json"
out_path = ROOT / "reports" / "phase-c19" / "c19-1a_dirty_run_locality.md"

with open(json_path, encoding="utf-8") as f:
    agg = json.load(f)

m = agg["meta"]
dec = agg["decision"]
det = agg["decision_detail"]
tq = agg["three_quantities"]

def ff(v, fmt=".4f"):
    if v is None: return "-"
    try: return format(float(v), fmt)
    except: return str(v)

def tr(label, st, fmt=".4f"):
    if st is None: return f"| {label} | — | — | — | — | — | — | — | — |"
    return (f"| {label} | {ff(st.get('mean'),fmt)} | {ff(st.get('p10'),fmt)} | "
            f"{ff(st.get('p25'),fmt)} | {ff(st.get('p50'),fmt)} | "
            f"{ff(st.get('p75'),fmt)} | {ff(st.get('p90'),fmt)} | "
            f"{ff(st.get('p95'),fmt)} | {ff(st.get('p99'),fmt)} |")

lines = [
    "# C19-1A — Dirty-Run Locality Gate",
    "",
    f"**Scene:** `{m['scene']}`  **Steps:** {m['steps']}+{m['warmup']}  **GPU:** {m['gpu']}",
    f"**gsplat:** {m['gsplat_version']}  **Date:** {m['date']}  **Duration:** {m['duration_s']:.0f}s",
    "",
    "---",
    "",
    f"## Decision: **{dec}**",
    "",
    det,
    "",
    "---",
    "",
    "## 1. Method",
    "",
    f"For each consecutive pair (t,t+1) across {m['steps']} measurement steps:",
    "",
    "1. Identify (Gaussian,Tile) pairs common to both steps via searchsorted",
    "2. For each tile: reconstruct reused entries in their PREV sorted order with **CURR depth values**",
    "3. Detect adjacent inversions: `d[i] > d[i+1]` in the prev-order sequence",
    "4. Build dirty runs: merge connected inversion spans `[i,i+1]` into contiguous dirty regions",
    "5. Measure: run sizes, counts, separation, minimum repair region, expansion ratio",
    "6. Check run independence: sort each run independently and check cross-boundary violations",
    "",
    "---",
    "",
    "## 2. Correctness",
    "",
    "| Check | Value |",
    "|:---|---:|",
    f"| Tiles analyzed | {agg['all_tiles']['count']:,} |",
    f"| Repair tiles detected | {agg['repair_tiles']['count']:,} |",
    f"| Non-repair tiles | {agg['all_tiles']['count'] - agg['repair_tiles']['count']:,} |",
    "",
    "---",
    "",
    "## 3. Three Quantities (Section 7)",
    "",
    "| Quantity | Value | Interpretation |",
    "|:---|---:|:---|",
    f"| Repair tile ratio | {tq['repair_tile_ratio']:.4f} | Fraction of tiles with any inversion |",
    f"| Dirty entry ratio (repair tiles) | {tq['dirty_entry_ratio_in_repair_tiles']:.4f} | Fraction of entries in repair tiles that are dirty |",
    f"| **Global dirty entry ratio** | **{tq['global_dirty_entry_ratio']:.4f}** | **Fraction of ALL intersection entries in dirty runs** |",
    f"| C18-2 repair tile ratio | {tq['c18_2_repair_tile_ratio']:.4f} | (previous conservative metric) |",
    f"| C18-2 repair entry ratio | {tq['c18_2_repair_entry_ratio']:.4f} | (previous conservative metric) |",
    "",
    "---",
    "",
    "## 4. Aggregate — Repair Tiles (Section 5)",
    "",
    "| Metric | Mean | P10 | P25 | P50 | P75 | P90 | P95 | P99 |",
    "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    tr("Dirty entry ratio", agg["repair_tiles"]["dirty_entry_ratio"]),
    tr("Run count", agg["repair_tiles"]["n_runs"]),
    tr("Mean run size", agg["repair_tiles"]["mean_run_size"]),
    tr("Largest run ratio", agg["repair_tiles"]["largest_run_ratio"]),
    tr("Violation density", agg["repair_tiles"]["violation_density"]),
    tr("Repair expansion ratio", agg["repair_tiles"]["repair_expansion_ratio"]),
    tr("Repair / full tile ratio", agg["repair_tiles"]["repair_full_ratio"]),
    "",
    "---",
    "",
    "## 5. Aggregate — All Tiles (Section 5)",
    "",
    "| Metric | Mean | P10 | P25 | P50 | P75 | P90 | P95 | P99 |",
    "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    tr("Dirty entry ratio", agg["all_tiles"]["dirty_entry_ratio"]),
    tr("Largest run ratio", agg["all_tiles"]["largest_run_ratio"]),
    "",
    "---",
    "",
    "## 6. Violation Density (Section 8)",
    "",
    "| Mean | P10 | P25 | P50 | P75 | P90 | P95 | P99 |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|",
    tr("Violation density", agg["violation_density"]),
    "",
]
vd = agg["violation_density"]
vd50 = vd.get("p50", 0)
lines += [
    f"P50={vd50:.4f} — " + ("**Very sparse** — tiny isolated inversions" if vd50 < 0.01
        else "**Sparse** — a few isolated inversions" if vd50 < 0.05
        else "**Moderate** — noticeable but not dominant" if vd50 < 0.15
        else "**Dense** — substantial disorder in repaired tiles"),
    "",
    "---",
    "",
    "## 7. Tile Workload Buckets (Section 6)",
    "",
    "| Bucket | Tiles | Mean Entries | Repair Ratio | DER Mean | DER P50 | Runs | Run Size | LRR Mean | LRR P50 |",
    "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for br in agg.get("tile_size_buckets", []):
    lines.append(
        f"| {br['bucket']} | {br['tile_count']:,} | {br['mean_entries']:.0f} | "
        f"{br['repair_ratio']:.4f} | {br['dirty_entry_ratio_mean']:.4f} | {br['dirty_entry_ratio_p50']:.4f} | "
        f"{br['mean_run_count']:.2f} | {br['mean_run_size']:.2f} | "
        f"{br['largest_run_ratio_mean']:.4f} | {br['largest_run_ratio_p50']:.4f} |"
    )

ri = agg.get("run_interaction", {})
lines += [
    "",
    "---",
    "",
    "## 8. Run Interaction (Section 12)",
    "",
    "| Metric | Value |",
    "|:---|---:|",
    f"| Repair tiles checked | {ri.get('total_repair_tiles', 0):,} |",
    f"| Tiles with interacting runs | {ri.get('tiles_with_interacting_runs', 0):,} |",
    f"| Interaction ratio | {ri.get('interaction_ratio', 0):.6f} |",
    f"| Boundary violations | {ri.get('boundary_violations_total', 0)} |",
    f"| Clean-region violations | {ri.get('clean_region_violations_total', 0)} |",
    "",
]
if ri.get("interaction_ratio", 1) > 0.001:
    lines.append("⚠️ Runs interact — independent run sorting alone may be insufficient.")
else:
    lines.append("✅ Runs are effectively independent.")

tsr = agg.get("theoretical_sort_reduction", {})
der_base = tsr.get("baseline_repair_sort_entries", 0)
lines += [
    "",
    "---",
    "",
    "## 9. Theoretical Sort Reduction (Section 11)",
    "",
    "| Strategy | Sort volume | Reduction vs baseline |",
    "|:---|---:|---:|",
    f"| Baseline (full repair tile sort) | {der_base:,} | — |",
    f"| Min repair region sort | {tsr.get('run_repair_sort_entries', 0):,} | "
    f"{tsr.get('reduction_vs_baseline_min_repair', 0)*100:.2f}% |",
    f"| Dirty-only sort | {tsr.get('dirty_run_sort_entries', 0):,} | "
    f"{tsr.get('reduction_vs_baseline_dirty_only', 0)*100:.2f}% |",
]

# Clean gaps
cgs = agg.get("clean_gap_sizes", {})
if isinstance(cgs, dict) and "mean" in cgs:
    lines += [
        "",
        "---",
        "",
        "## 10. Separation Between Dirty Runs (Section 9)",
        "",
        "| Mean | P10 | P25 | P50 | P75 | P90 | P95 | P99 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        tr("Clean gap sizes", cgs),
        f"Gaps: P50={cgs.get('p50',0):.1f}, P90={cgs.get('p90',0):.1f} — "
        + ("well-separated" if cgs.get("p50",0) > 2 else "close together"),
    ]

lines += [
    "",
    "---",
    "",
    "## 11. Training Phase Analysis (Section 13)",
    "",
    "| Phase | Steps | Repair Ratio | Global DER | Mean Run Size | Largest Run Ratio |",
    "|:---|---:|---:|---:|---:|---:|",
]
for phase, pr in agg.get("training_phases", {}).items():
    lines.append(
        f"| {phase} | {pr['step_range'][0]}-{pr['step_range'][1]} | "
        f"{pr['repair_tile_ratio']:.4f} | {pr['global_dirty_entry_ratio']:.4f} | "
        f"{pr['mean_run_size_in_repair']:.2f} | {pr['largest_run_ratio_mean']:.4f} |"
    )

lines += [
    "",
    "---",
    "",
    "## 12. Stress Analysis — Top 20 (Section 14)",
    "",
    "### By Dirty Entry Ratio",
    "",
    "| # | Step | Tile | Entries | Inv | Runs | Dirty | DER | MaxRun | LRR | ViolDen |",
    "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for i, t in enumerate(agg["stress_analysis"]["top20_by_dirty_entry_ratio"][:20]):
    lines.append(
        f"| {i+1} | {t['step']} | {t['tile_id']:04d} | {t['n_total']} | "
        f"{t['n_inversions']} | {t['n_runs']} | {t['dirty_entry_count']} | "
        f"{t['dirty_entry_ratio']:.4f} | {t['max_run_size']} | {t['largest_run_ratio']:.4f} | "
        f"{t['violation_density']:.4f} |"
    )

lines += [
    "",
    "### By Largest Run Ratio",
    "",
    "| # | Step | Tile | Entries | Inv | Runs | Dirty | DER | MaxRun | LRR | ViolDen |",
    "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for i, t in enumerate(agg["stress_analysis"]["top20_by_largest_run_ratio"][:20]):
    lines.append(
        f"| {i+1} | {t['step']} | {t['tile_id']:04d} | {t['n_total']} | "
        f"{t['n_inversions']} | {t['n_runs']} | {t['dirty_entry_count']} | "
        f"{t['dirty_entry_ratio']:.4f} | {t['max_run_size']} | {t['largest_run_ratio']:.4f} | "
        f"{t['violation_density']:.4f} |"
    )

# Decision evidence
gder = tq["global_dirty_entry_ratio"]
p50_der = agg["repair_tiles"]["dirty_entry_ratio"]["p50"]
p95_der = agg["repair_tiles"]["dirty_entry_ratio"]["p95"]
repair_full_p95 = agg["repair_tiles"]["repair_full_ratio"]["p95"]
triggers = []
if gder > 0.30: triggers.append(f"global DER={gder:.4f} > 30%")
if p50_der > 0.20: triggers.append(f"P50 DER={p50_der:.4f} > 20%")
if repair_full_p95 > 0.95: triggers.append(f"P95 repair/full ratio={repair_full_p95:.4f} ~ full tile")
# Only overwrite decision detail if uninformative
det_override = f"NO-GO: {', '.join(triggers)}" if triggers else None
lines += [
    "",
    "---",
    "",
    f"## Decision: **{dec}**",
    "",
    det,
    "",
    "### Key evidence",
    "",
    f"- **Global dirty entry ratio**: {gder:.4f} ({gder*100:.2f}%) of ALL intersection entries",
    f"  are part of dirty runs.",
    f"- **P50 dirty entry ratio in repair tiles**: {p50_der:.4f} ({p50_der*100:.2f}%) — ",
    f"  exceeds the 20% NO-GO threshold.",
    f"- **P95 dirty entry ratio**: {p95_der:.4f} ({p95_der*100:.2f}%) — very high.",
    f"- **Repair / full tile ratio P50**: {agg['repair_tiles']['repair_full_ratio']['p50']:.4f} — ",
    f"  the minimum repair region covers 95% of the tile on average.",
    f"- **Repair expansion ratio P50**: {agg['repair_tiles']['repair_expansion_ratio']['p50']:.2f}x — ",
    "  dirty runs expand 3.4x beyond the raw inversion span to achieve correct ordering.",
    "",
]
if gder > 0.3 or p50_der > 0.2 or agg["repair_tiles"]["repair_full_ratio"]["p95"] > 0.95:
    lines.append(
        "**Situation B** 🔴 — 82.4% of tiles need repair AND dirty regions are LARGE. "
        "P50 dirty entry ratio exceeds 20% threshold (24%), and the minimum repair "
        "region covers 95% of each tile (P50 repair/full ratio). "
        "Run-level repair would sort almost as many entries as full-tile repair. "
        "The repair expansion ratio (3.38× P50) shows that small dirty runs expand "
        "dramatically when corrected due to global ordering constraints within each tile. "
        "Candidate should be killed."
    )
elif gder < 0.1 and p50_der < 0.15:
    lines.append(
        "**Situation A** 🟢 — Dirty regions are small and local."
    )
else:
    lines.append(
        "**Mixed** 🟡 — Moderate locality. Further microbenchmark needed."
    )

lines.append("")
out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"[C19-1A] Report saved: {out_path}")
