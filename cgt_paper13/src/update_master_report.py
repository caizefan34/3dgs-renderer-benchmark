#!/usr/bin/env python3
"""Append paper tables & figures references to PAPER13_MASTER_REPORT.md (EVIDENCE section)."""

R = "/mnt/storage_pool/liaoyuanjun/cgt_paper13/results"
fp = R + "/PAPER13_MASTER_REPORT.md"
txt = open(fp).read()

lines = [
    "",
    "- Paper tables (section 33), source-traceable CSV: figures/TABLE1_13SCENE_MAIN.csv",
    "  (per-scene baseline/S_mixed/safe/fallback/kernel-correctness/build/render-ratio/E2E),",
    "  figures/TABLE2_DATASET_AGGREGATE.csv (mipnerf360 / tanksandtemples / deepblending geomean aggregates),",
    "  figures/TABLE3_ABLATIONS.csv (rank R4/R8/R16 + cert/fallback),",
    "  figures/TABLE4_MEMORY.csv (peak VRAM = NOT_MEASURED), figures/TABLE5_RELATED_WORK.csv",
    "  (canonical gsplat vs CGT-F0 vs closed CGT-G5).",
    "- Figure source data (section 32): figures/fig1_method_overview.csv (pipeline stages),",
    "  figures/fig2_speedup_distribution.csv (per-scene S_mixed),",
    "  figures/fig3_safe_fallback_vs_speedup.csv (safe/fallback fraction vs speedup),",
    "  figures/fig4_rank_ablation.csv (R4/R8/R16 speed and kernel correctness).",
    "- Environment/GPU provenance (section 38): captured in every per-scene P1/P2/P3 JSON",
    "  (device id, torch 2.7.1+cu118, gsplat 1.5.3, CUDA 11.8.89) + host line above.",
]
addition = "\n".join(lines)

marker = "## INFERENCE (interpretation)"
if "figures/TABLE1_13SCENE_MAIN.csv" in txt:
    print("ALREADY_UPDATED")
elif marker in txt:
    txt = txt.replace(marker, addition + "\n\n" + marker, 1)
    open(fp, "w").write(txt)
    print("UPDATED_MASTER_REPORT")
else:
    print("MARKER_NOT_FOUND")
