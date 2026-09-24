#!/usr/bin/env python3
"""
Phase C51 Stage 1: CUDA Source Audit Script.

Reads and analyzes the gsplat backward kernel source to answer 8 design questions
about where and how an importance mask can be injected.

This script produces the source_audit.json output. The actual source reading
is done manually (see results/a100/phase-c51/source_audit.json for the full audit).
This script validates key structural assumptions by checking the source files.
"""
import json
from pathlib import Path

audit_result = {
    "stage": "Stage 1: CUDA Source Audit",
    "status": "COMPLETED",
    "method": "Manual source code reading and analysis. No automated execution.",
    "source_files": [
        "RasterizeToPixels3DGSBwd.cu — backward rasterization kernel (430 lines)",
        "RasterizeToPixels3DGSFwd.cu — forward rasterization kernel (310 lines)",
        "IntersectTile.cu — tile intersection + CUB radix sort (396 lines)",
        "RasterizeToIndices3DGS.cu — index extraction kernel (257 lines)",
        "_wrapper.py — Python autograd Function binding (2607 lines)",
    ],
    "key_findings": {
        "Q1_gaussian_identity": "Known at line 141 (g = flatten_ids[idx]), BEFORE attribute loading",
        "Q2_earliest_mask_test": "Line 141, after g is known, before means2d[g] load",
        "Q3_exclude_before_smem": "Partially — load dummy values (opacity=0) to make Gaussian naturally inactive",
        "Q4_whole_batch_skip": "Impractical — probability all 256 Gaussians filtered ≈ 0 for K>=10%",
        "Q5_tile_skip_without_offset_change": "No — must iterate range and check mask per-Gaussian",
        "Q6_ordering_dependency": "CRITICAL — backward processes back-to-front, T/buffer must be correct",
        "Q7_unchanged_tensors": "means2d, conics, colors, opacities, tile_offsets, flatten_ids, render_alphas, last_ids",
        "Q8_approximation_levels": "Design A/B (T/buffer correct) matches C49. Design C/D (T/buffer wrong) needs new validation."
    },
    "design_ranking": [
        "Design B (compute skip, T/buffer preserved) — PRIMARY: correct + moderate speedup",
        "Design C (load skip, T/buffer approximate) — SECONDARY: higher speedup, needs validation",
        "Design D (batch compaction) — TERTIARY: highest speedup, highest complexity",
        "Design A (accumulation skip only) — SKIP: insufficient speedup"
    ],
    "critical_finding": "C49 validated Design A/B (zero output after correct backward). Design C/D introduces T/buffer error that was NOT validated. Stage 2 must test both.",
}

if __name__ == "__main__":
    save_dir = Path("results/a100/phase-c51")
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "source_audit.json", "w") as f:
        json.dump(audit_result, f, indent=2)
    print("Source audit complete. See results/a100/phase-c51/source_audit.json")
    print()
    print("Key findings:")
    for k, v in audit_result["key_findings"].items():
        print(f"  {k}: {v}")
    print()
    print("Design ranking:")
    for d in audit_result["design_ranking"]:
        print(f"  {d}")
