#!/usr/bin/env python3
"""R17-P0: independent arithmetic audit of the R16 opacity-semantics replay.

Recomputes, from RAW artifacts only (R16 CSV / oracle summaries / R15 capture):
  - GPU count sums per mode (raw / sigmoid / sigmoid_threshold_skip)
  - RAW -> sigmoid reduction (sum difference AND elementwise difference)
  - oracle reference sums and mismatch pair counts
  - the opacity branch-coverage table (t-classes of the CUDA AccuTile formula)
  - the max-count primitive case (R16 "Gaussian 1196" claim) from the CSV itself

Also reconciles against numbers claimed in the R16 report, flagging any figure
that cannot be reproduced. NOT_EXECUTED / MISSING is reported explicitly.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (OUT, R15_CAPTURE, R16_DIR, ensure_out_dirs,
                    read_common_constants, sha256_file, write_json)

MODES = ("raw", "sigmoid", "sigmoid_threshold_skip")


def load_csv(path: Path):
    if not path.is_file():
        return None
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mode_counts(rows, mode):
    vals = np.array([int(r["count"]) for r in rows if r["mode"] == mode], dtype=np.int64)
    return vals


def branch_table(rows_mode, alpha):
    """Classify each sampled primitive by the CUDA AccuTile t-formula branch."""
    o = np.array([float(r["raw_opacity"]) for r in rows_mode], dtype=np.float64)
    a = np.array([float(r["activated_opacity"]) for r in rows_mode], dtype=np.float64)
    t_raw = np.full_like(o, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = o / alpha
        t_raw = np.where(ratio > 0, 2.0 * np.log(np.where(ratio > 0, ratio, 1.0)), np.nan)
    cap = np.log(np.e ** 0.5)
    raw_neg = int((o <= 0).sum())
    raw_na_skip = int(((o > 0) & (o <= alpha)).sum())
    raw_mid = int(((o > alpha) & (np.log(o / alpha) < cap)).sum())
    raw_cap = int((o > alpha).sum() - raw_mid)
    oa_skip = int((a <= alpha).sum())
    return {
        "alpha_threshold": alpha,
        "n": int(o.size),
        "raw_logit_le_0__logf_of_negative__t_forced_by_fminf_nan": raw_neg,
        "raw_logit_in_(0,alpha]__logf_negative__t_negative": raw_na_skip,
        "raw_logit_capped_t_equals_1": raw_cap,
        "raw_logit_in_(alpha, alpha*e^0.5)__t_in_(0,1)": raw_mid,
        "activated_opacity_le_alpha__true_skip_branch_coverage": oa_skip,
        "activated_opacity_gt_alpha": int(o.size - oa_skip),
        "activation_fixes_skip_coverage": bool(oa_skip > 0 and raw_neg + raw_na_skip > 0),
    }


def main():
    ensure_out_dirs()
    audit = {
        "deliverable": "R17_R16_ARITHMETIC_AUDIT.json",
        "purpose": "Recompute every R16-reported arithmetic quantity from raw data; flag non-reproducible figures.",
        "sources": {
            "r16_dir": str(R16_DIR),
            "r16_csv": str(R16_DIR / "R16_REPLAY_COUNTS.csv"),
            "r16_csv_exists": (R16_DIR / "R16_REPLAY_COUNTS.csv").is_file(),
            "r16_report_json": str(R16_DIR / "R16_OPACITY_SEMANTICS_REPORT.json"),
            "r16_report_json_exists": (R16_DIR / "R16_OPACITY_SEMANTICS_REPORT.json").is_file(),
            "r15_capture": str(R15_CAPTURE),
            "r15_capture_exists": R15_CAPTURE.is_file(),
            "r15_capture_sha256": sha256_file(R15_CAPTURE),
            "r16_csv_sha256": sha256_file(R16_DIR / "R16_REPLAY_COUNTS.csv"),
        },
        "constants": read_common_constants(),
        "status": "NOT_EXECUTED",
    }

    rows = load_csv(R16_DIR / "R16_REPLAY_COUNTS.csv")
    csv_present = rows is not None
    if not csv_present:
        audit["status"] = "NOT_EXECUTED_R16_CSV_MISSING"
        write_json(OUT / "R17_R16_ARITHMETIC_AUDIT.json", audit)
        print("R16 CSV missing — audit marked NOT_EXECUTED; run R16 replay first "
              "(r16_replay.py may be unavailable; see README).", flush=True)
        return

    alpha = audit["constants"].get("alpha_threshold")
    per_mode = {}
    for mode in MODES:
        v = mode_counts(rows, mode)
        per_mode[mode] = {
            "rows": int(v.size),
            "gpu_count_sum": int(v.sum()) if v.size else None,
            "per_primitive_max": int(v.max()) if v.size else None,
            "per_primitive_mean": float(v.mean()) if v.size else None,
            "per_primitive_median": float(np.median(v)) if v.size else None,
            "per_primitive_p99": float(np.quantile(v, 0.99)) if v.size else None,
        }

    recomputed = {
        "gpu_raw_sum": per_mode["raw"]["gpu_count_sum"],
        "gpu_sigmoid_sum": per_mode["sigmoid"]["gpu_count_sum"],
        "gpu_threshold_mode_sum": per_mode["sigmoid_threshold_skip"]["gpu_count_sum"],
    }
    if None not in (recomputed["gpu_raw_sum"], recomputed["gpu_sigmoid_sum"]):
        raw_v = mode_counts(rows, "raw")
        sig_v = mode_counts(rows, "sigmoid")
        same_selection = raw_v.size == sig_v.size
        recomputed["raw_minus_sigmoid_sum_difference"] = int(raw_v.sum() - sig_v.sum())
        if same_selection:
            d = raw_v - sig_v
            recomputed["raw_minus_sigmoid_elementwise_abs_sum"] = int(np.abs(d).sum())
            recomputed["changed_pairs"] = int((d != 0).sum())
            recomputed["max_single_pair_reduction"] = int(d.max())
            recomputed["activation_never_increases_counts"] = bool((d >= 0).all())
    if None not in (recomputed["gpu_sigmoid_sum"], recomputed["gpu_threshold_mode_sum"]):
        sig_v = mode_counts(rows, "sigmoid")
        thr_v = mode_counts(rows, "sigmoid_threshold_skip")
        if sig_v.size == thr_v.size:
            recomputed["sigmoid_minus_threshold_sum_difference"] = int(sig_v.sum() - thr_v.sum())
            recomputed["sigmoid_vs_threshold_identical"] = bool(np.array_equal(sig_v, thr_v))

    # oracle reference sums (R16 oracle dirs)
    oracle = {}
    for mode in MODES:
        p = R16_DIR / f"oracle_{mode}" / "oracle_summary.json"
        if p.is_file():
            s = json.loads(p.read_text())
            oracle[mode] = {
                "reference_sum_valid": s.get("reference_sum_valid"),
                "gpu_sample_sum_all": s.get("gpu_sample_sum_all"),
                "mismatch_pairs": s.get("mismatch_pairs"),
                "count_above_physical_tiles": s.get("count_above_physical_tiles"),
                "count_above_padded_aabb_upper_bound": s.get("count_above_padded_aabb_upper_bound"),
                "verdict": s.get("verdict"),
            }
        else:
            oracle[mode] = {"status": "NOT_EXECUTED_MISSING"}

    # threshold-branch coverage from the CSV rows themselves
    raw_rows = [r for r in rows if r["mode"] == "raw"]
    branch = branch_table(raw_rows, alpha) if alpha and raw_rows else {"status": "NO_ALPHA_OR_ROWS"}
    radii_skipped_true = int(sum(1 for r in rows if r.get("radii_skipped") == "True"))

    # max-count primitive case from the CSV
    max_case = None
    if raw_rows:
        r = max(raw_rows, key=lambda x: int(x["count"]))
        max_case = {
            "mode": "raw", "camera": r.get("camera"), "gaussian_id": r.get("gaussian_id"),
            "gpu_count": int(r["count"]), "raw_opacity": float(r["raw_opacity"]),
            "activated_opacity": float(r["activated_opacity"]),
            "tile_bound_per_image": 60 * 34,
        }

    # cross-check against R15 capture gpu_counts (raw mode must match exactly)
    r15_check = {"status": "NOT_EXECUTED_NO_R15_CAPTURE"}
    if R15_CAPTURE.is_file():
        import torch
        pkg = torch.load(str(R15_CAPTURE), map_location="cpu", weights_only=False)
        ref = pkg["gpu_counts"].to(torch.int64).numpy().reshape(-1)
        raw_v = mode_counts(rows, "raw")
        if ref.size == raw_v.size:
            r15_check = {
                "status": "EXECUTED",
                "n": int(ref.size),
                "r15_capture_sha256": audit["sources"]["r15_capture_sha256"],
                "exact_match": bool(np.array_equal(ref, raw_v)),
                "sum_abs_diff": int(np.abs(ref - raw_v).sum()),
                "changed_pairs": int((ref != raw_v).sum()),
            }
        else:
            r15_check = {"status": "SHAPE_MISMATCH", "r15_n": int(ref.size), "csv_n": int(raw_v.size)}

    # reconcile against the R16 report claims
    claims = {"status": "NO_REPORT_JSON"}
    rp = R16_DIR / "R16_OPACITY_SEMANTICS_REPORT.json"
    if rp.is_file():
        rep = json.loads(rp.read_text())
        modes_rep = rep.get("modes", {})
        claimed_raw = modes_rep.get("raw", {}).get("gpu_count_sum")
        claimed_sig = modes_rep.get("sigmoid", {}).get("gpu_count_sum")
        claims = {
            "status": "EXECUTED",
            "report_status_field": rep.get("status"),
            "claimed_raw_gpu_sum": claimed_raw,
            "claimed_sigmoid_gpu_sum": claimed_sig,
            "raw_matches_csv": bool(claimed_raw == recomputed["gpu_raw_sum"]) if claimed_raw is not None else None,
            "sigmoid_matches_csv": bool(claimed_sig == recomputed["gpu_sigmoid_sum"]) if claimed_sig is not None else None,
            "note": ("Any reduction figure quoted in R16 prose must equal "
                     "raw_minus_sigmoid_sum_difference above; the CSV is authoritative."),
        }

    audit.update({
        "status": "EXECUTED",
        "per_mode": per_mode,
        "recomputed": recomputed,
        "oracle": oracle,
        "branch_coverage": branch,
        "radii_skipped_rows_in_threshold_mode": radii_skipped_true,
        "max_count_case": max_case,
        "r15_capture_raw_exact_match": r15_check,
        "report_claims_reconciliation": claims,
        "arithmetic_note": (
            "R17 recomputation definitions: reduction = raw_sum - sigmoid_sum over the SAME "
            "seeded 1000-sample set; threshold-mode difference = sigmoid_sum - threshold_sum. "
            "A claimed reduction that equals neither is flagged in R17_FINAL_REPORT.md."),
    })
    write_json(OUT / "R17_R16_ARITHMETIC_AUDIT.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k not in ("per_mode",)}, indent=2,
                     default=str), flush=True)


if __name__ == "__main__":
    main()
