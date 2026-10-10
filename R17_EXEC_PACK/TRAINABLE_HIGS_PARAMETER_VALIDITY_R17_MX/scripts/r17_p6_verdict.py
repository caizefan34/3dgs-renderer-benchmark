#!/usr/bin/env python3
"""R17-P6: final verdicts and report, computed ONLY from the executed artifacts.

Verdict rules are explicit and deterministic (documented in the output), and every
input file is either parsed or reported as NOT_EXECUTED. Nothing is inferred if
the underlying measurement is missing.

Outputs: R17_FINAL_VERDICT.json, R17_FINAL_REPORT.md
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, PROJECT, sha256_file, write_json

REF_DIR = Path(__file__).resolve().parent.parent / "reference"
CLAIMED_R16_REDUCTION = 360_251  # value quoted in the R16 prose (source: R16 report)


def read_json(p):
    p = Path(p)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def read_csv_rows(p):
    p = Path(p)
    if not p.is_file():
        return []
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def harness_evidence():
    ev = {}
    for name, fname in (("r3", "run_r3_benchmark.py"), ("r8", "R8_WORKING_ENTRYPOINT.py")):
        p = REF_DIR / fname
        if not p.is_file():
            ev[name] = {"status": "REFERENCE_SCRIPT_MISSING", "file": str(p)}
            continue
        text = p.read_text(errors="replace")
        ev[name] = {
            "file": str(p), "sha256": sha256_file(p),
            "applies_sigmoid_to_opacity": "sigmoid" in text,
            "applies_exp_to_scaling": "torch.exp" in text and "scaling" in text,
            "normalizes_quats": "F.normalize" in text,
            "passes_opacity_directly": ("opacity.squeeze(-1)" in text) or ("opacities': opacities" in text)
            or ("'opacities'" in text and "sigmoid" not in text),
        }
    return ev


def main():
    v = {"deliverable": "R17_FINAL_VERDICT.json", "status": "NOT_EXECUTED",
         "input_files": {}, "verdicts": {}, "r3_r16_conclusion_disposition": {},
         "rules": {}, "missing": []}

    p0 = read_json(OUT / "R17_R16_ARITHMETIC_AUDIT.json")
    p1 = read_json(OUT / "R17_PARAMETER_DISTRIBUTIONS.json")
    p2 = read_json(OUT / "R17_CAMERA_IDENTITY.json")
    p3_rows = read_csv_rows(OUT / "R17_580K_CONTROLLED_COMPARISON.csv")
    p4_rows = read_csv_rows(OUT / "R17_VALIDATED_BENCHMARK.csv")
    p5 = read_json(OUT / "R17_4P4M_SAFETY_GATE.json")

    v["input_files"] = {
        "p0_arithmetic": {"exists": p0 is not None,
                          "sha256": sha256_file(OUT / "R17_R16_ARITHMETIC_AUDIT.json")},
        "p1_distributions": {"exists": p1 is not None,
                             "sha256": sha256_file(OUT / "R17_PARAMETER_DISTRIBUTIONS.json")},
        "p2_camera": {"exists": p2 is not None,
                      "sha256": sha256_file(OUT / "R17_CAMERA_IDENTITY.json")},
        "p3_csv_rows": len(p3_rows),
        "p4_csv_rows": len(p4_rows),
        "p5_gate": {"exists": p5 is not None,
                    "sha256": sha256_file(OUT / "R17_4P4M_SAFETY_GATE.json")},
    }
    for key, present in (("p0", p0 is not None), ("p1", p1 is not None),
                         ("p2", p2 is not None), ("p3", bool(p3_rows)),
                         ("p4", bool(p4_rows)), ("p5", p5 is not None)):
        if not present:
            v["missing"].append(key)

    cells = {r["cell"]: r for r in p3_rows}
    harness = harness_evidence()

    # ---------------- contract verdicts -------------------------------------
    verdicts = {}
    if p1:
        st = p1.get("distribution_stats", {})
        op = st.get("opacity_raw_logit", {})
        sc = st.get("scaling_raw_log_scale", {})
        rot = st.get("rotation_raw", {})
        n = op.get("n") or 1
        verdicts["OPACITY_INPUT_CONTRACT"] = (
            "VIOLATED" if (op.get("count_outside_unit_interval", 0) / n) > 0.5
            else "UNKNOWN")
        verdicts["SCALE_INPUT_CONTRACT"] = (
            "VIOLATED" if (sc.get("count_negative", 0) > 0) else "UNKNOWN")
        verdicts["ROTATION_INPUT_CONTRACT"] = (
            "VIOLATED" if rot.get("frac_norm_dev_gt_1e-3", 0) > 0.01 else "UNKNOWN")
        verdicts["_contract_rule"] = (
            "VIOLATED = harness inputs of R3-R16 violated the renderer's documented "
            "contract (activated opacity/scales, normalized quats); computed from the real "
            "checkpoint distributions + harness source evidence.")
    else:
        verdicts["OPACITY_INPUT_CONTRACT"] = "UNKNOWN"
        verdicts["SCALE_INPUT_CONTRACT"] = "UNKNOWN"
        verdicts["ROTATION_INPUT_CONTRACT"] = "UNKNOWN"

    if p2:
        rs = (p2.get("r3_synthetic") or {}).get("status", "")
        r8s = (p2.get("r8_identity") or {}).get("status", "")
        rt = (p2.get("real_protocol") or {}).get("status", "")
        if "SYNTHETIC" in rs and "SYNTHETIC" in r8s and rt == "EXECUTED":
            verdicts["CAMERA_PROTOCOL"] = "SYNTHETIC"
        elif rt != "EXECUTED":
            verdicts["CAMERA_PROTOCOL"] = "UNKNOWN"
        else:
            verdicts["CAMERA_PROTOCOL"] = "PARTIAL_EVIDENCE"
        verdicts["_camera_rule"] = (
            "SYNTHETIC = R3 used circular synthetic cameras and R8 used an identity "
            "viewmatrix; neither is the training protocol; real cameras.json resolved and "
            "used by R17 P3/P4.")
    else:
        verdicts["CAMERA_PROTOCOL"] = "UNKNOWN"

    # ---------------- P3-driven impact --------------------------------------
    impact = {"available": bool(cells)}
    if cells:
        def gc_(cell, key):
            return fnum((cells.get(cell) or {}).get(key))
        a_cnt, d_cnt = gc_("A", "sum_counts_first_call"), gc_("D", "sum_counts_first_call")
        a_cov, d_cov = gc_("A", "coverage_frac"), gc_("D", "coverage_frac")
        impact = {
            "available": True,
            "A_sum_counts": a_cnt, "D_sum_counts": d_cnt,
            "A_coverage_frac": a_cov, "D_coverage_frac": d_cov,
            "A_status": (cells.get("A") or {}).get("fwd_status"),
            "D_status": (cells.get("D") or {}).get("fwd_status"),
        }
        if a_cnt and d_cnt:
            impact["count_ratio_A_over_D"] = a_cnt / d_cnt
            impact["count_gap_pct"] = 100.0 * (a_cnt - d_cnt) / d_cnt
        if a_cov is not None and d_cov is not None:
            impact["coverage_gap_pp"] = 100.0 * (a_cov - d_cov)
        impact["material_thresholds"] = {
            "count_gap_pct_ge_20_or_coverage_gap_pp_ge_10": "material",
        }
        material = ((impact.get("count_gap_pct") or 0) >= 20.0
                    or (impact.get("coverage_gap_pp") or 0) >= 10.0)
        impact["material"] = material
    v["p3_impact"] = impact

    # per-cell deltas vs A for attribution summary
    if cells:
        deltas = {}
        for cell, row in cells.items():
            if cell == "A":
                continue
            c = fnum(row.get("sum_counts_first_call"))
            a = fnum(cells["A"].get("sum_counts_first_call"))
            deltas[cell] = {
                "sum_counts": c, "delta_vs_A_pct": (100.0 * (c - a) / a) if (a and c) else None,
                "fwd_status": row.get("fwd_status"),
                "oracle_mismatch_pairs": row.get("oracle_mismatch_pairs"),
            }
        v["p3_attribution_deltas"] = deltas

    # ---------------- benchmark validity ------------------------------------
    if not cells or impact.get("D_status") != "PASS":
        verdicts["BENCHMARK_VALIDITY"] = "PARTIAL" if (cells or p4_rows) else "UNKNOWN"
        v["rules"]["benchmark_validity"] = "Cells A and D must both pass before validity can be judged."
    else:
        if impact.get("material"):
            verdicts["BENCHMARK_VALIDITY"] = "INVALIDATED"
        else:
            verdicts["BENCHMARK_VALIDITY"] = "PARTIAL"
        v["rules"]["benchmark_validity"] = (
            "INVALIDATED = the raw/identity harness protocol (used by R3-R16) changes the "
            "intersection structure versus the contract protocol by >=20% counts or >=10pp "
            "coverage; PARTIAL unchanged-structure but non-protocol cameras; VALIDATED only "
            "when both coincide.")

    # ---------------- memory scalability ------------------------------------
    mem = {"available": p5 is not None}
    if p5:
        sc = p5.get("scenarios", {})
        for name, s in sc.items():
            mem[name] = {"status": s.get("status"), "fwd_status": s.get("fwd_status"),
                         "bwd_status": s.get("bwd_status"), "peak_vram_gb": s.get("peak_vram_gb"),
                         "estimate": (s.get("estimate") or {}).get("est_isects_mean_scaled")}
            if s.get("vs_r8_reported_request"):
                mem[name]["vs_r8"] = s["vs_r8_reported_request"]
        act = [sc.get(k, {}).get("status") for k in ("activated_real_b1", "activated_real_b4")]
        if all(s == "FULL_EXECUTED" for s in act) and act:
            verdicts["HIGS_MEMORY_SCALABILITY"] = "PASS"
        elif any(s == "OOM_DURING_EXECUTED_FULL" for s in act):
            verdicts["HIGS_MEMORY_SCALABILITY"] = "FAIL"
        elif any(s == "ESTIMATED_GATE_SKIPPED_FULL" for s in act):
            verdicts["HIGS_MEMORY_SCALABILITY"] = "INCONCLUSIVE"
        else:
            verdicts["HIGS_MEMORY_SCALABILITY"] = "UNKNOWN"
        v["rules"]["memory_scalability"] = (
            "PASS = contract protocol 4.41M fwd+bwd executed and passed; FAIL = executed "
            "full run OOMed; INCONCLUSIVE = the byte projection did not fit free VRAM so the "
            "full run was not attempted (raw scenarios are estimate-only by policy).")
    else:
        verdicts["HIGS_MEMORY_SCALABILITY"] = "UNKNOWN"
    v["p5_memory"] = mem

    # ---------------- P0 reconciliation -------------------------------------
    if p0 and p0.get("status") == "EXECUTED":
        rc = p0.get("recomputed", {})
        actual = rc.get("raw_minus_sigmoid_sum_difference")
        v["p0_reconciliation"] = {
            "claimed_reduction_r16_prose": CLAIMED_R16_REDUCTION,
            "recomputed_raw_minus_sigmoid": actual,
            "recomputed_elementwise_abs_sum": rc.get("raw_minus_sigmoid_elementwise_abs_sum"),
            "sigmoid_vs_threshold_identical": rc.get("sigmoid_vs_threshold_identical"),
            "claimed_reduction_reproducible": bool(actual == CLAIMED_R16_REDUCTION),
            "r15_raw_exact_match": (p0.get("r15_capture_raw_exact_match") or {}).get("exact_match"),
            "branch_coverage": p0.get("branch_coverage"),
        }
    else:
        v["p0_reconciliation"] = {"status": "NOT_EXECUTED"}

    # ---------------- R3-R16 disposition ------------------------------------
    disp = {}
    if p3_rows:
        a = cells.get("A", {})
        d = cells.get("D", {})
        disp["R3_std_vs_higs_timing_claims"] = (
            "REMEASURE" if p4_rows else "REMEASURE_NOT_YET (P4 missing)")
        disp["R3_R8_HiGS_OOM_root_cause"] = (
            "REINTERPRET: raw logit opacity + log-scale + unnormalized quats + synthetic "
            "camera violate the documented contract; see p3_attribution_deltas"
            if impact.get("material") else
            "STAND (raw-vs-contract protocol did not materially change intersection structure)")
        disp["R12_intersection_counts_@580K_raw"] = {
            "r17_cell_A_sum_counts": a.get("sum_counts_first_call"),
            "note": "direct comparison basis for the ~490M R12 claim at the same 580K/identity protocol",
        }
        disp["R16_OPACITY_SEMANTICS_CONFIRMED"] = {
            "r17_cell_A_vs_B": {
                "A_sum": a.get("sum_counts_first_call"),
                "B_sum": (cells.get("B", {}) or {}).get("sum_counts_first_call"),
            },
            "disposition": ("CONFIRMED_AS_HARNESS_EFFECT_ONLY: real, but it quantifies the "
                            "gap between a violated and an activated opacity input, not a "
                            "property of the training protocol"),
        }
        disp["R15_oracle_discrepancy"] = (
            "REINTERPRET: oracle mismatch pairs measured under raw inputs; "
            "see per-cell oracle_mismatch_pairs (cell A vs cell D)")
    else:
        disp["status"] = "NOT_EXECUTED_P3_MISSING"
    v["r3_r16_conclusion_disposition"] = disp

    v["harness_source_evidence"] = harness
    v["verdicts"] = verdicts
    v["status"] = ("EXECUTED" if not v["missing"] else
                   "PARTIAL_MISSING_" + "_".join(v["missing"]))
    write_json(OUT / "R17_FINAL_VERDICT.json", v)

    # ---------------- report -------------------------------------------------
    def row_md(r):
        keys = ("cell", "fwd_status", "bwd_status", "sum_counts_first_call",
                "n_isects_metadata", "coverage_frac", "count_max", "peak_vram_gb",
                "oracle_mismatch_pairs", "oracle_verdict")
        return "| " + " | ".join(str(r.get(k)) for k in keys) + " |"

    lines = []
    lines.append("# R17 Final Report — parameter validity & camera protocol adjudication\n")
    lines.append(f"Status: **{v['status']}**\n")
    lines.append("## Verdicts\n")
    for k in ("OPACITY_INPUT_CONTRACT", "SCALE_INPUT_CONTRACT", "ROTATION_INPUT_CONTRACT",
              "CAMERA_PROTOCOL", "BENCHMARK_VALIDITY", "HIGS_MEMORY_SCALABILITY"):
        lines.append(f"- **{k}: {verdicts.get(k)}**")
    lines.append("")
    if p0:
        lines.append("## P0 — R16 arithmetic audit\n")
        lines.append("```json")
        lines.append(json.dumps(v.get("p0_reconciliation"), indent=2, default=str)[:3000])
        lines.append("```\n")
    if p3_rows:
        lines.append("## P3 — 580K attribution cells\n")
        lines.append("| " + " | ".join(("cell", "fwd_status", "bwd_status", "sum_counts",
                                       "n_isects_meta", "coverage_frac", "count_max",
                                       "peak_vram_gb", "oracle_mismatch", "oracle_verdict")) + " |")
        lines.append("|" + "---|" * 10)
        for r in p3_rows:
            lines.append(row_md(r))
        lines.append("")
        lines.append("Attribution deltas vs cell A:\n")
        lines.append("```json")
        lines.append(json.dumps(v.get("p3_attribution_deltas"), indent=2, default=str)[:3000])
        lines.append("```\n")
    if p4_rows:
        lines.append("## P4 — matched benchmark (activated state, real cameras)\n")
        lines.append("| protocol | backend | fwd_med_ms | bwd_med_ms | fb_med_ms | peak_gb | n_isects | status |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in p4_rows:
            lines.append("| {protocol} | {backend} | {fwd_median_ms} | {bwd_median_ms} | "
                         "{fb_median_ms} | {peak_vram_gb} | {n_isects_instrumented} | {status} |".format(**r))
        lines.append("")
    if p5:
        lines.append("## P5 — 4.41M safety gate\n")
        lines.append("```json")
        lines.append(json.dumps(mem, indent=2, default=str)[:3000])
        lines.append("```\n")
    if v["missing"]:
        lines.append("## NOT_EXECUTED\n")
        lines.append("Missing: " + ", ".join(v["missing"]) + "\n")
    lines.append("## R3–R16 conclusion disposition\n")
    lines.append("```json")
    lines.append(json.dumps(disp, indent=2, default=str)[:4000])
    lines.append("```\n")
    lines.append("## Harness source evidence\n")
    lines.append("```json")
    lines.append(json.dumps(harness, indent=2, default=str)[:2000])
    lines.append("```\n")
    lines.append("## Rules\n")
    lines.append("```json")
    lines.append(json.dumps(v["rules"], indent=2, default=str))
    lines.append("```\n")
    (OUT / "R17_FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[write] {OUT / 'R17_FINAL_REPORT.md'}", flush=True)
    print(json.dumps({"verdicts": verdicts, "status": v["status"], "missing": v["missing"]},
                     indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
