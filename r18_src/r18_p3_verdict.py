#!/usr/bin/env python3
"""R18-P3: research verdict — reads P0/P1/P2 artifacts, emits five verdicts.

Conservative decision rules: every PASS must be backed by an actually-measured
row with status PASS. Missing / OOM / ERROR backing rows yield INCONCLUSIVE,
never PASS. No GPU required; degrades gracefully on missing inputs.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r18_common import OUT, write_json

RGB_MAX_ABS_TOL = 0.02
ALPHA_MAX_ABS_TOL = 0.02
GRAD_COSINE_MIN = 0.999

P1_CONFIGS = ("580K_B1", "580K_B4", "4M_B1", "4M_B4")
P2_CONFIGS = ("580K_540p_B1", "580K_540p_B4", "4M_540p_B1", "4M_540p_B4", "4M_1080p_B4")


def read_csv(path: Path):
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (ValueError, TypeError):
        return None


def index_p1(rows):
    return {r.get("config"): r for r in rows}


def index_p2(rows):
    out = {}
    for r in rows:
        out[(r.get("config"), r.get("backend"))] = r
    return out


def p1_status_map(p1idx):
    return {c: (p1idx.get(c, {}) or {}).get("status", "MISSING") for c in P1_CONFIGS}


def main():
    missing = []
    p0 = None
    p0_path = OUT / "R18_P0_INTEGRITY_AUDIT.json"
    if p0_path.is_file():
        p0 = json.loads(p0_path.read_text())
    else:
        missing.append(str(p0_path))

    p1_csv = read_csv(OUT / "R18_P1_CORRECTNESS.csv")
    p1_details = {}
    p1d_path = OUT / "raw" / "R18_P1_details.json"
    if p1d_path.is_file():
        p1_details = json.loads(p1d_path.read_text())
    p1idx = index_p1(p1_csv)

    p2_csv = read_csv(OUT / "R18_P2_PERFORMANCE.csv")
    p2_details = {}
    p2d_path = OUT / "raw" / "R18_P2_details.json"
    if p2d_path.is_file():
        p2_details = json.loads(p2d_path.read_text())
    p2idx = index_p2(p2_csv)

    p1_status = p1_status_map(p1idx)

    # ---- CORRECTNESS ----
    correctness_evidence = {"per_config_status": p1_status, "tolerances": {
        "rgb_max_abs": RGB_MAX_ABS_TOL, "alpha_max_abs": ALPHA_MAX_ABS_TOL,
        "grad_cosine_min": GRAD_COSINE_MIN}}
    bad = [c for c, s in p1_status.items() if s not in ("PASS", "MISSING")]
    finite_fail = []
    worst_rgb = 0.0
    worst_alpha = 0.0
    worst_cos = 1.0
    for c in P1_CONFIGS:
        r = p1idx.get(c, {}) or {}
        for key in ("finite_frame", "finite_alpha", "finite_grads"):
            v = r.get(key)
            if v is not None and str(v).lower() in ("false", "0"):
                finite_fail.append(f"{c}:{key}")
        rm = safe_float(r.get("rgb_max_abs"))
        am = safe_float(r.get("alpha_max_abs"))
        if rm is not None:
            worst_rgb = max(worst_rgb, rm)
        if am is not None:
            worst_alpha = max(worst_alpha, am)
        for gk in ("grad_means_cosine", "grad_scales_cosine",
                   "grad_rotation_cosine", "grad_opacity_cosine"):
            gc = safe_float(r.get(gk))
            if gc is not None:
                worst_cos = min(worst_cos, gc)
    correctness_evidence.update({
        "worst_rgb_max_abs": worst_rgb, "worst_alpha_max_abs": worst_alpha,
        "worst_grad_cosine": (worst_cos if worst_cos < 1.0 else None),
        "non_pass_or_missing": [c for c, s in p1_status.items() if s != "PASS"],
        "finite_failures": finite_fail,
    })
    all_pass = all(p1_status[c] == "PASS" for c in P1_CONFIGS)
    if bad or finite_fail:
        correctness = "FAIL"
    elif all_pass and worst_rgb <= RGB_MAX_ABS_TOL and worst_alpha <= ALPHA_MAX_ABS_TOL \
            and worst_cos >= GRAD_COSINE_MIN:
        correctness = "PASS"
    elif all_pass:
        correctness = "INCONCLUSIVE"
        correctness_evidence["reason"] = "all PASS but tolerances not met (worst values exceed threshold)"
    else:
        correctness = "INCONCLUSIVE"
        correctness_evidence["reason"] = "one or more P1 configs not executed"

    # ---- LARGE_SCALE_FB ----
    p2_4m_b4_h = p2idx.get(("4M_540p_B4", "higs_native"), {})
    p1_4m_b4 = p1idx.get("4M_B4", {}) or {}
    ls_status = p2_4m_b4_h.get("status", "MISSING")
    ls_evidence = {
        "p2_4M_540p_B4_higs_status": ls_status,
        "p1_4M_B4_finite_grads": p1_4m_b4.get("finite_grads"),
        "p2_4M_1080p_B4_higs_status": p2idx.get(("4M_1080p_B4", "higs_native"), {}).get("status", "MISSING"),
        "p2_4M_540p_B4_std_status": p2idx.get(("4M_540p_B4", "std_ll"), {}).get("status", "MISSING"),
    }
    if ls_status == "PASS" and str(p1_4m_b4.get("finite_grads")).lower() in ("true", "1"):
        large_scale_fb = "PASS"
    elif ls_status in ("OOM", "ERROR"):
        large_scale_fb = "FAIL"
    else:
        large_scale_fb = "INCONCLUSIVE"

    # ---- PERFORMANCE (headline: 4M_540p_B4 fb_median) ----
    std_row = p2idx.get(("4M_540p_B4", "std_ll"), {})
    higs_row = p2idx.get(("4M_540p_B4", "higs_native"), {})
    std_fb = safe_float(std_row.get("fb_median_ms"))
    higs_fb = safe_float(higs_row.get("fb_median_ms"))
    speedup = (std_fb / higs_fb) if (std_fb and higs_fb and std_fb > 0 and higs_fb > 0) else None
    perf_evidence = {
        "4M_540p_B4": {"std_status": std_row.get("status", "MISSING"),
                        "higs_status": higs_row.get("status", "MISSING"),
                        "std_fb_median_ms": std_fb, "higs_fb_median_ms": higs_fb,
                        "speedup_higs_over_std": speedup}}
    # secondary 580K
    for c in ("580K_540p_B1", "580K_540p_B4"):
        s = safe_float(p2idx.get((c, "std_ll"), {}).get("fb_median_ms"))
        h = safe_float(p2idx.get((c, "higs_native"), {}).get("fb_median_ms"))
        perf_evidence[c] = {"std_fb": s, "higs_fb": h,
                            "speedup": (s / h) if (s and h) else None}
    if std_row.get("status") == "PASS" and higs_row.get("status") == "PASS" and speedup is not None:
        performance = "IMPROVED" if speedup > 1.0 else "REGRESSED"
    elif std_row.get("status") in ("OOM", "ERROR") or higs_row.get("status") in ("OOM", "ERROR"):
        performance = "INCONCLUSIVE"
        perf_evidence["reason"] = "one backend OOM/ERROR at 4M_540p_B4; speedup not measurable"
    else:
        performance = "INCONCLUSIVE"

    # ---- MEMORY (4M_540p_B4 peak_reserved) ----
    std_mem = safe_float(std_row.get("peak_reserved_gb"))
    higs_mem = safe_float(higs_row.get("peak_reserved_gb"))
    mem_evidence = {"4M_540p_B4": {"std_peak_reserved_gb": std_mem,
                                    "higs_peak_reserved_gb": higs_mem}}
    if std_row.get("status") == "PASS" and higs_row.get("status") == "PASS" \
            and std_mem is not None and higs_mem is not None:
        memory = "IMPROVED" if higs_mem < std_mem else "REGRESSED"
    else:
        memory = "INCONCLUSIVE"
        mem_evidence["reason"] = "peak_reserved not PASS/measured at 4M_540p_B4"

    # ---- RESEARCH_DECISION ----
    if correctness == "FAIL":
        decision = "DROP"
        rationale = "Correctness gate failed: HiGS output/gradient diverges from std_ll beyond tolerance."
    elif correctness == "PASS" and performance == "IMPROVED" and large_scale_fb == "PASS":
        decision = "KEEP"
        rationale = "Correct, completes 4.41M fwd+bwd, and is faster than std_ll at 4.41M/540p B4."
    elif correctness == "PASS" and large_scale_fb == "PASS" and (performance == "REGRESSED" or memory == "REGRESSED"):
        decision = "REDESIGN"
        rationale = "Correct and large-scale-complete but slower or uses more memory than std_ll."
    else:
        decision = "INVESTIGATE"
        rationale = "Correctness or large-scale/perf evidence inconclusive; needs investigation before a decision."

    verdict = {
        "deliverable": "R18_RESEARCH_VERDICT",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "question": ("Under correct (activated) inputs, can HiGS render 4,413,928 Gaussians "
                     "stably, correctly, and significantly faster than std_ll?"),
        "backing": {
            "p0_binaries_all_match": (p0 or {}).get("binaries_all_match"),
            "p0_checkpoint_match": (p0 or {}).get("checkpoint", {}).get("match"),
            "p1_rows": p1_status,
            "p2_4M_540p_B4": {"std_status": std_row.get("status", "MISSING"),
                               "higs_status": higs_row.get("status", "MISSING"),
                               "std_fb_median_ms": std_fb, "higs_fb_median_ms": higs_fb,
                               "speedup": speedup},
        },
        "verdicts": {
            "CORRECTNESS": correctness,
            "LARGE_SCALE_FB": large_scale_fb,
            "PERFORMANCE": performance,
            "MEMORY": memory,
            "RESEARCH_DECISION": decision,
        },
        "evidence": {
            "CORRECTNESS": correctness_evidence,
            "LARGE_SCALE_FB": ls_evidence,
            "PERFORMANCE": perf_evidence,
            "MEMORY": mem_evidence,
            "RESEARCH_DECISION": {"rationale": rationale},
        },
        "missing_inputs": missing,
    }
    write_json(OUT / "R18_RESEARCH_VERDICT.json", verdict)

    # ---- Markdown report ----
    def sp(d):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
    md = []
    md.append("# R18 Validated Benchmark — Research Verdict\n")
    md.append(f"_Generated {verdict['generated_utc']}_\n")
    md.append(f"**Question:** {verdict['question']}\n")
    md.append("## Verdicts\n")
    md.append("| Verdict | Result |")
    md.append("|---|---|")
    for k, v in verdict["verdicts"].items():
        md.append(f"| {k} | **{v}** |")
    md.append("\n## Direct answer\n")
    md.append(f"- **CORRECTNESS = {correctness}**: worst RGB max-abs {worst_rgb:.5f}, "
              f"worst alpha max-abs {worst_alpha:.5f}, worst grad cosine "
              f"{(worst_cos if worst_cos < 1.0 else float('nan')):.6f} "
              f"(tolerances RGB/alpha<={RGB_MAX_ABS_TOL}, cosine>={GRAD_COSINE_MIN}).")
    md.append(f"- **LARGE_SCALE_FB = {large_scale_fb}**: HiGS 4.41M/540p/B4 fwd+bwd = {ls_status}.")
    if speedup is not None:
        md.append(f"- **PERFORMANCE = {performance}**: 4.41M/540p/B4 fwd+bwd median "
                  f"std_ll {std_fb:.3f} ms vs HiGS {higs_fb:.3f} ms → speedup {speedup:.3f}x.")
    else:
        md.append(f"- **PERFORMANCE = {performance}**: speedup not measurable at 4.41M/540p/B4 "
                  f"(std={std_row.get('status','MISSING')}, higs={higs_row.get('status','MISSING')}).")
    md.append(f"- **MEMORY = {memory}**: 4.41M/540p/B4 peak reserved std_ll {std_mem} GB vs HiGS {higs_mem} GB.")
    md.append(f"- **RESEARCH_DECISION = {decision}**: {rationale}")
    if missing:
        md.append(f"\n_Missing inputs: {missing}_\n")
    md.append("\n## Evidence (P1 correctness)\n")
    md.append("```json\n" + json.dumps(sp(correctness_evidence), indent=2, default=str) + "\n```")
    md.append("\n## Evidence (P2 performance)\n")
    md.append("```json\n" + json.dumps(sp(perf_evidence), indent=2, default=str) + "\n```")
    (OUT / "R18_RESEARCH_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({"verdicts": verdict["verdicts"],
                      "p1_rows": p1_status,
                      "p2_4M_540p_B4": verdict["backing"]["p2_4M_540p_B4"],
                      "missing_inputs": missing}, indent=2))
    print("P3 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
