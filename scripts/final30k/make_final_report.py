#!/usr/bin/env python3
"""make_final_report.py — Render reports/higs/final-30k-c0-v3-vs-b1a.md from the
frozen FINAL-30K aggregate artifacts (all produced by final30k_aggregate.py).

Report discipline (frozen):
  - per-scene tables BEFORE aggregates (skill §22)
  - speedup (ratio) and reduction (%) reported SEPARATELY
  - dLPIPS sign NOT flipped (negative = improvement)
  - failed/contaminated scenes never silently dropped
  - TTQ with R1 censoring semantics
  - no causal H8/E2 language (§32/§35); DSH-E P3 temporal model not used (§33)
"""
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
F30K = os.path.join(REPO, "artifacts", "final-30k")
OUT_REPORT = os.path.join(REPO, "reports", "higs", "final-30k-c0-v3-vs-b1a.md")
SCENES = ["bicycle", "bonsai", "counter", "flowers", "garden", "kitchen",
          "room", "stump", "treehill", "train", "truck", "drjohnson", "playroom"]


def jload(name):
    with open(os.path.join(F30K, name), encoding="utf-8") as f:
        return json.load(f)


def f2(x, nd=2):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def main():
    agg = jload("aggregate_performance.json")
    ttq = jload("ttq.json")
    n_gs = jload("n_gs_comparison.json")
    ph = jload("phase_timing.json")
    fm = jload("failure_manifest.json")
    verdict = jload("final_verdict.json")
    manifest = jload("manifest.json")

    lines = []
    A = lines.append
    A("# FINAL-30K: C0_V3_FINAL30K vs B1A_ACCUTILE — 13-Scene × 30K Matched Benchmark")
    A("")
    A(f"**Verdict: {verdict['verdict']}**")
    A("")
    A("Frozen protocol: seed 42, 1080p-class (max_side 1920), 13 scenes, reference_v1 "
      "recipe (densify 500–15000/100, absgrad threshold 0.0008, opacity reset 3000, SH/1000, "
      "λ_dssim 0.2, COLMAP SfM init), 30000 iterations, identical seed-42 camera sequence per "
      "scene-pair. BASE = B1A_ACCUTILE (`0471fbd9…`); TEST = C0_V3_FINAL30K (`9baf8655…`, "
      "F9 + SCALAR_ADJOINT + H8-MR, PX=2, HIGS_BWD_ABSGRAD=1). Both arms run the same unified "
      "trainer; only the render call and binary bootstrap differ. Scheduling: dynamic clean-GPU "
      "selection (zero processes + <500 MiB in two scans ≥120 s apart), per-run contamination "
      "snapshots; contaminated runs are downgraded and listed, never dropped.")
    A("")
    A("## 1. Per-scene results (before any aggregate)")
    A("")
    A("| scene | T b1a (min) | T c0 (min) | speedup (ratio) | reduction (%) | PSNR b1a | PSNR c0 | ΔPSNR (dB) | SSIM b1a | SSIM c0 | ΔSSIM | LPIPS b1a | LPIPS c0 | ΔLPIPS | final N b1a | final N c0 | status |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    sp_by_scene = {x["scene"]: x for x in agg["speedup"]["per_scene"]}
    q_by_scene = {x["scene"]: x for x in agg["quality_delta"]["per_scene"]}
    def jloadp(p):
        if not os.path.exists(p):
            return {}
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def fmt_int(x):
        return f"{x:,}" if isinstance(x, int) else "—"

    status_by = {}
    for cand_id in ("B1A_ACCUTILE", "C0_V3_FINAL30K"):
        for s in SCENES:
            fs_p = os.path.join(F30K, cand_id, s, "final_status.json")
            if not os.path.exists(fs_p):
                status_by[(cand_id, s)] = "MISSING"
                continue
            fs = jloadp(fs_p)
            tag = "OK" if fs.get("status") == "SUCCESS" and not fs.get("contaminated") else \
                (fs.get("status", "?") + ("/CONTAM" if fs.get("contaminated") else ""))
            status_by[(cand_id, s)] = tag
    for s in SCENES:
        sp = sp_by_scene.get(s, {})
        q = q_by_scene.get(s, {})
        n = n_gs["per_scene"].get(s, {})
        tb = jloadp(os.path.join(F30K, "B1A_ACCUTILE", s, "timing.json")).get("total_wall_s")
        tc = jloadp(os.path.join(F30K, "C0_V3_FINAL30K", s, "timing.json")).get("total_wall_s")
        qb = jloadp(os.path.join(F30K, "B1A_ACCUTILE", s, "quality.json"))
        qc = jloadp(os.path.join(F30K, "C0_V3_FINAL30K", s, "quality.json"))
        stat = status_by.get(("C0_V3_FINAL30K", s), "?")
        if status_by.get(("B1A_ACCUTILE", s), "?") != "OK":
            stat += " (b1a: " + status_by.get(("B1A_ACCUTILE", s), "?") + ")"
        A(f"| {s} | {f2((tb or 0)/60,1) if tb else '—'} | {f2((tc or 0)/60,1) if tc else '—'} | "
          f"{f2(sp.get('speedup'),4) if sp.get('speedup') else '—'} | "
          f"{f2(sp.get('reduction_pct'),2) if sp.get('reduction_pct') is not None else '—'} | "
          f"{f2(qb.get('psnr'))} | {f2(qc.get('psnr'))} | "
          f"{f2(q.get('dPSNR'),3) if q.get('dPSNR') is not None else '—'} | "
          f"{f2(qb.get('ssim'),4)} | {f2(qc.get('ssim'),4)} | "
          f"{f2(q.get('dSSIM'),4) if q.get('dSSIM') is not None else '—'} | "
          f"{f2(qb.get('lpips'),4)} | {f2(qc.get('lpips'),4)} | "
          f"{f2(q.get('dLPIPS'),4) if q.get('dLPIPS') is not None else '—'} | "
          f"{fmt_int(n.get('final_N',{}).get('b1a'))} | {fmt_int(n.get('final_N',{}).get('c0'))} | {stat} |")
    A("")
    A("ΔLPIPS = candidate − baseline; **negative is improvement** (sign not flipped). "
      "Speedup (ratio) and reduction (%) are separate representations of the same T pair.")
    A("")
    A("## 2. Aggregate performance")
    A("")
    sp = agg["speedup"]
    A(f"- geomean speedup (primary): **{f2(sp.get('geomean'),4)}×**")
    A(f"- arithmetic mean speedup (secondary): {f2(sp.get('arithmetic_mean'),4)}×")
    A(f"- geomean time reduction: {f2(agg['time_reduction_pct'].get('geomean'),2)}%")
    A(f"- success pairs: {agg.get('n_success_pairs')}/{agg.get('n_scenes')}")
    if agg.get("missing_or_failed_scenes"):
        A(f"- missing/failed scenes: {json.dumps(agg['missing_or_failed_scenes'])}")
    A("")
    A("## 3. Aggregate quality")
    A("")
    qd = agg["quality_delta"]
    A(f"- mean ΔPSNR: **{f2(qd.get('mean_dPSNR'),3)} dB**")
    A(f"- mean ΔSSIM: {f2(qd.get('mean_dSSIM'),4)}")
    A(f"- mean ΔLPIPS: {f2(qd.get('mean_dLPIPS'),4)} (negative = improvement)")
    A("- Garden (primary sensitivity scene) is reported in the per-scene table above, "
      "in the primary aggregate (no exclusion).")
    A("")
    A("## 4. Time-to-quality (TTQ, R1 censoring semantics)")
    A("")
    A("- Thresholds are derived from the FROZEN B1A final-eval quality per scene "
      "(not chosen after seeing results).")
    ts = ttq.get("ttq_speedup_to_PSNR", {})
    rc = ts.get("reach_counts", {})
    A(f"- geomean TTQ speedup to PSNR (reached/reached pairs only): "
      f"**{f2(ts.get('geomean'),4) if ts.get('geomean') else '—'}×** "
      f"({rc.get('n_reached_pairs_psnr', 0)}/{rc.get('n_scenes', 13)} comparable pairs)")
    A("")
    A("| scene | threshold PSNR (b1a final) | TTQ c0 (s) | status | censor_time (s) |")
    A("|---|---|---|---|---|")
    for s in SCENES:
        row = ttq.get("per_scene", {}).get(s, {})
        if row.get("status") == "NO_CURVE":
            A(f"| {s} | — | — | NO_CURVE | — |")
            continue
        th = row.get("thresholds", {})
        ttq_s = row.get("time_to_PSNR_s")
        A(f"| {s} | {f2(th.get('psnr')) if th.get('psnr') else '—'} | "
          f"{f2(ttq_s,1) if ttq_s else '—'} | {row.get('status_PSNR')} | "
          f"{f2(row.get('censor_time_PSNR_s'),1) if row.get('censor_time_PSNR_s') else '—'} |")
    A("")
    A("Censored scenes (TTQ_NOT_REACHED) contribute their censor time as a bound only and are "
      "excluded from geomeans; reach counts are stated explicitly. Baseline TTQ to its own "
      "threshold is 30,000-step total wall time by construction (threshold = baseline final).")
    A("")
    A("## 5. Phase timing (means over 30K iterations)")
    A("")
    A("| phase | b1a mean (ms) | c0 mean (ms) |")
    A("|---|---|---|")
    for phase in ("forward", "backward", "loss", "densify", "optimizer"):
        a = ph["aggregate"].get(phase, {})
        A(f"| {phase} | {f2(a.get('b1a_mean_ms'),3)} | {f2(a.get('c0_mean_ms'),3)} |")
    A("")
    A("Phase timing describes WHERE time goes; it does not by itself attribute causality "
      "to individual composed modules (§35 discipline).")
    A("")
    A("## 6. N_GS comparison")
    A("")
    A(f"- geomean final-N ratio (c0/b1a): **{f2(n_gs['aggregate'].get('geomean_ratio'),4)}×**")
    A(f"- mean final N: b1a {int(n_gs['aggregate']['mean_final_N']['b1a'] or 0):,} / "
      f"c0 {int(n_gs['aggregate']['mean_final_N']['c0'] or 0):,}")
    A("- Per-scene values in the table above; gate E preflight showed ≤3.9% max N_GS "
      "trajectory deviation at 2K on room/bicycle/garden.")
    A("")
    A("## 7. Failure manifest")
    A("")
    if fm.get("n_failures", 0) == 0:
        A("- none (26/26 SUCCESS, zero contamination)")
    else:
        A(f"- {fm['n_failures']} entries:")
        for f in fm.get("failures", []):
            A(f"  - {f.get('candidate')}/{f.get('scene')}: {f.get('status')}"
              + (" (CONTAMINATED)" if f.get("contaminated") else ""))
    A("")
    A("## 8. Verdict")
    A("")
    A(f"**{verdict['verdict']}**")
    A("")
    A(f"Rule: {verdict['rule']}")
    A("")
    A("Inputs: " + json.dumps(verdict.get("inputs", {}), indent=2))
    if verdict.get("reasons"):
        A("")
        A("Reasons:")
        for r in verdict["reasons"]:
            A(f"- {r}")
    A("")
    A("## 9. Provenance and limitations")
    A("")
    A("- Binary identities: B1A_ACCUTILE `0471fbd95cae4cfa658bc1e2e3f7abe801c6f58a36a2e681a6d16af84279986b`; "
      "C0_V3_FINAL30K `9baf8655f859f3e0f456e99a2d8e5b3ddfe414fe17a1a3e9d8072e0afed95b91` "
      "(gates C+D+E PASS, `artifacts/higs-final30k-compat/final_gate.json`).")
    A("- eps2d: B1A 0.1 (frozen recipe), C0 0.3 (frozen validated stack) — renderer-internal, "
      "disclosed and quantified by gate D1.")
    A("- DSH-E P3 temporal speedup model: INVALID_SUPERSEDED_MODEL/DO_NOT_USE (§33). "
      "No E2 in the tested stack (§32). No causal H8 claims (§35).")
    A("- Timing is publication-grade only for runs launched on verified-clean GPUs without "
      "foreign-process contamination; contaminated runs are downgraded and listed in §7.")
    A("- Historical planning figures (B1A 13-scene cohort) were used for capacity planning "
      "only, never as results.")

    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
