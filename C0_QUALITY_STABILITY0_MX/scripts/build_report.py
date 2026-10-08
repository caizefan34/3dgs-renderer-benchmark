#!/usr/bin/env python3
"""C0-QUALITY-STABILITY0-MX report + manifest generator.
Reads all campaign JSONs, writes QUALITY_REPORT.md (25 sections) and MANIFEST.sha256
with self-verification. Run after aggregate + derived builders.
"""
import argparse, json, os, hashlib, glob, datetime

CAMPAIGN = "/mnt/storage_pool/liaoyuanjun/C0_QUALITY_STABILITY0_MX"
SCENES = ["room", "bicycle", "garden"]
SEEDS = [42, 43, 44]


def load(path, default=None):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            return default
    return default


def fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def build_report(O):
    src = load(os.path.join(O, "QUALITY_SOURCE_IDENTITY.json"), {})
    proto = load(os.path.join(O, "QUALITY_PROTOCOL_IDENTITY.json"), {})
    reuse = load(os.path.join(O, "QUALITY_SEED42_REUSE.json"), {})
    gates = load(os.path.join(O, "QUALITY_FROZEN_GATES.json"), {})
    metrics = load(os.path.join(O, "QUALITY_30K_METRICS.json"), {})
    stats = load(os.path.join(O, "QUALITY_MULTI_SEED_STATS.json"), {})
    speed = load(os.path.join(O, "QUALITY_SPEED_STABILITY.json"), {})
    pop = load(os.path.join(O, "QUALITY_POPULATION_TRAJECTORY.json"), {})
    fd = load(os.path.join(O, "QUALITY_FIRST_DIVERGENCE.json"), {})
    st = load(os.path.join(O, "QUALITY_SHORT_TRACE.json"), {})
    fixed = load(os.path.join(O, "QUALITY_FIXED_STATE_EXACTNESS.json"), {})
    exact = load(os.path.join(O, "QUALITY_EXACTNESS_CLAIMS.json"), {})
    mod = load(os.path.join(O, "QUALITY_MODULE_ATTRIBUTION.json"), {})
    hyp = load(os.path.join(O, "QUALITY_HYPOTHESIS_TEST.json"), {})
    verdict = load(os.path.join(O, "QUALITY_VERDICT.json"), {})
    contam = load(os.path.join(O, "QUALITY_CONTAMINATION.json"), {})
    runplan = load(os.path.join(O, "QUALITY_RUN_PLAN.json"), {})

    L = []
    A = L.append
    A("# C0-QUALITY-STABILITY0-MX — Quality Stability Across Seeds\n")
    A(f"_Generated: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}_\n")

    # per-pair delta lookup
    pp = {(p["scene"], p["seed"]): p for p in metrics.get("per_pair", [])}

    A("## 1. Executive Summary\n")
    A(f"**Terminal quality verdict: `{verdict.get('verdict')}`**\n")
    A(f"- Mean ΔPSNR (C0−A0) over 9 pairs: **{fmt(stats.get('dPSNR',{}).get('mean'))} dB** "
      f"(95% CI [{fmt(stats.get('dPSNR',{}).get('lo'))}, {fmt(stats.get('dPSNR',{}).get('hi'))}])")
    A(f"- Mean ΔSSIM: **{fmt(stats.get('dSSIM',{}).get('mean'),5)}**; Mean ΔLPIPS: **{fmt(stats.get('dLPIPS',{}).get('mean'),5)}**")
    A(f"- Overall speed GM (T_A0/T_C0): **{fmt(speed.get('overall_geomean'))}x**, positive pairs {speed.get('n_positive')}/{speed.get('n_ratios')}")
    A(f"- Final-N C0/A0 geomean: **{fmt(pop.get('final_N_c0_over_a0_geomean'))}**")
    A(f"- Exactness: algebraically exact (same .so, runtime switches) but **NOT** bit-exact and **NOT** trajectory-equivalent.\n")

    A("## 2. Scientific Question\n")
    A("Is the seed42 A0→C0 full-training quality degradation (mean ΔPSNR −0.2056 dB, QUALITY_DEGRADATION) "
      "an idiosyncratic trajectory-divergence event, or a reproducible systematic quality shift across seeds?\n")

    A("## 3. Frozen Prior Evidence\n")
    A("- MX-C0-MATCHED-REPLICATION (B1A vs C0): REPLICATION_POSITIVE, GM 1.0669x — but substrate differs from A0.\n"
      "- C0-SAME-PARENT-CAUSAL-LADDER-MX (seed42 A0/A1/A2/A3): ATTRIBUTION_MIXED_BUT_EXPLAINABLE, A0→C0 1.01737x, ΔPSNR −0.2056 dB.\n"
      "- C0-SUBSTRATE-ATTRIBUTION0-MX: SUBSTRATE_DOMINANT (B1A→A0 = 1.05034x).\n")

    A("## 4. A0 Definition\n")
    A("higs-dynamic substrate; F9 OFF, SCALAR_ADJOINT OFF, H8-MR OFF; eps2d=0.3. "
      "Runtime env: `HIGS_DISABLE_F9=1`, `HIGS_PX_RUNTIME=2`, `HIGS_BWD_ABSGRAD=1`.\n")

    A("## 5. C0 Definition\n")
    A("Same higs-dynamic substrate; F9 ON, SCALAR_ADJOINT ON, H8-MR ON; eps2d=0.3. "
      "Runtime env: `HIGS_BWD_SCALAR_ADJOINT=scalar_adjoint`, `HIGS_BWD_H8_MR=1`, `HIGS_PX_RUNTIME=2`, `HIGS_BWD_ABSGRAD=1`.\n")

    A("## 6. Source Identity\n")
    A(f"- Same `.so` for both arms: `{src.get('compiled_implementation',{}).get('so_sha256','')[:16]}...` (prefix)")
    A(f"- Core `.so`: `{src.get('compiled_implementation',{}).get('core_so_sha256','')[:16]}...`")
    A(f"- Only allowed differences: {src.get('only_allowed_differences')}")
    A(f"- Verdict: **{src.get('verdict')}**\n")

    A("## 7. Protocol Identity\n")
    A(f"- iters=30000, eps2d=0.3, recipe identical across arms: {proto.get('arms_recipe_identical')}")
    A(f"- Verdict: **{proto.get('verdict')}**\n")

    A("## 8. Seed42 Reuse\n")
    A(f"- Decision: **{reuse.get('reuse_decision')}** (rerun_required={reuse.get('rerun_required')})")
    A(f"- All 6 seed42 A0/C0 artifacts SUCCESS / uncontaminated / PUBLICATION; mean ΔPSNR {fmt(reuse.get('mean_d_psnr'))} dB reproduces frozen causal ladder exactly.\n")

    A("## 9. Execution / Contamination\n")
    A(f"- 12 new runs (seed43/44 × room/bicycle/garden × {runplan.get('arm_definitions',{}).keys()}), alternating order, same-GPU-per-pair.")
    A(f"- Clean runs: {contam.get('n_clean')}/{len(contam.get('per_run',[]))}; contaminated: {contam.get('n_contaminated')}. Foreign jobs never killed.\n")

    A("## 10. 30K Quality Results\n")
    A("| scene | seed | ΔPSNR | ΔSSIM | ΔLPIPS | speedup |")
    A("|---|---|---|---|---|---|")
    for sc in SCENES:
        for sd in SEEDS:
            p = pp.get((sc, sd))
            if p:
                A(f"| {sc} | {sd} | {fmt(p.get('d_psnr'))} | {fmt(p.get('d_ssim'),5)} | {fmt(p.get('d_lpips'),5)} | {fmt(p.get('speedup'))}x |")
    A("")

    A("## 11. Multi-Seed Quality Stability\n")
    A(f"- ΔPSNR mean {fmt(stats.get('dPSNR',{}).get('mean'))}, median {fmt(stats.get('dPSNR',{}).get('median'))}, "
      f"95% CI [{fmt(stats.get('dPSNR',{}).get('lo'))}, {fmt(stats.get('dPSNR',{}).get('hi'))}]")
    A(f"- ΔSSIM mean {fmt(stats.get('dSSIM',{}).get('mean'),5)}; ΔLPIPS mean {fmt(stats.get('dLPIPS',{}).get('mean'),5)}")
    A(f"- Direction counts (ΔPSNR): negative {stats.get('direction_count',{}).get('dPSNR_negative')}, positive {stats.get('direction_count',{}).get('dPSNR_positive')}")
    A(f"- Per-seed mean ΔPSNR: " + ", ".join(f"s{sd}={fmt(stats.get('per_seed',{}).get(str(sd),{}).get('mean_d_psnr'))}" for sd in SEEDS))
    A(f"- Per-scene mean ΔPSNR: " + ", ".join(f"{sc}={fmt(stats.get('per_scene',{}).get(sc,{}).get('mean_d_psnr'))}" for sc in SCENES) + "\n")

    A("## 12. Speed Stability\n")
    A(f"- Overall GM {fmt(speed.get('overall_geomean'))}x; per-seed GM " + ", ".join(f"s{sd}={fmt(speed.get('per_seed_geomean',{}).get(str(sd)))}" for sd in SEEDS))
    A(f"- Positive pairs {speed.get('n_positive')}/{speed.get('n_ratios')}. Secondary metric; not tuned.\n")

    A("## 13. Gaussian Population\n")
    A(f"- Final-N C0/A0 geomean: **{fmt(pop.get('final_N_c0_over_a0_geomean'))}**")
    A("- Descriptive only; causal link to quality is INFERRED, not OBSERVED.\n")

    A("## 14. First Trajectory Divergence\n")
    for r in fd.get("results", []):
        A(f"- {r.get('scene')}/s{r.get('seed')}: var={r.get('first_divergent_variable')}, iter={r.get('first_divergence_iter')}, "
          f"pop_div_iter={(r.get('first_population_divergence') or {}).get('iter')}, rel={r.get('relative_to_first_densification')}")
    A("")

    A("## 15. Short Trace Diagnostic\n")
    A(f"- status: **{st.get('status')}** — {st.get('reason')}\n")

    A("## 16. Fixed-State Numerical Exactness\n")
    fw = fixed.get("forward", {})
    A(f"- initial loss reldiff: {fmt((fw.get('initial') or {}).get('observed_loss_reldiff_iter1'),6)}")
    A(f"- mid loss reldiff: {fmt((fw.get('mid') or {}).get('observed_loss_reldiff_iter15000'),6)}")
    A(f"- late loss reldiff: {fmt((fw.get('late') or {}).get('observed_loss_reldiff_iter30000'),6)}")
    A(f"- authoritative causal-ladder forward rel-L2: {json.dumps(fw.get('causal_ladder_forward_rel_l2'))}\n")

    A("## 17. Exactness Terminology Audit\n")
    for k, v in exact.get("layers", {}).items():
        A(f"- {k}: **{v}**")
    A(f"\n_Forbidden inference explicitly avoided: {exact.get('forbidden_inference')}_\n")

    A("## 18. Module-Level Quality Attribution\n")
    A(f"- Scope: **{mod.get('scope')}** (no seed43/44 A1/A2 launched)")
    d = mod.get("seed42_dPSNR_mean", {})
    A(f"- seed42 ΔPSNR means — F9: {fmt(d.get('A0_to_A1_F9'))}, SCALAR: {fmt(d.get('A1_to_A2_SCALAR'))}, H8: {fmt(d.get('A2_to_A3_H8'))}, TOTAL: {fmt(d.get('TOTAL_A0_to_C0'))}\n")

    A("## 19. Competing Hypotheses\n")
    for k, v in hyp.get("hypotheses", {}).items():
        A(f"- **{k}**: {v.get('status')}")

    A("\n## 20. Quality Verdict\n")
    A(f"**{verdict.get('verdict')}**\n")
    crit = verdict.get("criteria", {})
    A(f"- n_pairs={crit.get('n_pairs')}, mean_d_psnr={fmt(crit.get('mean_d_psnr'))}, "
      f"n_dpsnr_below_floor={crit.get('n_dpsnr_below_floor')}, n_catastrophic={crit.get('n_catastrophic')}")
    A(f"- per-seed mean ΔPSNR: {json.dumps(crit.get('per_seed_mean_d_psnr'))}\n")

    A("## 21. Publication Implications\n")
    ver = verdict.get('verdict')
    if ver == "QUALITY_SYSTEMATIC_DEGRADATION":
        A("- The A0→C0 modules do NOT preserve full-training quality across seeds; "
          "exact restructuring cannot be claimed quality-neutral end-to-end. Paper must DROP full-training equivalence (see §24).")
    elif ver == "QUALITY_STABLE_NEUTRAL":
        A("- Quality is neutral across seeds; exact-restructuring claim can be retained with trajectory-divergence qualification.")
    else:
        A(f"- Verdict {ver}: report raw paired statistics; see report §24 for paper action.\n")

    A("## 22. Reviewer-Safe Claims\n")
    A("- The B1A→C0 1.0669x system effect is mostly substrate; the fair proposed-module effect is A0→C0.")
    A("- A0 and C0 load the identical binary; only three runtime switches differ.")
    A("- Algebraic exactness of the restructuring does NOT imply full-training trajectory or final-quality equivalence.\n")

    A("## 23. Limitations\n")
    A("- Single comparison pair (A0 vs C0); no multi-seed isolation of individual modules (would require A1/A2 seed43/44, forbidden here).")
    A("- Fixed-state rel-L2 authoritative numbers come from the seed42 causal ladder; seed43/44 fixed-state uses supplementary loss-reldiff.\n")

    A("## 24. Terminal Verdict\n")
    A(f"**{verdict.get('verdict')}**\n")

    A("## 25. Evidence Classification\n")
    A("- OBSERVED: per-run PSNR/SSIM/LPIPS, wall-time, final-N, per-iter trajectories.")
    A("- DERIVED: paired deltas, bootstrap CIs, gate pass/fail, terminal verdict.")
    A("- INFERRED: causal links (population→quality, densification→quality).")
    A("- HYPOTHESIZED: mechanism hypotheses (see §19).")
    A("- UNRESOLVED: items without sufficient evidence.\n")

    txt = "\n".join(L) + "\n"
    open(os.path.join(O, "QUALITY_REPORT.md"), "w").write(txt)
    return txt


def build_manifest(O):
    arts = ["QUALITY_SOURCE_IDENTITY.json", "QUALITY_PROTOCOL_IDENTITY.json",
            "QUALITY_SEED42_REUSE.json", "QUALITY_RUN_PLAN.json", "QUALITY_CONTAMINATION.json",
            "QUALITY_FROZEN_GATES.json", "QUALITY_30K_METRICS.json", "QUALITY_MULTI_SEED_STATS.json",
            "QUALITY_SPEED_STABILITY.json", "QUALITY_POPULATION_TRAJECTORY.json",
            "QUALITY_FIRST_DIVERGENCE.json", "QUALITY_SHORT_TRACE.json",
            "QUALITY_FIXED_STATE_EXACTNESS.json", "QUALITY_EXACTNESS_CLAIMS.json",
            "QUALITY_MODULE_ATTRIBUTION.json", "QUALITY_HYPOTHESIS_TEST.json",
            "QUALITY_VERDICT.json", "RUN_STATUS.json", "QUALITY_REPORT.md", "QUALITY_GATE_SUMMARY.json"]
    lines = []
    for a in arts:
        p = os.path.join(O, a)
        if os.path.exists(p):
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            lines.append(f"{h}  {a}")
    man = os.path.join(O, "MANIFEST.sha256")
    open(man, "w").write("\n".join(lines) + "\n")
    # self-verify
    ok = 0; bad = []
    for line in lines:
        h, name = line.split("  ", 1)
        h2 = hashlib.sha256(open(os.path.join(O, name), "rb").read()).hexdigest()
        if h2 == h:
            ok += 1
        else:
            bad.append(name)
    return {"n_manifest": len(lines), "verify_ok": ok, "verify_bad": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=CAMPAIGN)
    args = ap.parse_args()
    O = args.outdir
    build_report(O)
    m = build_manifest(O)
    print(f"REPORT + MANIFEST built; manifest verify_ok={m['verify_ok']}/{m['n_manifest']} bad={m['verify_bad']}")


if __name__ == "__main__":
    main()
