#!/usr/bin/env python3
"""P12 final report assembler.

Reads the P12A_*/P12B_* stats/verdict/quality/trajectory artifacts and emits
the EXACT final report format required by the P12 protocol (P12_FINAL_REPORT.txt).
"""
import json
import os
import sys

ART = "/root/epic05_p12/artifacts"


def j(name):
    return json.load(open(os.path.join(ART, name)))


def fmt(x, nd=4):
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def main():
    lock = j("P12_FUSED_LOSS_LOCK.json")
    ident = j("P12_SOURCE_IDENTITY.json")
    pref = j("P12_PREFLIGHT.json")
    contam = j("P12_CONTAMINATION.json")
    a_stats = j("P12A_SYSTEM_STATS.json")
    a_qual = j("P12A_SYSTEM_QUALITY.json")
    a_traj = j("P12A_SYSTEM_TRAJECTORY.json")
    a_verd = j("P12A_SYSTEM_VERDICT.json")
    b_stats = j("P12B_F9_STATS.json")
    b_qual = j("P12B_F9_QUALITY.json")
    b_traj = j("P12B_F9_TRAJECTORY.json")
    b_verd = j("P12B_F9_VERDICT.json")

    fs = lock["p11_identity_reused"]
    sem = lock["semantic_lock"]

    # ---- P12-A interpretation (descriptive only) ----
    a_gm = a_stats["geometric_mean"]
    a_old = a_stats["old_reference"]["gm"]
    if a_verd["verdict"].startswith("SYSTEM_REVALIDATED"):
        a_interp = (f"the C0 system endpoint SURVIVES the strong fused-loss "
                    f"baseline: new GM {a_gm:.4f} vs old SepSSIM P8 GM {a_old} "
                    f"(dlog {a_stats['new_vs_old']['delta_log_ratio']:+.4f}, "
                    f"{a_stats['new_vs_old']['relative_change_pct']:+.2f}%); "
                    f"old and new GMs are never pooled")
    elif a_verd["verdict"] == "SYSTEM_NEUTRAL":
        a_interp = (f"the C0 system endpoint does NOT clearly survive the "
                    f"strong fused-loss baseline (GM {a_gm:.4f}, approx parity); "
                    f"the P8 SepSSIM GM {a_old} is not confirmed under the "
                    f"modernized loss")
    else:
        a_interp = (f"REVERSED under the strong fused-loss baseline: current "
                    f"gsplat main is faster (GM {a_gm:.4f}); the P8 SepSSIM "
                    f"GM {a_old} does not survive")

    # ---- P12-B interpretation ----
    b_gm = b_stats["geometric_mean"]
    b_old = b_stats["old_reference"]["gm"]
    if b_verd["verdict"] in ("F9_REVALIDATED_STRONG", "F9_REVALIDATED_POSITIVE_SMALL"):
        b_interp = (f"F9 retains a reproducible full-training benefit under the "
                    f"strong fused-loss baseline: new GM {b_gm:.4f} vs old P2 "
                    f"SepSSIM GM {b_old} (dlog "
                    f"{b_stats['new_vs_old']['delta_log_ratio']:+.4f}, "
                    f"{b_stats['new_vs_old']['relative_change_pct']:+.2f}%); "
                    f"never pooled")
    elif b_verd["verdict"] == "F9_NEUTRAL":
        b_interp = (f"F9 does NOT retain a clear full-training benefit once the "
                    f"common SepSSIM bottleneck is removed (GM {b_gm:.4f}, "
                    f"parity/mixed); the old P2 GM {b_old} was within "
                    f"system-level noise")
    else:
        b_interp = (f"F9 is consistently SLOWER under the strong fused-loss "
                    f"baseline (GM {b_gm:.4f}); the old P2 result does not "
                    f"reproduce")

    def confound(traj):
        corr = traj["corr_logR_logNratio"]
        spread = (f"N-ratio span [{traj['N_ratio_min']:.3f}, "
                  f"{traj['N_ratio_max']:.3f}]")
        if corr is None:
            return f"{spread}; corr not computable"
        strong = abs(corr) > 0.5
        return (f"{spread}; corr(logR, logN-ratio)={corr:+.3f}; "
                + ("population confound MATERIAL -- the 30K wall ratio must NOT "
                   "be called an intrinsic renderer-kernel ratio"
                   if strong else
                   "population confound weak -- wall-ratio direction is not "
                   "explained by trajectory population differences"))

    a_n = a_stats["n_positive"]
    b_n = b_stats["n_positive"]
    a_ci = a_stats["ci95_log"]
    b_ci = b_stats["ci95_log"]

    def valid_line(camp):
        cc = contam["campaigns"][camp]
        pres = cc.get("preserved_contaminated_attempts") or []
        n = len(cc["runs"])
        assert n == 26 and cc["n_dirty"] == 0, (camp, n, cc["n_dirty"])
        note = ""
        if pres:
            scenes = sorted({os.path.basename(p).split(".contaminated_")[0]
                             for p in pres})
            note = (f"; contamination events: {len(pres)} run attempt(s) "
                    f"({', '.join(scenes)}) preserved verbatim and the pair "
                    f"rerun unchanged; final 26/26 clean")
        return f"valid scenes: 13/13 (clean paired 30K runs; " \
               f"contamination join clean{note})"

    lines = []
    A = lines.append
    A("P12 Strong-Baseline Revalidation")
    A("")
    A("FUSED LOSS LOCK")
    A(f"P11 identity reused: rahul-goel/fused-ssim @ {fs['commit']} "
      f"(working tree {'clean' if fs['working_tree_clean'] else 'DIRTY'})")
    A(f"padding: {sem['padding']}")
    A(f"semantic lock: {sem['loss_formula']}; 11x11 Gaussian sigma=1.5, "
      f"C1={sem['C1']}, C2={sem['C2']}, {sem['image_range']}, {sem['reduction']}; "
      f"P11 classification {sem['p11_equivalence_classification']} "
      f"(S_loss 15.9x per P11 L2); eval SSIM metric = SepSSIM (frozen procedure)")
    A(f"source hash: {fs['module_sha256']} (fused_ssim/__init__.py)")
    A(f"binary hash: {fs['so_sha256']} (fused_ssim_cuda.so; hard-locked in "
      f"trainer; matches P11)")
    A("")
    A("P12-A — CURRENT GSPLAT vs C0")
    A(valid_line("p12a"))
    A(f"C0 faster: {a_n}/13")
    A(f"GM GMAIN/C0: {fmt(a_gm)}")
    A(f"95% CI: [{fmt(a_ci['ratio_lo'])}, {fmt(a_ci['ratio_hi'])}] "
      f"({'entirely > 1' if a_ci['entirely_above_1'] else 'overlaps 1'})")
    A(f"median: {fmt(a_stats['median_ratio'])}")
    A(f"min/max: {fmt(a_stats['min_ratio'])} / {fmt(a_stats['max_ratio'])}")
    A(f"equivalent C0 wall reduction: {a_stats['equivalent_wall_reduction']*100:.2f}%")
    A("")
    A(f"mean ΔPSNR (gmain-C0): {fmt(a_qual['mean_d_psnr'], 3)} dB "
      f"(median {fmt(a_qual['median_d_psnr'], 3)})")
    A(f"mean ΔSSIM: {fmt(a_qual['mean_d_ssim'], 5)}")
    A(f"mean ΔLPIPS: {fmt(a_qual['mean_d_lpips'], 5)}")
    A(f"trajectory/population confound: {confound(a_traj)}")
    A("")
    A(f"old P8 GM: {a_old} (SepSSIM common loss, 13/13)")
    A(f"new fused-loss GM: {fmt(a_gm)} ({a_n}/13)")
    A(f"interpretation: {a_interp}")
    A("")
    A(f"P12-A verdict: {a_verd['verdict']}")
    A("")
    A("P12-B — F9")
    A(valid_line("p12b"))
    A(f"F9 faster: {b_n}/13")
    A(f"GM A0/A1: {fmt(b_gm)}")
    A(f"95% CI: [{fmt(b_ci['ratio_lo'])}, {fmt(b_ci['ratio_hi'])}] "
      f"({'entirely > 1' if b_ci['entirely_above_1'] else 'overlaps 1'})")
    A(f"median: {fmt(b_stats['median_ratio'])}")
    A(f"min/max: {fmt(b_stats['min_ratio'])} / {fmt(b_stats['max_ratio'])}")
    A(f"equivalent wall reduction: {b_stats['equivalent_wall_reduction']*100:.2f}%")
    A("")
    A(f"mean ΔPSNR (A1-A0): {fmt(b_qual['mean_d_psnr'], 3)} dB "
      f"(median {fmt(b_qual['median_d_psnr'], 3)})")
    A(f"mean ΔSSIM: {fmt(b_qual['mean_d_ssim'], 5)}")
    A(f"mean ΔLPIPS: {fmt(b_qual['mean_d_lpips'], 5)}")
    A(f"trajectory/population confound: {confound(b_traj)}")
    A("")
    A(f"old P2 F9 GM: {b_old} (SepSSIM common loss, 13/13, SXM)")
    A(f"new fused-loss F9 GM: {fmt(b_gm)} ({b_n}/13)")
    A(f"interpretation: {b_interp}")
    A("")
    A(f"P12-B verdict: {b_verd['verdict']}")
    A("")
    A("PAPER CONSEQUENCE")

    # system endpoint claim
    if a_verd["verdict"].startswith("SYSTEM_REVALIDATED"):
        sys_claim = (f"KEEP (updated numbers): the frozen C0 complete system "
                     f"remains faster than current gsplat main "
                     f"(512d366b) under the fused-SSIM strong baseline "
                     f"(GM {fmt(a_gm)}, {a_n}/13); report the fused-loss GM as "
                     f"the modern endpoint; do NOT attribute the system ratio "
                     f"to F9/H8")
    elif a_verd["verdict"] == "SYSTEM_NEUTRAL":
        sys_claim = (f"WEAKEN: the C0-vs-current-gsplat system claim does not "
                     f"clearly survive the fused-SSIM strong baseline "
                     f"(GM {fmt(a_gm)}); state parity under the strong baseline "
                     f"and keep the P8 SepSSIM result as the loss-conditioned "
                     f"comparison")
    else:
        sys_claim = (f"RETRACT/REVERSE: under the fused-SSIM strong baseline "
                     f"current gsplat main is faster (GM {fmt(a_gm)}); the "
                     f"system endpoint claim must be removed or reversed")
    # F9 claim
    if b_verd["verdict"].startswith("F9_REVALIDATED"):
        f9_claim = (f"KEEP (bounded): F9 retains a small reproducible "
                    f"full-training effect under the strong fused-loss baseline "
                    f"(GM {fmt(b_gm)}, {b_n}/13); present it as a "
                    f"materialization-free producer improvement, not a "
                    f"headline speedup")
    elif b_verd["verdict"] == "F9_NEUTRAL":
        f9_claim = (f"WEAKEN: F9's full-training effect does not survive the "
                    f"strong fused-loss baseline (GM {fmt(b_gm)}); keep F9 as a "
                    f"mechanism/representation contribution and report the "
                    f"revalidation as within system-level noise")
    else:
        f9_claim = (f"RETRACT: F9 is consistently slower under the strong "
                    f"fused-loss baseline (GM {fmt(b_gm)}); the full-training "
                    f"performance claim must be removed")
    h8_claim = ("UNCHANGED: H8 remains an exact-representation contribution "
                "with context-bound fixed-state backward gain (P10 direct "
                "backward GM 1.0193x) and E2E-neutral full training "
                "(EPIC 1.0011x FULLTRAIN_NEUTRAL; mx 0.9971x "
                "REPLICATION_NEUTRAL); no H8 rescue was attempted in P12; "
                "Scalar remains an implementation simplification")

    A(f"system endpoint claim: {sys_claim}")
    A(f"F9 claim: {f9_claim}")
    A(f"H8 claim: {h8_claim}")
    changes = []
    changes.append(
        "abstract: state the system endpoint with the fused-loss GM "
        f"{fmt(a_gm)} ({a_verd['verdict']});"
        if a_verd["verdict"].startswith("SYSTEM_REVALIDATED") else
        "abstract: soften the system endpoint claim to the P8 loss-conditioned "
        "result and disclose the fused-loss revaluation;")
    changes.append(
        "credit fused-SSIM as a strong-baseline modernization (NOT our "
        "contribution, rahul-goel/fused-ssim @ a7c48d6d, padding=same, "
        "P11-validated SEMANTICALLY_MATCHED_WITHIN_FP);")
    changes.append(
        f"results: add the P12-A/P12-B fused-loss revalidation tables; report "
        f"old SepSSIM and new fused-loss GMs separately, never pooled;")
    if b_verd["verdict"].startswith("F9_REVALIDATED"):
        changes.append(
            f"results: F9 full-training effect {fmt(b_gm)} GM under the strong "
            f"baseline ({b_verd['verdict']});")
    else:
        changes.append(
            "results/limitations: disclose that F9's ~1% full-training effect "
            "is within system-level noise once the common loss bottleneck is "
            "removed;")
    changes.append(
        "limitations: A100-SXM4 cohort only; co-tenant contamination events "
        "preserved and rerun; eval SSIM metric is SepSSIM while the training "
        "loss is fused-SSIM (semantically matched within FP).")
    A("what must change in abstract/introduction/results/limitations: " +
      " ".join(changes))
    A("")
    A("OVERALL:")
    A("P12_COMPLETE")

    out = os.path.join(ART, "P12_FINAL_REPORT.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWROTE {out}")


if __name__ == "__main__":
    main()
