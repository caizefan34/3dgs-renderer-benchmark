#!/usr/bin/env python
"""TI0 finalize: given TI0A_UPDATE_COST.json, produce all lane deliverables (fail-closed).

Downstream stages that have not run are NOT_REACHED. Verdict logic per prereg S5/S23.
Does NOT modify F0, does NOT change gates, does NOT reopen native CGT.
"""
import argparse, hashlib, json, os, time


def not_reached(reason):
    return {"status": "NOT_REACHED", "reason": reason}


def build_report(a, dirty, verdict, agg, S_B, ti0a_gate, proceed, ts, prov):
    pv = a.get("per_view", [])
    L = []
    L.append("# CGT-TI0 — Training Integration Oracle: REPORT")
    L.append("")
    L.append("- stage: TI0-A (UPDATE-COST ORACLE) complete; lane %s" % ("PROCEEDING_TO_TI0_B" if proceed else "TERMINAL_AT_TI0_A"))
    L.append("- ts: %s" % ts)
    L.append("- scene: %s | ckpt: %s | N=%s | views=%s | rank=%s | fit_iters=%s" % (
        a.get("scene"), a.get("ckpt"), a.get("n_gaussians"), a.get("views"),
        a.get("rank"), a.get("fit_iters")))
    if prov:
        L.append("- gpu_provenance: physical_gpu=%s mem_used_mib=%s util_pct=%s (clean<500MiB required, S19)" % (
            prov.get("physical_gpu"), prov.get("mem_used_mib"), prov.get("util_pct")))
    L.append("")
    L.append("## OBSERVED")
    L.append("- TI0-A per-view (footprint-stable, theta_t footprints reused at theta_t1):")
    for r in pv:
        L.append("  - view %s: footprints=%s surv=%s match=%s dirty=%s (%.1f%%) wdirty=%.1f%% shift_med=%s Tcan=%.3fms Tcgt=%.3fms refit_all=%.1fms cert_dirty=%.1fms" % (
            r.get("view"), r.get("n_footprints"), r.get("n_tiles_t1"), r.get("n_match"), r.get("n_dirty"),
            100*(r.get("dirty_fraction") or 0), 100*(r.get("weighted_dirty") or 0),
            ("%.2e" % r["teacher_shift_median"]) if r.get("teacher_shift_median") is not None else "NA",
            r.get("T_canonical_ms") or float("nan"), r.get("T_cgt_ms") or float("nan"),
            r.get("refit_all_ms") or float("nan"), r.get("cert_dirty_ms") or float("nan")))
    L.append("- aggregate: dirty_fraction=%s weighted_dirty=%s teacher_shift_median=%s teacher_shift_max=%s" % (
        agg.get("dirty_fraction"), agg.get("weighted_dirty"), agg.get("teacher_shift_median"), agg.get("teacher_shift_max")))
    L.append("- aggregate timing: T_canonical=%s ms T_cgt=%s ms T_update_A=%s T_update_B=%s T_update_C=%s" % (
        agg.get("T_canonical_ms"), agg.get("T_cgt_ms"), agg.get("T_update_A_full_ms"),
        agg.get("T_update_B_dirty_ms"), agg.get("T_update_C_cert_ms")))
    L.append("- ratios: R_update_A=%s R_update_B=%s R_update_C=%s | S_effective_A=%s S_effective_B=%s S_effective_C=%s" % (
        agg.get("R_update_A"), agg.get("R_update_B"), agg.get("R_update_C"),
        agg.get("S_effective_A"), agg.get("S_effective_B"), agg.get("S_effective_C")))
    L.append("")
    L.append("## EVIDENCE")
    L.append("- TI0-A gate (S5): S_effective_B=%s -> %s (thresholds: PASS>=1.15, MARGINAL>=1.0, DROP<=1.0)" % (S_B, ti0a_gate))
    L.append("- dirty-work oracle (S6): OLD coeffs (route-A L-BFGS-B fit at theta_t) re-certified against NEW teacher at theta_t1 after one real canonical optimizer step; failed=dirty.")
    L.append("- T_update_A=full per-tile refit; T_update_B=certify-all + refit-dirty-only (realistic); T_update_C=certify-all + reuse-passing.")
    L.append("- T_canonical/T_CGT = F0 fused vs teacher kernel on SAME per-tile scope (inherits frozen F0; matches upstream ~3x).")
    L.append("")
    L.append("## INFERENCE")
    if proceed:
        L.append("- Update path has >=1.15x effective forward headroom -> justify TI0-B (forward+backward composition).")
    elif ti0a_gate == "MARGINAL_UPDATE_HEADROOM":
        L.append("- Update path marginal (1.0<=S_effective_B<1.15); proceed only if one obvious low-risk removable overhead exists (S5).")
    else:
        L.append("- Update/rebuild cost erases the F0 forward gain (S_effective_B<=1.0) -> DROP_UPDATE_COST, close integration lane (S5).")
    L.append("- weighted_dirty indicates whether incremental update (strategy B/C) can help: high weighted dirty => refit dominates.")
    L.append("")
    L.append("## HYPOTHESIS")
    L.append("- If weighted_dirty is low and teacher_shift < cert gate (0.02), CGT state is largely reusable across steps -> incremental update viable.")
    L.append("- If weighted_dirty ~1.0 every step, incremental update unlikely to help; full refit required.")
    L.append("")
    L.append("## NOT_MEASURED")
    L.append("- TI0-B (forward+backward composition), TI0-C (short time-to-quality): %s" % verdict.get("downstream_status"))
    L.append("- Peak VRAM / DRAM traffic / L2 / occupancy / launch count: NOT_MEASURED (S20; requires nsys/pynvml profiling run).")
    L.append("- Projected end-to-end training speedup (Amdahl): NOT_COMPUTED until TI0-B measured renderer/training fraction.")
    L.append("")
    L.append("## CONCLUSION")
    L.append("- TI0-A verdict: %s (S_effective_B=%s). Final verdict: %s" % (ti0a_gate, S_B, verdict.get("final_verdict")))
    L.append("- F0 forward speed is NOT substituted for training speed; update + backward + recompute costs accounted separately (S24).")
    L.append("- Downstream artifacts are NOT_REACHED until their stages execute on a clean GPU.")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ti0a", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ts", default=time.strftime("%Y-%m-%d %H:%M:%S"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    a = json.load(open(args.ti0a))
    agg = a.get("aggregate", {})
    gate = a.get("gate", {})
    S_B = gate.get("S_effective_B", agg.get("S_effective_B"))
    pv = a.get("per_view", [])
    prov = a.get("gpu_provenance", None)

    dirty = {
        "stage": "TI0-A-DIRTY-WORK", "ts": args.ts, "source": os.path.basename(args.ti0a),
        "ckpt": a.get("ckpt"), "n_gaussians": a.get("n_gaussians"), "views": a.get("views"),
        "gpu_provenance": prov,
        "per_view": [{k: r.get(k) for k in ("view","n_footprints","n_tiles_t","n_tiles_t1","n_match",
                       "n_dirty","dirty_fraction","M_dirty","M_total","weighted_dirty",
                       "teacher_shift_median","teacher_shift_max","cert_t_ms","cert_dirty_ms")} for r in pv],
        "aggregate": {"dirty_fraction": agg.get("dirty_fraction"), "weighted_dirty": agg.get("weighted_dirty"),
                      "teacher_shift_median": agg.get("teacher_shift_median"), "teacher_shift_max": agg.get("teacher_shift_max")},
        "notes": ("Dirty-work oracle (S6): per footprint, certify OLD coeffs (route-A L-BFGS-B fit at theta_t) against NEW "
                  "teacher at theta_t1 after one real canonical optimizer step. passed=reusable, failed=dirty. teacher_shift = "
                  "max|teacher_field(theta_t1)-teacher_field(theta_t)| per footprint (CPU, deterministic). weighted_dirty = sum M "
                  "over dirty / sum M over matched. Footprint-stable matching (theta_t footprints reused at theta_t1). Cert gate "
                  "GATE_MAX_ABS=0.02 inherited from frozen upstream; not changed."),
    }
    json.dump(dirty, open(os.path.join(args.outdir, "TI0A_DIRTY_WORK.json"), "w"), indent=2)

    if S_B is None:
        ti0a_gate = "INCONCLUSIVE"; final_verdict = "BLOCKED_GPU_OCCUPIED"; proceed = False
    elif S_B >= 1.15:
        ti0a_gate = "PASS_TI0_B"; final_verdict = None; proceed = True
    elif S_B >= 1.0:
        ti0a_gate = "MARGINAL_UPDATE_HEADROOM"; final_verdict = "MARGINAL_UPDATE_HEADROOM"; proceed = False
    else:
        ti0a_gate = "DROP_UPDATE_COST"; final_verdict = "DROP_UPDATE_COST"; proceed = False

    downstream_status = "PENDING_TI0_B_CLEAN_GPU" if proceed else "NOT_REACHED"
    reason = ("TI0-A S_effective_B=%s >= 1.15 -> proceed to TI0-B (forward+backward composition) on a clean GPU." % S_B) if proceed else \
             ("TI0-A gate=%s (S_effective_B=%s); downstream not run (fail-closed)." % (ti0a_gate, S_B))

    verdict = {
        "stage": "TI0-VERDICT", "ts": args.ts, "scene": a.get("scene"),
        "ti0a_S_effective_B": S_B, "ti0a_gate": ti0a_gate, "final_verdict": final_verdict,
        "proceed_to_ti0_b": proceed, "downstream_status": downstream_status, "downstream_reason": reason,
        "allowed_verdicts": ["PASS_TO_CGT_TI1","DROP_UPDATE_COST","MARGINAL_UPDATE_HEADROOM",
                             "DROP_TRAINING_COMPOSITION","DROP_TRAINING_QUALITY","DROP_TIME_TO_QUALITY",
                             "BLOCKED_GPU_OCCUPIED","BLOCKED_SOURCE_INTEGRITY"],
        "gpu_provenance": prov,
        "notes": ("Final verdict is terminal only when the lane stops. If TI0-A passes, final_verdict=PENDING until TI0-B/C run; "
                  "TI0-B/TI0-C/TI0-MEMORY remain NOT_REACHED until their stages execute. Historical upstream verdicts unchanged."),
    }
    json.dump(verdict, open(os.path.join(args.outdir, "TI0_VERDICT.json"), "w"), indent=2)

    nr_reason = reason
    for name in ("TI0B_COMPOSITION_TIMING","TI0B_PROFILE_BREAKDOWN","TI0C_SHORT_TRAINING","TI0C_TIME_TO_QUALITY"):
        json.dump({"stage": name, "ts": args.ts, **not_reached(nr_reason)},
                  open(os.path.join(args.outdir, name + ".json"), "w"), indent=2)
    mem = {"stage":"TI0_MEMORY_ACCOUNTING","ts":args.ts,"status":"NOT_MEASURED",
           "reason":"Peak VRAM / DRAM traffic not captured by TI0-A harness; requires a profiling run (nsys/pynvml). NOT_MEASURED per S20."}
    json.dump(mem, open(os.path.join(args.outdir, "TI0_MEMORY_ACCOUNTING.json"), "w"), indent=2)

    report = build_report(a, dirty, verdict, agg, S_B, ti0a_gate, proceed, args.ts, prov)
    open(os.path.join(args.outdir, "TI0_REPORT.md"), "w").write(report)

    files = ["TI0_PREREG.json","TI0_SOURCE_AUDIT.json","TI0A_UPDATE_COST.json","TI0A_DIRTY_WORK.json",
             "TI0B_COMPOSITION_TIMING.json","TI0B_PROFILE_BREAKDOWN.json","TI0C_SHORT_TRAINING.json",
             "TI0C_TIME_TO_QUALITY.json","TI0_MEMORY_ACCOUNTING.json","TI0_VERDICT.json","TI0_REPORT.md"]
    lines = []
    for fn in files:
        fp = os.path.join(args.outdir, fn)
        if os.path.exists(fp):
            lines.append("%s  %s" % (hashlib.sha256(open(fp,"rb").read()).hexdigest(), fn))
    open(os.path.join(args.outdir, "MANIFEST.sha256"), "w").write("\n".join(lines) + "\n")
    print("finalize: ti0a_gate=%s S_effective_B=%s proceed=%s final_verdict=%s" % (ti0a_gate, S_B, proceed, final_verdict))
    print("wrote 11 deliverables + MANIFEST.sha256 to", args.outdir)


if __name__ == "__main__":
    main()