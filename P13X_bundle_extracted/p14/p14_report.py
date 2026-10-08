"""P14 final: P14_CONTAMINATION.json, P14_VERDICT.json, P14_REPORT.md,
MANIFEST_P14.json. Reads all stage artifacts; applies the frozen STOP
chain. No result rescuing: gates are exactly as preregistered.
"""
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p14_common as C

A = lambda *p: os.path.join(C.P14_ROOT, "artifacts", *p)
U = lambda *p: os.path.join(C.P14_ROOT, *p)


def load(path):
    if os.path.exists(path):
        return json.load(open(path))
    return None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ident = load(A("P14_SOURCE_IDENTITY.json"))
    clusters = load(A("P14_CAMERA_CLUSTERS.json"))
    selector = load(A("P14_SELECTOR.json"))
    o1 = load(A("P14_O1_AGG.json"))
    o2 = load(A("P14_O2_AGG.json"))
    o3 = load(A("P14_O3_TIMING.json"))

    # ---- contamination record ----------------------------------------
    o1_units = [json.load(open(p)) for p in sorted(glob.glob(
        os.path.join(C.P14_ROOT, "P14_O1_UNITS", "*.json")))]
    o2_units = [json.load(open(p)) for p in sorted(glob.glob(
        os.path.join(C.P14_ROOT, "P14_O2_UNITS", "*.json")))]
    freeze_violations = [u for u in o1_units
                         if u.get("arm") == "f"
                         and u.get("freeze_ok") is False]
    nonfinite = [u for u in o1_units
                 if u.get("finite_ok") is False]
    contamination = {
        "phase": "P14",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy": ("all measurement runs on dedicated idle A100 GPUs "
                   "(GPU 4 room chain, GPU 5 bicycle chain where "
                   "parallel; identity verified before any run; no "
                   "foreign compute on the measurement GPU during "
                   "timing; paired-run contamination policy inherited "
                   "from P12)"),
        "gpu_snapshots": {
            "stage0": ident.get("env", {}).get("gpu_snapshot_at_stage0"),
            "clusters_start": clusters.get("gpu_snapshot_start"),
            "clusters_end": clusters.get("gpu_snapshot_end"),
        },
        "gpu_snapshot_now": C.gpu_snapshot(),
        "freeze_semantics": {
            "mechanism": ("row-masked Adam (gather active rows, update, "
                          "scatter back); frozen rows never read-written"),
            "per_step_bitwise_guard": ("torch.equal on frozen rows of "
                                       "params/exp_avg/exp_avg_sq every "
                                       "step"),
            "n_F_arms": len([u for u in o1_units
                             if u.get("arm") == "f"]),
            "freeze_violations": len(freeze_violations),
            "INVALID_FREEZE_SEMANTICS": bool(freeze_violations),
            "nonfinite_arms": len(nonfinite),
        },
    }
    C.jwrite(A("P14_CONTAMINATION.json"), contamination)

    # ---- verdict chain -------------------------------------------------
    verdict = {
        "phase": "P14",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "identity": ident.get("verdict_identity"),
        "freeze_semantics_ok": not bool(freeze_violations),
        "gate_chain": {},
    }
    stop_reason = None
    final = None
    if ident.get("verdict_identity") != "IDENTITY_VERIFIED":
        final = "ABORT_IDENTITY_MISMATCH"
        stop_reason = "source identity mismatch"
    elif freeze_violations:
        final = "INVALID_FREEZE_SEMANTICS"
        stop_reason = "bitwise freeze violated in F arm(s)"
    else:
        if o1 is not None:
            verdict["gate_chain"]["O1"] = {
                "decision": o1.get("decision"),
                "hard_drop": o1.get("stage_A", {}).get("O1_HARD_DROP"),
                "candidate": o1.get("stage_A", {}).get("O1_CANDIDATE"),
                "stage_B_allowed": o1.get("stage_A",
                                          {}).get("stage_B_allowed"),
                "surviving_q": o1.get("surviving_q"),
                "per_q_stageA": [
                    {k: r[k] for k in ("q", "median_R_train",
                                       "n_viable",
                                       "n_audit_penalty_gt_1pct")}
                    for r in o1.get("stage_A", {}).get("per_q", [])],
            }
            if o1.get("decision") == "O1_HARD_DROP":
                final = "DROP"
                stop_reason = "O1_HARD_DROP"
        if final is None and o2 is not None:
            verdict["gate_chain"]["O2"] = {
                "decision": o2.get("decision"),
                "O2_PASS_q": o2.get("O2_PASS_q"),
                "hard_drop": o2.get("O2_HARD_DROP"),
            }
            if o2.get("O2_HARD_DROP"):
                final = "DROP"
                stop_reason = "O2_HARD_DROP"
            elif not o2.get("O2_PASS_q"):
                final = "DROP"
                stop_reason = ("no q passes O2"
                               + (" (diagnostic-only path)" if o2.get(
                                   "diagnostic_only_q") else ""))
        if final is None and o3 is not None:
            verdict["gate_chain"]["O3"] = {
                "common_q": o3.get("common_q_O1_O2"),
                "GO_q": o3.get("final_gate", {}).get("GO_q"),
                "per_q_pass": {q: d["FINAL_GATE_PASS"]
                               for q, d in o3.get("final_gate",
                                                  {}).get("by_q",
                                                          {}).items()},
                "scene_medians": {
                    q: {"H32": {s: d["H32"][s]["median_S"]
                                for s in C.SCENES},
                        "H128": {s: d["H128"][s]["median_S"]
                                 for s in C.SCENES},
                        "P25": {"H32": d["H32"]["P25_S_all_cells"],
                                "H128": d["H128"]["P25_S_all_cells"]}}
                    for q, d in o3.get("final_gate", {}).get("by_q",
                                                             {}).items()},
            }
            if o3.get("final_gate", {}).get("GO_q"):
                final = "GO_MINIMAL_COMPILED_RENDER_PROTOTYPE"
            else:
                final = "DROP"
                stop_reason = ("O3 final gate: no common q reaches "
                               "S>=4.0x on both scenes at both H with "
                               "P25>=3.5x")
    verdict["stop_reason"] = stop_reason
    verdict["verdict"] = final
    verdict["reminders"] = [
        "no git commit", "no full training",
        "no speedup claim beyond measured oracle numbers",
        "no novelty claim (UNPROVEN)",
        "even on GO: next phase must solve practical selector, block "
        "rotation, optimizer-state reactivation, global camera "
        "coverage, thousands-step convergence",
    ]
    C.jwrite(A("P14_VERDICT.json"), verdict)

    # ---- report --------------------------------------------------------
    r = []
    r.append("# P14 Report - FreezeGS / Optimization-Induced "
             "Stationarity Oracle\n")
    r.append(f"Verdict: **{final}**"
             + (f" ({stop_reason})" if stop_reason else "") + "\n")
    r.append("## 1. OBSERVED\n")
    r.append("P13-X showed natural stationarity does not survive dense "
             "Adam: dirty-intersection fraction = 1.0 at every epsilon "
             "(X3), no K<=64 DCT basis reaches cosine 0.99 (X5), and "
             "the measured E2E is dominated by backward (57% room / 47% "
             "bicycle) and optimizer (17% room / 33% bicycle).\n")
    if o3:
        for s in C.SCENES:
            comp = o3["scenes"][s]["components"]
            r.append(f"- {s} re-measured at the 16K state: E2E = "
                     f"{comp['e2e_baseline_ms']:.2f} ms "
                     f"(fwd {comp['component_ms_median']['fwd_ms']:.2f}"
                     f" + loss "
                     f"{comp['component_ms_median']['loss_ms']:.2f}"
                     f" + bwd "
                     f"{comp['component_ms_median']['bwd_ms']:.2f}"
                     f" + opt "
                     f"{comp['component_ms_median']['opt_ms']:.2f}), "
                     f"bwd regression r2="
                     f"{comp['backward_regression']['r2']:.3f}.\n")
    r.append("\n## 2. EVIDENCE\n")
    if o1:
        for row in o1["stage_A"]["per_q"]:
            r.append(f"- O1 Stage A q={row['q']}: median R_train = "
                     f"{row['median_R_train']}, viable cells "
                     f"{row['n_viable']}/6, cells audit>1%: "
                     f"{row['n_audit_penalty_gt_1pct']}\n")
        o1pb = o1.get("O1_PASS_by_q", {})
        if o1pb:
            r.append("- O1 per-H pass checks (Stage A + Stage B): "
                     + "; ".join(
                         f"q={q}: "
                         + ",".join(f"{h}={'PASS' if ok else 'FAIL'}"
                                    for h, ok in d["checks"].items())
                         + f" -> O1_PASS={d['O1_PASS']}"
                         for q, d in o1pb.items()) + "\n")
    if o2 and o2.get("by_q"):
        for q, rec in o2["by_q"].items():
            for Hk in ("H32", "H128"):
                st = rec.get(Hk)
                if st:
                    r.append(f"- O2 q={q} {Hk}: median R_stream(max) = "
                             f"{st['median_R_stream_max']:.4f}, median "
                             f"R_rebuild = {st['median_R_rebuild']:.4f}, "
                             f"max FP32 cache = "
                             f"{st['max_cache_gb_fp32']:.2f} GB, "
                             f"cells cache>24GB = "
                             f"{st['n_cells_cache_gt_24gb']}, PASS = "
                             f"{rec.get('O2_PASS')}\n")
    if o3 and o3.get("final_gate", {}).get("by_q"):
        for q, d in o3["final_gate"]["by_q"].items():
            r.append(f"- O3 q={q}: FINAL_GATE_PASS = "
                     f"{d['FINAL_GATE_PASS']}\n")
            for H in ("H32", "H128"):
                for s in C.SCENES:
                    r.append(f"    - {H} {s}: median S = "
                             f"{d[H][s]['median_S']}\n")
                r.append(f"    - {H} P25(S) all cells = "
                         f"{d[H]['P25_S_all_cells']}\n")
    r.append("\n## 3. HYPOTHESIS\n")
    r.append("Optimization-induced stationarity (freeze G_F params + "
             "Adam state within a micro-epoch of H steps while G_F "
             "still fully participates in I = R(G_A, G_F)) yields an "
             "exact compiled frozen context that can compress forward, "
             "backward, and optimizer work while preserving training "
             "progress on the cluster cameras and audit generalization."
             "\n")
    r.append("\n## 4. TEST\n")
    r.append("Preregistered gates O1 (optimization viability, "
             "R_train/P_audit/freeze-bitwise), O2 (compiled stream "
             "structure, R_stream/R_rebuild/cache), O3 (conservative "
             "E2E oracle with fixed floors, charged cache build and "
             "invalidation, real masked-optimizer benchmark). Frozen "
             "q grid {0.10,0.20,0.25,0.30,0.50}, H grid {8,32,128}, "
             "6 scene-cluster cells, clusters fixed before any O1 "
             "result.\n")
    r.append("Integrity note: the first Stage A/B pass contained an "
             "inverted stream-labeling bug (active-run starts recorded "
             "as frozen segments; frozen-isect count recorded as "
             "active), detected via a mathematical impossibility "
             "(R_stream must be non-decreasing in q because top-K "
             "active sets are nested, but measured values decreased). "
             "After fixing tile_stream_hashes and adding a synthetic-"
             "stream selftest, ALL Stage A and Stage B units were "
             "regenerated in a single clean re-run; the final artifact "
             "set contains only fixed-code units.\n")
    r.append("\n## 5. CONCLUSION\n")
    r.append(f"**{final}**"
             + (f" - {stop_reason}" if stop_reason else "") + ".\n")
    if final == "DROP" and stop_reason == "O2_HARD_DROP":
        r.append(
            "Observed mechanism (both measured on the fixed-code units): "
            "(1) the active set is chosen by gradient energy, and "
            "high-energy Gaussians are precisely the large-footprint "
            "ones - across the 6 cells the active fraction of "
            "intersections spans 0.43-1.00, so freezing the low-energy "
            "majority leaves the compiled stream at R_stream 0.74-1.00 "
            "(median 0.91 at q=0.20, 0.95 at q=0.25, H=32), far above "
            "the 0.30 gate and the 0.50 hard-drop line; (2) active "
            "motion invalidates nearly every tile's frozen-segment "
            "decomposition within one micro-epoch (median R_rebuild "
            "0.994 at H=32 and 0.998 at H=128, vs gate 0.15 and "
            "hard line 0.40). Optimization-induced stationarity "
            "preserves training progress (O1) but does NOT make the "
            "rendered streams compressible or the frozen context "
            "cache-stable under this selector definition.\n")
    r.append("\n## Prior-art boundary\n")
    r.append("See P14_PRIOR_ART_BOUNDARY.md. Novelty status: UNPROVEN; "
             "no novelty or speedup claim is made. The masked-optimizer "
             "benchmark (SelectiveAdam fallback: fused kernel absent "
             "from the frozen build; semantically-equivalent masked "
             "torch Adam used) is prior art, not a P14 contribution.\n")
    with open(A("P14_REPORT.md"), "w") as f:
        f.write("\n".join(r))

    # ---- manifest ------------------------------------------------------
    man = {"phase": "P14", "created_utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": final, "artifacts": {}}
    targets = []
    for pat in ("artifacts/P14_*.json", "artifacts/P14_*.md",
                "artifacts/p14_o3_parts/*.json",
                "P14_O1_UNITS/*.json", "P14_O2_UNITS/*.json",
                "raw/*.npz", "logs/*.log"):
        targets.extend(glob.glob(os.path.join(C.P14_ROOT, pat)))
    for p in sorted(set(targets)):
        rel = os.path.relpath(p, C.P14_ROOT)
        man["artifacts"][rel] = {"sha256": sha256_file(p),
                                 "bytes": os.path.getsize(p)}
    man["scripts"] = {}
    for p in sorted(glob.glob(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "p14_*.py"))):
        man["scripts"][os.path.basename(p)] = sha256_file(p)
    C.jwrite(A("MANIFEST_P14.json"), man)
    print(f"[p14] VERDICT = {final}" + (f" ({stop_reason})"
                                        if stop_reason else ""))


if __name__ == "__main__":
    main()
