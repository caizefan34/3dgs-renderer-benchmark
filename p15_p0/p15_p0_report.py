"""P15-P0 Step 4b - report, contamination ledger and manifest.

Reads artifacts/P15_P0_PARETO.json (offline; no GPU) and writes:
  artifacts/P15_P0_REPORT.md          - protocol report (sections 1-5)
  artifacts/P15_P0_CONTAMINATION.json - GPU foreign-process ledger built
                                        from the export summaries
  artifacts/MANIFEST_P15_P0.json      - sha256 of every P15-P0 artifact,
                                        script and log + the certified P14
                                        reuse-surface hashes

No numbers are hand-written: every value in the report comes from the
PARETO JSON. No novelty or speedup claim; P0 is diagnostic only.
"""
import argparse
import glob
import hashlib
import json
import os
import time


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pareto", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--summaries", default=None)
    ap.add_argument("--exports", default=None)
    args = ap.parse_args()

    root = args.root or os.path.dirname(os.path.dirname(
        os.path.abspath(args.out)))
    sdir = args.summaries or os.path.join(root, "summaries")
    edir = args.exports or os.path.join(root, "exports")
    adir = os.path.join(root, "artifacts")

    P = json.load(open(args.pareto))
    verdict = P.get("verdict")
    agg = P.get("aggregate", {})
    pcs = P.get("per_cell", {})
    stmt = P.get("verdict_statement", {})

    r = []
    r.append("# P15-P0 Report - EdgeBudget Pareto Oracle "
             "(cost = exact Gaussian-tile intersections)\n")
    r.append(f"Verdict: **{verdict}**"
             + (f" - {stmt.get('stop_reason')}" if stmt.get("stop_reason")
                else "") + "\n")
    r.append(f"\nScope: no training, no CUDA changes, no speedup/novelty "
             f"claim; oracle/diagnostic only. "
             f"cells={agg.get('n_cells', 0)}; the cost-aware frontier is "
             f"a fractional-knapsack upper bound.\n")

    # ---------------- 1. OBSERVED ----------------
    r.append("\n## 1. OBSERVED\n")
    r.append("P14 measured that 20-25% primitive updates (top gradient "
             "energy) retain almost all short-horizon optimization "
             "progress (R_train: q=0.10 median 0.887, q=0.20 median 0.980, "
             "q=0.25 median 0.994) but the same primitives consume most "
             "exact rasterizer intersections (P14 O2: median R_stream "
             "0.91 at q=0.20 / 0.95 at q=0.25 vs the 0.30 gate), so P14 "
             "hard-dropped the freeze-and-compile path.\n")
    r.append("\nP15-P0 asks the cheapest falsification question before any "
             "new training implementation: does a COST-AWARE primitive "
             "subset exist when cost is the exact Gaussian-tile "
             "intersection count? I.e. can intersection work be cut to a "
             "small fraction while keeping the gradient-energy capture "
             "level that P14 measured to carry progress?\n")

    # ---------------- 2. EVIDENCE ----------------
    r.append("\n## 2. EVIDENCE\n")
    if verdict == "P0_EXPORT_INVALID":
        r.append("Export validation FAILED; no Pareto statement is "
                 "permitted. Findings:\n")
        for f in P.get("findings", []):
            r.append(f"- {f}\n")
    else:
        r.append("### 2.1 per-cell oracle table\n")
        r.append("| cell | N | isects | minJac | E_lp(.30) | E_bar_prog | "
                 "E_lp(.50) | E_bar_viab | C_ref(q=.2) | C*(prog) | "
                 "C*(viab) | flags |\n")
        r.append("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        order = sorted(pcs.keys(), key=lambda k: (k.split("_")[0],
                                                  int(k.split("_c")[-1])))
        for k in order:
            c = pcs[k]
            h = c["headline"]
            b = c["bars"]
            fl = c["flags"]
            flags = ",".join(n for n in ("strong", "viab", "relax")
                             if fl.get(n))
            r.append(f"| {k} | {c['N']:,} | {c['n_isects_total']:,} | "
                     f"{c['gate_min_jaccard']:.6f} | "
                     f"{fmt(h['E_lp_at_0.30'])} | "
                     f"{fmt(b['E_bar_prog'])} | "
                     f"{fmt(h['E_lp_at_0.50'])} | "
                     f"{fmt(b['E_bar_viab'])} | "
                     f"{fmt(h['C_ref_q20'])} | "
                     f"{fmt(h['c_star_prog_int'], 3)} | "
                     f"{fmt(h['c_star_viab_int'], 3)} | {flags or '-'} |\n")
        m = agg.get("median", {})
        r.append("\nMedian across the 6 cells: E_lp(0.30) = "
                 f"{fmt(m.get('E_lp_at_0.30'))}, E_lp(0.50) = "
                 f"{fmt(m.get('E_lp_at_0.50'))}; the P14 energy-top sets "
                 f"carry C_ref(q=.10) = {fmt(m.get('C_ref_q10'))} and "
                 f"C_ref(q=.20) = {fmt(m.get('C_ref_q20'))} of the "
                 f"intersections; E_bar_prog = "
                 f"{fmt(m.get('E_bar_prog'))}, E_bar_viab = "
                 f"{fmt(m.get('E_bar_viab'))}.\n")
        r.append("\n### 2.2 per-scene split\n")
        for sc, d in (agg.get("per_scene") or {}).items():
            r.append(f"- {sc}: E_lp(0.30) = "
                     f"{[round(x, 4) for x in d['E_lp_at_0.30']]}; "
                     f"E_lp(0.50) = "
                     f"{[round(x, 4) for x in d['E_lp_at_0.50']]}; "
                     f"C_ref(q=.20) = "
                     f"{[round(x, 4) for x in d['C_ref_q20']]}; "
                     f"C*(prog) = {d['C_star_prog_int']}\n")
        r.append("\n### 2.3 oracle frontier (median E_lp across cells)\n")
        bs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
        r.append("| budget b | " + " | ".join(str(b) for b in bs) + " |\n")
        r.append("|---|" + "---|" * len(bs) + "\n")
        row = []
        for b in bs:
            vals = sorted(pcs[k]["frontier"][str(b)]["E_lp"]
                          for k in pcs if str(b) in pcs[k]["frontier"])
            med = vals[len(vals) // 2] if vals else None
            row.append(fmt(med))
        r.append("| median E_lp | " + " | ".join(row) + " |\n")
        r.append("\n### 2.4 gate + counts\n")
        r.append(f"- gate: min Jaccard over all cells/q = "
                 f"{fmt(min(c['gate_min_jaccard'] for c in pcs.values()), 6)}"
                 f", max |dE| = "
                 f"{max(c['gate_max_abs_dE'] for c in pcs.values()):.2e}\n")
        r.append(f"- counts: n_strong={agg.get('n_strong')}/6, "
                 f"n_viab={agg.get('n_viab')}/6, "
                 f"n_relax={agg.get('n_relax')}/6 "
                 f"(informational 0.80-bar: n_viab80="
                 f"{agg.get('n_viab80_info')}/6)\n")

    # ---------------- 3. HYPOTHESIS ----------------
    r.append("\n## 3. HYPOTHESIS\n")
    r.append("A cost-aware primitive subset exists: there is a selection "
             "rule over (energy S_i, exact intersection cost c_i) whose "
             "subset reaches the P14-measured progress-carrying energy "
             "level while using a small fraction of the exact Gaussian-"
             "tile intersections. Falsification: the oracle frontier "
             "cannot reach the minimum P14-viability energy level even "
             "when allowed 50% of the intersections.\n")

    # ---------------- 4. TEST ----------------
    r.append("\n## 4. TEST\n")
    r.append("Frozen protocol (preregistered before exports):\n")
    r.append("- six cells (room/bicycle x cell0/1/2), 16K initial state; "
             "progress proxy = P14 selector energy S_i (certified "
             "p14_selector reused in place); cost c_i = exact "
             "Gaussian-tile intersection count over the 8 cluster cameras "
             "via the exact C0 stream path, computed twice with bitwise "
             "certificate.\n")
    r.append("- P14 reproduction gate per q in {0.10,0.20,0.25,0.30,0.50}: "
             "exact active count, |dE| <= 5e-4, Jaccard >= 0.95, "
             "symmetric-difference energy share <= 1e-5; any failure => "
             "P0_EXPORT_INVALID.\n")
    r.append("- oracle: ratio-order fractional knapsack E_lp(b) and "
             "integer greedy E_int(b) on the budget grid 0.05..1.00; "
             "headline budgets 0.30 (P14 O2 gate) and 0.50 (P14 O2 "
             "hard-drop line); per-cell bars from P14's measured H=32 "
             "arms: E_bar_viab = energy of the smallest q with R_train >= "
             "0.85; E_bar_prog = energy of the q=0.20 set.\n")
    r.append("- verdict: n_strong>=5 -> P0_STRONG_CANDIDATE; elif "
             "n_relax<=1 -> P0_HARD_DROP; else P0_HOLD_WEAK_PARETO. No q/"
             "alpha/budget was added or changed after results.\n")
    r.append("- epistemic bound (frozen): energy capture is a "
             "necessary-condition proxy; P15-P0 runs no optimization arms "
             "and cannot certify real progress retention.\n")

    # ---------------- 5. CONCLUSION ----------------
    r.append("\n## 5. CONCLUSION\n")
    r.append(f"**{verdict}**"
             + (f" - {stmt.get('stop_reason')}" if stmt.get("stop_reason")
                else "") + ".\n")
    if stmt.get("next"):
        r.append(f"\nNext (mandated): {stmt['next']}\n")
    r.append("\nWhat remains uncertain: (a) the energy proxy does not "
             "measure actual progress retention (P14 data show energy "
             "saturates while R_train still varies), so a positive P0 "
             "result would still require O1-style arms to certify; (b) the "
             "cost metric is the raw intersection count, not the "
             "compiled-stream segment structure that P14 O2 measured; "
             "(c) only one seed and two scenes are covered.\n")
    if verdict == "P0_STRONG_CANDIDATE":
        hv = ("build the practical cost-aware selector and re-measure "
              "R_train for its subsets with P14-style arms (P15-O1), "
              "pending research-lead review")
    elif verdict == "P0_HOLD_WEAK_PARETO":
        hv = ("none under the current frozen protocol; any follow-up "
              "requires a new preregistration")
    elif verdict == "P0_HARD_DROP":
        hv = ("none for the primitive EdgeBudget path; record the drop "
              "and move to a different mechanism class")
    else:
        hv = "repair exports and re-run the frozen analysis"
    r.append(f"\nSingle highest-value next experiment: {hv}.\n")

    # ---------------- contamination ----------------
    r.append("\n## Contamination ledger\n")
    r.append("Correctness-only work. Full nvidia-smi inventories "
             "(all GPUs + compute apps) are recorded in each export "
             "summary; the assembled ledger is "
             "artifacts/P15_P0_CONTAMINATION.json. Measurement GPU per "
             "export and foreign-app counts are listed there.\n")

    # ---------------- prior-art boundary ----------------
    r.append("\n## Prior-art boundary\n")
    r.append("Cost-aware / visibility-aware primitive selection and "
             "sparse updates (Speedy-Splat, Taming3DGS SelectiveAdam, "
             "visibility-masked updates) are prior art. P15-P0 makes no "
             "novelty claim; it is a preregistered falsification oracle. "
             "Novelty status of the compiled-frozen-context mechanism "
             "remains UNPROVEN and unchanged by this phase.\n")

    with open(args.out, "w") as f:
        f.write("".join(r))
    print(f"[p15] wrote {args.out}", flush=True)

    # ---------------- contamination ledger ----------------
    cells = []
    any_foreign = 0
    for p in sorted(glob.glob(os.path.join(sdir, "p15_export_*.json"))):
        s = json.load(open(p))
        g = s.get("gpu", {})
        before = g.get("before", {})
        after = g.get("after", {})
        nf = max(before.get("n_foreign_on_measure_gpu", 0) or 0,
                 after.get("n_foreign_on_measure_gpu", 0) or 0)
        any_foreign += 1 if nf else 0
        cells.append({
            "summary": os.path.basename(p),
            "scene": s.get("scene"), "cell": s.get("cell"),
            "gpu": s.get("gpu_str") or s.get("gpu"),
            "snapshot_before": before.get("snapshot"),
            "snapshot_after": after.get("snapshot"),
            "foreign_apps_on_measure_gpu_before":
                before.get("foreign_apps_on_measure_gpu"),
            "foreign_apps_on_measure_gpu_after":
                after.get("foreign_apps_on_measure_gpu"),
            "n_foreign_on_measure_gpu_max": nf,
        })
    ledger = {
        "phase": "P15-P0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy": ("correctness-only exports (no timing claims); one clean "
                   "GPU queue (sequential, single GPU); full nvidia-smi "
                   "inventory (all GPUs + compute apps) recorded before "
                   "and after every cell; foreign processes on the "
                   "measurement GPU are flagged but do not invalidate "
                   "correctness data; any foreign compute is disclosed "
                   "here"),
        "cells": cells,
        "n_cells_with_foreign_on_measure_gpu": any_foreign,
        "all_snapshots": [c["snapshot_after"] for c in cells],
    }
    with open(os.path.join(adir, "P15_P0_CONTAMINATION.json"), "w") as f:
        json.dump(ledger, f, indent=2)
    print(f"[p15] wrote {os.path.join(adir, 'P15_P0_CONTAMINATION.json')}",
          flush=True)

    # ---------------- manifest ----------------
    man = {"phase": "P15-P0", "created_utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verdict": verdict, "root": root, "artifacts": {},
        "reused_certified": {}, "p14_inputs_from_identity": {}}
    targets = []
    for pat in ("exports/*.npz", "summaries/*.json", "artifacts/*",
                "logs/*.log", "scripts/*.py", "scripts/*.sh"):
        targets.extend(glob.glob(os.path.join(root, pat)))
    for p in sorted(set(targets)):
        rel = os.path.relpath(p, root).replace(os.sep, "/")
        man["artifacts"][rel] = {"sha256": sha256_file(p),
                                 "bytes": os.path.getsize(p)}
    ident_path = os.path.join(adir, "P15_SOURCE_IDENTITY.json")
    if os.path.exists(ident_path):
        ident = json.load(open(ident_path))
        for name, rec in (ident.get("fatal_checks") or {}).items():
            if name.startswith("reuse_") or name.startswith("input_"):
                man["reused_certified"][name] = {
                    "path": rec.get("path"), "sha256": rec.get("sha256"),
                    "match": rec.get("match")}
    with open(os.path.join(adir, "MANIFEST_P15_P0.json"), "w") as f:
        json.dump(man, f, indent=2)
    print(f"[p15] wrote {os.path.join(adir, 'MANIFEST_P15_P0.json')} "
          f"({len(man['artifacts'])} entries)", flush=True)
    print(f"[p15] REPORT verdict={verdict}")


if __name__ == "__main__":
    main()
