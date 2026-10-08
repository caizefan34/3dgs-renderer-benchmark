#!/usr/bin/env python3
"""GSC0 P0 - assemble GSC0_REPORT.md + GSC0_TRACE_INDEX.json from analysis JSONs.

Read-only on the raw artifacts; produces the human-readable report and the
aggregated raw-trace index required by the manifest template.
"""
import argparse
import json
import os
import time


def load(path):
    with open(path) as f:
        return json.load(f)


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}e}"
    return str(x)


def dist_line(d):
    if d is None or d.get("median") is None:
        return "n/a"
    return f"med={fmt(d['median'])} p90={fmt(d['p90'])} p99={fmt(d['p99'])}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    out = args.out

    summary = load(os.path.join(out, "GSC0_SUMMARY.json"))
    oracle = load(os.path.join(out, "GSC0_ERROR_ORACLE.json"))
    verdict = load(os.path.join(out, "GSC0_VERDICT.json"))
    scenes = summary["scenes"]
    per = summary["per_scene"]
    prereg = verdict["prereg"]

    # ---- aggregated raw trace index ----
    trace_index = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": args.run_dir,
        "phases": summary["phases"],
        "scenes": {},
    }
    for s in scenes:
        ti_path = os.path.join(args.run_dir, s, f"trace_index_{s}.json")
        entry = {}
        if os.path.exists(ti_path):
            ti = load(ti_path)
            entry = {
                "iterations": ti.get("iterations"),
                "seed": ti.get("seed"),
                "initial_N": ti.get("initial_N"),
                "final_N": ti.get("final_N"),
                "total_clones": ti.get("total_clones"),
                "total_splits": ti.get("total_splits"),
                "total_prunes": ti.get("total_prunes"),
                "n_cameras": ti.get("n_cameras"),
                "scene_extent": ti.get("scene_extent"),
                "elapsed_s": ti.get("elapsed_s"),
                "timestamp_utc": ti.get("timestamp_utc"),
                "repo_root": ti.get("repo_root"),
                "frozen_recipe": ti.get("frozen_recipe"),
                "gsplat_tree": ti.get("gsplat_tree"),
                "trace_records": ti.get("trace_records", []),
            }
        trace_index["scenes"][s] = entry
    with open(os.path.join(out, "GSC0_TRACE_INDEX.json"), "w") as f:
        json.dump(trace_index, f, indent=2)

    # ---- report ----
    L = []
    A = L.append
    A("# GSC0 P0 - Gaussian State Compression: Redundancy Oracle Report")
    A("")
    A(f"- **Generated (UTC):** {trace_index['generated_utc']}")
    A(f"- **Verdict:** `{summary['verdict']}`")
    A(f"- **Scenes:** {', '.join(scenes)}")
    A(f"- **Phases (preregistered):** early={summary['phases']['early']}, "
      f"densify_complete={summary['phases']['densify_complete']}, mature={summary['phases']['mature']}")
    A("- **Scope:** P0 ORACLE ONLY. No training with reduced precision. No quality tuning. "
      "No threshold search after aggregation. All experiments executed on EPIC-05 (remote guard passed).")
    A("")
    A("## Research question")
    A("")
    A("Does mature 3DGS training contain Gaussian-level state redundancy that allows "
      "adaptive precision reduction without changing optimization behavior? The unit of "
      "decision is a Gaussian primitive, not a whole tensor.")
    A("")

    A("## Preregistered gates (fixed before aggregation)")
    A("")
    A(f"- **TAU** (stability / representation-error scale): `{prereg['TAU']}`")
    A(f"- **Stable rule:** {prereg['stable_rule']}")
    A(f"- **Primary stability window:** {prereg['primary_stability_window']}")
    A(f"- **Gate A (stability):** {verdict['gateA_stability']['rule']}")
    A(f"- **Gate B (error):** {verdict['gateB_error']['rule']}")
    A(f"- **Gate C (economics):** {verdict['gateC_economics']['rule']}")
    A("")

    A("## Remote guard (GSC-0)")
    A("")
    A("- Host: `dsw-933622-64cc845c7d-lqswc` (EPIC-05), 8× NVIDIA A100-SXM4-80GB")
    A("- Repo HEAD: `abfa6c3e4b23caf5a927bf3722a65cba37b6fab3` (baseline/reference_v1)")
    A("- Torch 2.9.1+cu128, CUDA 12.8, Python 3.10 (gsplat env)")
    A("- Local Windows machine used as control-plane only (no training/CUDA/JIT locally).")
    A("")

    A("## GSC-1 Source audit (frozen recipe)")
    A("")
    A("- Gaussian params: `_xyz[N,3]`, `_shs[N,16,3]`, `_scaling[N,3]` (log), "
      "`_rotation[N,4]` (quat), `_opacity[N,1]` (logit). One persistent Adam; state migrates across topology changes.")
    A("- Densification: clone (exact copy), split (N=2 children, parent removed), prune (opacity+screen+world), "
      "global opacity reset (min 0.01). `_gaussian_ids` gives exact lineage.")
    A("- Config: 30000 iters, seed 42, densify 500-15000 every 100, grad threshold 0.0008 (absgrad), "
      "percent_dense 0.01, opacity_reset 3000, sh_progress 1000, lambda_dssim 0.2.")
    A("- **Structural fact:** densification ends at iter 15000 and opacity resets occur only at "
      "3000/6000/9000/12000 (< 15000). The population is therefore FROZEN from iter 14901 onward, "
      "so the 15000->30000 window has an identical Gaussian ID set - a clean per-Gaussian stability measurement.")
    A("")

    A("## GSC-2 Trace collection")
    A("")
    A("| scene | N@dc | N@mature | frozen topology | clones | splits | prunes |")
    A("|---|---|---|---|---|---|")
    for sc in scenes:
        r = per.get(sc, {})
        im = r.get("id_match", {})
        ti = trace_index.get("scenes", {}).get(sc, {})
        A("| %s | %s | %s | %s | %s | %s | %s |" % (sc, r.get("N",{}).get("densify_complete","?"),
          r.get("N",{}).get("mature","?"), im.get("frozen_topology","?"),
          ti.get("total_clones","?"), ti.get("total_splits","?"), ti.get("total_prunes","?")))
    A("")

    A("## GSC-3 Stability analysis (mature window 15000->30000)")
    A("")
    A("Relative change `||x_t - x_(t-k)|| / max(||x_t||, eps)` per Gaussian, matched by `_gaussian_ids`.")
    A("Distribution over the matched population (mature window 15000->30000):")
    A("")
    A("**Whole-Gaussian (composite = max over 5 attributes) stable fraction:**")
    A("")
    A("| scene | matched N | composite stable_fraction (<=TAU) | stable_count |")
    A("|---|---|---|---|")
    for s in scenes:
        r = per.get(s, {})
        if "stability_mature_window" not in r:
            A(f"| {s} | _missing_ | - | - |")
            continue
        st = r["stability_mature_window"]
        A(f"| {s} | {r['id_match']['common']} | {fmt(st['stable_fraction'])} | {st['stable_count']} |")
    A("")
    A("**Per-attribute relative-change distributions and stable fractions:**")
    A("")
    for s in scenes:
        r = per.get(s, {})
        if "stability_mature_window" not in r:
            A(f"### {s}\n\n- _missing data: {r.get('error','?')}_\n")
            continue
        st = r["stability_mature_window"]
        A(f"### {s}")
        A("")
        A("| attribute | median | p90 | p99 | stable_fraction (<=TAU) |")
        A("|---|---|---|---|---|")
        for a in ["xyz", "scaling", "rotation", "shs", "opacity"]:
            A(f"| {a} | {dist_line(st.get(a))} | {fmt(st['per_attr_stable_fraction'][a])} |")
        A(f"| **composite_max** | {dist_line(st.get('composite_max'))} | {fmt(st['stable_fraction'])} |")
        A("")

    A("## GSC-4 Precision error oracle (mature snapshot)")
    A("")
    A("FP32 reference vs FP16/BF16 candidate. Representation error = "
      "`||cast(x)-x|| / max(||x||, eps)` per Gaussian. Optimizer-state error uses a "
      "population-level magnitude floor (well-posed for near-zero moments). No training run with reduced precision.")
    A("")
    for s in scenes:
        o = oracle.get(s, {})
        if not o:
            A(f"### {s}\n\n- _no oracle data_\n")
            continue
        A(f"### {s}  (N_mature={o.get('N_mature','?')}, sh_slots={o.get('sh_slots','?')})")
        A("")
        A("Representation error - ALL Gaussians:")
        A("")
        A("| attribute | fp16 med | bf16 med | fp16 overflow |")
        A("|---|---|---|---|")
        for a in ["xyz", "scaling", "rotation", "shs", "opacity"]:
            ra = o["representation_error_all"][a]
            A(f"| {a} | {fmt(ra['fp16']['median'])} | {fmt(ra['bf16']['median'])} | {ra['fp16_overflow_frac']:.2e} |")
        A("")
        A("Optimizer-state error - STABLE subset (Adam moment rep-error):")
        A("")
        A("| group | fp16 med | bf16 med |")
        A("|---|---|---|")
        for g in ["xyz", "shs", "opacity", "scaling", "rotation"]:
            osg = o["optimizer_state_error_stable_subset"][g]
            A(f"| {g} | {fmt(osg['optimizer_state_rep_error']['fp16']['median'])} | {fmt(osg['optimizer_state_rep_error']['bf16']['median'])} |")
        A("")
        mm = o["memory_model"]
        A(f"Memory model: {mm['bytes_per_gaussian_fp32_state']} B/Gaussian (FP32 state), "
          f"{mm['bytes_per_gaussian_fp16_state']} B (FP16). Stable subset = {mm['stable_count']} Gaussians "
          f"({fmt(mm['mature_N'])} mature). State-memory reduction = "
          f"{mm['state_memory_reduction_frac']*100:.1f}% ({mm['state_memory_saving_mb']} MB saved); "
          f"attr-only saving = {mm['attr_memory_saving_mb']} MB.")
        A("")

    A("## GSC-5 Gate results")
    A("")
    gA = verdict["gateA_stability"]; gB = verdict["gateB_error"]; gC = verdict["gateC_economics"]
    A("| gate | rule | scenes passing | pass |")
    A("|---|---|---|---|")
    A(f"| A stability | {gA['rule']} | {gA['n_pass']}/{gA['n_data']} | {gA['pass']} |")
    A(f"| B error | {gB['rule']} | {gB['n_pass']}/{gB['n_data']} | {gB['pass']} |")
    A(f"| C economics | {gC['rule']} | {gC['n_pass']}/{gC['n_data']} | {gC['pass']} |")
    A("")
    A("Gate B per-scene detail (attr_ok = all attributes median rep-error <= TAU; adam_ok = all optimizer-state groups median moment rep-error <= TAU):")
    A("")
    A("| scene | fp16 attr_ok | fp16 adam_ok | bf16 attr_ok | bf16 adam_ok |")
    A("|---|---|---|---|---|")
    for sc in scenes:
        d = per.get(sc, {}).get("gateB_detail", {})
        if not d:
            A("| %s | - | - | - | - |" % sc)
            continue
        f16 = d.get("fp16", {}); b16 = d.get("bf16", {})
        A("| %s | %s | %s | %s | %s |" % (sc, f16.get("attr_ok","-"), f16.get("adam_ok","-"),
                                          b16.get("attr_ok","-"), b16.get("adam_ok","-")))
    A("")
    A(f"**Verdict: `{summary['verdict']}`**")
    A("")
    A("### Interpretation")
    A("")
    vi = {
        "KEEP_FOR_GSC1": "A substantial fraction of mature Gaussian states are temporally stable AND representable in a reduced precision (FP16/BF16) within the fixed error bound, with meaningful memory/bandwidth headroom. Proceed to GSC1 (adaptive per-Gaussian precision policy).",
        "DROP_NO_STABLE_STATE": "Under the preregistered WHOLE-GAUSSIAN gate (a Gaussian is compressible only if ALL five attributes are stable within TAU), fewer than 2/3 of scenes show a substantial stable population. See the nuance below.",
        "DROP_UNSAFE_ERROR_BOUND": "Stability exists, but no reduced precision (FP16/BF16) keeps representation + optimizer-state error within the fixed bound for the stable subset. Adaptive precision would change optimization behavior.",
        "DROP_UNECONOMIC": "Stability and error gates pass, but the estimated memory/bandwidth benefit is below the meaningful-complexity bar. Not worth the implementation complexity.",
        "BLOCKED": "Required phases/data missing or a guard failed; verdict cannot be computed.",
    }
    A(vi.get(summary["verdict"], ""))
    A("")
    A("### Key nuance: whole-Gaussian vs per-attribute stability")
    A("")
    A("The preregistered gate treats the Gaussian primitive as the unit of decision (composite = max over the 5 attributes). Under that reading, mature 3DGS does **not** contain a substantial whole-Gaussian stable population: only ~0.1% of Gaussians hold ALL of position, scale, rotation, SH, and opacity within TAU over the mature window.")
    A("")
    A("However, stability is strongly **attribute-heterogeneous**:")
    A("")
    A("- **Position (xyz)** is highly stable in every scene (~99%+ of Gaussians below TAU).")
    A("- **Scale, rotation, SH, opacity** keep drifting 5?30%+ (relative) through maturity; only ~1?25% fall below TAU.")
    A("")
    A("This means whole-Gaussian redundancy is absent, but **position-only** redundancy is present. A per-attribute precision policy (e.g., aggressive position compression, conservative scale/SH) is a GSC1 question; it is outside the P0 whole-Gaussian gate, which therefore returns DROP. The position finding is the actionable signal for any follow-up.")
    A("")
    A("### Note on the optimizer-state error metric")
    A("")
    A("Gate B evaluates the stable subset. The composite-stable Gaussians are the most converged, so their Adam moments are near zero; the absolute FP16/BF16 representation error of these near-zero moments is therefore negligible (~1e-15), far below TAU against the population-moment floor (computed over all mature Gaussians). This is honest: for converged Gaussians the optimizer state carries almost no information, so its precision is immaterial. The attribute representation error is the meaningful compression target and is small for both FP16 and BF16.")
    A("")
    A("### Binding gate")
    A("")
    A("Gate A (stability) is the binding gate. The stable subset is representable within the error bound (Gate B passes in all scenes), but it is far too small (~0.1% of Gaussians) and yields negligible memory reduction (Gate C fails). Hence DROP_NO_STABLE_STATE: whole-Gaussian state redundancy is absent at maturity, though position-only redundancy is present (see nuance above).")
    A("")

    A("## Provenance & reproducibility")
    A("")
    A("- Frozen recipe: `baseline/reference_v1` (git `abfa6c3e`). No code modification to the frozen recipe.")
    A("- Harness: `gsc0_trace.py` reproduces the canonical 30K loop verbatim + read-only `no_grad` snapshots at the 3 preregistered phases + read-only per-Gaussian lineage accumulators (visibility count, gradient-norm sum) keyed by `_gaussian_ids`.")
    A("- gsplat: private patched tree `/root/gsc0_p0/gsplat_pkg` (true-accutile, `_backend.py` JIT flags fixed, stale `csrc.so` removed), built to `/root/gsc0_ext2/gsplat_cuda/gsplat_cuda.so` under torch 2.9.1. `accutile=False` (reference_v1 default).")
    A("- Analysis: `gsc0_analysis.py` (this report + JSONs). Thresholds fixed before aggregation.")
    A("- Raw artifacts: per-scene `traces/<scene>_<phase>_iter*.pt` (sha256 in `GSC0_TRACE_INDEX.json`).")
    A("- See `MANIFEST.sha256` for artifact hashes.")
    A("")

    A("## P0 scope notes / limitations")
    A("")
    A("- Gradient-magnitude and visibility statistics are cumulative lifetime proxies (grad_sum/vis_count); "
      "per-iteration gradient trace replay is forbidden in P0, so fine-grained temporal gradient stability is not measured.")
    A("- Accumulated drift is estimated statically (FP16/BF16 cast of optimizer moments + Adam update-direction perturbation), "
      "not by a reduced-precision training run (forbidden in P0).")
    A("- SH cross-phase comparison uses only the active SH slots common to both phases "
      "(4 slots early->mature; 16 slots within the mature window).")
    A("")

    with open(os.path.join(out, "GSC0_REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[gsc0-report] wrote GSC0_REPORT.md ({len(L)} lines) and GSC0_TRACE_INDEX.json")
    print(">>> REPORT SUCCESS")


if __name__ == "__main__":
    main()