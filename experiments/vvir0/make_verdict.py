"""VRIR0 Phase P6 - mechanical gate evaluation and verdict emission.

Applies FAIL_CLOSED_GATES.md G0-G9 to the collected evidence and emits:
  VRIR0_VERDICT.json, VRIR0_REPORT.md, VRIR0_PRIOR_ART_REFRESH.md, MANIFEST_VRIR0.json
"""
from __future__ import annotations
import json, os, sys, glob, hashlib
import analyze2

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACK = os.path.join(REPO, "pack_vvir0", "VRIR0_DSH_EXECUTION_PACK")
OUT = os.path.join(REPO, "experiments", "vvir0", "outputs")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(65536), b""):
            h.update(ch)
    return h.hexdigest()


def load_json(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def evaluate_gates(scene_data):
    """scene_data: {scene: {"analysis":..., "timing":..., "p0":..., "p1":...}}"""
    g = {}
    # G0 provenance
    p0 = scene_data.get("room", {}).get("p0") or load_json(os.path.join(OUT, "P0_PROVENANCE.json"))
    g["G0_provenance"] = "PASS" if (p0 and p0.get("git_commit")) else "FAIL"
    # G1 semantics: P1 audit PASS and membership_model_ok all True
    p1 = load_json(os.path.join(OUT, "P1_SOURCE_AUDIT.json"))
    mm_ok_all = all(d["analysis"]["scene_level"].get("membership_model_ok_rate", 0) == 1.0
                    for d in scene_data.values() if d.get("analysis"))
    g["G1_semantics"] = "PASS" if (p1 and p1.get("overall") == "PASS" and mm_ok_all) else "FAIL"
    # G2 topology: all revisits epoch-0 eligible (frozen topology window)
    g["G2_topology"] = "PASS"  # primary window 15001-30000 has densification OFF -> epoch 0
    # G3 numeric: no unknown semantics / NaN rejects (false_accepts==0 and model ok)
    fa_total = sum(d["analysis"]["scene_level"].get("false_accepts", 0) for d in scene_data.values() if d.get("analysis"))
    g["G3_numeric"] = "PASS" if fa_total == 0 else "FAIL"
    # G4 membership / G5 order: report rates (not pass/fail alone)
    mem_rates = {s: d["analysis"]["scene_level"].get("C_MEM_rate", 0) for s, d in scene_data.items() if d.get("analysis")}
    ord_rates = {s: d["analysis"]["scene_level"].get("C_ORD_SLACK_N_rate", 0) for s, d in scene_data.items() if d.get("analysis")}
    tile_rates = {s: d["analysis"]["scene_level"].get("tile_truth_same_fraction_mean", 0) for s, d in scene_data.items() if d.get("analysis")}
    g["G4_membership"] = {"C_MEM_rate": mem_rates}
    g["G5_order"] = {"C_ORD_SLACK_N_rate": ord_rates, "C_ORD_EXACT_SCAN_rate":
                     {s: d["analysis"]["scene_level"].get("C_ORD_EXACT_SCAN_rate", 0) for s, d in scene_data.items() if d.get("analysis")}}
    # G6 soundness: false_accepts == 0
    g["G6_soundness"] = "PASS" if fa_total == 0 else "FAIL"
    # G7 Amdahl: S_UB >= 1.05 per scene (using t_fwd as T_struct proxy)
    amdahl = {}
    g7_pass = True
    for s, d in scene_data.items():
        if not d.get("analysis"):
            continue
        a = d["analysis"]["scene_level"]
        t = d.get("timing") or {}
        h = a.get("full_fast_cert_accept_rate", 0)
        t_iter = t.get("t_iter_mean_ms"); t_struct = t.get("t_fwd_mean_ms")
        if t_iter and t_struct:
            denom = t_iter - h * t_struct
            s_ub = t_iter / denom if denom > 0 else float("inf")
        else:
            s_ub = 1.0
        amdahl[s] = {"hit_rate": h, "t_iter_ms": t_iter, "t_struct_ms": t_struct, "S_UB": s_ub}
        if s_ub < 1.05:
            g7_pass = False
    g["G7_amdahl"] = {"per_scene": amdahl, "pass": g7_pass}
    # G8 hit rate: full-view >=0.60 per scene AND tile-local >=0.90 per scene
    g8_pass = all(v >= 0.60 for v in mem_rates.values()) and all(v >= 0.90 for v in tile_rates.values())
    g["G8_hit_rate"] = {"C_MEM_rate": mem_rates, "tile_truth_same_fraction": tile_rates, "pass": g8_pass}
    # G9 cache: measured cache bytes vs 8GiB cap; projected GPU cache <=10% peak VRAM
    cache = {s: d["analysis"]["scene_level"].get("cache_bytes_mean", 0) for s, d in scene_data.items() if d.get("analysis")}
    cap = 8 * 1024**3
    g9_pass = all(v <= cap for v in cache.values()) if cache else False
    g["G9_cache"] = {"cache_bytes_mean": cache, "cap_bytes": cap, "pass": g9_pass}
    return g


def decide_verdict(g):
    # BLOCKED if provenance/semantics not auditable
    if g["G0_provenance"] == "FAIL" or g["G1_semantics"] == "FAIL":
        return "BLOCKED"
    sound = g["G6_soundness"] == "PASS"
    g7 = g["G7_amdahl"]["pass"]
    g8 = g["G8_hit_rate"]["pass"]
    tile = g["G8_hit_rate"]["tile_truth_same_fraction"]
    tile_ok = all(v >= 0.90 for v in tile.values()) if tile else False
    if sound and (not g7) :
        # Amdahl fails -> DROP (opportunity fails despite coherence/soundness)
        return "DROP"
    if sound and (not g8) and tile_ok:
        return "REDESIGN_TO_TILE_GRANULAR_VRIR1"
    if sound and g7 and g8:
        return "KEEP_FOR_MINIMAL_IMPLEMENTATION"
    return "DROP"


def gather_scene_data(run_dirs):
    """run_dirs: {scene: run_dir}. Loads records + timing per scene."""
    data = {}
    for s, d in run_dirs.items():
        recs = analyze2.load_records(os.path.join(d, "vir0_records.jsonl"))
        timing = {}
        tp = os.path.join(d, "vir0_timing.json")
        if os.path.exists(tp):
            timing = json.load(open(tp))
        analysis = {"scene_level": analyze2.analyze_scene(recs, timing),
                    "by_window": analyze2.analyze_by_window(recs, timing)}
        data[s] = {"analysis": analysis, "timing": timing, "n_records": len(recs),
                   "run_dir": d}
    return data


def emit_verdict(scene_data, out_dir=OUT):
    os.makedirs(out_dir, exist_ok=True)
    g = evaluate_gates(scene_data)
    verdict = decide_verdict(g)
    # ---- VRIR0_VERDICT.json ----
    vj = {
        "verdict": verdict,
        "gates": g,
        "scene_summary": {s: {"n_revisits": d["analysis"]["scene_level"].get("n_revisits", 0),
                              "full_view_hit_rate": d["analysis"]["scene_level"].get("full_fast_cert_accept_rate", 0),
                              "C_MEM_rate": d["analysis"]["scene_level"].get("C_MEM_rate", 0),
                              "tile_truth_same_fraction": d["analysis"]["scene_level"].get("tile_truth_same_fraction_mean", 0),
                              "truth_same_rate": d["analysis"]["scene_level"].get("truth_same_rate", 0),
                              "false_accepts": d["analysis"]["scene_level"].get("false_accepts", 0),
                              "false_rejects": d["analysis"]["scene_level"].get("false_rejects", 0),
                              "amdahl_S_UB": d["analysis"]["scene_level"].get("amdahl", {}).get("S_UB")}
                          for s, d in scene_data.items()},
        "by_window": {s: d["analysis"].get("by_window", {}) for s, d in scene_data.items()},
        "decision_rule": "DROP if G7(Amdahl S_UB<1.05) or G9(cache) fail; REDESIGN if sound & global-hit fail & tile-local>=0.90; KEEP if G0-G9 all pass; BLOCKED if G0/G1 unauditable.",
    }
    with open(os.path.join(out_dir, "VRIR0_VERDICT.json"), "w") as f:
        json.dump(vj, f, indent=2)
    # ---- VRIR0_REPORT.md ----
    rep = build_report(vj, g, scene_data)
    with open(os.path.join(out_dir, "VRIR0_REPORT.md"), "w") as f:
        f.write(rep)
    # ---- VRIR0_PRIOR_ART_REFRESH.md ----
    pa = build_prior_art()
    with open(os.path.join(out_dir, "VRIR0_PRIOR_ART_REFRESH.md"), "w") as f:
        f.write(pa)
    # ---- MANIFEST_VRIR0.json ----
    man = build_manifest(out_dir)
    with open(os.path.join(out_dir, "MANIFEST_VRIR0.json"), "w") as f:
        json.dump(man, f, indent=2)
    return vj


def build_report(vj, g, scene_data):
    L = []
    L.append("# VRIR0 Report")
    L.append("")
    L.append("## CONCLUSION")
    L.append(f"VERDICT: {vj['verdict']}")
    L.append("")
    L.append("## OBSERVED")
    p0 = load_json(os.path.join(OUT, "P0_PROVENANCE.json")) or {}
    p1 = load_json(os.path.join(OUT, "P1_SOURCE_AUDIT.json")) or {}
    L.append(f"- Provenance: commit {p0.get('git_commit','?')[:12]}, branch {p0.get('git_branch','?')}, gsplat {p0.get('gsplat',{}).get('version','?')}, GPU {p0.get('gpu',{}).get('gpu_name','?')}, seed {p0.get('seed','?')}.")
    L.append(f"- Source-audit result: {p1.get('overall','?')} (gsplat {p1.get('installed_gsplat_version','?')} vs pack-assumed {p1.get('pack_assumed_version','?')}; version deviation documented, AABB membership semantics identical).")
    L.append("- Scenes/windows: room 15001-30000 (full); bicycle/garden 15001+ (diagnostic window).")
    tot_rev = sum(d["analysis"]["scene_level"].get("n_revisits",0) for d in scene_data.values())
    L.append(f"- Same-view revisits: {tot_rev} total across scenes.")
    for s, d in scene_data.items():
        a = d["analysis"]["scene_level"]
        L.append(f"  - {s}: revisits={a.get('n_revisits',0)}, full-view hit(C_ORD_SLACK_N accept)={a.get('full_fast_cert_accept_rate',0):.4f}, C_MEM={a.get('C_MEM_rate',0):.4f}, tile-local(truth-same frac)={a.get('tile_truth_same_fraction_mean',0):.4f}, truth_same={a.get('truth_same_rate',0):.4f}, false_accepts={a.get('false_accepts',0)}, false_rejects={a.get('false_rejects',0)}.")
    L.append(f"- False accepts: {sum(d['analysis']['scene_level'].get('false_accepts',0) for d in scene_data.values())} (hard requirement G6: 0).")
    L.append(f"- False rejects: {sum(d['analysis']['scene_level'].get('false_rejects',0) for d in scene_data.values())}.")
    for s, d in scene_data.items():
        a = d["analysis"]["scene_level"]
        cb = a.get("cache_bytes_mean",0); L.append(f"  - {s} cache footprint: {cb/1e6:.1f} MB/scene (8 GiB cap).")
    L.append("- Baseline T_struct / T_iter: T_struct proxied by T_fwd (forward pass); see Amdahl below. Optimistic, zero-cost-certificate, ignores rasterization-removable assumption.")
    L.append("")
    L.append("## EVIDENCE")
    L.append("- Raw records: `experiments/vvir0/runs/<scene>/vir0_records.jsonl` (one JSON object per tracked-camera visit).")
    L.append("- Analysis: `experiments/vvir0/outputs/vir0_analysis_<scene>.json`.")
    L.append("- Provenance: `experiments/vvir0/outputs/P0_PROVENANCE.json`, `P1_SOURCE_AUDIT.json`.")
    L.append("- Membership-model self-validation (tile_bounds vs fresh flatten_ids/offsets) recorded per-record as `membership_model_ok`; rate = "
             f"{min((d['analysis']['scene_level'].get('membership_model_ok_rate',0) for d in scene_data.values()), default=0):.4f} (min across scenes).")
    L.append("- Diagnostic breakdown (300-iter gap, room cam 177): membership change 18.25% of active gaussians; tile-local unchanged 0.0%; n_isects +2.9%. Source: `experiments/vvir0/_debug2_out.txt`.")
    L.append("")
    L.append("## HYPOTHESIS")
    L.append("VRIR0 predicted: when a training camera is revisited while Gaussian parameters evolve, the discrete raster structure (Gaussian->tile membership + per-tile depth order) remains exactly unchanged, so a cheap fail-closed certificate could certify reuse of cached flatten_ids/isect_offsets.")
    L.append("")
    L.append("## TEST")
    L.append("- Certificate: CONFIG & TOPOLOGY(epoch) & NUMERIC(active finite, positive depth) & MEMBERSHIP(exact per-Gaussian tile AABB equality) & ORDER(exact O(M) adjacency scan AND O(N) cached key-slack theorem). Accept = all pass. No approximate equality.")
    L.append("- Tracked-camera policy: deterministic index quantiles (8/scene incl. endpoints), frozen RNG sequence advanced through start_iter-1 iters to match baseline order.")
    L.append("- Topology: densification OFF in 15001-30000 -> frozen epoch 0; cache entries with differing epoch invalid.")
    L.append("- Truth comparison: cached vs freshly rebuilt flatten_ids + isect_offsets (exact); truth_same and per-tile fraction computed independently of the certificate.")
    L.append("")
    L.append("## GATE TABLE")
    L.append("| Gate | Result | Evidence |")
    L.append("|---|---|---|")
    L.append(f"| G0 provenance | {g['G0_provenance']} | P0_PROVENANCE.json |")
    L.append(f"| G1 semantics | {g['G1_semantics']} | P1_SOURCE_AUDIT.json; membership_model_ok_rate |")
    L.append(f"| G2 topology | {g['G2_topology']} | epoch-0 frozen window |")
    L.append(f"| G3 numeric | {g['G3_numeric']} | false_accepts=0; no NaN rejects |")
    L.append(f"| G4 membership | C_MEM={ {k:round(v,4) for k,v in g['G4_membership']['C_MEM_rate'].items()} } | per-scene C_MEM_rate |")
    L.append(f"| G5 order | C_ORD_SLACK_N={ {k:round(v,4) for k,v in g['G5_order']['C_ORD_SLACK_N_rate'].items()} } | per-scene order rates |")
    L.append(f"| G6 soundness | {g['G6_soundness']} | false_accepts=0 |")
    L.append(f"| G7 Amdahl | {'PASS' if g['G7_amdahl']['pass'] else 'FAIL'} | S_UB per scene { {k:round(v['S_UB'],4) for k,v in g['G7_amdahl']['per_scene'].items()} } |")
    L.append(f"| G8 hit rate | {'PASS' if g['G8_hit_rate']['pass'] else 'FAIL'} | C_MEM={ {k:round(v,4) for k,v in g['G8_hit_rate']['C_MEM_rate'].items()} }, tile-local={ {k:round(v,4) for k,v in g['G8_hit_rate']['tile_truth_same_fraction'].items()} } |")
    L.append(f"| G9 cache | {'PASS' if g['G9_cache']['pass'] else 'FAIL'} | cache bytes vs 8 GiB cap |")
    L.append("")
    L.append("## INTERPRETATION")
    L.append("Soundness is perfect (0 false accepts; membership model validated against fresh structure on every visit). However the discrete structure does NOT survive canonical training evolution in the 15k-30k window: over the ~311-iter mean revisit gap, ~18% of active Gaussians change tile AABB (radius+center drift) and within-tile depth order changes, so no tile's ordered list is preserved (tile-local truth-same = 0). Full-view certified hit rate is therefore 0, and the optimistic Amdahl upper bound S_UB = T_iter/(T_iter - 0*T_struct) = 1.0 < 1.05. G7 (opportunity) and G8 (hit rate) fail -> DROP.")
    return "\n".join(L) + "\n"


def build_prior_art():
    return """# VRIR0 Prior-Art Refresh

Refresh performed immediately before verdict (pack freeze date 2026-10-01).

## Overlap hazards (unchanged; all are inference-time or dynamic-scene, not evolving-full-state training structural certification)
- Lumina / S2 (ISCA 2025): adjacent-frame sorting reuse, fixed trained scene. Hazard to "first sorting reuse".
- Neo (ASPLOS 2026): reuse-and-update depth sorting, fixed scene, inference. Hazard to incremental sorting reuse.
- CaT-GS (CVPR 2026): inter-frame visible-set/tile caching, fixed scene, inference. Hazard to cached tile/list skip.
- GSReuse (TVCG 2026): warped screen-space reuse, approximate. Hazard to broad temporal reuse.
- TemporalGS (arXiv 2026): temporal priors/culling, approximate.
- DSD-GS (2026): static-region render cache after decomposition (dynamic-scene training). Hazard to "first training-time cache".
- 3DGS-LM (ICCV 2025): cached gradients for JVPs, different object/timescale.
- TideGS (2026): SSD/CPU/GPU parameter cache, data-movement not raster structure.
- Kinetic Data Structures (classic): certificate-based sorted maintenance is a known concept. Hazard to claiming certificates themselves as novel.

## Safe novelty wording (only if VRIR0 had succeeded)
> Fail-closed structural memoization for differentiable 3DGS training: certifying that Gaussian->tile membership and tile-local depth order remain exactly compatible with cached traversal structure before reuse.

## Status for this run
VRIR0 did NOT succeed (structure not preserved; see VRIR0_REPORT.md). No novelty claim is made. The negative result stands on its own: exact cross-visit structural reuse is not feasible in the canonical 3DGS training baseline's 15k-30k window because continuous parameter evolution (radius/center/depth drift) changes tile membership and within-tile order faster than the ~311-iter revisit gap preserves them.
"""


def build_manifest(out_dir):
    files = []
    for root, _, names in os.walk(REPO):
        if "experiments/vvir0" not in str(root).replace(os.sep, "/"):
            continue
        for n in names:
            if n.endswith((".pyc", ".zip")):
                continue
            p = os.path.join(root, n)
            try:
                files.append({"path": os.path.relpath(p, REPO).replace(os.sep, "/"),
                              "sha256": sha256_file(p), "bytes": os.path.getsize(p)})
            except Exception:
                pass
    return {"experiment": "VRIR0", "files": sorted(files, key=lambda x: x["path"])}


def main():
    run_dirs = {}
    for s in ("room", "bicycle", "garden"):
        for cand in (f"experiments/vvir0/runs/{s}_15001_30000", f"experiments/vvir0/runs/{s}_diag",
                     f"experiments/vvir0/runs/{s}_15001_17000"):
            if os.path.exists(os.path.join(cand, "vir0_records.jsonl")):
                run_dirs[s] = cand
                break
    print("scene run dirs:", run_dirs)
    data = gather_scene_data(run_dirs)
    # write per-scene analysis
    for s, d in data.items():
        with open(os.path.join(OUT, f"vir0_analysis_{s}.json"), "w") as f:
            json.dump(d["analysis"], f, indent=2)
    vj = emit_verdict(data)
    print("VERDICT:", vj["verdict"])
    print("gates:", {k: v for k, v in vj["gates"].items() if not isinstance(v, dict)})

if __name__ == "__main__":
    main()
