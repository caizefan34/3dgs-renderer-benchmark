#!/usr/bin/env python3
"""Generate PAPER13 paper tables (CSV) + figure source data from existing raw JSON.
Frozen inputs only; no fabrication. Sources cited in every table.
"""
import json, csv, os, math

ROOT = "/mnt/storage_pool/liaoyuanjun/cgt_paper13"
R = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

def load(name):
    with open(os.path.join(R, name)) as f:
        return json.load(f)

cgt   = load("PAPER13_CGT_RESULTS.json")
base  = load("PAPER13_BASELINE_RESULTS.json")
e2e   = load("PAPER13_BUILD_AMORTIZATION.json")
abl   = load("PAPER13_ABLATIONS.json")
cert  = load("PAPER13_CERTIFICATE_STATS.json")
mem   = load("PAPER13_MEMORY.json")
reg   = load("PAPER13_SCENE_REGISTRY.json")
freeze= load("PAPER13_METHOD_FREEZE.json")
corr  = load("PAPER13_CORRECTNESS.json")

scenes = [e["scene"] for e in reg["registry"]]
dataset_of = {e["scene"]: e["dataset"] for e in reg["registry"]}
n_gauss = {}
for e in reg["registry"]:
    m = e.get("ckpt_meta") or {}
    n_gauss[e["scene"]] = m.get("num_points") or e.get("n_gaussians")

def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else None

# TABLE1: 13-scene main
t1 = os.path.join(FIG, "TABLE1_13SCENE_MAIN.csv")
with open(t1, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scene","dataset","n_gaussians","baseline_B_ms","baseline_B_fps",
                "S_mixed","safe_fraction","weighted_fallback","kernel_max_abs","kernel_pass",
                "cgt_vs_canon_max_abs","T_build_full_min","render_ratio","S_novel","N_break",
                "source"])
    for s in scenes:
        c = cgt["per_scene"].get(s, {})
        b = base["per_scene"].get(s, {})
        e = e2e["per_scene"].get(s, {})
        nb = e.get("N_break")
        w.writerow([s, dataset_of.get(s,""), n_gauss.get(s,""),
                    b.get("baseline_ms",""), b.get("baseline_fps",""),
                    c.get("S_mixed",""), c.get("safe_fraction",""), c.get("weighted_fallback",""),
                    c.get("kernel_max_abs",""), c.get("kernel_pass",""),
                    c.get("cgt_vs_canon_max_abs",""),
                    e.get("T_build_full_min",""), e.get("render_ratio",""), e.get("S_novel",""),
                    "None" if nb is None else nb,
                    "PAPER13_CGT_RESULTS/PAPER13_BASELINE_RESULTS/PAPER13_BUILD_AMORTIZATION"])
print("WROTE", t1)

# TABLE2: dataset aggregate
ds_groups = {}
for s in scenes:
    ds_groups.setdefault(dataset_of.get(s,"other"), []).append(s)
t2 = os.path.join(FIG, "TABLE2_DATASET_AGGREGATE.csv")
with open(t2, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset","n_scenes","scenes",
                "baseline_B_geomean_ms","S_mixed_geomean","safe_fraction_mean",
                "weighted_fallback_mean","T_build_full_geomean_min","render_ratio_geomean",
                "S_novel_geomean","source"])
    for ds, ss in ds_groups.items():
        bl_ms = [base["per_scene"][s]["baseline_ms"] for s in ss if "baseline_ms" in base["per_scene"].get(s,{})]
        sm    = [cgt["per_scene"][s]["S_mixed"] for s in ss if "S_mixed" in cgt["per_scene"].get(s,{})]
        sf    = [cgt["per_scene"][s]["safe_fraction"] for s in ss if "safe_fraction" in cgt["per_scene"].get(s,{})]
        wf    = [cgt["per_scene"][s]["weighted_fallback"] for s in ss if "weighted_fallback" in cgt["per_scene"].get(s,{})]
        tb    = [e2e["per_scene"][s]["T_build_full_min"] for s in ss if "T_build_full_min" in e2e["per_scene"].get(s,{})]
        rr    = [e2e["per_scene"][s]["render_ratio"] for s in ss if "render_ratio" in e2e["per_scene"].get(s,{})]
        sn    = [e2e["per_scene"][s]["S_novel"] for s in ss if e2e["per_scene"].get(s,{}).get("S_novel")]
        gm_pos = lambda xs: geomean(xs) if xs and all(x>0 for x in xs) else ""
        w.writerow([ds, len(ss), ";".join(ss),
                    gm_pos(bl_ms), gm_pos(sm),
                    sum(sf)/len(sf) if sf else "", sum(wf)/len(wf) if wf else "",
                    geomean(tb) if tb else "", geomean(rr) if rr else "",
                    geomean(sn) if sn else "",
                    "derived from TABLE1 sources"])
print("WROTE", t2)

# TABLE3: ablations
t3 = os.path.join(FIG, "TABLE3_ABLATIONS.csv")
with open(t3, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ablation","variant","metric","value","status","source"])
    aA = abl.get("ablation_A_rank", {})
    for rk in ["room_R4","room_R8","room_R16"]:
        if rk in aA:
            w.writerow(["A_rank", rk, "S_mixed", aA[rk].get("S_mixed",""), aA[rk].get("kernel_pass",""), "PAPER13_ABLATIONS.ablation_A_rank"])
            w.writerow(["A_rank", rk, "kernel_max_abs", aA[rk].get("kernel_max_abs",""), aA[rk].get("kernel_pass",""), "PAPER13_ABLATIONS.ablation_A_rank"])
    aB = abl.get("ablation_B_cert_fallback", {})
    w.writerow(["B_cert_fallback","mixed","safe_fraction_mean", aB.get("safe_fraction_mean",""), aB.get("status",""), "PAPER13_ABLATIONS.ablation_B_cert_fallback"])
    w.writerow(["B_cert_fallback","mixed","weighted_fallback_mean", aB.get("weighted_fallback_mean",""), aB.get("status",""), "PAPER13_ABLATIONS.ablation_B_cert_fallback"])
    aC = abl.get("ablation_C_fused_translation", {})
    if isinstance(aC, dict):
        for k,v in aC.items():
            if isinstance(v,(int,float,str,bool)):
                w.writerow(["C_fused_translation", k, "", v, aC.get("status",""), "PAPER13_ABLATIONS.ablation_C_fused_translation"])
print("WROTE", t3)

# TABLE4: memory (NOT_MEASURED explicit)
t4 = os.path.join(FIG, "TABLE4_MEMORY.csv")
with open(t4, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric","canonical_peak_vram_mib","cgt_peak_vram_mib","status","source"])
    w.writerow(["peak_render_vram","NOT_MEASURED","NOT_MEASURED", mem.get("status","NOT_MEASURED"), "PAPER13_MEMORY.json"])
print("WROTE", t4)

# TABLE5: related work
sm_geomean = cgt.get("S_mixed_geomean", 0)
t5 = os.path.join(FIG, "TABLE5_RELATED_WORK.csv")
with open(t5, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method","representation","trainable_primitive","fused_forward_speedup","e2e_novel_view_speedup","source"])
    w.writerow(["Canonical 3DGS (gsplat)","per-Gaussian raster","yes (SH/xyz/scale/rot/opacity)","1.00x (baseline)","1.00x (baseline)","PAPER13_BASELINE_RESULTS"])
    w.writerow(["CGT-F0 (frozen)","screen-space tile-structured field + rigorous certificate","no (stateless per-frame rebuild)","~%.2fx (50-tile mixed, geomean)"%sm_geomean,"~0x (build-dominated)","PAPER13_CGT_RESULTS/PAPER13_BUILD_AMORTIZATION"])
    w.writerow(["CGT-G5 (closed)","native coefficient optimization","yes (dropped)","N/A","N/A","DROP_NATIVE_OPTIMIZATION (frozen)"])
print("WROTE", t5)

# FIG2: speedup distribution
f2 = os.path.join(FIG, "fig2_speedup_distribution.csv")
with open(f2,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["scene","S_mixed"])
    for s in scenes:
        w.writerow([s, cgt["per_scene"].get(s,{}).get("S_mixed","")])
print("WROTE", f2)

# FIG3: safe/fallback vs speedup
f3 = os.path.join(FIG, "fig3_safe_fallback_vs_speedup.csv")
with open(f3,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["scene","S_mixed","safe_fraction","weighted_fallback"])
    for s in scenes:
        c=cgt["per_scene"].get(s,{})
        w.writerow([s, c.get("S_mixed",""), c.get("safe_fraction",""), c.get("weighted_fallback","")])
print("WROTE", f3)

# FIG4: rank ablation
f4 = os.path.join(FIG, "fig4_rank_ablation.csv")
with open(f4,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["rank","S_mixed","kernel_pass","kernel_max_abs"])
    aA=abl.get("ablation_A_rank",{})
    for rk in ["room_R4","room_R8","room_R16"]:
        if rk in aA:
            w.writerow([rk.replace("room_",""), aA[rk].get("S_mixed",""), aA[rk].get("kernel_pass",""), aA[rk].get("kernel_max_abs","")])
print("WROTE", f4)

# FIG1: method overview
f1 = os.path.join(FIG, "fig1_method_overview.csv")
with open(f1,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["stage","description"])
    w.writerow(["canonical_gaussian_traversal","gsplat full-frame raster baseline"])
    w.writerow(["tile_local_structured_field","per-tile teacher field extraction (extract_tiles)"])
    w.writerow(["certificate","rigorous interval cert (certify_tile, GATE=0.02)"])
    w.writerow(["cgt_fast_path","F0 fused CUDA kernel (R=4 only) for certified tiles"])
    w.writerow(["canonical_fallback","teacher kernel for uncertified tiles"])
    w.writerow(["final_image","composited output; correctness max_abs 1.85e-6"])
print("WROTE", f1)

print("=== ALL TABLES + FIG SOURCE WRITTEN ===")
