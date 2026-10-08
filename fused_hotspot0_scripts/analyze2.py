#!/usr/bin/env python3
"""FUSED-HOTSPOT0 analysis stage 2: scaling, amdahl, redundancy, prior-art, opportunity, verdict, docs."""
import json, os, math, datetime

P = "/mnt/storage_pool/liaoyuanjun/fused_profile0"
HOT = P + "/FUSED_HOTSPOT0"
RAW = HOT + "/raw"
SCENES = ["room", "bicycle", "garden"]
STAGES = ["s1_5k", "s2_10k", "s3_mature"]
STAGE_ITERS = {"s1_5k": 5000, "s2_10k": 5000, "s3_mature": 5000}
GEN_UTC = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
COMPS = ["P", "I", "S", "F", "L", "PB", "B", "BO", "D", "M", "U"]

def load(p):
    with open(p) as f: return json.load(f)
def rnd(x, n=4):
    try: return round(float(x), n)
    except Exception: return x
def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    if not xs: return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))

CB = load(f"{HOT}/FUSED_HOTSPOT0_COMPONENT_BREAKDOWN.json")
WS = load(f"{HOT}/FUSED_HOTSPOT0_WORKLOAD_STATS.json")
KC = load(f"{HOT}/FUSED_HOTSPOT0_KERNEL_CHARACTERIZATION.json")
PS = CB["per_scene_stage"]

def frac(scene, stage, c):
    return PS[f"{scene}_{stage}"]["components"][c]["fraction_of_iteration"]
def cpms(scene, stage, c):
    return PS[f"{scene}_{stage}"]["components"][c]["critical_path_ms"]
def Titer(scene, stage):
    return PS[f"{scene}_{stage}"]["T_iteration_ms"]

# ---- geomean fractions (mature s3) and full-training-weighted ----
geo_mature = {}
for c in COMPS:
    vals = [frac(s, "s3_mature", c) for s in SCENES]
    geo_mature[c] = rnd(geomean([v for v in vals if v is not None]), 4)
# full-training weighted: F_X_train = sum_stage(iter_s*T_X_s)/sum_stage(iter_s*T_iter_s)
ftw = {}
for c in COMPS:
    num = sum(STAGE_ITERS[s] * cpms(sc, s, c) for sc in SCENES for s in STAGES)
    den = sum(STAGE_ITERS[s] * Titer(sc, s) for sc in SCENES for s in STAGES)
    ftw[c] = rnd(num / den, 4) if den else None

# ---- SCALING ----
def scaling():
    per = {}
    for sc in SCENES:
        for s in STAGES:
            st = WS["per_scene_stage"][f"{sc}_{s}"]
            ni = st["n_instances_tile_intersections"]; ng = st["n_gaussians_total"]
            W, Hh = st["image_wh"]; npx = W * Hh
            per[f"{sc}_{s}"] = {
                "scene": sc, "stage": s, "T_iteration_ms": rnd(Titer(sc, s)),
                "n_gaussians": ng, "image_wh": [W, Hh], "n_pixels": npx, "n_tiles": st["n_tiles_image"],
                "n_instances": ni, "instances_per_gaussian": rnd(ni / ng, 3) if ni else None,
                "T_B_ms": rnd(cpms(sc, s, "B")), "T_F_ms": rnd(cpms(sc, s, "F")),
                "T_BO_ms": rnd(cpms(sc, s, "BO")), "T_PB_ms": rnd(cpms(sc, s, "PB")),
                "T_forward_raster_ms": rnd(sum(cpms(sc, s, c) for c in ["P","I","S","F"])),
                "T_backward_fused_ms": rnd(sum(cpms(sc, s, c) for c in ["PB","B","BO"])),
                "us_per_instance_B": rnd(cpms(sc, s, "B") * 1000 / ni, 6) if ni else None,
                "ms_per_gaussian_B": rnd(cpms(sc, s, "B") / ng * 1e6, 6) if ng else None,
            }
    # scaling summaries (mature)
    summ = {}
    for c, key in [("B","T_B_ms"),("F","T_F_ms"),("forward_raster","T_forward_raster_ms"),("backward_fused","T_backward_fused_ms")]:
        pts = [(per[f"{sc}_s3_mature"]["n_instances"], per[f"{sc}_s3_mature"][key]) for sc in SCENES if per[f"{sc}_s3_mature"]["n_instances"]]
        # ratio T_B / n_instances across scenes (mature)
        ratios = [per[f"{sc}_s3_mature"][key] / per[f"{sc}_s3_mature"]["n_instances"] for sc in SCENES if per[f"{sc}_s3_mature"]["n_instances"]]
        summ[c] = {"mature_T_ms": {sc: per[f"{sc}_s3_mature"][key] for sc in SCENES},
                   "mature_n_instances": {sc: per[f"{sc}_s3_mature"]["n_instances"] for sc in SCENES},
                   "T_per_ninstance": {sc: rnd(r, 8) for sc, r in zip([sc for sc in SCENES if per[f"{sc}_s3_mature"]["n_instances"]], ratios)}}
    return {"schema": "FUSED_HOTSPOT0_SCALING", "generated_utc": GEN_UTC,
            "method": "Diagnostic scaling of component critical-path time vs workload variables (n_gaussians, n_instances, n_pixels). Not a claim of universal asymptotic complexity.",
            "per_scene_stage": per, "mature_scaling_summary": summ,
            "observations": [
                "blend_backward critical-path time grows with instances/Gaussian: room (703 inst/G, T_B largest absolute but small scene) vs bicycle (207 inst/G at s3, T_B=~4.3ms). The per-instance cost is roughly stable, indicating T_B ~ O(n_contributions) = O(instances).",
                "Forward raster (P+I+S+F) is a large share in bicycle/garden (~0.31-0.38) but smaller in room; create_instances + sort scale with n_instances.",
                "n_instances itself scales with n_gaussians * instances_per_gaussian; room has few Gaussians but very large footprints (703 tiles/Gaussian), bicycle has many Gaussians with moderate footprints."]}

json.dump(scaling(), open(f"{HOT}/FUSED_HOTSPOT0_SCALING.json", "w"), indent=2)

# ---- AMDAHL ORACLE ----
SPEEDS = [1.25, 1.5, 2.0, 3.0, 4.0, float("inf")]
def amdahl(f, s):
    if s == float("inf"): return 1.0 / (1.0 - f) if f < 1 else float("inf")
    return 1.0 / ((1.0 - f) + f / s)
ORACLE_COMPS = {
    "B": "blend_backward_cu (raster backward blend)",
    "F": "blend_cu (forward raster blend)",
    "BO": "fused optimizer (adam_step, fused in backward)",
    "PB": "preprocess_backward_cu",
    "forward_raster_PISF": "P+I+S+F (forward projection/instance/sort/blend)",
    "backward_fused_PBBBO": "PB+B+BO (entire backward + fused optimizer)",
    "L": "loss (l1 + fusedssim)",
}
def comp_frac_coupled(sc, s, clist):
    return sum(frac(sc, s, c) for c in clist)
amdahl_per = {}
for key, label in ORACLE_COMPS.items():
    if "+" in key:
        clist = key.split("_")[0] if False else None
    # resolve coupled
    if key == "forward_raster_PISF":
        fvals = {sc: comp_frac_coupled(sc, "s3_mature", ["P","I","S","F"]) for sc in SCENES}
    elif key == "backward_fused_PBBBO":
        fvals = {sc: comp_frac_coupled(sc, "s3_mature", ["PB","B","BO"]) for sc in SCENES}
    else:
        fvals = {sc: frac(sc, "s3_mature", key) for sc in SCENES}
    fm = geomean(list(fvals.values()))
    row = {"label": label, "mature_fraction_per_scene": {sc: rnd(fvals[sc]) for sc in SCENES},
           "mature_fraction_geomean": rnd(fm),
           "S_E2E": {}}
    for sp in SPEEDS:
        # per-scene S_E2E then geomean
        per_sc = [amdahl(fvals[sc], sp) for sc in SCENES]
        row["S_E2E"][str(sp)] = rnd(geomean(per_sc), 4)
    amdahl_per[key] = row
json.dump({"schema": "FUSED_HOTSPOT0_AMDAHL_ORACLE", "generated_utc": GEN_UTC,
           "formula": "S_E2E(s) = 1 / ((1 - f) + f / s); f = component critical-path fraction of iteration (mature geomean across room/bicycle/garden). Coupled groups use summed fraction.",
           "speedups": SPEEDS, "per_component": amdahl_per,
           "promotion_gate": {"rule_2x": "S_E2E(2x) >= 1.15", "rule_perfect": "S_E2E(inf) >= 1.25",
                              "passing_components": [k for k, v in amdahl_per.items() if v["S_E2E"]["2.0"] >= 1.15 or v["S_E2E"]["inf"] >= 1.25]}},
          open(f"{HOT}/FUSED_HOTSPOT0_AMDAHL_ORACLE.json", "w"), indent=2)

# ---- REDUNDANCY_AUDIT ----
redundancy = {
    "schema": "FUSED_HOTSPOT0_REDUNDANCY_AUDIT", "generated_utc": GEN_UTC,
    "method": "Source inspection (kernels_forward.cuh, kernels_backward.cuh) + census + workload scaling to identify genuinely removable work per component. Objective section 14.",
    "per_component": {
        "B": {"verdict": "REMOVABLE_WORK_PRESENT", "mechanism": "Per-primitive atomicAdd gradient accumulation (grad_mean2d/conic/opacity/colors) across all tiles a primitive touches. Contention scales with instances/Gaussian (room 703, bicycle 207).",
              "removable": "Atomic contention overhead: a deterministic privatized / sorted segmented accumulation yields bit-identical gradients with fewer serialized atomic conflicts. Forward early-stop (last_contributor) and 1/255 alpha early-stop ALREADY present, so chain-replay redundancy is largely eliminated; the residual removable work is the atomic serialization, not the blend math.",
              "evidence": "cuobjdump: blend_backward 64 reg, 50% occupancy; source: 6 atomicAdd per primitive at kernel end; workload: instances/Gaussian 703 (room) / 207 (bicycle)."},
        "F": {"verdict": "MOSTLY_INTRINSIC", "mechanism": "Forward alpha-blend per tile with shared staging; early-stop at 1/255 and transmittance 1e-4 already present.",
              "removable": "Little exact-render removable work; early-stop mechanisms already intrinsic. Not a strong standalone target.", "evidence": "kernels_forward.cuh:354 blend_cu; config min_alpha_threshold=1/255, transmittance_threshold=1e-4."},
        "BO": {"verdict": "INTRINSIC_FUSION", "mechanism": "Adam fused INSIDE backward per-Gaussian (backward/optimizer fusion already intrinsic to FasterGSFused).",
               "removable": "Fusion is already applied; no obvious removable work without changing convergence. Prior-art occupied.", "evidence": "kernels_backward.cuh:464 adam_step_invisible; mechanisms_present: backward_optimizer_fusion."},
        "P": {"verdict": "MOSTLY_INTRINSIC", "mechanism": "Projection/covariance/SH per Gaussian; exact-render required.", "removable": "SH evaluation could be pruned by active-sh bases (already bounded by active_sh_bases). Limited removable work.", "evidence": "kernels_forward.cuh:14 preprocess_cu."},
        "I": {"verdict": "MOSTLY_INTRINSIC", "mechanism": "create_instances with EXACT per-tile contribution reject (will_primitive_contribute) + warp cooperative culling; radix sort. Exact reject is INTRINSIC (AABB-PROD0 closed: C32==B32, S_reject=1.0).",
              "removable": "Exact reject already minimal; do not reopen. Bookkeeping (ranges/bucket counts) required for bucketed backward.", "evidence": "kernels_forward.cuh:218 create_instances_cu; kernel_utils.cuh:62 will_primitive_contribute."},
        "S": {"verdict": "INTRINSIC", "mechanism": "DeviceRadixSort orders instances by (tile,depth). Required for correct back-to-front blend.", "removable": "Sort is required; low self-time (~2%). Not a target.", "evidence": "buffer_utils.h radix sort; census ~2.1-2.5%."},
        "L": {"verdict": "INTRINSIC", "mechanism": "L1 + DSSIM loss (LAMBDA 0.8/0.2).", "removable": "Loss is required by training objective; not removable without changing convergence.", "evidence": "Loss.py; census fusedssim ~2.6-3.3%."},
        "PB": {"verdict": "PARTIALLY_REMOVABLE", "mechanism": "Backward projection (dL/dmean2d, covariance/scale/rotation, SH grad). 95 reg, 31% occupancy (register-bound).",
               "removable": "Recomputes gaussian expf already computed in forward (no alpha checkpoint stored) -> redundant compute vs memory tradeoff; register pressure limits occupancy. Possible but changes numerics only if checkpointed exactly.", "evidence": "kernels_backward.cuh:15 preprocess_backward_cu; cuobjdump 95 reg, 31% occ."}
    },
    "summary": "Strongest genuinely-removable work = blend_backward (B) atomic contention (identical gradients, deterministic accumulation alternative). All obvious forward-side optimizations (exact reject, early-stop, bucketing, backward/optimizer fusion) are ALREADY intrinsic to FasterGSFused. Per baseline-overlap rule (section 24), mechanisms already present must NOT be promoted."
}
json.dump(redundancy, open(f"{HOT}/FUSED_HOTSPOT0_REDUNDANCY_AUDIT.json", "w"), indent=2)

# ---- OPPORTUNITY_RANKING ----
def opp():
    rows = []
    scores = {
        "B": {"A_amdahl": 5, "B_removable": 4, "C_priorart_gap": 3, "D_risk": 3, "E_complexity": 3, "F_convergence": 5},
        "F": {"A_amdahl": 3, "B_removable": 1, "C_priorart_gap": 1, "D_risk": 2, "E_complexity": 2, "F_convergence": 5},
        "BO": {"A_amdahl": 2, "B_removable": 1, "C_priorart_gap": 1, "D_risk": 2, "E_complexity": 2, "F_convergence": 4},
        "PB": {"A_amdahl": 2, "B_removable": 2, "C_priorart_gap": 2, "D_risk": 3, "E_complexity": 3, "F_convergence": 3},
        "forward_raster": {"A_amdahl": 3, "B_removable": 1, "C_priorart_gap": 1, "D_risk": 2, "E_complexity": 2, "F_convergence": 5},
    }
    labels = {"B": "blend_backward_cu (raster backward blend + atomic gradient accumulation)",
              "F": "blend_cu (forward raster blend)", "BO": "fused optimizer (adam_step)",
              "PB": "preprocess_backward_cu", "forward_raster": "P+I+S+F coupled forward raster"}
    for k, sc in scores.items():
        if k == "forward_raster":
            f = geomean([comp_frac_coupled(sc_, "s3_mature", ["P","I","S","F"]) for sc_ in SCENES])
        else:
            f = geomean([frac(sc_, "s3_mature", k) for sc_ in SCENES])
        # total = A+B+C + F - D - E  (higher better; D risk and E complexity are costs)
        total = sc["A_amdahl"] + sc["B_removable"] + sc["C_priorart_gap"] + sc["F_convergence"] - sc["D_risk"] - sc["E_complexity"]
        rows.append({"target": k, "label": labels[k], "mature_geomean_fraction": rnd(f),
                     "amdahl_2x": amdahl_per[k]["S_E2E"]["2.0"] if k in amdahl_per else rnd(amdahl(f, 2.0)),
                     "amdahl_perfect": amdahl_per[k]["S_E2E"]["inf"] if k in amdahl_per else rnd(amdahl(f, float("inf"))),
                     "scores": sc, "opportunity_score": total})
    rows.sort(key=lambda r: -r["opportunity_score"])
    for i, r in enumerate(rows, 1): r["rank"] = i
    return {"schema": "FUSED_HOTSPOT0_OPPORTUNITY_RANKING", "generated_utc": GEN_UTC,
            "scoring": "A=Amdahl headroom, B=removable-work, C=prior-art gap, D=risk(cost), E=complexity(cost), F=convergence-safety. score = A+B+C+F - D - E. Qualitative; Amdahl gate (section 21) is the hard filter.",
            "hard_gate": "major hotspot: geomean f>=0.25 OR f>=0.30 in >=2 scenes (section 18).",
            "ranked": rows}
json.dump(opp(), open(f"{HOT}/FUSED_HOTSPOT0_OPPORTUNITY_RANKING.json", "w"), indent=2)

# stash key numbers for verdict/report
summary_nums = {"geo_mature": geo_mature, "ftw": ftw,
                "B_mature_frac": {sc: rnd(frac(sc, "s3_mature", "B")) for sc in SCENES},
                "B_geomean": geo_mature["B"],
                "amdahl_B": amdahl_per["B"]["S_E2E"],
                "resolved_min": min(PS[k]["resolved_fraction"] for k in PS)}
json.dump(summary_nums, open(f"{HOT}/_summary_nums.json", "w"), indent=2)
print("STAGE2_DONE scaling/amdahl/redundancy/opportunity written")
print("geo_mature:", json.dumps(geo_mature))
print("ftw:", json.dumps(ftw))
print("B_mature_frac:", summary_nums["B_mature_frac"], "geomean", geo_mature["B"])
print("amdahl_B:", summary_nums["amdahl_B"])
print("resolved_min:", summary_nums["resolved_min"])