#!/usr/bin/env python3
"""FUSED-HOTSPOT0 analysis stage 4: REPORT.md + MANIFEST.sha256."""
import json, os, hashlib, datetime
P="/mnt/storage_pool/liaoyuanjun/fused_profile0"; HOT=P+"/FUSED_HOTSPOT0"
GEN=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
def load(p):
    with open(p) as f: return json.load(f)

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda: f.read(1<<20), b''): h.update(c)
    return h.hexdigest()
CB=load(f"{HOT}/FUSED_HOTSPOT0_COMPONENT_BREAKDOWN.json"); PS=CB["per_scene_stage"]
KC=load(f"{HOT}/FUSED_HOTSPOT0_KERNEL_CENSUS.json")
WS=load(f"{HOT}/FUSED_HOTSPOT0_WORKLOAD_STATS.json")
CHAR=load(f"{HOT}/FUSED_HOTSPOT0_KERNEL_CHARACTERIZATION.json")
AM=load(f"{HOT}/FUSED_HOTSPOT0_AMDAHL_ORACLE.json")
OR=load(f"{HOT}/FUSED_HOTSPOT0_OPPORTUNITY_RANKING.json")
RED=load(f"{HOT}/FUSED_HOTSPOT0_REDUNDANCY_AUDIT.json")
SC=load(f"{HOT}/FUSED_HOTSPOT0_SCALING.json")
VD=load(f"{HOT}/FUSED_HOTSPOT0_VERDICT.json")
PF=load(f"{P}/FUSED_PROFILE0_FREEZE.json")
SCENES=["room","bicycle","garden"]; STAGES=["s1_5k","s2_10k","s3_mature"]
def f(sc,s,c): return PS[f"{sc}_{s}"]["components"][c]["fraction_of_iteration"]
def cp(sc,s,c): return PS[f"{sc}_{s}"]["components"][c]["critical_path_ms"]
def ti(sc,s): return PS[f"{sc}_{s}"]["T_iteration_ms"]

L=[]
def w(x=""): L.append(x)
w("# FUSED-HOTSPOT0 — FasterGSFused Production Raster Residual Hotspot Decomposition + Opportunity Oracle")
w()
w(f"> Generated {GEN}. Host bms-39468022-001, 8x A100-PCIE-40GB. Baseline FasterGSFused TS16, commit 9c8b5a6, backend .so sha b12c6596... (frozen).")
w()
w("## EXECUTIVE RESULT")
w()
w(f"**Verdict: {VD['verdict']}**")
w()
w(f"- Single dominant residual hotspot: **blend_backward_cu (B)** — mature geomean E2E fraction **{VD['major_hotspot_criteria']['geomean_fraction_mature']}** (room {VD['major_hotspot_criteria']['per_scene_mature_fraction']['room']}, bicycle {VD['major_hotspot_criteria']['per_scene_mature_fraction']['bicycle']}, garden {VD['major_hotspot_criteria']['per_scene_mature_fraction']['garden']}); >=0.40 in all 3 core scenes, all 3 stages.")
w(f"- Amdahl ceiling (E2E): 1.25x->{AM['per_component']['B']['S_E2E']['1.25']}, 1.5x->{AM['per_component']['B']['S_E2E']['1.5']}, 2x->{AM['per_component']['B']['S_E2E']['2.0']}, 3x->{AM['per_component']['B']['S_E2E']['3.0']}, 4x->{AM['per_component']['B']['S_E2E']['4.0']}, perfect->{AM['per_component']['B']['S_E2E']['inf']}. Passes promotion gate (2x>=1.15, perfect>=1.25).")
w(f"- Dominant cost mechanism: per-pixel reverse blend-chain replay (serial, warp.shfl-dependent) + per-primitive atomicAdd gradient accumulation whose contention scales with instances-per-Gaussian (room 703, bicycle 207).")
w(f"- Removable work: atomic-contention overhead — a deterministic privatized/sorted segmented accumulation yields bit-identical gradients. Forward early-stop (last_contributor, 1/255) already intrinsic, so chain-replay redundancy is largely eliminated.")
w(f"- Prior-art: contention-reducing accumulation is NOT applied/validated in the Faster-GS/fast-GS/gsplat lineage (naive atomics) -> plausible gap.")
w(f"- Component accounting complete (min resolved fraction {VD['accounting']['min_resolved_fraction']} >= 0.90). No baseline drift for the raster-dominant conclusion.")
w()
w("## BASELINE")
w()
w(f"- Repo {PF['baseline_repo']} commit {PF['baseline_git']['commit']}; backend .so sha256 {PF['backend_build']['B16_TS16']['sha256'][:16]}... (B16/TS16).")
w(f"- Env: python {PF['environment']['python']}, torch {PF['environment']['torch']}, cuda {PF['environment']['torch_cuda']}, {PF['environment']['gpu_model']}.")
w(f"- Scenes/checkpoints: room 1.027M, bicycle 4.430M, garden 3.105M gaussians (final.pt S3). Config sha256 frozen. Tile 16x16, block_size_blend=256, min_alpha=1/255, transmittance=1e-4.")
w(f"- Preloading: PL2 (VRAM) for authoritative timing; data decode excluded (SAT-AUDIT0 correction).")
w()
w("## MEASUREMENT VALIDITY")
w()
w("- Clean-GPU guard before each block: mem<=512MiB, util<=2%, observe>=20s (recorded in raw/guard_*.json). All 8 GPUs observed 0 MiB/0% at guard.")
w("- H1 warmup 50 / iters 200, CUDA events, one exclusive A100. H2 census warmup 30 / 30 profiled iters under torch.profiler chrome trace.")
w("- ncu hardware counters UNAVAILABLE (ncu 2021.3.1 sections/rules path missing, no sudo) -> COUNTERS_UNAVAILABLE; used cuobjdump ptxas resource report + census + scaling + source inspection (objective section 15 fallback).")
w("- Accounting completeness (section 13): explained GPU critical-path fraction >= 0.90 on all 9 scene/stages (min 0.944).")
w("- H1 reproduction gate (section 9): bicycle/garden reproduce SAT-AUDIT0 PL2 within 0.3%; room absolute T_iter differs (3.86 vs 6.12ms) but room flagged non-reproducible in SAT-AUDIT0 runlog and f_raster fraction matches (0.910 vs 0.894). Raster-dominant conclusion holds -> no INCONCLUSIVE_BASELINE_DRIFT.")
w()
w("## PRODUCTION TIMING")
w()
w("| scene | stage | T_iter(ms) | fwd(ms) | bwd(ms) | loss(ms) | nG |")
w("|---|---|---|---|---|---|---|")
for sc in SCENES:
    for s in STAGES:
        seg=PS[f"{sc}_{s}"]["segment_anchors_ms"]
        w(f"| {sc} | {s} | {ti(sc,s):.3f} | {seg['forward']:.3f} | {seg['backward']:.3f} | {seg['loss']:.3f} | {WS['per_scene_stage'][f'{sc}_{s}']['n_gaussians_total']:,} |")
w()
w("All three scenes are backward-dominant (backward > forward). f_raster=(fwd+bwd)/T_iter: room 0.910, bicycle 0.943, garden 0.879 (mature) — matches SAT-AUDIT0 corrected 0.894/0.936/0.880.")
w()
w("## KERNEL CENSUS")
w()
w("Cross-scene mature mean (% of iteration time / % of GPU kernel time):")
w()
w("| kernel | %iter | %kernel | component |")
w("|---|---|---|---|")
for r in KC["mature_top_kernels_cross_scene"][:12]:
    comp=RED["per_component"].get(r["name"].split("::")[-1].split("(")[0], {}).get("verdict","")
    w(f"| `{r['name'][:58]}` | {r['mean_pct_iteration_time_mature']:.2f} | {r['mean_pct_kernel_time_mature']:.2f} | {r['name'].split('kernels::')[-1].split('(')[0] if 'kernels::' in r['name'] else 'other'} |")
w()
w("blend_backward is the #1 kernel (~50% of iteration time cross-scene). Full per-kernel tables in FUSED_HOTSPOT0_KERNEL_CENSUS.json + raw/census_*_trace.json.")
w()
w("## COMPONENT BREAKDOWN")
w()
w("Critical-path-allocated component fractions of iteration (mature s3):")
w()
w("| component | room | bicycle | garden | geomean | definition |")
w("|---|---|---|---|---|---|")
defs=CB["component_defs"]
for c in ["P","I","S","F","L","PB","B","BO","U"]:
    vals=[f(sc,"s3_mature",c) for sc in SCENES]
    import math
    gm=math.exp(sum(math.log(x) for x in vals if x>0)/len([x for x in vals if x>0])) if any(vals) else 0
    w(f"| {c} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {gm:.3f} | {defs.get(c,'')[:40]} |")
w()
w("Full per-stage table in FUSED_HOTSPOT0_COMPONENT_BREAKDOWN.json. Backward+optimizer fused (BO) reported as fused block; B and BO separated by census self-time but overlap (Adam fused in backward).")
w()
w("## WORKLOAD SCALING")
w()
w("| scene | stage | nG | n_tiles | n_instances | inst/Gaussian | T_B(ms) | T_F(ms) |")
w("|---|---|---|---|---|---|---|---|")
for sc in SCENES:
    for s in STAGES:
        st=WS["per_scene_stage"][f"{sc}_{s}"]
        ni=st["n_instances_tile_intersections"]
        w(f"| {sc} | {s} | {st['n_gaussians_total']:,} | {st['n_tiles_image']} | {ni if ni else 'OVF'} | {st['n_instances_tile_intersections']/st['n_gaussians_total'] if ni else 'n/a'} | {cp(sc,s,'B'):.3f} | {cp(sc,s,'F'):.3f} |")
w()
w("- T_B scales with n_contributions (instances): room has few Gaussians but huge footprints (703 inst/Gaussian) -> large absolute instance count (721M) and heavy atomic contention; bicycle has many Gaussians, inst/Gaussian grows 60->207 across training.")
w("- garden n_instances UNAVAILABLE (production forward overflows int32 / OOMs on full-res instance buffers).")
w("- Full ratios in FUSED_HOTSPOT0_SCALING.json.")
w()
w("## BOTTLENECK CHARACTERIZATION")
w()
w("| kernel | reg/thr | shared(B) | occupancy% | regime |")
w("|---|---|---|---|---|---|")
# bicycle mature kernel detail
for k in CHAR["per_scene_stage"]["bicycle_s3_mature"]["kernels"][:10]:
    nm=k["name"]
    short=nm.split("kernels::")[-1].split("(")[0] if "kernels::" in nm else nm[:24]
    reg=k["registers_per_thread"]; sh=k["static_shared_bytes"]; occ=k["occupancy_pct_est"]; reg_m=k["regime"]["regime"] if k["regime"] else "?"
    w(f"| `{short}` | {reg} | {sh} | {occ} | {reg_m} |")
w()
w("- blend_backward: 64 reg, 50% occupancy, MIXED (serial chain replay + atomic contention). blend_cu: 100% occ, 9.3KB shared, MEMORY_BANDWIDTH_BOUND. preprocess_backward: 95 reg, 31% occ, COMPUTE_BOUND (register-limited). adam_step: 14 reg, 100% occ, MEMORY_BANDWIDTH_BOUND. (COUNTERS_UNAVAILABLE; regimes inferred from ptxas resources + census + source, not name.)")
w()
w("## AMDAHL ORACLES")
w()
w("S_E2E(s)=1/((1-f)+f/s), f=mature critical-path fraction (geomean across scenes):")
w()
w("| component | f(geomean) | 1.25x | 1.5x | 2x | 3x | 4x | perfect |")
w("|---|---|---|---|---|---|---|---|")
for key,row in AM["per_component"].items():
    se=row["S_E2E"]
    w(f"| {key} | {row['mature_fraction_geomean']:.3f} | {se['1.25']} | {se['1.5']} | {se['2.0']} | {se['3.0']} | {se['4.0']} | {se['inf']} |")
w()
w(f"- Passing promotion gate (2x>=1.15 OR perfect>=1.25): {AM['promotion_gate']['passing_components']}.")
w()
w("## REDUNDANCY / REMOVABLE WORK")
w()
for c in ["B","F","BO","P","I","S","L","PB"]:
    r=RED["per_component"][c]
    w(f"- **{c} — {r['verdict']}**: {r['removable'][:180]}")
w()
w("## PRIOR-ART COLLISION CHECK")
w()
w("See FUSED_HOTSPOT0_PRIOR_ART_MAP.md. Baseline-overlap rule: exact reject, early-stop, bucketing, backward/optimizer fusion, shared staging are ALL intrinsic -> not promotable. Only blend_backward contention reduction leaves a plausible gap.")
w()
w("## CROSS-SCENE / CROSS-STAGE ROBUSTNESS")
w()
w("- blend_backward is the top kernel in all 3 scenes x 3 stages (fraction 0.40-0.52).")
w("- Backward-dominant in every scene/stage; f_raster 0.88-0.94 mature.")
w("- Room vs bicycle differ in regime (room = extreme footprint/contention; bicycle = many Gaussians) but blend_backward dominates both -> robust.")
w()
w("## RANKED OPPORTUNITIES")
w()
w("| rank | target | f(geomean) | Amdahl 2x | Amdahl perfect | score |")
w("|---|---|---|---|---|---|")
for r in OR["ranked"]:
    w(f"| {r['rank']} | {r['label'][:40]} | {r['mature_geomean_fraction']} | {r['amdahl_2x']} | {r['amdahl_perfect']} | {r['opportunity_score']} |")
w()
w("## NEGATIVE FINDINGS")
w()
for n in VD["negative_findings"]: w(f"- {n}")
w()
w("## NOT MEASURED")
w()
for n in VD["not_measured"]: w(f"- {n}")
w()
w("## VERDICT")
w()
w(f"**{VD['verdict']}** — {VD['one_line']}")
w()
w("### Mandatory final recommendation (single next target; NOT implemented in this lane)")
w()
rt=VD["recommended_next_target"]
pf_scene = ", ".join(f"{k}={v}" for k, v in rt["measured_e2e_fraction_mature_per_scene"].items())
amd_str = ", ".join(f"{k}x->{v}" for k, v in rt["amdahl_ceiling_E2E"].items())
w(f"- **Target component**: {rt['target_component']}")
w(f"- **Measured E2E fraction**: {rt['measured_e2e_fraction_mature_geomean']} (mature geomean); per-scene {pf_scene}; full-training-weighted {rt['measured_e2e_fraction_full_training_weighted']}.")
w(f"- **Dominant cost mechanism**: {rt['dominant_cost_mechanism']}")
w(f"- **Measured removable-work opportunity**: {rt['measured_removable_work_opportunity']}")
w(f"- **Amdahl ceiling**: {amd_str}")
w(f"- **Prior-art gap**: {rt['prior_art_gap']}")
w(f"- **Minimal falsification experiment**: {rt['minimal_falsification_experiment']}")
w()
w("Per objective section 0/1, FUSED-HOTSPOT0 authorizes NO optimization; the method above is for PI to design in a subsequent lane.")
w()
open(f"{HOT}/FUSED_HOTSPOT0_REPORT.md","w").write("\n".join(L)+"\n")
print("REPORT written, lines:", len(L))

# ---------------- MANIFEST.sha256 ----------------
req=["FUSED_HOTSPOT0_PREREG.json","FUSED_HOTSPOT0_BASELINE_FREEZE.json","FUSED_HOTSPOT0_TIMING_BOUNDARY.md",
     "FUSED_HOTSPOT0_CLEAN_TIMING.json","FUSED_HOTSPOT0_SOURCE_KERNEL_MAP.json","FUSED_HOTSPOT0_KERNEL_CENSUS.json",
     "FUSED_HOTSPOT0_COMPONENT_BREAKDOWN.json","FUSED_HOTSPOT0_WORKLOAD_STATS.json","FUSED_HOTSPOT0_KERNEL_CHARACTERIZATION.json",
     "FUSED_HOTSPOT0_SCALING.json","FUSED_HOTSPOT0_AMDAHL_ORACLE.json","FUSED_HOTSPOT0_REDUNDANCY_AUDIT.json",
     "FUSED_HOTSPOT0_PRIOR_ART_MAP.md","FUSED_HOTSPOT0_OPPORTUNITY_RANKING.json","FUSED_HOTSPOT0_VERDICT.json",
     "FUSED_HOTSPOT0_REPORT.md"]
lines=[]
for fn in req:
    p=f"{HOT}/{fn}"
    if os.path.exists(p):
        lines.append(f"{sha256_file(p)}  {fn}")
    else:
        lines.append(f"MISSING  {fn}")
# also include the copied source kernel map + scripts + raw as evidence (hashes)
extra=[]
for d in ["scripts","source_map","raw","profiler","logs"]:
    for root,_,files in os.walk(f"{HOT}/{d}"):
        for fl in sorted(files):
            fp=os.path.join(root,fl)
            extra.append(f"{sha256_file(fp)}  {os.path.relpath(fp,HOT)}")
lines+=sorted(extra)
open(f"{HOT}/MANIFEST.sha256","w").write("\n".join(lines)+"\n")
print("MANIFEST written, entries:", len(lines))