#!/usr/bin/env python3
"""FUSED-HOTSPOT0 analysis: derive all artifacts from raw measurements."""
import json, os, glob, math, statistics, hashlib, datetime

P = "/mnt/storage_pool/liaoyuanjun/fused_profile0"
HOT = P + "/FUSED_HOTSPOT0"
RAW = HOT + "/raw"
SCENES = ["room", "bicycle", "garden"]
STAGES = ["s1_5k", "s2_10k", "s3_mature"]
STAGE_ITERS = {"s1_5k": 5000, "s2_10k": 5000, "s3_mature": 5000}  # 0-5k,5k-10k,10k-15k
GEN_UTC = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def load(p):
    with open(p) as f: return json.load(f)

def rnd(x, n=4):
    try: return round(float(x), n)
    except Exception: return x

# ---- census kernel-name substring -> component ----
COMP_MAP = [
    ("blend_backward", "B"),
    ("blend_cu", "F"),
    ("preprocess_backward", "PB"),
    ("preprocess_cu", "P"),
    ("adam_step", "BO"),
    ("create_instances", "I"),
    ("apply_depth_ordering", "I"),
    ("extract_instance_ranges", "I"),
    ("extract_bucket_counts", "I"),
    ("RadixSort", "S"),
    ("radix", "S"),
    ("ssim", "L"),
    ("l1_loss", "L"),
]
def comp_of(name):
    for sub, c in COMP_MAP:
        if sub in name: return c
    return "M"

# forward-path components vs backward-path vs loss
FWD_COMPS = {"P", "I", "S", "F"}
BWD_COMPS = {"PB", "B", "BO"}
LOSS_COMPS = {"L"}

# ---- cuobj resource report (re-derive by calling cuobjdump) ----
import subprocess, re
SO = P + "/pylibs/FasterGSFusedCudaBackend/_C.cpython-310-x86_64-linux-gnu.so"
so_sha = hashlib.sha256(open(SO, "rb").read()).hexdigest()
cuobj = {}
BS = {"blend_backward":256,"blend_cu":256,"preprocess_cu":128,"preprocess_backward":128,
      "adam_step":256,"create_instances":256,"apply_depth_ordering":256,"extract_instance_ranges":256,"extract_bucket_counts":256}
def est_occ(reg, sh, bs):
    if reg <= 0 or bs <= 0: return None
    by_regs = 65536 // (reg * bs)
    by_sh = (164 * 1024) // sh if sh > 0 else 32
    by_thr = 2048 // bs
    blocks = max(1, min(32, by_regs, by_sh, by_thr))
    warps = (blocks * bs) // 32
    return {"blocks_per_sm": blocks, "threads_per_sm": blocks * bs, "occupancy_pct": round(100 * warps / 64.1)}
def parse_cuobj():
    out = subprocess.run(["cuobjdump", "-res-usage", SO], capture_output=True, text=True).stdout
    lines = out.splitlines(); i = 0; recs = []
    while i < len(lines):
        m = re.match(r"\s*Function\s+(\S+)", lines[i])
        if m:
            name = m.group(1).rstrip(":"); reg = sh = None; j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("Function"):
                mm = re.search(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)", lines[j])
                if mm: reg = int(mm.group(1)); sh = int(mm.group(3)); break
                j += 1
            if reg is not None: recs.append((name, reg, sh))
            i = j
        else: i += 1
    seen = {}
    for name, reg, sh in recs:
        for w in BS:
            if w in name and w not in seen:
                seen[w] = {"registers_per_thread": reg, "static_shared_bytes": sh, "block_size": BS[w],
                           "occupancy_est": est_occ(reg, sh, BS[w])}
    return seen
CUOBJ = parse_cuobj()

# ---- assemble per scene/stage ----
DATA = {}  # (scene,stage) -> dict
for scene in SCENES:
    for stage in STAGES:
        cl = load(f"{RAW}/clean_{scene}_{stage}.json")
        ce = load(f"{RAW}/census_{scene}_{stage}.json")
        st = load(f"{RAW}/stats_{scene}_{stage}.json")
        seg = cl["segment_medians_ms"]
        T_iter = cl["median_total_ms"]
        # census component self-time (us/iter)
        comp_us = {}; comp_kern = {}
        for k in ce["kernels"]:
            c = comp_of(k["name"])
            comp_us[c] = comp_us.get(c, 0.0) + k["total_us_per_iter"]
            comp_kern.setdefault(c, []).append(k)
        fwd_us = sum(v for c, v in comp_us.items() if c in FWD_COMPS)
        bwd_us = sum(v for c, v in comp_us.items() if c in BWD_COMPS)
        fwd_seg = seg["forward"]; bwd_seg = seg["backward"]; loss_seg = seg["loss"]
        # critical-path allocation
        cp = {c: 0.0 for c in ["P","I","S","F","L","PB","B","BO","D","M","U"]}
        for c, v in comp_us.items():
            if c in FWD_COMPS and fwd_us > 0:
                cp[c] = fwd_seg * (v / fwd_us)
            elif c in BWD_COMPS and bwd_us > 0:
                cp[c] = bwd_seg * (v / bwd_us)
            elif c in LOSS_COMPS:
                cp[c] = loss_seg
            # M (torch glue) absorbed into forward/backward segment wall time; D (densification) inactive in this timed boundary
        seg_sum = fwd_seg + bwd_seg + loss_seg
        cp["U"] = max(0.0, T_iter - seg_sum)
        resolved = seg_sum
        DATA[(scene, stage)] = {"clean": cl, "census": ce, "stats": st, "seg": seg, "T_iter": T_iter,
                                "comp_us": comp_us, "comp_kern": comp_kern, "cp_ms": cp,
                                "fwd_seg": fwd_seg, "bwd_seg": bwd_seg, "loss_seg": loss_seg,
                                "fwd_us": fwd_us, "bwd_us": bwd_us, "resolved_frac": resolved / T_iter if T_iter else 0}

# ---- COMPONENT_BREAKDOWN ----
def comp_artifact():
    per = {}
    for scene in SCENES:
        for stage in STAGES:
            d = DATA[(scene, stage)]
            comps = {}
            for c in ["P", "I", "S", "F", "L", "PB", "B", "BO", "D", "M", "U"]:
                cp_ms = d["cp_ms"].get(c, 0.0)
                comps[c] = {"critical_path_ms": rnd(cp_ms), "fraction_of_iteration": rnd(cp_ms / d["T_iter"], 4),
                            "self_time_ms_from_census": rnd(d["comp_us"].get(c, 0.0) / 1000.0)}
            per[f"{scene}_{stage}"] = {
                "scene": scene, "stage": stage, "T_iteration_ms": rnd(d["T_iter"]),
                "segment_anchors_ms": {"forward": rnd(d["fwd_seg"]), "loss": rnd(d["loss_seg"]), "backward": rnd(d["bwd_seg"])},
                "components": comps,
                "resolved_fraction": rnd(d["resolved_frac"], 4),
                "accounting_gate_ge_0.90": d["resolved_frac"] >= 0.90,
                "note": "Critical-path allocation: forward segment split across P/I/S/F by census self-time share; backward segment split across PB/B/BO; loss segment -> L; residual -> U. Backward+optimizer fused (BO) reported as fused block; B and BO separated by census self-time but overlap (adam fused in backward)."
            }
    return {"schema": "FUSED_HOTSPOT0_COMPONENT_BREAKDOWN", "generated_utc": GEN_UTC,
            "method": "H0 source map + H2 chrome-trace census + H1 CUDA-event segment anchors. Critical-path (not naive summed) allocation per objective section 11/12.",
            "component_defs": {"P": "projection/preprocess (preprocess_cu)", "I": "instance gen/binning/contribution test/bookkeeping (create_instances, apply_depth_ordering, extract_instance_ranges, extract_bucket_counts)",
                               "S": "sort (DeviceRadixSort)", "F": "forward raster/blend (blend_cu)", "L": "loss (l1+fusedssim)",
                               "PB": "projection/SH/param backward outside blend (preprocess_backward)", "B": "raster backward blend (blend_backward_cu)",
                               "BO": "fused optimizer (adam_step, fused inside backward)", "D": "densification/strategy (inactive at S3)",
                               "M": "memcpy/misc torch elementwise", "U": "unresolved (sample/sync/python gap)"},
            "per_scene_stage": per}

json.dump(comp_artifact(), open(f"{HOT}/FUSED_HOTSPOT0_COMPONENT_BREAKDOWN.json", "w"), indent=2)

# ---- WORKLOAD_STATS consolidated ----
def wl_artifact():
    per = {}
    for scene in SCENES:
        for stage in STAGES:
            st = DATA[(scene, stage)]["stats"]
            per[f"{scene}_{stage}"] = st
    return {"schema": "FUSED_HOTSPOT0_WORKLOAD_STATS", "generated_utc": GEN_UTC,
            "method": "H4: single probe camera per scene/stage via _C.forward; n_instances = total per-tile intersections after intrinsic exact reject. Production forward materializes full instance buffers -> int32 overflow / OOM on large scenes (garden, bicycle s1).",
            "caveat": "n_instances is a SINGLE probe camera, not an average; per-tile list-length percentiles require retaining instance_buffers (OOM-prone) and were not measured. n_instances/n_gaussians = avg tiles covered per Gaussian (amplification factor).",
            "per_scene_stage": per}
json.dump(wl_artifact(), open(f"{HOT}/FUSED_HOTSPOT0_WORKLOAD_STATS.json", "w"), indent=2)

# ---- KERNEL_CHARACTERIZATION ----
def regime_for(w, reg, sh, occ, comp):
    if w == "blend_backward":
        return {"regime": "MIXED", "rationale": "Per-pixel reverse blend-chain replay is serial (warp.shfl transmittance/color/grad dependence) -> LATENCY/SERIAL_BOUND; per-primitive atomicAdd gradient accumulation across tiles -> ATOMIC/CONTENTION_BOUND scaling with instances/Gaussian (room 703, bicycle 207). 50% occupancy (64 reg) leaves headroom but chain is latency-limited."}
    if w == "blend_cu":
        return {"regime": "MEMORY_BANDWIDTH_BOUND", "rationale": "100% occupancy but 9.3KB static shared staging; per-pixel dependent alpha-blend chain reads/writes shared + global. High occupancy + shared staging -> bandwidth-bound, not compute-bound."}
    if w == "preprocess_backward":
        return {"regime": "COMPUTE_BOUND", "rationale": "95 reg/thread, block 128 -> 31% occupancy (register-limited). Covariance/scale/rotation/SH gradient math is ALU-heavy with few threads in flight."}
    if w == "preprocess_cu":
        return {"regime": "COMPUTE_BOUND", "rationale": "56 reg/thread, block 128 -> 56% occupancy. Projection + covariance/conic + SH evaluation per Gaussian; moderate register pressure."}
    if w == "adam_step":
        return {"regime": "MEMORY_BANDWIDTH_BOUND", "rationale": "14 reg/thread, block 256 -> 100% occupancy. Fused Adam elementwise moment updates; trivially parallel, bandwidth-limited."}
    if w == "create_instances":
        return {"regime": "MIXED", "rationale": "40 reg + 8KB shared, block 256 -> 75% occupancy. Exact contribution test (will_primitive_contribute) + warp cooperative culling; branch divergence on tile boundaries."}
    return {"regime": "UNRESOLVED", "rationale": "Bookkeeping kernel; low self-time, not characterized in detail."}
def char_artifact():
    per = {}
    for scene in SCENES:
        for stage in STAGES:
            d = DATA[(scene, stage)]
            ks = []
            for k in d["census"]["kernels"][:12]:
                w = None
                for cand in BS:
                    if cand in k["name"]: w = cand; break
                reg = CUOBJ.get(w, {}).get("registers_per_thread") if w else None
                sh = CUOBJ.get(w, {}).get("static_shared_bytes") if w else None
                occ = CUOBJ.get(w, {}).get("occupancy_est", {}).get("occupancy_pct") if w else None
                ks.append({"name": k["name"][:80], "count_per_iter": k["count_per_iter"],
                           "self_us_per_iter": k["total_us_per_iter"], "pct_kernel_time": k["pct_kernel_time"],
                           "pct_iteration_time": k["pct_iteration_time"], "component": comp_of(k["name"]),
                           "registers_per_thread": reg, "static_shared_bytes": sh, "occupancy_pct_est": occ,
                           "regime": regime_for(w, reg, sh, occ, comp_of(k["name"])) if w else None})
            per[f"{scene}_{stage}"] = {"T_iter_ms": rnd(d["T_iter"]), "total_kernel_ms_per_iter": rnd(d["census"]["total_kernel_ms_per_iter"]),
                                       "kernel_to_wall_ratio": rnd(d["census"]["total_kernel_ms_per_iter"] / d["T_iter"], 3), "kernels": ks}
    return {"schema": "FUSED_HOTSPOT0_KERNEL_CHARACTERIZATION", "generated_utc": GEN_UTC,
            "counters_status": "COUNTERS_UNAVAILABLE (ncu 2021.3.1 sections/rules path missing, no sudo). Used cuobjdump ptxas resource report + census timing + scaling + source inspection per objective section 15.",
            "so_sha256": so_sha, "resource_report_source": "cuobjdump -res-usage on frozen backend .so",
            "per_scene_stage": per}
json.dump(char_artifact(), open(f"{HOT}/FUSED_HOTSPOT0_KERNEL_CHARACTERIZATION.json", "w"), indent=2)

print("STAGE1_DONE component/workload/characterization written")
print("resolved_frac per scene/stage:")
for scene in SCENES:
    for stage in STAGES:
        print(f"  {scene}/{stage}: {DATA[(scene,stage)]['resolved_frac']:.3f}")