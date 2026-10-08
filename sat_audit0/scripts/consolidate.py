import json, os, datetime

ROOT = "/mnt/storage_pool/liaoyuanjun/fused_profile0"
RAW = os.path.join(ROOT, "SAT_AUDIT0", "raw")
SYN = os.path.join(ROOT, "SAT_AUDIT0")
NOW = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def load(p):
    with open(p) as f: return json.load(f)
def dump(name, obj):
    with open(os.path.join(SYN, name), "w") as f: json.dump(obj, f, indent=2)
def sload(scene, kind):
    if scene == "room":
        m = {"clean": "smoke_room_clean.json", "preload_pl2": "smoke_room_preload.json",
             "preload_pl1": "smoke_room_preloadram.json", "sync": "smoke_room_sync.json",
             "cpu": "smoke_room_cpu.json", "timeline": "smoke_room_timeline.json"}
        fn = m[kind]
    else:
        fn = f"{scene}_{kind}.json"
    p = os.path.join(RAW, fn)
    return load(p) if os.path.exists(p) else None

SCENES = ["room", "bicycle", "garden"]
CKPT = {
 "room": "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_room_2026-10-06-01-01-33/checkpoints/final.pt",
 "bicycle": "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_bicycle_2026-10-06-01-01-33/checkpoints/final.pt",
 "garden": "/mnt/storage_pool/liaoyuanjun/strong_baselines/nerficg/output/FasterGSFused/fp1_garden_2026-10-06-01-32-48/checkpoints/final.pt",
}
fp2 = load(f"{ROOT}/FUSED_PROFILE0_COMPONENT_PROFILE.json")
clean = {s: sload(s, "clean") for s in SCENES}
pl1   = {s: sload(s, "preload_pl1") for s in SCENES}
pl2   = {s: sload(s, "preload_pl2") for s in SCENES}
sync  = {s: sload(s, "sync") for s in SCENES}
cpu   = {s: sload(s, "cpu") for s in SCENES}
tl    = {s: sload(s, "timeline") for s in SCENES}

acc = {}
for s in SCENES:
    c = clean[s]; p2 = pl2[s]; p1 = pl1[s]
    cseg = c["segment_medians_ms"]; p2seg = p2["segment_medians_ms"]
    T_pl0 = c["median_total_ms"]; T_data = cseg["rgb"]
    T_fwd0 = cseg["forward"]; T_loss0 = cseg["loss"]; T_bwd0 = cseg["backward"]; gap0 = c["unaccounted_gap_ms"]
    T_corr = p2["median_total_ms"]
    T_gpu_active = p2seg["forward"] + p2seg["loss"] + p2seg["backward"]
    T_raster_plus_opt = p2seg["forward"] + p2seg["backward"]
    f_raster_corr = T_raster_plus_opt / T_corr if T_corr else None
    io_materiality = abs(T_pl0 - p1["median_total_ms"]) / T_pl0 if T_pl0 else None
    pct_removed_pl2 = (T_pl0 - T_corr) / T_pl0 if T_pl0 else None
    acc[s] = {
        "n_gaussians": c["n_gaussians"], "checkpoint": c["checkpoint"],
        "T_iteration_wall_PL0_ms": round(T_pl0, 3),
        "T_data_input_image_decode_ms": round(T_data, 3),
        "T_forward_PL0_ms": round(T_fwd0, 3), "T_loss_PL0_ms": round(T_loss0, 3),
        "T_backward_PL0_ms": round(T_bwd0, 3),
        "T_unexplained_gap_PL0_ms": round(gap0, 3), "gap_fraction_PL0": c["gap_fraction"],
        "T_iteration_corrected_preloadedPL2_ms": round(T_corr, 3),
        "T_gpu_active_PL2_ms": round(T_gpu_active, 3),
        "T_raster_plus_optimizer_PL2_ms": round(T_raster_plus_opt, 3),
        "corrected_f_raster": round(f_raster_corr, 4) if f_raster_corr else None,
        "original_f_raster_FP2": fp2["scenes"][s]["f_raster"],
        "original_T_iteration_ms_FP1": round(fp2["scenes"][s]["T_iteration_ms_median"], 3),
        "preload_PL1_total_ms": round(p1["median_total_ms"], 3),
        "preload_PL2_total_ms": round(T_corr, 3),
        "pct_time_removed_by_preload_PL2": round(pct_removed_pl2, 4) if pct_removed_pl2 is not None else None,
        "io_materiality_fraction_PL1": round(io_materiality, 4) if io_materiality is not None else None,
        "sync_calls_per_iter": sync[s]["sync_calls_per_iter"],
        "rgb_load_alone_ms": cpu[s]["rgb_load_alone_ms_per_iter"],
    }

scene_cmp = {
  "schema": "SAT_AUDIT0_SCENE_COMPARISON_V1", "generated_utc": NOW,
  "method": "Same baseline (FasterGSFused TS16), same checkpoints, same iteration semantics. PL0 (disk decode in window) vs PL1 (RAM preload) vs PL2 (VRAM preload).",
  "per_scene": acc,
  "summary": {
    "pct_time_removed_by_preload": {s: acc[s]["pct_time_removed_by_preload_PL2"] for s in SCENES},
    "io_materiality_all_ge_10pct": all(acc[s]["io_materiality_fraction_PL1"] >= 0.10 for s in SCENES),
    "note": "Preloading removes 94-98% of PL0 iteration time on every scene. The 'non-kernel/framework' time is per-iteration image decode + adaptive_avg_pool2d resize + H2D copy, not framework compute.",
    "room_control_note": "Room is the mandatory control. Room PL0 is also decode-dominated (98.9ms of 104.7ms). The ORIGINAL FP1 room (3.835ms) did NOT reproduce (reproduced 102.4ms with the identical original harness), evidencing a reproducibility problem in the original profile independent of the bicycle/garden artifact.",
  },
}
dump("SAT_AUDIT0_SCENE_COMPARISON.json", scene_cmp)

comp = {"schema": "SAT_AUDIT0_COMPONENT_ACCOUNTING_V1", "generated_utc": NOW,
        "definition": "T_iteration_wall decomposed. PL0 = original-style timed window (includes data load). Corrected = preloaded PL2 (true production iteration).",
        "per_scene": {}}
for s in SCENES:
    a = acc[s]; p2 = pl2[s]; p2seg = p2["segment_medians_ms"]
    comp["per_scene"][s] = {
        "PL0_timed_window": {
            "T_iteration_wall_ms": a["T_iteration_wall_PL0_ms"],
            "T_data_input_image_decode_ms": a["T_data_input_image_decode_ms"],
            "T_forward_ms": a["T_forward_PL0_ms"], "T_loss_ms": a["T_loss_PL0_ms"],
            "T_backward_optimizer_ms": a["T_backward_PL0_ms"],
            "T_sync_wait_ms": None, "T_logging_metrics_ms": 0.0,
            "T_python_other_ms": round(a["T_iteration_wall_PL0_ms"] - (a["T_data_input_image_decode_ms"]+a["T_forward_PL0_ms"]+a["T_loss_PL0_ms"]+a["T_backward_PL0_ms"]), 4),
            "T_unexplained_ms": a["T_unexplained_gap_PL0_ms"],
        },
        "corrected_preloaded_PL2": {
            "T_iteration_wall_ms": a["T_iteration_corrected_preloadedPL2_ms"],
            "T_gpu_active_ms": a["T_gpu_active_PL2_ms"],
            "T_raster_forward_ms": p2seg["forward"], "T_loss_ms": p2seg["loss"],
            "T_backward_optimizer_ms": p2seg["backward"], "T_data_input_ms": p2seg["rgb"],
            "T_unexplained_ms": round(a["T_iteration_corrected_preloadedPL2_ms"] - a["T_gpu_active_PL2_ms"] - p2seg["sample"] - p2seg["bg"], 4),
        },
        "accounting_residual_fraction_PL0": round(a["T_unexplained_gap_PL0_ms"] / a["T_iteration_wall_PL0_ms"], 4) if a["T_iteration_wall_PL0_ms"] else None,
    }
comp["summary"] = {
    "max_accounting_residual_PL0": max(comp["per_scene"][s]["accounting_residual_fraction_PL0"] for s in SCENES),
    "note": "PL0 residual (unexplained gap / wall) is <0.2% on all scenes: the timed window is FULLY accounted for by image-decode + forward + loss + backward. The large non-kernel time is classified as T_data_input, not 'framework'.",
}
dump("SAT_AUDIT0_COMPONENT_ACCOUNTING.json", comp)

def amdahl(f, s): return 1.0/((1.0-f)+f/s) if f is not None else None
ch = {"schema": "SAT_AUDIT0_CORRECTED_HEADROOM_V1", "generated_utc": NOW,
      "standalone_reference_speedup": 1.742, "formula": "E2E_upper = 1/((1-f_raster)+f_raster/S)", "per_scene": {}}
for s in SCENES:
    f = acc[s]["corrected_f_raster"]
    ch["per_scene"][s] = {
        "corrected_f_raster": f, "original_f_raster_FP2": acc[s]["original_f_raster_FP2"],
        "corrected_T_iteration_ms": acc[s]["T_iteration_corrected_preloadedPL2_ms"],
        "amdahl_ceiling_standalone_1.742x": round(amdahl(f, 1.742), 4) if f else None,
        "amdahl_ceiling_reject_intrinsic_1.0x": 1.0,
        "note": "Corrected f_raster uses the preloaded PL2 denominator. All three scenes become raster-dominant (f~0.88-0.94), inverting the original 'framework-bound' reading for bicycle/garden.",
    }
ch["summary"] = {
    "corrected_f_raster_range": [min(acc[s]["corrected_f_raster"] for s in SCENES), max(acc[s]["corrected_f_raster"] for s in SCENES)],
    "original_f_raster_range": [min(acc[s]["original_f_raster_FP2"] for s in SCENES), max(acc[s]["original_f_raster_FP2"] for s in SCENES)],
    "key_correction": "bicycle f_raster 0.042 -> ~0.94; garden 0.023 -> ~0.88. The baseline is NOT framework-bound/saturated; it is raster-bound once data loading is excluded. The original saturation conclusion was a measurement artifact.",
    "caveat": "Per objective section 21, corrected substantial raster headroom is REPORTED but does NOT automatically reopen AABB-PROD0 or any optimization lane. Return to PI.",
}
dump("SAT_AUDIT0_CORRECTED_HEADROOM.json", ch)

ct = {"schema": "SAT_AUDIT0_CLEAN_TIMING_V1", "generated_utc": NOW,
      "method": "MODE A: clean uninstrumented production iteration (CUDA events, per-iter sync, warmup 50, iters 200), PRELOADING_LEVEL=0. Reproduces the FP1/FP2 timed window. Re-run on bicycle and garden per section 24.",
      "per_scene": {}}
for s in SCENES:
    c = clean[s]
    ct["per_scene"][s] = {
        "median_total_ms": c["median_total_ms"], "iters": c["iters"], "warmup": c["warmup"],
        "segment_medians_ms": c["segment_medians_ms"], "cpu_perf_segment_medians_ms": c["cpu_perf_segment_medians_ms"],
        "sum_segment_medians_ms": c["sum_segment_medians_ms"], "unaccounted_gap_ms": c["unaccounted_gap_ms"],
        "gap_fraction": c["gap_fraction"], "gpu_pci_bus_id": c["gpu_pci_bus_id"],
        "original_FP1_median_ms": acc[s]["original_T_iteration_ms_FP1"],
        "reproduced_original": abs(c["median_total_ms"] - acc[s]["original_T_iteration_ms_FP1"]) / acc[s]["original_T_iteration_ms_FP1"] < 0.10,
    }
ct["summary"] = {
    "bicycle_reproduced": ct["per_scene"]["bicycle"]["reproduced_original"],
    "garden_reproduced": ct["per_scene"]["garden"]["reproduced_original"],
    "room_reproduced": ct["per_scene"]["room"]["reproduced_original"],
    "note": "bicycle/garden reproduce the original ~277-300ms (decode-dominated). room does NOT reproduce (102ms now vs 3.84ms original) -> original room profile was not reproducible.",
}
dump("SAT_AUDIT0_CLEAN_TIMING.json", ct)

pt = {"schema": "SAT_AUDIT0_PRELOAD_TEST_V1", "generated_utc": NOW,
      "method": "MODE B: identical iteration with images preloaded into VRAM (PL2) or RAM (PL1) before timing. No file reads inside the timed window.",
      "per_scene": {}}
for s in SCENES:
    pt["per_scene"][s] = {
        "T_PL0_ms": acc[s]["T_iteration_wall_PL0_ms"], "T_PL1_ram_ms": acc[s]["preload_PL1_total_ms"],
        "T_PL2_vram_ms": acc[s]["preload_PL2_total_ms"], "PL1_segments_ms": pl1[s]["segment_medians_ms"],
        "PL2_segments_ms": pl2[s]["segment_medians_ms"], "io_materiality_PL1": acc[s]["io_materiality_fraction_PL1"],
        "classification": "material (>=10%)" if acc[s]["io_materiality_fraction_PL1"] >= 0.10 else ("moderate (3-10%)" if acc[s]["io_materiality_fraction_PL1"] >= 0.03 else "non-material (<3%)"),
        "pct_removed_PL2": acc[s]["pct_time_removed_by_preload_PL2"],
    }
pt["summary"] = {"exoneration": "Data loading is MATERIAL on all scenes (94-97% of PL0 time). It is data loading, not framework; the official FasterGSFused training excludes it (INCLUDE_DATALOADING_IN_TOTAL=false, production preloads images)."}
dump("SAT_AUDIT0_PRELOAD_TEST.json", pt)

tml = {"schema": "SAT_AUDIT0_TIMELINE_V1", "generated_utc": NOW,
       "method": "MODE C: torch.profiler chrome trace (nsys unavailable), ~20 iters. NOT used as headline wall time.", "per_scene": {}}
for s in SCENES:
    rows = tl[s]["rows"]; n = tl[s]["iters_profiled"]
    decode_cpu_us = sum(r["self_cpu_time_us"] for r in rows if "decode_image" in r["name"])
    pool_cpu_us = sum(r["self_cpu_time_us"] for r in rows if "adaptive_avg_pool2d" in r["name"])
    gpu_idle_during_decode = all(r["self_cuda_time_us"] == 0 for r in rows if "decode_image" in r["name"])
    tml["per_scene"][s] = {
        "trace": tl[s]["trace"], "iters_profiled": n,
        "image_decode_cpu_ms_per_iter": round(decode_cpu_us/1000.0/n, 3),
        "adaptive_avg_pool2d_cpu_ms_per_iter": round(pool_cpu_us/1000.0/n, 3),
        "gpu_idle_during_decode": gpu_idle_during_decode, "top_rows": rows[:10],
        "case": "Case 1: GPU mostly idle during CPU image decode/resize; CPU blocks next launch.",
    }
dump("SAT_AUDIT0_TIMELINE.json", tml)

sa = {"schema": "SAT_AUDIT0_SYNC_AUDIT_V1", "generated_utc": NOW,
      "method": "MODE D: count torch.cuda.synchronize / Tensor.item during the timed window. Determine whether apparent framework time is synchronization overhead.",
      "per_scene": {}}
for s in SCENES:
    sa["per_scene"][s] = {"median_total_PL0_ms": acc[s]["T_iteration_wall_PL0_ms"], "sync_calls_per_iter": sync[s]["sync_calls_per_iter"]}
sa["summary"] = {"synchronize_per_iter": 1.01, "item_per_iter": 2.5,
                 "conclusion": "Synchronization frequency is minimal (~1 synchronize + ~2.5 .item per iter). The 275-288ms is NOT synchronization overhead; it is CPU image decode."}
dump("SAT_AUDIT0_SYNC_AUDIT.json", sa)

cp = {"schema": "SAT_AUDIT0_CPU_PROFILE_V1", "generated_utc": NOW,
      "method": "MODE E: cProfile the timed window + isolated rgb-load timing.", "per_scene": {}}
for s in SCENES:
    cp["per_scene"][s] = {"rgb_load_alone_ms_per_iter": cpu[s]["rgb_load_alone_ms_per_iter"],
                          "cprofile_top": "\n".join(cpu[s]["cprofile_text"].splitlines()[:25])}
cp["summary"] = {"note": "Isolated view.rgb (image decode) alone equals the PL0 rgb segment (~99-290ms/iter), confirming the decode is the cost. cProfile top hotspots are image decode / PIL / resize."}
dump("SAT_AUDIT0_CPU_PROFILE.json", cp)

print("OK consolidated 8 JSON")
print("corrected f_raster:", {s: acc[s]["corrected_f_raster"] for s in SCENES})
print("pct removed by preload:", {s: acc[s]["pct_time_removed_by_preload_PL2"] for s in SCENES})
print("max residual PL0:", comp["summary"]["max_accounting_residual_PL0"])