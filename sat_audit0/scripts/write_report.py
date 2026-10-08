import json, os, datetime

ROOT = "/mnt/storage_pool/liaoyuanjun/fused_profile0"
SYN = os.path.join(ROOT, "SAT_AUDIT0")
NOW = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
def load(p):
    with open(p) as f: return json.load(f)

sc = load(f"{SYN}/SAT_AUDIT0_SCENE_COMPARISON.json")
ca = load(f"{SYN}/SAT_AUDIT0_COMPONENT_ACCOUNTING.json")
ch = load(f"{SYN}/SAT_AUDIT0_CORRECTED_HEADROOM.json")
ct = load(f"{SYN}/SAT_AUDIT0_CLEAN_TIMING.json")
pt = load(f"{SYN}/SAT_AUDIT0_PRELOAD_TEST.json")
tl = load(f"{SYN}/SAT_AUDIT0_TIMELINE.json")
sa = load(f"{SYN}/SAT_AUDIT0_SYNC_AUDIT.json")
cpu = load(f"{SYN}/SAT_AUDIT0_CPU_PROFILE.json")
vd = load(f"{SYN}/SAT_AUDIT0_VERDICT.json")
SCENES = ["room", "bicycle", "garden"]
acc = sc["per_scene"]

L = []; w = L.append
w("# SAT-AUDIT0 Report — FasterGSFused Non-Kernel Time Audit")
w("")
w("> Generated %s. Terminal status: **PROFILE_ARTIFACT_FOUND**." % NOW)
w("> Diagnostic-only audit of the ~275-288ms non-kernel/framework time on bicycle/garden in the frozen FUSED_PROFILE0 profile.")
w("")
w("## OBSERVED")
w("")
w("The original FUSED_PROFILE0/COMPONENT_PROFILE reported per-iteration wall time with only ~8.8-14.4ms of GPU kernels, leaving ~275-288ms 'non-kernel/framework' on bicycle/garden (and ~0.1ms residual on room). Re-running the identical clean harness (PRELOADING_LEVEL=0, CUDA events, warmup 50, 200 iters, per-iter sync) on a clean A100 reproduced the cost and isolated where it sits:")
w("")
w("| scene | T_iter PL0 (ms) | rgb=data-decode (ms) | forward | loss | backward | unexplained gap |")
w("|---|---|---|---|---|---|---|")
for s in SCENES:
    a = acc[s]; sm = a
    w("| %s | %.1f | %.1f | %.2f | %.2f | %.2f | %.3f (%.1f%%) |" % (
        s, a["T_iteration_wall_PL0_ms"], a["T_data_input_image_decode_ms"],
        a["T_forward_PL0_ms"], a["T_loss_PL0_ms"], a["T_backward_PL0_ms"],
        a["T_unexplained_gap_PL0_ms"], 100*a["gap_fraction_PL0"]))
w("")
w("The timed window is FULLY accounted for (residual <0.2%); there is no missing 'framework' bucket. The entire non-kernel time is the `rgb` segment = per-iteration image decode.")
w("")
w("## EVIDENCE")
w("")
w("**MODE A (clean PL0) reproduces the cost.** bicycle %.1fms, garden %.1fms, room %.1fms — matching the original ~289/296ms on bicycle/garden. (§24 reproduction satisfied for bicycle and garden.)" % (
    acc["bicycle"]["T_iteration_wall_PL0_ms"], acc["garden"]["T_iteration_wall_PL0_ms"], acc["room"]["T_iteration_wall_PL0_ms"]))
w("")
w("**MODE B (preload) collapses the cost.** With images preloaded into VRAM (PL2) or RAM (PL1) before timing, no file reads occur in the window:")
w("")
w("| scene | PL0 (ms) | PL1 RAM (ms) | PL2 VRAM (ms) | % removed (PL2) | I/O materiality |")
w("|---|---|---|---|---|---|")
for s in SCENES:
    p = pt["per_scene"][s]
    w("| %s | %.1f | %.2f | %.2f | %.1f%% | %s |" % (
        s, p["T_PL0_ms"], p["T_PL1_ram_ms"], p["T_PL2_vram_ms"],
        100*p["pct_removed_PL2"], p["classification"]))
w("")
w("**MODE C (timeline, torch.profiler chrome trace; nsys unavailable)** shows the cost is CPU-side image decode with the GPU idle (Case 1):")
w("")
for s in SCENES:
    t = tl["per_scene"][s]
    w("- %s: `image::decode_image` = %.0fms CPU/iter (0 CUDA), `adaptive_avg_pool2d` = %.0fms CPU/iter, GPU idle during decode = %s." % (
        s, t["image_decode_cpu_ms_per_iter"], t["adaptive_avg_pool2d_cpu_ms_per_iter"], t["gpu_idle_during_decode"]))
w("")
w("**MODE D (sync audit):** ~1.01 `torch.cuda.synchronize` + ~2.5 `.item` per iter — the cost is NOT synchronization overhead.")
w("")
w("**MODE E (CPU profile):** isolated `view.rgb` (image decode) alone = %.0f/%.0f/%.0f ms/iter for room/bicycle/garden, equal to the PL0 `rgb` segment." % (
    cpu["per_scene"]["room"]["rgb_load_alone_ms_per_iter"], cpu["per_scene"]["bicycle"]["rgb_load_alone_ms_per_iter"], cpu["per_scene"]["garden"]["rgb_load_alone_ms_per_iter"]))
w("")
w("**Reproduction anomaly (room control):** the original FP1 room total was 3.835ms, but the identical harness now gives %.1fms — the original room profile was NOT reproducible, independent of the bicycle/garden artifact." % ct["per_scene"]["room"]["median_total_ms"])
w("")
w("## DIAGNOSIS")
w("")
w("`ImageData.image` returns `self._load() if self._data is None else self._data`. Under `PRELOADING_LEVEL=0` (set in `mat_*.yaml`), `_prepare_dataset` skips preloading, so `_data` stays None and **every** `view.rgb` re-decodes the JPEG from disk (`io.decode_image`) and rescales (`apply_image_scale_factor` -> `adaptive_avg_pool2d`). The FP1/FP2 `iter_body` accesses `view.rgb` inside the `t0->t1` CUDA-event boundary, charging the decode to the iteration. Official FasterGSFused training excludes dataloading (`TIMING.INCLUDE_DATALOADING_IN_TOTAL=false`) and preloads images (PRELOADING_LEVEL default 1). **The 275-288ms is per-iteration data loading, not framework compute.**")
w("")
w("## ARTIFACT CHECKS")
w("")
w("| artifact | present | evidence |")
w("|---|---|---|")
w("| data loading inside timed region | YES | preload removes 94-98% of PL0 time |")
w("| profiler-induced serialization | NO | sync ~1/iter; GPU idle during decode |")
w("| accidental synchronize | NO | sync_calls_per_iter ~1.0 |")
w("| different scene timing boundaries | YES | FP1/FP2 includes view.rgb; official excludes dataloading; room not reproducible |")
w("| debug instrumentation | NO | none in timed path |")
w("| periodic maintenance as iteration | NO | steady-state mature ckpt, update_densification_info=False |")
w("| unclassified GPU kernel time | NO | decode rows cuda=0; kernels separately accounted |")
w("| incorrect CPU+GPU summation | NO | PL0 residual <0.2% |")
w("")
w("## CORRECTED PROFILE")
w("")
w("Excluding data loading (preloaded PL2 denominator), the baseline is **raster-dominant on ALL scenes**, inverting the original 'framework-bound/saturated' reading for bicycle/garden:")
w("")
w("| scene | original f_raster (FP2) | corrected f_raster | original T_iter (ms) | corrected T_iter (ms) |")
w("|---|---|---|---|---|")
for s in SCENES:
    w("| %s | %.3f | %.3f | %.1f | %.2f |" % (
        s, acc[s]["original_f_raster_FP2"], acc[s]["corrected_f_raster"],
        acc[s]["original_T_iteration_ms_FP1"], acc[s]["T_iteration_corrected_preloadedPL2_ms"]))
w("")
w("Corrected Amdahl ceilings (if the standalone 1.742x were applicable): " +
  ", ".join("%s %.3fx" % (s, ch["per_scene"][s]["amdahl_ceiling_standalone_1.742x"]) for s in SCENES) +
  ". (The exact per-tile AABB reject remains intrinsic, C32==B32, so the realistic ceiling stays 1.0 — unchanged by this audit.)")
w("")
w("## IMPLICATIONS")
w("")
w("- The original `FUSED_PROFILE0 = DROP_AABB_PRODUCTION_HEADROOM` and `bicycle/garden framework-bound/saturated` conclusions were drawn on an artifact-inflated denominator. The Q1 gate that passed via the optimistic naive-Amdahl clause is affected: the corrected baseline is raster-bound, so raster-side levers are not obviously headroom-less.")
w("- **Per objective §21, this does NOT automatically reopen AABB-PROD0 or any optimization lane.** The AABB exact-reject-intrinsic fact (C32==B32, S_reject=1.0) is unchanged; the corrected profile changes the denominator, not that fact. Return to PI for direction.")
w("- The negative-paper backbone must be amended (§18, new amendment, no silent overwrite). See `SATURATION_SYNTHESIS/AMENDMENT_20261006_SAT_AUDIT0.json`.")
w("- Room remains reported as raster-heavy under corrected evidence (corrected f_raster=0.894), consistent with §20 (do not generalize all scenes identically).")
w("")
w("## NOT MEASURED")
w("")
w("- Nsight Systems timeline (unavailable on host); used torch.profiler chrome trace instead (lower overhead, sufficient for localization).")
w("- Production `run()` end-to-end wall time (the audit reproduces the FP1/FP2 harness boundary, not the official training loop's timer summation).")
w("- Backward/convergence re-validation under corrected timing (out of scope; A1 gate already failed on S_prod).")
w("- Whether the original 3.835ms room was measured under warm page-cache / a preloaded config (not recoverable from existing artifacts).")
w("")
w("## VERDICT")
w("")
w("**PROFILE_ARTIFACT_FOUND** — the large non-kernel time on bicycle/garden is per-iteration image decoding (data loading) inside the timed region, introduced by PRELOADING_LEVEL=0 in the FP1/FP2 harness. It is not intrinsic FasterGSFused framework work. Corrected clean timing and Amdahl ceilings are reported; no optimization lane is reopened; return to PI.")
w("")

with open(os.path.join(SYN, "SAT_AUDIT0_REPORT.md"), "w") as f:
    f.write("\n".join(L) + "\n")
print("WROTE REPORT.md lines=%d" % len(L))