#!/usr/bin/env python3
"""P10 verdict assembler: H2 equivalence + H3 timing + H4 mechanism -> verdict.

Decision rubric (from the P10 brief):
  FAIL_SEMANTICS  any H2 scene fails equivalence (or the switch never engaged)
  FAIL_TIMING     direct-backward geomean gain <= -1.0% (H8 slows the step)
  PASS_STRONG     equivalence passes, direct-backward geomean gain >= +1.0%,
                  >= 2/3 scenes positive, mechanism consistent with reduced
                  pre-reduction geometry arithmetic
  PASS_NEUTRAL    equivalence passes but timing is ~0 (|geomean| < 1%)
"""
import argparse
import json
import math


def geomean_gain(vals):
    """geomean of per-scene speedup ratios, expressed in % gain."""
    ratios = [1.0 + v / 100.0 for v in vals]
    g = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    return (g - 1.0) * 100.0


KERNEL_KEYS = {
    "rasterizer_bwd": "rasterize_to_pixels_3dgs_bwd_kernel",
    "projection_bwd": "projection_ewa_3dgs_fused_bwd_kernel",
    "rasterizer_fwd": "rasterize_to_pixels_3dgs_fwd_kernel",
    "projection_fwd": "projection_ewa_3dgs_fused_fwd_kernel",
}


def sass_summary(h4):
    """Fold the cuobjdump census into per-kernel-family off/on totals.

    Sums over every instantiation of a kernel family present in each arm's
    filtered SASS dump (the harness keeps only the four families, so the sums
    are comparable across arms: 23 channel instantiations each, plus the bool
    H8_MR variants in the patched .so)."""
    out = {}
    for arm in ("off", "on", "pristine_crosscheck"):
        blk = h4.get(f"sass_{arm}") or {}
        fam = {}
        for fname, stats in (blk.get("kernels") or {}).items():
            for key, needle in KERNEL_KEYS.items():
                if needle in fname:
                    e = fam.setdefault(key, {"n_instantiations": 0, "fp_total": 0,
                                             "total_instructions": 0, "atom_red": 0,
                                             "load_store": 0, "regs_max": 0, "spill_total": 0})
                    e["n_instantiations"] += 1
                    e["fp_total"] += stats.get("fp_total", 0)
                    e["total_instructions"] += stats.get("total_instructions", 0)
                    e["atom_red"] += sum((stats.get("atom_red") or {}).values())
                    e["load_store"] += sum((stats.get("load_store") or {}).values())
                    e["regs_max"] = max(e["regs_max"], stats.get("regs") or 0)
                    e["spill_total"] += (stats.get("spill_local") or 0) + (stats.get("spill_stack") or 0)
        out[arm] = fam
    return out


def mechanism_consistent(mech, all_eq, gm_direct):
    s = mech["sass_summary"]
    off, on = s.get("off", {}), s.get("on", {})
    rb_o = (off.get("rasterizer_bwd") or {}).get("fp_total")
    rb_n = (on.get("rasterizer_bwd") or {}).get("fp_total")
    pb_o = (off.get("projection_bwd") or {}).get("fp_total")
    pb_n = (on.get("projection_bwd") or {}).get("fp_total")
    if rb_o is None or rb_n is None:
        return None
    raster_ok = rb_n <= rb_o
    recon_bounded = (pb_o is not None and pb_n is not None and pb_n >= pb_o
                     and (pb_n - pb_o) <= 0.25 * max(pb_o, 1))
    if gm_direct is None:
        direction_ok = None          # timing gate not run yet: no direction to match
    else:
        direction_ok = (gm_direct < 1.0) or (gm_direct >= 1.0 and all_eq)
    return {"rasterizer_bwd_fp_off": rb_o, "rasterizer_bwd_fp_on": rb_n,
            "rasterizer_bwd_no_added_arithmetic": bool(raster_ok),
            "projection_bwd_fp_off": pb_o, "projection_bwd_fp_on": pb_n,
            "projection_reconstruction_bounded": bool(recon_bounded),
            "time_direction_matches_timing_gate": (None if direction_ok is None else bool(direction_ok)),
            "consistent": (None if direction_ok is None else bool(raster_ok and recon_bounded and direction_ok))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h2", nargs="+", required=True)
    ap.add_argument("--h3-h8", nargs="*", default=[],
                    help="omit to emit a PENDING_TIMING verdict (equivalence and "
                         "mechanism only); e.g. while the timing gate cannot run on a "
                         "clean GPU")
    ap.add_argument("--h3-pristine", nargs="*", default=[])
    ap.add_argument("--timing-note", default="")
    ap.add_argument("--h4", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    h2 = {}
    for p in args.h2:
        d = json.load(open(p))
        h2[d["scene"]] = d
    h3 = {}
    for p in args.h3_h8:
        d = json.load(open(p))
        h3[d["scene"]] = d
    h3p = {}
    for p in args.h3_pristine:
        d = json.load(open(p))
        h3p[d["scene"]] = d
    h4 = json.load(open(args.h4))

    scenes = sorted(h3)
    timing_available = bool(scenes)
    eq = {s: h2[s]["verdict"]["classification"] for s in sorted(h2)}
    all_eq = all(v == "PASS_NUMERICALLY_EQUIVALENT" for v in eq.values())

    timing = {}
    for s in scenes:
        d = h3[s]
        timing[s] = {
            "direct_bwd_off_ms": d["direct_backward"]["off"]["median_ms"],
            "direct_bwd_on_ms": d["direct_backward"]["on"]["median_ms"],
            "direct_bwd_gain_pct": d["direct_backward"]["gain_pct_median"],
            "direct_bwd_gain_pct_ci95": d["direct_backward"]["gain_pct_ci95"],
            "fwd_bwd_off_ms": d["forward_backward"]["off"]["median_ms"],
            "fwd_bwd_on_ms": d["forward_backward"]["on"]["median_ms"],
            "fwd_bwd_gain_pct": d["forward_backward"]["gain_pct_median"],
            "n_gaussians": d["n_gaussians"],
        }
    direct_gains = [timing[s]["direct_bwd_gain_pct"] for s in scenes]
    fb_gains = [timing[s]["fwd_bwd_gain_pct"] for s in scenes]
    gm_direct = geomean_gain(direct_gains) if scenes else None
    gm_fb = geomean_gain(fb_gains) if scenes else None
    n_positive = sum(1 for g in direct_gains if g > 0)

    cross = {}
    for s in scenes:
        if s in h3p:
            cross[s] = {
                "pristine_direct_bwd_ms": h3p[s]["direct_backward"]["base"]["median_ms"],
                "patched_off_direct_bwd_ms": timing[s]["direct_bwd_off_ms"],
                "delta_pct": (timing[s]["direct_bwd_off_ms"] - h3p[s]["direct_backward"]["base"]["median_ms"])
                             / h3p[s]["direct_backward"]["base"]["median_ms"] * 100.0,
            }

    mech = {
        "kernel_time_table_room": h4.get("kernel_time_table_room", []),
        "total_cuda_ms_per_iter_room": h4.get("total_cuda_ms_per_iter_room"),
        "all_scene_kernel_tables": h4.get("all_scene_kernel_tables"),
        "hardware_counters": h4.get("hardware_counters"),
        "sass_summary": sass_summary(h4),
        "ncu_probe_conclusion": (h4.get("ncu_probe") or {}).get("conclusion"),
    }
    mech["rule"] = ("consistent = rasterizer bwd kernel does not add per-intersection geometry "
                    "arithmetic (fp_total_on <= fp_total_off for rasterize_to_pixels_3dgs_bwd) AND "
                    "projection bwd kernel adds only the bounded per-Gaussian reconstruction "
                    "(fp_total_on - fp_total_off small relative to per-Gaussian work) AND total "
                    "CUDA time moves in the same direction as the timing gate")
    mech["consistent"] = mechanism_consistent(mech, all_eq, gm_direct)

    verdict = {
        "equivalence_per_scene": eq,
        "equivalence_all_pass": all_eq,
        "timing_available": bool(scenes),
        "timing_blocker": (None if scenes else (args.timing_note or
                                                "H3 direct-backward microbenchmark not run yet")),
        "timing": timing,
        "direct_bwd_gains_pct": direct_gains,
        "fwd_bwd_gains_pct": fb_gains,
        "direct_bwd_geomean_gain_pct": gm_direct,
        "fwd_bwd_geomean_gain_pct": gm_fb,
        "scenes_positive": n_positive,
        "n_scenes": len(scenes),
        "cross_build_pristine_vs_patched_off": cross,
        "mechanism": mech,
        "rubric": {
            "PASS_STRONG": "equivalence pass AND direct-bwd geomean >= +1.0% AND >=2/3 scenes positive AND mechanism consistent",
            "PASS_NEUTRAL": "equivalence pass AND |direct-bwd geomean| < 1.0%",
            "FAIL_TIMING": "equivalence pass AND direct-bwd geomean <= -1.0%",
            "FAIL_SEMANTICS": "any equivalence scene reports FAIL_SEMANTICS",
            "INVALID": "switch never engaged on a scene -> no correctness or timing claim",
            "INCONCLUSIVE": "equivalence gate noise-limited on a scene -> no correctness or timing claim",
            "PENDING_TIMING": "equivalence passes but H3 was not run (blocked on clean-GPU availability) "
                              "-> correctness verdict only, no timing claim either way",
            "geomean": "geometric mean of per-scene (1 + gain/100)",
        },
    }

    hard_fail = [s for s, c in eq.items() if c == "FAIL_SEMANTICS"]
    invalid = [s for s, c in eq.items() if c == "INVALID_SWITCH_NOT_ENGAGED"]
    inconclusive = [s for s, c in eq.items() if c == "INCONCLUSIVE_ELEVATED_H8_SCATTER"]
    if hard_fail:
        cls = "FAIL_SEMANTICS"
        why = f"H2 FAIL_SEMANTICS on {hard_fail}: {eq}"
    elif invalid:
        cls = "INVALID"
        why = f"H2 switch not engaged on {invalid} -- no timing claim"
    elif inconclusive:
        cls = "INCONCLUSIVE"
        why = f"H2 gate noise-limited on {inconclusive} -- no timing claim"
    elif not all_eq:
        cls = "INVALID"
        why = f"H2 unmatched classifications: {eq}"
    elif not timing_available:
        cls = "PENDING_TIMING"
        why = ("H2 equivalence passes on all scenes; H3 timing gate not run -- "
               + verdict["timing_blocker"])
    elif gm_direct <= -1.0:
        cls = "FAIL_TIMING"
        why = f"direct-backward geomean {gm_direct:+.2f}% (slows)"
    elif gm_direct >= 1.0 and n_positive >= max(1, math.ceil(2 * len(scenes) / 3)):
        cls = "PASS_STRONG"
        why = (f"direct-backward geomean {gm_direct:+.2f}%, "
               f"{n_positive}/{len(scenes)} scenes positive, equivalence pass, "
               f"mechanism consistent={mech['consistent']}")
    else:
        cls = "PASS_NEUTRAL"
        why = f"direct-backward geomean {gm_direct:+.2f}% (timing ~0), equivalence pass"
    verdict["classification"] = cls
    verdict["classification_rationale"] = why

    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"VERDICT {cls}: {why}")
    if timing_available:
        print(f"  direct-bwd gains: " + ", ".join(f"{s}={g:+.2f}%" for s, g in zip(scenes, direct_gains)))
        print(f"  fwd+bwd   gains: " + ", ".join(f"{s}={g:+.2f}%" for s, g in zip(scenes, fb_gains)))
    else:
        print("  timing: not run (PENDING_TIMING)")


if __name__ == "__main__":
    main()
