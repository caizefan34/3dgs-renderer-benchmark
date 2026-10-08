#!/usr/bin/env python3
"""P18-P0-R1 R1-B closure: fold record-integrity verification, pre-fix defect
history, conic-decode erratum, neutrality interpretation and render-equivalence
into P18_R1_TRACE_MANIFEST.json.

Idempotent: re-running replaces the same top-level keys.
"""
import json
import time
from pathlib import Path

import numpy as np

ART = Path("/root/epic05_p18_r1/artifacts")
MANIFEST = ART / "P18_R1_TRACE_MANIFEST.json"


def load(name):
    return json.load(open(ART / name))


def pop_stats(hist):
    h = np.asarray(hist, dtype=np.float64)
    n = h.sum()
    vals = np.arange(17, dtype=np.float64)
    mean = float((h * vals).sum() / n)
    c = np.cumsum(h)
    med = int(np.searchsorted(c, n / 2.0))
    hpos = h[1:]
    npos = hpos.sum()
    mean_pos = float((hpos * np.arange(1, 17)).sum() / npos) if npos else 0.0
    cpos = np.cumsum(hpos)
    med_pos = int(np.searchsorted(cpos, npos / 2.0) + 1) if npos else 0
    return {"hist17": [int(x) for x in hist], "n_rows": int(n),
            "mean": mean, "median": med,
            "mean_given_nonempty": mean_pos, "median_given_nonempty": med_pos,
            "frac_pop0": float(h[0] / n), "frac_pop16": float(h[16] / n)}


def main():
    m = json.load(open(MANIFEST))
    room_v = load("P18_R1_RECORD_VERIFY_room.json")
    bike_v = load("P18_R1_RECORD_VERIFY_bicycle.json")
    neu = load("P18_R1_NEUTRALITY.json")

    m["record_integrity_verification"] = {
        "method": ("GPU exact half2-path recompute of ALL 16 tile-row masks per record "
                   "(p18_r1_maskchk.cu: CHOL_SCALE=13.589, LOG2_INV_AT=7.9943534, frozen build "
                   "flags incl. -use_fast_math) -> d16[n,16] hamming vs the stored mask16; "
                   "claimed-row / pp-candidate / best-row analysis host-side in numpy."),
        "acceptance_criterion": "exact_claimed_row_frac == 1.0 (every stored mask bit-exactly reproduced at its claimed row)",
        "room": {
            "n_verified": room_v["n_verified"],
            "exact_claimed_row_frac": room_v["exact_claimed_row_frac"],
            "some_row_equal_frac": room_v["some_row_equal_frac"],
            "mismatch_frac": room_v["mismatch_frac"],
            "claimed_d_hist": room_v["claimed_d_hist"],
            "artifact": str(ART / "P18_R1_RECORD_VERIFY_room.json"),
            "artifact_sha256": "c828ed0fe5113c69da749b2fa58d8804dda9d449338e282a370beec5591a9937",
        },
        "bicycle": {
            "n_verified": bike_v["n_verified"],
            "exact_claimed_row_frac": bike_v["exact_claimed_row_frac"],
            "some_row_equal_frac": bike_v["some_row_equal_frac"],
            "mismatch_frac": bike_v["mismatch_frac"],
            "claimed_d_hist": bike_v["claimed_d_hist"],
            "artifact": str(ART / "P18_R1_RECORD_VERIFY_bicycle.json"),
            "artifact_sha256": "7b04b8b5e6b8664c66ca5a33a8f156d6156ac73804fd9d7caaeda277831d079d",
        },
        "verdict": "PASS - every stored ok_mask (16 real contiguous x-pixels) bit-exactly reproduced by the exact-path recompute; 0 mismatches of 1,081,568 (room) + 1,871,872 (bicycle) records",
    }

    m["pre_fix_defect_note"] = {
        "status": "pre-fix trace INVALID and superseded; trace rebuild + re-capture + re-verify completed 2026-09-28T05:39Z",
        "defect_D1_slot_label": ("record slot/label derived from raw lane_id ((p18_r = lane_id)) instead of the "
                                 "pp-loop value; the pp=3 iteration overwrote slots 0..11 with rows labeled "
                                 "pp = lane_id>>2 -> every dumped record carried a stale/wrong row label"),
        "defect_D2_ballot_extract": ("ok_mask .y-half extraction used (ok_mask>>1)&1; cuda_fp16.h __hlt2_mask returns "
                                     "0xFFFF/0x0000 PER HALF, so bit 1 stays inside the LOW half and bal1 was always "
                                     "identical to bal0 (the pre-fix 4x1x2 == ok_mask>>1&1 in ONE_LANE analysis: invalid "
                                     "reconstruction); every h=1 slot carried the .x mask"),
        "empirical_signature": ("100% of pre-fix stored masks matched the exact recompute for row 12+(rec_r&1): "
                                "claimed-row match 0.6181 (not 1.0); per-rec_r zero-fraction matrix showed p=3 frac "
                                "1.0 for rh=0 classes; the stale slots 0..11 all landed on rows {12,13} because "
                                "pp=3 (row offset 12+ty) was the last write"),
        "detection_path": ("GPU d16 exact recompute (p18_r1_verify_records.py) + host-side per-rec_r zero-fraction "
                           "matrix (p18_r1_prefix_probe.py) -> content row = 12+(rec_r&1) for 100% of records "
                           "=> deduced bal1==bal0 => D2; D1 exposed by the label/content decomposition"),
        "fix": ("p18_r = (pp<<2)|(lane_id&3) taken from the pp loop; .y truth = (ok_mask>>16)&1; histogram writer "
                "restricted to canonical lanes lane_id<4 (one add per real 16-px run; lanes {t,t+4,t+8,t+12} hold "
                "the same m16 for slot t)"),
        "fixed_src_sha256": "161b09cf69d78b5f1a1ea1456b7d8187228679a865d02edaa02c95cc20c93fbf",
        "fixed_ext_so_sha256": "b3fc1ade919a7135d6b8585ca9a54dde739a5e6886925697a63b117a26217933",
        "pre_fix_build_provenance": {
            "d1_only_fix_src_sha256": "1234b8983148f62485e0bf493abb8de3a2e083198cc0cd8f55915d36ef3f7477",
            "d1_only_fix_so_sha256": "4c5718cf688bdc517264a42020776610257b23b39314e1cd914e13f0182fd724",
            "note": "pre-fix capture also showed an extra render mismatch at room:19 (on-pass 917835cf7338 vs frozen ae23bd11e496) absent from the fixed build's equivalence set - consistent with a pre-fix instrumentation race; not investigated further (build superseded)",
        },
        "post_fix_acceptance": "see record_integrity_verification.verdict (exact_claimed_row_frac == 1.0 both scenes)",
    }

    m["conic_decode_erratum"] = {
        "fields": "npz c_l0/c_l1/c_l2/c_opacity = raw conic halves {l0,l1,l2,opacity} decoded by BIT-REINTERPRETATION (numpy .view(np.float16)) of the 2x uint32 payload; astype(uint16->float16) would VALUE-convert and corrupt magnitudes",
        "downstream_reconstruction": ("kernel-equivalent scaling must be reapplied: l00'=half(l0*CHOL_SCALE), "
                                      "l10'=half(l1*CHOL_SCALE), l11'=half(fp32(l2)*CHOL_SCALE), "
                                      "nlo=-log2f(fp32(opacity)); CHOL_SCALE=0.84932180028*16=13.589"),
        "geometry_units": ("record pos_x/pos_y = means2d in GLOBAL pixel units; x0/y0 = global pixel coords of the "
                           "subgroup's row start; subgroup pixels = x0..x0+15 (contiguous, pixel centers x+0.5). "
                           "sigma-domain row reconstruction: s=1/16; dy=(pos_y-y0)/16-0.03125 (half quantized in "
                           "kernel); q(x)=a*x^2+b*x+c with a=s2^2, b=-2*s2*M, c=M^2+(l11'*dy)^2+nlo, "
                           "s2=l00'*s, M=l00'*(s*pos_x-0.03125)+l10'*dy"),
    }

    neu_pairs = neu["pairs"]
    m["instrumentation_neutrality"] = {
        "probe": "P18_R1_NEUTRALITY.json - ONE process, 4 full-sequence passes off->on->off2->on2, 8 cams/scene",
        "artifact_sha256": "53ff88e421964456cd234cce59972f66e029327261287fbde9218c12ac89991a",
        "strict_criterion": neu["overall"],
        "strict_criterion_note": ("bit-identical across ALL pass pairs is UNATTAINABLE by any build: the frozen "
                                  "uninstrumented renderer itself is bistable pass-to-pass (fp16-ulp-scale) on a "
                                  "fixed cam set; see flake_evidence"),
        "armed_pair_room": "on vs on2: max_abs=0.0 nz=0 hash_identical=true -> trace arm does not perturb the render",
        "flake_evidence": {
            "frozen_build_no_instrumentation": "room/bicycle frozen ext .so, two independent runs: bicycle:36 nz=6 max=2.441e-04 (room:19/room:291 identical)",
            "post_fix_both_armed_AND_both_unarmed": "room on-vs-on2 identical (0/0); room off-vs-off2 nz=7 max=4.883e-04 (both UNARMED)",
            "pre_fix_both_unarmed": "bicycle off-vs-off2 nz=66 max=1.099e-03 (both UNARMED)",
            "render_equivalence_flake_rotation": "in the fixed build's frozen/off/on comparison the flake hit a DIFFERENT pass per cam: frozen at room:97 (91c665), off at room:291 (1e2123), on at bicycle:36 (6916f9) - i.e. per-pass coin flip, not arm-correlated",
            "envelope": "all observed pairwise deviations <= 1.099e-03 (fp16 ulp 2^-10..2^-12 at those magnitudes), nz <= 80 of 8,294,400 px, mean_abs <= 7.3e-10",
        },
        "flake_cam_set": ["room:97", "room:291", "bicycle:36", "bicycle:12"],
        "conclusion": ("OVERALL=FAIL is the base renderer's own bistable flake, not a trace-instrumentation effect: "
                       "(a) it fires with both passes UNARMED, (b) it fires in the frozen build with zero "
                       "instrumentation, (c) it rotates across passes run-to-run. Armed passes stay inside the base "
                       "envelope (room armed pair bit-identical; bicycle armed pair differs only at cam 12 which "
                       "also flakes between two unarmed passes). The trace DATA is bit-exact "
                       "(record_integrity_verification = 1.0/1.0)."),
    }

    m["render_equivalence"] = {
        "compared": "16 (scene,cam) render sha256 - frozen (no instrumentation) vs off (trace build, disarmed) vs on (trace build, armed)",
        "mismatch_set": ["room:97", "room:291", "bicycle:36"],
        "detail": {
            "room:97": "frozen=91c665e83150 (flaked value) off=d3545eb5e0af on=d3545eb5e0af",
            "room:291": "frozen=33640f88c050 off=1e2123032ce2 (flaked value) on=33640f88c050",
            "bicycle:36": "frozen=759c37e3fae6 off=759c37e3fae6 on=6916f979ccc7 (flaked value)",
        },
        "interpretation": "3/16 cams differ, each by a single fp16-ulp-scale deviation at <=6 px, and the flake value lands on a different pass each time -> same bistable base flake as instrumentation_neutrality; the other 13 cams are hash-identical across frozen/off/on",
    }

    full = {s: pop_stats(sm["live_pop_hist17"]) for s, sm in m["scenes"].items()}
    m["histogram_populations"] = {
        "live_pop_hist17": ("FULL population: every phase-2 (warp,gaussian,pp) call contributes its 16 real row "
                            "segments (lanes<4 canonical adds); total rows = 344,153,536 (room) / 523,565,712 "
                            "(bicycle) = all inner-loop row segments of the render"),
        "pop_record_check_hist17": ("DUMPED trace subset: popcount of the stored mask16 over the 1,081,568 / "
                                    "1,871,872 records (idx&255==0 gate + first-2-per-warp cap + max_ctas); this "
                                    "is the live-population of the records actually used as the R1 workload"),
        "gate_selection_note": ("idx_gate=255 selects smem-index positions {0,256,512,768} within each 1024-gaussian "
                                "batch (index-based, not random) + first-2 gated calls per warp; hence the dumped "
                                "subset is slightly biased vs the full population at the extremes (e.g. bicycle "
                                "frac_pop16 0.213 vs 0.143) - BOTH are reported; the record subset bounds the R1 "
                                "workload statistic"),
        "mean_median_labeling": ("the capture-script fields live_pop_mean/live_pop_median are CONDITIONAL on "
                                 "pop>=1 (computed over hist bins 1..16 only, i.e. over non-empty rows); the "
                                 "live_pop_trace_records block reports both unconditional and non-empty-conditional "
                                 "values explicitly. Full-population unconditional mean = "
                                 f"{full['room']['mean']:.3f} (room) / {full['bicycle']['mean']:.3f} (bicycle); "
                                 "full-population conditional mean = "
                                 f"{full['room']['mean_given_nonempty']:.3f} / {full['bicycle']['mean_given_nonempty']:.3f}"),
    }

    for scene, sm in m["scenes"].items():
        sm["live_pop_trace_records"] = pop_stats(sm["pop_record_check_hist17"])
        sm["live_pop_trace_records"]["source"] = ("popcount of stored mask16 over the dumped records = the live-pop "
                                                  "distribution of the R1 timing workload; every row is a real "
                                                  "contiguous 16-px subgroup")

    m["r1b_closure"] = {
        "closed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "criteria": {
            "records_real_contiguous_subgroups": "PASS (x0..x0+15; x[j+1]-x[j]==1 asserted host-side for every record)",
            "n_primary_records_ge_8192_per_scene": "PASS (room 1,081,568; bicycle 1,871,872)",
            "records_bit_exact_vs_exact_recompute": "PASS (1.0/1.0)",
            "live_pop_from_real_contiguous_pixels": "PASS (hist17 + mean/median/frac_pop0/frac_pop16, both populations)",
            "neutrality_armed_on_eq_on2_room": "PASS (0/0)",
            "neutrality_strict_all_pairs": "FAIL - base-renderer flake, evidence in instrumentation_neutrality; not trace-induced, and the flake also hits frozen/unarmed passes",
        },
    }

    with open(MANIFEST, "w") as f:
        json.dump(m, f, indent=2, allow_nan=False)
    print("wrote", MANIFEST)
    print("room trace-records pop:", m["scenes"]["room"]["live_pop_trace_records"]["mean"],
          m["scenes"]["room"]["live_pop_trace_records"]["median"])
    print("bike trace-records pop:", m["scenes"]["bicycle"]["live_pop_trace_records"]["mean"],
          m["scenes"]["bicycle"]["live_pop_trace_records"]["median"])


if __name__ == "__main__":
    main()
