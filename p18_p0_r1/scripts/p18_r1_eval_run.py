#!/usr/bin/env python3
"""P18-P0-R1 R1-C numeric validation + R1-E timing (S_eval gate).

Refuses to time unless P18_R1_SASS_AUDIT.json verdict == PASS (spec item 4).

Validation (R1-C): RAW QSCAN output (no guard, no fallback) vs DIRECT on the real
trace workload: max_abs / max_rel, cull-branch agreement (G < 1/255), NaN/Inf.

Timing (R1-E): same method as P18-P0 (CUDA graphs, 7 reps x 10 iters, median).
Cells = {room, bicycle} x {exp2 (forward domain), exp (backward domain)}.
Workload = the real trace rows, tiled K times per scene so the per-launch duration
matches the P18-P0 measurement scale (~64M+ items) and launch overhead is
negligible; the raw 1x-trace result is also reported. S_eval = T_direct/T_qscan.
Gate (unchanged): per-variant median S_eval over the 4 cells: >=4.0 STRONG;
<2.5 HARD_DROP (all variants below 2.5 => P18_P0_HARD_DROP).
Also timed: an exp-free control kernel (audited 0 MUFU, 0 exp calls) giving the
upper bound S_eval_noexp on any exp-elimination speedup in this proxy.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from p18_r1_eval_common import ART, AT, build_ext, build_workload, jwrite, log, pop_stats

ITERS = 10
REPS = 7
TILE_TARGET_ITEMS = 64_000_000


def to_dev(w, k_tile=1):
    a = np.tile(w["a"], k_tile)
    b = np.tile(w["b"], k_tile)
    c = np.tile(w["c"], k_tile)
    x = np.tile(w["x"], k_tile)
    return (torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda(),
            torch.from_numpy(c).cuda(), torch.from_numpy(x).cuda())


def validate(mod, a, b, c, x, exp2, tag):
    out_d = torch.zeros_like(a)
    mod.eval_direct(a, b, c, x, out_d, exp2, 1)
    torch.cuda.synchronize()
    res = {}
    for sub in (4, 8, 16):
        out_q = torch.zeros_like(a)
        mod.eval_qscan(a, b, c, x, out_q, sub, exp2, 1)
        torch.cuda.synchronize()
        oq = out_q.double()
        od = out_d.double()
        n = a.numel()
        fin = torch.isfinite(oq)
        n_nan = int(torch.isnan(oq).sum())
        n_inf = int(torch.isinf(oq).sum())
        d = (oq[fin] - od[fin]).abs()
        max_abs = float(d.max()) if d.numel() else 0.0
        od_f = od[fin]
        big = od_f.abs() > 1e-20
        max_rel = float((d[big] / od_f[big].abs()).max()) if bool(big.any()) else 0.0
        bd = out_d < AT
        bq = out_q < AT
        mm = bd ^ bq
        mm_n = int(mm.sum())
        far = torch.abs(out_d.double() - AT) > 1e-5
        mm_far = int((mm & far).sum())
        mm_fin = int((mm & fin).sum())
        nan_d = int((~torch.isfinite(out_d)).sum())
        res[str(sub)] = {"n_items": int(n),
                         "n_nan_qscan": n_nan, "n_inf_qscan": n_inf,
                         "max_abs_finite": max_abs, "max_rel_finite": max_rel,
                         "branch_mismatch": mm_n, "branch_mismatch_far_from_at": mm_far,
                         "branch_mismatch_excluding_nonfinite": mm_fin,
                         "nan_inf_direct": nan_d}
        log(f"validate {tag} sub{sub}: nan={n_nan} inf={n_inf} max_abs_fin={max_abs:.3e} "
            f"max_rel_fin={max_rel:.3e} mm={mm_n} mm_far={mm_far} mm_fin={mm_fin}")
        del out_q
    del out_d
    torch.cuda.empty_cache()
    return res


def time_variant(mod, a, b, c, x, out, sub, exp2):
    def launch():
        if sub == 0:
            mod.eval_direct(a, b, c, x, out, exp2, 1)
        elif sub == -1:
            mod.eval_noexp(a, b, c, x, out, 1)
        else:
            mod.eval_qscan(a, b, c, x, out, sub, exp2, 1)

    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        launch()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        launch()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        s_ev = torch.cuda.Event(enable_timing=True)
        e_ev = torch.cuda.Event(enable_timing=True)
        s_ev.record()
        for _ in range(ITERS):
            g.replay()
        e_ev.record()
        torch.cuda.synchronize()
        ts.append(s_ev.elapsed_time(e_ev) / ITERS)
    return float(np.median(ts))


def main():
    audit = json.load(open(ART / "P18_R1_SASS_AUDIT.json"))
    if audit["verdict"] != "PASS":
        log(f"SASS audit verdict = {audit['verdict']} -> STOP, do not time")
        sys.exit(2)
    log("SASS audit PASS -> proceeding")

    mod = build_ext(name="p18_r1_eval")
    validation, timing, timing_raw = {}, {}, {}
    stats = {}

    for scene in ("room", "bicycle"):
        w = build_workload(scene)
        st = w["stats"]
        ps = pop_stats(st["live_pop_hist17"])
        ps.update({"n_records_total": st["n_records_total"],
                   "n_records_used": st["n_records_used"], "n_items": st["n_items"]})
        stats[scene] = ps
        log(f"{scene}: n_records_used={st['n_records_used']} n_items={st['n_items']} "
            f"frac_pop0={ps['frac_pop0']:.4f} frac_pop16={ps['frac_pop16']:.4f} "
            f"mean_live_pop={ps['mean']:.3f} med={ps['median']} "
            f"mean|nonempty={ps['mean_given_nonempty']:.3f}")
        validation[scene] = {}
        for exp2, tag in ((True, "exp2_forward"), (False, "exp_backward")):
            a, b, c, x = to_dev(w, 1)
            validation[scene][tag] = validate(mod, a, b, c, x, exp2, f"{scene}/{tag}")
            del a, b, c, x
            torch.cuda.empty_cache()

    for scene in ("room", "bicycle"):
        w = build_workload(scene)
        k_tile = max(1, -(-TILE_TARGET_ITEMS // w["stats"]["n_items"]))
        log(f"{scene}: timing tile K={k_tile} (items={w['stats']['n_items']} -> {w['stats']['n_items']*k_tile})")
        timing[scene], timing_raw[scene] = {}, {}
        for exp2, tag in ((True, "exp2_forward"), (False, "exp_backward")):
            a, b, c, x = to_dev(w, k_tile)
            out_g = torch.zeros_like(a)
            t_direct = time_variant(mod, a, b, c, x, out_g, 0, exp2)
            t_noexp = time_variant(mod, a, b, c, x, out_g, -1, exp2)
            r = {"n_items": int(a.numel()), "k_tile": int(k_tile), "iters": ITERS, "reps": REPS,
                 "T_direct_ms": t_direct, "T_noexp_ms": t_noexp,
                 "S_eval_noexp_bound": t_direct / t_noexp}
            for sub in (4, 8, 16):
                tq = time_variant(mod, a, b, c, x, out_g, sub, exp2)
                r[f"T_qscan{sub}_ms"] = tq
                r[f"S_eval_q{sub}"] = t_direct / tq
            timing[scene][tag] = r
            log(f"{scene}/{tag}: K={k_tile} items={r['n_items']} S_eval q4={r['S_eval_q4']:.3f} "
                f"q8={r['S_eval_q8']:.3f} q16={r['S_eval_q16']:.3f} "
                f"noexp_bound={r['S_eval_noexp_bound']:.3f} "
                f"(ms D={t_direct:.3f} noexp={t_noexp:.3f} 4={r['T_qscan4_ms']:.3f} "
                f"8={r['T_qscan8_ms']:.3f} 16={r['T_qscan16_ms']:.3f})")
            del a, b, c, x, out_g
            torch.cuda.empty_cache()
            a, b, c, x = to_dev(w, 1)
            out_g = torch.zeros_like(a)
            t_direct = time_variant(mod, a, b, c, x, out_g, 0, exp2)
            t_noexp = time_variant(mod, a, b, c, x, out_g, -1, exp2)
            rr = {"n_items": int(a.numel()), "k_tile": 1, "iters": ITERS, "reps": REPS,
                  "T_direct_ms": t_direct, "T_noexp_ms": t_noexp,
                  "S_eval_noexp_bound": t_direct / t_noexp}
            for sub in (4, 8, 16):
                tq = time_variant(mod, a, b, c, x, out_g, sub, exp2)
                rr[f"T_qscan{sub}_ms"] = tq
                rr[f"S_eval_q{sub}"] = t_direct / tq
            timing_raw[scene][tag] = rr
            log(f"{scene}/{tag} RAW K=1: S_eval q4={rr['S_eval_q4']:.3f} "
                f"q8={rr['S_eval_q8']:.3f} q16={rr['S_eval_q16']:.3f} "
                f"noexp_bound={rr['S_eval_noexp_bound']:.3f} (ms D={t_direct:.3f})")
            del a, b, c, x, out_g
            torch.cuda.empty_cache()

    cells = [(s, t) for s in ("room", "bicycle") for t in ("exp2_forward", "exp_backward")]
    med = {}
    for variant in ("q4", "q8", "q16"):
        vals = [timing[s][t][f"S_eval_{variant}"] for s, t in cells]
        med[variant] = float(np.median(vals))
    med_noexp = float(np.median([timing[s][t]["S_eval_noexp_bound"] for s, t in cells]))
    gate = {}
    for variant, m in med.items():
        gate[variant] = "STRONG" if m >= 4.0 else ("HARD_DROP" if m < 2.5 else "MID")
    any_strong = any(v == "STRONG" for v in gate.values())
    all_hard = all(v == "HARD_DROP" for v in gate.values())
    all_near_1 = all(0.9 <= m <= 1.1 for m in med.values())
    close_out = bool(all_hard and all_near_1)

    im = {
        "phase": "P18-P0-R1", "artifact": "P18_R1_EVAL_MICROBENCH",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "S_eval_definition": ("median T_DIRECT / median T_QSCAN over the real R1 trace "
                              "(CUDA graphs, events, 7 reps x 10 iters, per-replay median; "
                              "same harness as P18-P0 p18_p0_trace_eval.py time_eval)"),
        "units": ("T_* keys are milliseconds (torch Event.elapsed_time unit); the P18-P0 "
                  "artifact's T_*_us keys held the same ms-scale numbers (mislabeled)"),
        "workload": ("real R1 trace rows (16 real contiguous px per record/subgroup), record-major; "
                     f"timing tiled to >= {TILE_TARGET_ITEMS} items per launch to match the P18-P0 "
                     "measurement scale (raw 1x numbers reported alongside)"),
        "sass_audit": "P18_R1_SASS_AUDIT.json verdict PASS (exp counts: DIRECT 1.0/px, QSCAN 3/SUB/px, exp-free control 0)",
        "validation_raw_no_guard": validation,
        "timing_tiled": timing,
        "timing_raw_1x": timing_raw,
        "S_eval_median_over_cells": med,
        "S_eval_noexp_bound_median": med_noexp,
        "gate": {"rule": "per-variant median S_eval over the 4 cells: >=4.0 STRONG; <2.5 HARD_DROP (unchanged)",
                 "per_variant": gate, "any_strong": any_strong, "all_below_2_5": all_hard},
        "p18_p0_hard_drop_close_out": {
            "criterion": "spec item 7: unchanged gate already <2.5 for all variants AND true QSCAN S_eval ~ 1",
            "all_below_2_5": all_hard, "all_medians_near_1": all_near_1,
            "expfree_upper_bound_near_1": bool(0.9 <= med_noexp <= 1.1),
            "closed": close_out},
        "dispatch_statistics": stats,
        "note": ("S_eval is the isolated inner-loop proxy (protocol); gate/floor unchanged from P18-P0. "
                 "S_eval_noexp_bound = T_direct/T_noexp with an audited exp-free kernel (identical "
                 "loads/store, 0 exps): an upper bound on any exp-elimination speedup in this proxy."),
    }
    jwrite(ART / "P18_R1_EVAL_MICROBENCH.json", im)
    print("GATE", json.dumps(gate), "median:", json.dumps(med),
          "noexp_bound:", f"{med_noexp:.3f}", "close_out:", close_out)


if __name__ == "__main__":
    main()
