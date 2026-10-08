#!/usr/bin/env python3
"""M1 cost model: T_ratio = eta*[(1-delta)*(u + c + (1-u)*r + h) + delta*d] + zeta.

Base unit = one complete strong-baseline training step on the same machine
(replay_*_14000/timing.json, FUNCTIONAL_ONLY timing on the shared A100).

Measured:
  u  dense fixed share that stays dense: optimizer + ADC stats + fused loss fwd
  c  complete coarse branch forward+loss+backward (fused loss, q=1/4)
  h  anchor A/B generation + device copies, amortized over the lag window
  raster work counts on the studied states (tile/intersection/pixel fractions)
HYPOTHESIS (never substituted as measured):
  eta (assumed 1.2 per protocol), delta*d, zeta, the raster share of the fine
  forward/backward inside the frozen binary, and future sparse-kernel overhead.
  r is therefore reported as an interval, not a point.
"""
import argparse
import csv as csvmod
import json
import os


def med(c, name):
    v = c["components_ms"].get(name)
    if not v:
        return None
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="+", required=True)
    ap.add_argument("--replays", nargs="+", required=True)
    ap.add_argument("--eta", type=float, default=1.2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    OUT = {"eta_assumed": args.eta, "base_unit": "one strong-baseline step (same machine)",
           "formula": "T_ratio = eta*[(1-delta)*(u + c + (1-u)*r + h) + delta*d] + zeta",
           "per_state": {}, "hypotheses": {}}
    gate_T = 1.0 / 1.3

    for prof_path in args.profiles:
        p = json.load(open(prof_path))
        scene = p["scene"]
        it = int(os.path.basename(prof_path).split("_it")[1].split("_")[0])
        base = None
        for r in args.replays:
            if f"replay_{scene}_" in os.path.basename(os.path.dirname(r)):
                base = json.load(open(r))
                break
        assert base is not None, f"no replay timing for {scene}"
        T = base["median_iter_ms"]
        comp = {k: med(p, k) for k in p["components_ms"]}
        rw = dict(p["counts"]["raster_work"])
        n_tiles_total = ((1080 + 15) // 16) * ((1920 + 15) // 16)
        rw["n_tiles_total_corrected"] = n_tiles_total
        rw["selected_tiles_fraction_of_total_corrected"] = rw["selected_tiles"] / n_tiles_total
        rw["note_grid"] = (f"tile grid is ceil(H/16) x ceil(W/16) = {n_tiles_total}; "
                           "fractions recomputed from raw counts here")
        f_isect = rw["selected_tile_intersections_fraction"]
        f_pix = rw["selected_unique_pixels_fraction"]

        fine_fb_ms = base["forward_ms"] + base["backward_ms"] + base["loss_ms"]
        u = {"optimizer": base["optimizer_ms"] / T,
             "adc_stats": base["phase_ms"]["densify"]["median_ms"] / T,
             "fused_loss_full_res_fwd": comp["fused_loss_at_full_res"] / T}
        u["sum_dense_fixed"] = sum(u.values())
        c = {"coarse_branch_fb_ms": comp["coarse_branch_forward_loss_backward"],
             "fraction": comp["coarse_branch_forward_loss_backward"] / T,
             "coarse_fwd_ms": comp["coarse_quarter_forward"],
             "fraction_coarse_fwd": comp["coarse_quarter_forward"] / T}
        anchor_gen_ms = comp["anchor_A_generation_full_res"] + comp["anchor_B_generation_coarse"]
        anchor_copy_ms = comp["anchor_A_d2d_copy"] + comp["anchor_B_d2d_copy"]
        h_per_step_ms = (anchor_gen_ms + anchor_copy_ms) / 16.0
        h = {"anchor_gen_ms": anchor_gen_ms, "anchor_copy_ms": anchor_copy_ms,
             "amortization_lag": 16, "h_per_step": h_per_step_ms / T,
             "bytes": p["counts"]["anchor_bytes"]}
        # r interval: variable fine-path work after restricting rasterization to the
        # selected tiles. Fine fwd+bwd+loss share is measured; the raster part of it is
        # NOT separable inside the frozen binary -> interval [0, max(f_isect, f_pix)].
        fine_share = fine_fb_ms / T
        f_hi = max(f_isect, f_pix)
        r_lo, r_hi = 0.0, f_hi * fine_share
        t_lo = args.eta * (u["sum_dense_fixed"] + c["fraction"] +
                           (1.0 - u["sum_dense_fixed"]) * r_lo + h["h_per_step"])
        t_hi = args.eta * (u["sum_dense_fixed"] + c["fraction"] +
                           (1.0 - u["sum_dense_fixed"]) * r_hi + h["h_per_step"])
        OUT["per_state"][f"{scene}_it{it}"] = {
            "T_base_ms": T, "components_ms": comp, "u": u, "c": c, "h": h,
            "fine_fwd_backward_loss_ms": fine_fb_ms, "fine_share_of_step": fine_share,
            "r": {"interval_of_base_step": [r_lo, r_hi],
                  "fraction_hi_basis": {"intersection_fraction": f_isect,
                                        "pixel_fraction": f_pix,
                                        "used": f_hi},
                  "note": ("lower bound 0: the raster share of the fine fwd/bwd cannot be "
                           "separated from projection/sort inside the frozen binary; upper "
                           "bound assumes the whole fine fwd+bwd is raster and scales with "
                           "the larger of intersection/pixel fractions")},
            "raster_work": rw,
            "T_ratio_interval_eta": [t_lo, t_hi],
            "speedup_interval": [1.0 / t_hi, (1.0 / t_lo if t_lo > 0 else None)],
            "within_1.3x_gate": {"gate_T_ratio": gate_T,
                                 "lo_within": t_lo <= gate_T, "hi_within": t_hi <= gate_T},
            "excluded_from_main_range": {"delta_d": 0.0, "zeta": 0.0,
                                         "note": "any dense calibration or init overhead "
                                                 "only increases T_ratio"},
        }

    OUT["hypotheses"] = {
        "eta": {"assumed": args.eta, "status": "HYPOTHESIS; not measured in M1"},
        "delta_d": {"status": "HYPOTHESIS; excluded (0) from the main range"},
        "zeta": {"status": "HYPOTHESIS; excluded (0) from the main range"},
        "raster_share_of_fine_fb": {"status": "HYPOTHESIS; unobservable inside the frozen "
                                              "binary; drives the r interval"},
        "sparse_kernel_overhead": {"status": "HYPOTHESIS; no sparse kernel exists; "
                                             "per-tile launch/indirection cost unmeasured"},
    }
    states = OUT["per_state"]
    per_state = {k: {"T_ratio_interval": s["T_ratio_interval_eta"],
                     "optimistic_within_gate": s["T_ratio_interval_eta"][0] <= gate_T,
                     "conservative_within_gate": s["T_ratio_interval_eta"][1] <= gate_T}
                 for k, s in states.items()}
    OUT["verdicts"] = {
        "eta": args.eta, "T_ratio_gate_for_1.3x": gate_T,
        "per_state": per_state,
        "conditional_speed_space_exists_in_some_state":
            bool(any(v["optimistic_within_gate"] for v in per_state.values())),
        "gate_met_in_all_states_conservative":
            bool(all(v["conservative_within_gate"] for v in per_state.values())),
        "interpretation": ("the model leaves conditional speed space at the 5K states but "
                           "not at the 14K states, where the coarse branch alone (c) grows "
                           "with scene size and pushes even the optimistic bound above the "
                           "1.3x gate; the sparse-kernel overhead that would realize the "
                           "5K-case margin has no measured path inside the frozen binary, "
                           "so the cost side alone would be INCONCLUSIVE_COST with a "
                           "DROP_COST reading at 14K"),
    }

    with open(args.out, "w") as f:
        json.dump(OUT, f, indent=2, allow_nan=False)
    with open(args.csv, "w", newline="") as f:
        w = csvmod.writer(f)
        w.writerow(["state", "T_base_ms", "u_sum", "c_fraction", "h_per_step", "r_lo",
                    "r_hi", "T_ratio_lo", "T_ratio_hi", "speedup_lo", "speedup_hi",
                    "intersection_frac", "pixel_frac", "tile_frac"])
        for k, s in states.items():
            r = s["r"]["interval_of_base_step"]
            t = s["T_ratio_interval_eta"]
            sp = s["speedup_interval"]
            w.writerow([k, s["T_base_ms"], s["u"]["sum_dense_fixed"], s["c"]["fraction"],
                        s["h"]["h_per_step"], r[0], r[1], t[0], t[1], sp[0], sp[1],
                        s["raster_work"]["selected_tile_intersections_fraction"],
                        s["raster_work"]["selected_unique_pixels_fraction"],
                        s["raster_work"]["selected_tiles_fraction_of_total_corrected"]])
    print(f"[model] wrote {args.out} + {args.csv}")
    for k, s in states.items():
        t = s["T_ratio_interval_eta"]
        print(f"  {k}: T_base={s['T_base_ms']:.2f}ms u={s['u']['sum_dense_fixed']:.3f} "
              f"c={s['c']['fraction']:.3f} h={s['h']['h_per_step']:.4f} "
              f"r=[{t[0] and s['r']['interval_of_base_step'][0]:.3f},"
              f"{s['r']['interval_of_base_step'][1]:.3f}] "
              f"T_ratio=[{t[0]:.3f},{t[1]:.3f}] speedup=[{1/t[1]:.2f},{1/t[0]:.2f}]")
    print(f"  gate(1.3x) T_ratio<={gate_T:.3f}; "
          f"speed_space_in_some_state="
          f"{OUT['verdicts']['conditional_speed_space_exists_in_some_state']} "
          f"gate_met_all_states_conservative="
          f"{OUT['verdicts']['gate_met_in_all_states_conservative']}")
    for k, v in OUT["verdicts"]["per_state"].items():
        print(f"  verdict {k}: T_ratio={v['T_ratio_interval']} "
              f"optimistic_within={v['optimistic_within_gate']} "
              f"conservative_within={v['conservative_within_gate']}")


if __name__ == "__main__":
    main()
