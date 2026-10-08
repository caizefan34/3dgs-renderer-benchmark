#!/usr/bin/env python3
"""GSC0 P0 analysis — stability (GSC-3), precision oracle (GSC-4), gate (GSC-5).

Runs on EPIC-05 only (large NumPy). Reads the read-only full-state snapshots
produced by gsc0_trace.py and computes:
  GSC-3  per-attribute relative-change distributions over the preregistered
         phases, matched by frozen _gaussian_ids.
  GSC-4  FP16/BF16 representation-error oracle + optimizer-state error + a
         first-order Adam update-direction perturbation (NO training run).
  GSC-5  preregistered gate -> verdict.

Preregistered thresholds (FIXED before aggregation; no threshold search):
  TAU = 1e-2        stability / representation-error scale
  STABLE_FRAC_MIN = 0.50   per-scene "substantial stable population"
  SCENE_FRAC_MIN = 2/3     >=2/3 scenes must satisfy a gate
  ECON_FRAC_MIN = 0.40     stable_fraction implying >=20% memory reduction
  ECON_COUNT_MIN = 10000   non-trivial compressible population
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

EPS = 1e-12
TAU = 1e-2
STABLE_FRAC_MIN = 0.50
SCENE_FRAC_MIN = 2.0 / 3.0
ECON_FRAC_MIN = 0.40
ECON_COUNT_MIN = 10000

ATTRS = ["xyz", "scaling", "rotation", "shs", "opacity"]
# native storage floats per Gaussian (shs = 16 coeffs * 3)
ATTR_FLOATS = {"xyz": 3, "scaling": 3, "rotation": 4, "shs": 48, "opacity": 1}
ATTR_FLOATS_TOTAL = sum(ATTR_FLOATS.values())          # 59
OPT_FLOATS_TOTAL = 2 * ATTR_FLOATS_TOTAL               # Adam exp_avg + exp_avg_sq
STATE_FLOATS_TOTAL = ATTR_FLOATS_TOTAL + OPT_FLOATS_TOTAL  # 177


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def attr_norm(x):
    if x.dim() <= 1:
        return x.abs()
    return x.norm(dim=tuple(range(1, x.dim())))


def rel_change(x_t, x_tk, nslots=None):
    """||x_t - x_tk|| / max(||x_t||, eps) per Gaussian."""
    if nslots is not None and x_t.dim() == 3:
        x_t = x_t[:, :nslots, :]
        x_tk = x_tk[:, :nslots, :]
    num = attr_norm(x_t - x_tk)
    den = attr_norm(x_t).clamp_min(EPS)
    return num / den


def cast_rel_error(x, dtype):
    xc = x.to(dtype).float()
    num = attr_norm(xc - x)
    den = attr_norm(x).clamp_min(EPS)
    return num / den


def cast_overflow_frac(x, dtype):
    xf = x.float()
    xc = x.to(dtype).float()
    finite = torch.isfinite(xc)
    nonfinite = (~finite).sum().item()
    total = xf.numel()
    return nonfinite / max(total, 1), nonfinite, total


def adam_update_dir(exp_avg, exp_avg_sq, dtype, eps=1e-12):
    """Adam update direction u = m/(sqrt(v)+eps); fp32 (ref) and reduced precision."""
    m_r = exp_avg.float()
    v_r = exp_avg_sq.float()
    m_l = exp_avg.to(dtype).float()
    v_l = exp_avg_sq.to(dtype).float()
    u_ref = m_r / (v_r.sqrt() + eps)
    u_low = m_l / (v_l.sqrt() + eps)
    return u_ref, u_low


def update_dir_perturbation(exp_avg, exp_avg_sq, dtype, floor, eps=1e-12):
    """||u_low-u_ref|| / (||u_ref|| + floor).

    `floor` = population median of ||u_ref|| (a FIXED positive scale), so the
    ratio stays bounded for near-zero-moment Gaussians (whose absolute update
    error is itself tiny). Well-posed representation of optimizer-state effect.
    """
    u_ref, u_low = adam_update_dir(exp_avg, exp_avg_sq, dtype, eps)
    num = attr_norm(u_low - u_ref)
    den = attr_norm(u_ref) + float(floor)
    return num / den


def optimizer_state_rep_error(exp_avg, exp_avg_sq, dtype, floor):
    """FP16/BF16 representation error of the Adam moments, absolute error over a

    population-level magnitude floor: (||cast(m)-m|| + ||cast(v)-v||) / floor.

    `floor` = median over the subset of (||m||+||v||). Robust to near-zero rows.
    """
    m = exp_avg.float()
    v = exp_avg_sq.float()
    abs_err = attr_norm(m.to(dtype).float() - m) + attr_norm(v.to(dtype).float() - v)
    return abs_err / float(floor)


def match_by_id(snap_a, snap_b):
    ids_a = snap_a["ids"].numpy()
    ids_b = snap_b["ids"].numpy()
    b_map = {int(v): i for i, v in enumerate(ids_b)}
    ia, ib, common = [], [], []
    for i, v in enumerate(ids_a):
        j = b_map.get(int(v))
        if j is not None:
            ia.append(i); ib.append(j); common.append(int(v))
    return np.array(common), np.array(ia, dtype=np.int64), np.array(ib, dtype=np.int64)


def dist_stats(r):
    r = np.asarray(r)[np.isfinite(r)]
    if r.size == 0:
        return {"n": 0, "median": None, "p90": None, "p99": None, "mean": None}
    return {
        "n": int(r.size),
        "median": float(np.median(r)),
        "p90": float(np.percentile(r, 90)),
        "p99": float(np.percentile(r, 99)),
        "mean": float(np.mean(r)),
    }


def load_phase(run_dir, scene, phase):
    files = glob.glob(os.path.join(run_dir, scene, "traces", f"{scene}_{phase}_iter*.pt"))
    if not files:
        return None
    return torch.load(sorted(files)[-1], map_location="cpu")


def n_active_slots(active_sh_degree):
    return (active_sh_degree + 1) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default="room,garden,bicycle")
    args = ap.parse_args()
    scenes = args.scenes.split(",")
    os.makedirs(args.out, exist_ok=True)

    PREREG = {
        "TAU": TAU,
        "STABLE_FRAC_MIN": STABLE_FRAC_MIN,
        "SCENE_FRAC_MIN": SCENE_FRAC_MIN,
        "ECON_FRAC_MIN": ECON_FRAC_MIN,
        "ECON_COUNT_MIN": ECON_COUNT_MIN,
        "stable_rule": "composite: Gaussian stable iff max-over-5-attrs rel_change <= TAU",
        "primary_stability_window": "densify_complete(15000) -> mature(30000), frozen topology",
        "gate_B_rule": "stable subset: exists p in {fp16,bf16} s.t. median p attr rep-error <= TAU AND median p optimizer-state moment rep-error <= TAU (all attrs/groups)",
        "gate_C_rule": "stable_fraction >= ECON_FRAC_MIN (>=20% mem reduction) AND stable_count >= ECON_COUNT_MIN",
        "note_thresholds_fixed_before_aggregation": True,
    }

    per_scene = {}
    oracle_by_scene = {}
    for scene in scenes:
        t0 = time.time()
        early = load_phase(args.run_dir, scene, "early")
        dc = load_phase(args.run_dir, scene, "densify_complete")
        mature = load_phase(args.run_dir, scene, "mature")
        rec = {"scene": scene}
        if mature is None or dc is None:
            rec["error"] = "missing required phases (densify_complete/mature)"
            per_scene[scene] = rec
            continue

        slots_m = n_active_slots(int(mature["active_sh_degree"]))
        slots_dc = n_active_slots(int(dc["active_sh_degree"]))
        slots_common = min(slots_m, slots_dc)
        rec["active_sh_degree"] = {"early": int(early["active_sh_degree"]) if early is not None else None,
                                   "densify_complete": int(dc["active_sh_degree"]),
                                   "mature": int(mature["active_sh_degree"])}
        rec["N"] = {"densify_complete": int(dc["N"]), "mature": int(mature["N"])}

        # ---- ID match mature window ----
        common, ia, ib = match_by_id(dc, mature)
        rec["id_match"] = {
            "densify_complete_N": int(dc["N"]),
            "mature_N": int(mature["N"]),
            "common": int(common.size),
            "only_densify_complete": int(dc["N"] - common.size),
            "only_mature": int(mature["N"] - common.size),
            "frozen_topology": bool(common.size == dc["N"] == mature["N"]),
        }

        # ---- GSC-3 stability: mature window (primary) ----
        stab = {}
        max_rel = torch.full((common.size,), -1.0)
        for a in ATTRS:
            x_m = mature[a][ib]
            x_d = dc[a][ia]
            nslots = slots_common if a == "shs" else None
            r = rel_change(x_m, x_d, nslots=nslots)
            stab[a] = dist_stats(r.numpy())
            max_rel = torch.maximum(max_rel, r)
        stable_mask = max_rel <= TAU
        stab["composite_max"] = dist_stats(max_rel.numpy())
        stab["stable_fraction"] = float(stable_mask.double().mean().item())
        stab["stable_count"] = int(stable_mask.sum().item())
        # per-attribute stable fractions
        stab["per_attr_stable_fraction"] = {}
        for a in ATTRS:
            x_m = mature[a][ib]; x_d = dc[a][ia]
            nslots = slots_common if a == "shs" else None
            r = rel_change(x_m, x_d, nslots=nslots)
            stab["per_attr_stable_fraction"][a] = float((r <= TAU).double().mean().item())
        rec["stability_mature_window"] = stab

        # ---- secondary windows (report only) ----
        rec["stability_secondary"] = {}
        if early is not None:
            slots_e = n_active_slots(int(early["active_sh_degree"]))
            common2, ie, im = match_by_id(early, mature)
            slots_em = min(slots_e, slots_m)
            sec = {"id_match": {"common": int(common2.size),
                                "only_early": int(early["N"] - common2.size),
                                "only_mature": int(mature["N"] - common2.size)}}
            for a in ATTRS:
                x_m = mature[a][im]; x_e = early[a][ie]
                nslots = slots_em if a == "shs" else None
                sec[a] = dist_stats(rel_change(x_m, x_e, nslots=nslots).numpy())
            rec["stability_early_to_mature"] = sec

        # ---- gradient magnitude + visibility (P0-safe cumulative proxies) ----
        grad_m = mature["grad_sum"][ib] / mature["vis_count"][ib].clamp_min(1)
        grad_d = dc["grad_sum"][ia] / dc["vis_count"][ia].clamp_min(1)
        rec["gradient_magnitude"] = {
            "mean_abs_grad_densify_complete": dist_stats(grad_d.numpy()),
            "mean_abs_grad_mature": dist_stats(grad_m.numpy()),
            "abs_delta_mature_minus_dc": dist_stats((grad_m - grad_d).abs().numpy()),
            "note": "cumulative lifetime mean |grad| (grad_sum/vis_count); per-iter replay forbidden in P0",
        }
        rec["visibility_mature_window"] = {
            "vis_count_mature_minus_dc": dist_stats(
                (mature["vis_count"][ib] - dc["vis_count"][ia]).numpy()),
            "vis_freq_mature_window": dist_stats(
                ((mature["vis_count"][ib] - dc["vis_count"][ia]) / 15000.0).numpy()),
            "note": "cumulative visibility count over the 15000-iter mature window",
        }

        # ---- GSC-4 precision oracle on mature snapshot ----
        oracle = {"N_mature": int(mature["N"]), "sh_slots": slots_m}
        # all-Gaussian representation error
        rep_all = {}
        for a in ATTRS:
            x = mature[a]
            nslots = slots_m if a == "shs" else None
            if nslots is not None:
                x = x[:, :nslots, :]
            rep_all[a] = {
                "fp16": dist_stats(cast_rel_error(x, torch.float16).numpy()),
                "bf16": dist_stats(cast_rel_error(x, torch.bfloat16).numpy()),
                "fp16_overflow_frac": cast_overflow_frac(x, torch.float16)[0],
                "bf16_overflow_frac": cast_overflow_frac(x, torch.bfloat16)[0],
            }
        oracle["representation_error_all"] = rep_all

        # stable subset
        stable_idx = ib[stable_mask.numpy()]
        rep_stable = {}
        adam_stable = {}
        for a in ATTRS:
            x = mature[a][stable_idx]
            nslots = slots_m if a == "shs" else None
            if nslots is not None:
                x = x[:, :nslots, :]
            rep_stable[a] = {
                "fp16": dist_stats(cast_rel_error(x, torch.float16).numpy()),
                "bf16": dist_stats(cast_rel_error(x, torch.bfloat16).numpy()),
            }
        # Optimizer-state error on the stable subset. Two well-posed metrics use a
        # POPULATION-LEVEL floor (median magnitude over the subset) as denominator so
        # that near-zero-moment Gaussians (tiny absolute update error) do not dominate.
        for gname in ["xyz", "shs", "opacity", "scaling", "rotation"]:
            # Floors computed over ALL mature Gaussians (a fixed, nonzero
            # population scale). The stable subset is the most-converged set,
            # so its own moment median can be ~0 (degenerate denominator).
            ea_all = mature["adam"][gname]["exp_avg"]
            eas_all = mature["adam"][gname]["exp_avg_sq"]
            ea = ea_all[stable_idx]
            eas = eas_all[stable_idx]
            with torch.no_grad():
                u_ref_all = ea_all.float() / (eas_all.float().sqrt() + 1e-12)
                floor_u = float(attr_norm(u_ref_all).median().item())
                floor_m = float((attr_norm(ea_all.float()) + attr_norm(eas_all.float())).median().item())
            adam_stable[gname] = {
                "floor_update_norm": floor_u,
                "floor_moment_norm": floor_m,
                "floor_scope": "all_mature_gaussians",
                "update_dir_perturbation": {
                    "fp16": dist_stats(update_dir_perturbation(ea, eas, torch.float16, floor_u).numpy()),
                    "bf16": dist_stats(update_dir_perturbation(ea, eas, torch.bfloat16, floor_u).numpy()),
                },
                "optimizer_state_rep_error": {
                    "fp16": dist_stats(optimizer_state_rep_error(ea, eas, torch.float16, floor_m).numpy()),
                    "bf16": dist_stats(optimizer_state_rep_error(ea, eas, torch.bfloat16, floor_m).numpy()),
                },
            }
        oracle["representation_error_stable_subset"] = rep_stable
        oracle["optimizer_state_error_stable_subset"] = adam_stable
        oracle["stable_subset_size"] = int(stable_idx.size)

        # ---- memory / bandwidth model ----
        bytes_fp32_state = STATE_FLOATS_TOTAL * 4
        bytes_fp16_state = STATE_FLOATS_TOTAL * 2
        bytes_fp32_attr = ATTR_FLOATS_TOTAL * 4
        bytes_fp16_attr = ATTR_FLOATS_TOTAL * 2
        stable_count = stab["stable_count"]
        oracle["memory_model"] = {
            "bytes_per_gaussian_fp32_state": bytes_fp32_state,
            "bytes_per_gaussian_fp16_state": bytes_fp16_state,
            "bytes_per_gaussian_fp32_attr_only": bytes_fp32_attr,
            "bytes_per_gaussian_fp16_attr_only": bytes_fp16_attr,
            "stable_count": stable_count,
            "mature_N": int(mature["N"]),
            "state_memory_total_mb": round(mature["N"] * bytes_fp32_state / 1e6, 1),
            "state_memory_stable_subset_mb": round(stable_count * bytes_fp32_state / 1e6, 1),
            "state_memory_saving_mb": round(stable_count * (bytes_fp32_state - bytes_fp16_state) / 1e6, 1),
            "state_memory_reduction_frac": round(stable_count * (bytes_fp32_state - bytes_fp16_state) /
                                                  max(mature["N"] * bytes_fp32_state, 1), 4),
            "attr_memory_saving_mb": round(stable_count * (bytes_fp32_attr - bytes_fp16_attr) / 1e6, 1),
        }

        oracle_by_scene[scene] = oracle
        rec["elapsed_s"] = round(time.time() - t0, 1)
        per_scene[scene] = rec
        print(f"[gsc0-analysis] {scene}: stable_fraction={stab['stable_fraction']:.3f} "
              f"stable_count={stab['stable_count']} frozen={rec['id_match']['frozen_topology']} "
              f"({rec['elapsed_s']}s)", flush=True)

    # ---- GSC-5 gate ----
    scenes_with_data = [s for s in scenes if "stability_mature_window" in per_scene.get(s, {})]
    gateA_scene_pass, gateB_scene_pass, gateC_scene_pass = {}, {}, {}
    for s in scenes_with_data:
        rec = per_scene[s]
        sf = rec["stability_mature_window"]["stable_fraction"]
        gateA_scene_pass[s] = bool(sf >= STABLE_FRAC_MIN)
        # Gate B: is there a SAFE precision policy for the stable subset?
        # A precision p is "safe" if, for ALL attributes AND all optimizer-state
        # groups, the median representation error <= TAU. Population-floor
        # denominators keep the ratio bounded (near-zero moments do not dominate).
        # Gate B passes if AT LEAST ONE of {fp16, bf16} is safe.
        orc = oracle_by_scene[s]
        b_detail = {}
        b_any_ok = False
        for p in ["fp16", "bf16"]:
            attr_ok = all(orc["representation_error_stable_subset"][a][p]["median"] is not None and
                          orc["representation_error_stable_subset"][a][p]["median"] <= TAU
                          for a in ATTRS)
            adam_ok = all(
                orc["optimizer_state_error_stable_subset"][g]["optimizer_state_rep_error"][p]["median"] is not None
                and orc["optimizer_state_error_stable_subset"][g]["optimizer_state_rep_error"][p]["median"] <= TAU
                for g in ["xyz", "shs", "opacity", "scaling", "rotation"])
            b_detail[p] = {"attr_ok": bool(attr_ok), "adam_ok": bool(adam_ok),
                           "safe": bool(attr_ok and adam_ok)}
            b_any_ok = b_any_ok or (attr_ok and adam_ok)
        gateB_scene_pass[s] = bool(b_any_ok)
        per_scene[s]["gateB_detail"] = b_detail
        # Gate C
        gateC_scene_pass[s] = bool(sf >= ECON_FRAC_MIN and
                                   rec["stability_mature_window"]["stable_count"] >= ECON_COUNT_MIN)

    def scenes_passing(d):
        return [s for s in scenes_with_data if d.get(s)]

    nA, nB, nC = len(scenes_passing(gateA_scene_pass)), len(scenes_passing(gateB_scene_pass)), len(scenes_passing(gateC_scene_pass))
    n_data = len(scenes_with_data)
    gateA = nA >= SCENE_FRAC_MIN * n_data if n_data else False
    gateB = nB >= SCENE_FRAC_MIN * n_data if n_data else False
    gateC = nC >= SCENE_FRAC_MIN * n_data if n_data else False

    if n_data == 0:
        verdict = "BLOCKED"
    elif not gateA:
        verdict = "DROP_NO_STABLE_STATE"
    elif not gateB:
        verdict = "DROP_UNSAFE_ERROR_BOUND"
    elif not gateC:
        verdict = "DROP_UNECONOMIC"
    else:
        verdict = "KEEP_FOR_GSC1"

    gate = {
        "prereg": PREREG,
        "scenes_with_data": scenes_with_data,
        "gateA_stability": {
            "rule": f">= {SCENE_FRAC_MIN:.2f} of scenes with stable_fraction >= {STABLE_FRAC_MIN}",
            "scene_pass": gateA_scene_pass, "n_pass": nA, "n_data": n_data, "pass": bool(gateA),
        },
        "gateB_error": {
            "rule": f"stable subset: exists p in {{fp16,bf16}} s.t. median {p} attr rep-error <= {TAU} AND median {p} optimizer-state moment rep-error <= {TAU} (all attrs/groups)",
            "scene_pass": gateB_scene_pass, "n_pass": nB, "n_data": n_data, "pass": bool(gateB),
        },
        "gateC_economics": {
            "rule": f"stable_fraction >= {ECON_FRAC_MIN} (>=20% mem reduction) AND stable_count >= {ECON_COUNT_MIN}",
            "scene_pass": gateC_scene_pass, "n_pass": nC, "n_data": n_data, "pass": bool(gateC),
        },
        "verdict": verdict,
    }

    # ---- assemble deliverables ----
    summary = {
        "verdict": verdict,
        "scenes": scenes,
        "phases": {"early": 1000, "densify_complete": 15000, "mature": 30000},
        "per_scene": per_scene,
        "gate": gate,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(args.out, "GSC0_SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "GSC0_ERROR_ORACLE.json"), "w") as f:
        json.dump(oracle_by_scene, f, indent=2)
    with open(os.path.join(args.out, "GSC0_VERDICT.json"), "w") as f:
        json.dump(gate, f, indent=2)
    print(f"[gsc0-analysis] VERDICT = {verdict}  (A={gateA} B={gateB} C={gateC}; "
          f"nA={nA} nB={nB} nC={nC} nData={n_data})", flush=True)
    print(">>> ANALYSIS SUCCESS", flush=True)


if __name__ == "__main__":
    main()