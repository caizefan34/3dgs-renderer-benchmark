#!/usr/bin/env python3
"""Gate E comparison: matched 2K smoke, B1A vs C0_V3_FINAL30K.

Reads the 6 results.json files produced by final30k_trainer.py and emits:
  artifacts/higs-final30k-compat/smoke_2k_results.json        (gate verdict + summary)
  artifacts/higs-final30k-compat/densification_trajectory.json (per-event matched tables)

Gate rule (operationalized from Addendum A gate E; documented in the artifact):
  FINITE     every recorded loss finite; every eval PSNR in (0,60);
             grad_inf_count == 0 at >= 75% of events per arm-scene.
  TRAJECTORY at each matched record (same iteration), r = max(N_c0/N_b1a, N_b1a/N_c0);
             FAIL if r > 4 for the final 5 matched records (persistent deviation).
  COLLAPSE   N_final / N_initial in [0.5, 50] for each arm-scene.
  GROWTH     total clones, total splits, and final N each within 4x across arms.
  QUALITY    final-eval PSNR within 3 dB across arms (2K sanity, not a quality claim).
Verdict per scene and overall: C0_V3_FINAL30K_PREFLIGHT_PASS / FAIL(reasons).
"""
import json
import os
import sys

SCENES = ["room", "bicycle", "garden"]
ARMS = ["b1a", "c0"]
SMOKE_DIR = "/mnt/storage_pool/liaoyuanjun/final30k_smoke"


def load(arm, scene):
    p = os.path.join(SMOKE_DIR, f"{arm}_{scene}_2k", "results.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def trajectory_ratio(b1a, c0):
    """Matched per-100-iter N_GS ratios."""
    b = {r["iter"]: r for r in b1a["per_iter"]}
    c = {r["iter"]: r for r in c0["per_iter"]}
    common = sorted(set(b) & set(c))
    rows = []
    for it in common:
        nb, nc = b[it]["n_gs"], c[it]["n_gs"]
        r = max(nc / nb, nb / nc) if nb > 0 and nc > 0 else float("inf")
        rows.append({"iter": it, "n_b1a": nb, "n_c0": nc, "ratio": r})
    return rows


def event_table(b1a, c0):
    """Matched densification events by iteration."""
    b = {e["iter"]: e for e in b1a["densification_events"]}
    c = {e["iter"]: e for e in c0["densification_events"]}
    common = sorted(set(b) & set(c))
    rows = []
    for it in common:
        eb, ec = b[it], c[it]
        rows.append({
            "iter": it,
            "n_before_b1a": eb["n_before"], "n_after_b1a": eb["n_after"],
            "cloned_b1a": eb["cloned"], "split_b1a": eb["split"], "pruned_b1a": eb["pruned_total"],
            "selected_b1a": eb["selected"],
            "grad_max_b1a": eb["grad_norm_stats"].get("max"),
            "grad_p99_b1a": eb["grad_norm_stats"].get("p99"),
            "n_before_c0": ec["n_before"], "n_after_c0": ec["n_after"],
            "cloned_c0": ec["cloned"], "split_c0": ec["split"], "pruned_c0": ec["pruned_total"],
            "selected_c0": ec["selected"],
            "grad_max_c0": ec["grad_norm_stats"].get("max"),
            "grad_p99_c0": ec["grad_norm_stats"].get("p99"),
            "grad_inf_count_b1a": eb["grad_inf_count"], "grad_inf_count_c0": ec["grad_inf_count"],
        })
    return rows


def evaluate_scene(scene, b1a, c0):
    reasons = []
    checks = {}
    # FINITE
    bad_losses_b = sum(1 for r in b1a["per_iter"] if not (r["loss"] == r["loss"] and abs(r["loss"]) < 1e30))
    bad_losses_c = sum(1 for r in c0["per_iter"] if not (r["loss"] == r["loss"] and abs(r["loss"]) < 1e30))
    ev_ok_b = all(0 < e["psnr"] < 60 for e in b1a["eval_rows"])
    ev_ok_c = all(0 < e["psnr"] < 60 for e in c0["eval_rows"])
    inf_b = sum(1 for e in b1a["densification_events"] if e["grad_inf_count"] > 0)
    inf_c = sum(1 for e in c0["densification_events"] if e["grad_inf_count"] > 0)
    nev_b = max(1, len(b1a["densification_events"]))
    nev_c = max(1, len(c0["densification_events"]))
    fin_b = bad_losses_b == 0 and ev_ok_b and inf_b <= 0.25 * nev_b
    fin_c = bad_losses_c == 0 and ev_ok_c and inf_c <= 0.25 * nev_c
    checks["finite"] = {"b1a": fin_b, "c0": fin_c,
                        "bad_losses": {"b1a": bad_losses_b, "c0": bad_losses_c},
                        "eval_psnr_ok": {"b1a": ev_ok_b, "c0": ev_ok_c},
                        "events_with_grad_inf": {"b1a": inf_b, "c0": inf_c}}
    if not fin_b or not fin_c:
        reasons.append("FINITE check failed")
    # TRAJECTORY (persistent deviation: last 5 matched records)
    traj = trajectory_ratio(b1a, c0)
    tail = traj[-5:]
    tail_bad = [t for t in tail if t["ratio"] > 4.0]
    checks["trajectory"] = {"n_matched": len(traj), "tail_ratios": [round(t["ratio"], 4) for t in tail],
                            "max_ratio": max((t["ratio"] for t in traj), default=None)}
    if tail_bad:
        reasons.append(f"persistent >4x N_GS deviation in last 5 records: "
                       f"{[(t['iter'], round(t['ratio'], 3)) for t in tail_bad]}")
    # COLLAPSE / EXPLOSION
    for arm, res in (("b1a", b1a), ("c0", c0)):
        g = res["final_N"] / res["initial_N"]
        if not (0.5 <= g <= 50.0):
            reasons.append(f"{arm}: N_final/N_initial = {g:.3f} outside [0.5, 50] (collapse/explosion)")
    checks["growth_factor"] = {"b1a": b1a["final_N"] / b1a["initial_N"],
                               "c0": c0["final_N"] / c0["initial_N"]}
    # GROWTH within 4x
    for name, vb, vc in (("total_clones", b1a["total_clones"], c0["total_clones"]),
                         ("total_splits", b1a["total_splits"], c0["total_splits"]),
                         ("final_N", b1a["final_N"], c0["final_N"])):
        r = max(vc / vb, vb / vc) if vb and vc else float("inf")
        checks[f"{name}_ratio"] = round(r, 4)
        if r > 4.0:
            reasons.append(f"{name} ratio {r:.2f} > 4x (b1a={vb}, c0={vc})")
    # QUALITY sanity (3 dB at 2K)
    pb = b1a["final_eval"]["psnr"]
    pc = c0["final_eval"]["psnr"]
    checks["final_psnr"] = {"b1a": pb, "c0": pc, "delta_db": pc - pb}
    if abs(pc - pb) > 3.0:
        reasons.append(f"final PSNR delta {abs(pc - pb):.2f} dB > 3 dB at 2K")
    return {
        "scene": scene,
        "gate_checks": checks,
        "fail_reasons": reasons,
        "verdict": "PASS" if not reasons else "FAIL",
        "summary": {
            "initial_N": b1a["initial_N"],
            "final_N": {"b1a": b1a["final_N"], "c0": c0["final_N"]},
            "total_clones": {"b1a": b1a["total_clones"], "c0": c0["total_clones"]},
            "total_splits": {"b1a": b1a["total_splits"], "c0": c0["total_splits"]},
            "total_prunes": {"b1a": b1a["total_prunes"], "c0": c0["total_prunes"]},
            "final_psnr": {"b1a": pb, "c0": pc},
            "final_ssim": {"b1a": b1a["final_eval"]["ssim"], "c0": c0["final_eval"]["ssim"]},
            "loss_first": {"b1a": b1a["per_iter"][0]["loss"], "c0": c0["per_iter"][0]["loss"]},
            "loss_last": {"b1a": b1a["per_iter"][-1]["loss"], "c0": c0["per_iter"][-1]["loss"]},
            "wall_min": {"b1a": b1a["timing"]["total_wall_min"], "c0": c0["timing"]["total_wall_min"]},
            "timing_grade": b1a["timing_grade"],
            "binary_identity": {
                "b1a_so_sha256": b1a["binary_identity"]["so_sha256"],
                "c0_so_sha256": c0["binary_identity"]["so_sha256"],
            },
        },
        "n_gaussian_trajectory": traj,
        "densification_events_matched": event_table(b1a, c0),
    }


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "artifacts/higs-final30k-compat"
    os.makedirs(out_dir, exist_ok=True)
    per_scene = {}
    missing = []
    for scene in SCENES:
        b1a = load("b1a", scene)
        c0 = load("c0", scene)
        if b1a is None or c0 is None:
            missing.append(f"{scene}: b1a={'ok' if b1a else 'MISSING'} c0={'ok' if c0 else 'MISSING'}")
            continue
        per_scene[scene] = evaluate_scene(scene, b1a, c0)
    all_pass = (not missing) and all(s["verdict"] == "PASS" for s in per_scene.values())
    result = {
        "gate": "E",
        "name": "matched_2k_smoke_b1a_vs_c0_v3_final30k",
        "verdict": "C0_V3_FINAL30K_PREFLIGHT_PASS" if all_pass else
                   ("INCOMPLETE" if missing else "C0_V3_FINAL30K_PREFLIGHT_FAIL"),
        "missing": missing,
        "protocol": {
            "iterations": 2000, "scenes": SCENES,
            "recipe": "reference_v1 frozen (seed 42, densify 500-15000/100, threshold 0.0008, "
                      "reset 3000, SH/1000, lambda_dssim 0.2, 1080p, COLMAP SfM init)",
            "b1a_render": "accutile=True, absgrad=True, eps2d=0.1, tile 16, packed=False",
            "c0_render": "V3 env (F9+SCALAR_ADJOINT+H8_MR, PX=2) + HIGS_BWD_ABSGRAD=1, eps2d=0.3, "
                         "backward_mode=higs_native",
            "gpu": "shared GPU (FUNCTIONAL_ONLY; timing not publication-grade)",
            "gate_rule": {
                "finite": "no NaN/Inf losses; eval PSNR in (0,60); grad inf at <=25% of events",
                "trajectory": "no >4x N_GS ratio in the final 5 matched 100-iter records (persistent)",
                "collapse": "N_final/N_initial in [0.5, 50] per arm",
                "growth": "total clones / total splits / final N each within 4x across arms",
                "quality": "final-eval PSNR within 3 dB across arms at 2K (sanity only)",
            },
        },
        "per_scene": per_scene,
    }
    p1 = os.path.join(out_dir, "smoke_2k_results.json")
    with open(p1, "w") as f:
        json.dump(result, f, indent=2)
    # trajectory artifact (per-event + per-100-iter, matched)
    traj_art = {
        "gate": "E",
        "densification_trajectory": {
            s: {"n_gaussian_trajectory": per_scene[s]["n_gaussian_trajectory"],
                "events": per_scene[s]["densification_events_matched"]}
            for s in per_scene
        },
    }
    p2 = os.path.join(out_dir, "densification_trajectory.json")
    with open(p2, "w") as f:
        json.dump(traj_art, f, indent=2)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ("verdict", "missing")}, indent=2))
    for s, r in per_scene.items():
        print(f"\n=== {s}: {r['verdict']}")
        print(json.dumps(r["summary"], indent=2))
        if r["fail_reasons"]:
            print("FAIL REASONS:", r["fail_reasons"])
    print(f"\nWROTE {p1}\nWROTE {p2}")


if __name__ == "__main__":
    main()
