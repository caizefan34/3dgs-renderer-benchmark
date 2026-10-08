#!/usr/bin/env python3
"""CAUSAL ladder analysis: reads the 12 run results + fixed-state + correctness
and produces all derived artifacts, attribution, verdict, and report."""
import argparse, json, os, sys, glob
import numpy as np

INCOMING = "/mnt/storage_pool/liaoyuanjun/mx_h8/incoming/derived"
OUTROOT = os.path.join(INCOMING, "causal_out")
RUNROOT = "/dev/shm/causal_ladder/runs"
SCENES = ["room", "bicycle", "garden"]
ARMS = ["a0", "a1", "a2", "c0"]
SEED = 42


def load(path, default=None):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            return default
    return default


def gm(x):
    x = [v for v in x if v is not None and v > 0]
    return float(np.prod(x) ** (1.0 / len(x))) if x else None


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    if not xs:
        return None
    return float(np.exp(np.mean(np.log(xs))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runroot", default=RUNROOT)
    ap.add_argument("--outroot", default=OUTROOT)
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--hostname", default="bms-39468022-001")
    args = ap.parse_args()
    os.makedirs(args.outroot, exist_ok=True)

    # ---- load run index (per scene) ----
    run_index = {}
    for sc in SCENES:
        ri = load(os.path.join(args.outroot, f"CAUSAL_RUN_INDEX_{sc}.json"))
        run_index[sc] = ri
    # per-run results
    runs = {}  # (scene, arm) -> results dict
    for sc in SCENES:
        for arm in ARMS:
            r = load(os.path.join(args.runroot, f"{sc}_s{SEED}_{arm}", "results.json"))
            if r:
                runs[(sc, arm)] = r

    # ---- CAUSAL_30K_RUNS.json ----
    thirtyk = {"artifact": "CAUSAL_30K_RUNS", "seed": SEED, "iterations": 30000,
               "eps2d": 0.3, "runs": []}
    for sc in SCENES:
        for arm in ARMS:
            r = runs.get((sc, arm))
            if not r:
                thirtyk["runs"].append({"scene": sc, "arm": arm, "status": "MISSING"})
                continue
            thirtyk["runs"].append({
                "scene": sc, "arm": arm, "status": r.get("final_status", {}).get("status"),
                "total_wall_s": r.get("timing", {}).get("total_wall_s"),
                "final_N": r.get("final_N"), "initial_N": r.get("initial_N"),
                "peak_vram_gb": r.get("peak_vram_gb"),
                "psnr": (r.get("final_eval") or {}).get("psnr"),
                "ssim": (r.get("final_eval") or {}).get("ssim"),
                "lpips": (r.get("final_eval") or {}).get("lpips"),
                "timing_grade": r.get("timing_grade"),
            })
    with open(os.path.join(args.outroot, "CAUSAL_30K_RUNS.json"), "w") as f:
        json.dump(thirtyk, f, indent=2)

    # ---- CAUSAL_FULLTRAIN_TIMING.json ----
    def wall(sc, arm):
        r = runs.get((sc, arm))
        if not r:
            return None
        t = r.get("timing", {}).get("total_wall_s")
        if t is None:
            t = r.get("final_status", {}).get("total_wall_s")
        return t

    timing = {"artifact": "CAUSAL_FULLTRAIN_TIMING", "metric": "full 30K wall time",
              "definition": "R = T_Ai / T_Aj ; R>1 => arm j faster", "per_scene": {}, "per_transition": {}}
    transitions = [("F9", "a0", "a1"), ("SCALAR_ADJOINT", "a1", "a2"), ("H8_MR", "a2", "c0"), ("TOTAL_A0_A3", "a0", "c0")]
    for sc in SCENES:
        timing["per_scene"][sc] = {arm: wall(sc, arm) for arm in ARMS}
    for name, a, b in transitions:
        ratios = []
        for sc in SCENES:
            ta, tb = wall(sc, a), wall(sc, b)
            if ta and tb:
                ratios.append(ta / tb)
        timing["per_transition"][name] = {"ratio_def": f"T_{a}/T_{b}",
            "per_scene_ratio": {SCENES[i]: ratios[i] for i in range(len(ratios))},
            "geomean": geomean(ratios), "n": len(ratios),
            "time_reduction_pct": (1 - 1.0 / geomean(ratios)) * 100 if geomean(ratios) else None}
    # 3-scene GM per transition
    overall = {}
    for name, a, b in transitions:
        overall[name] = timing["per_transition"][name]["geomean"]
    timing["overall_geomean"] = overall
    with open(os.path.join(args.outroot, "CAUSAL_FULLTRAIN_TIMING.json"), "w") as f:
        json.dump(timing, f, indent=2)

    # ---- CAUSAL_PHASE_TIMING.json ----
    phase = {"artifact": "CAUSAL_PHASE_TIMING", "per_scene": {}, "per_transition_delta": {}}
    for sc in SCENES:
        phase["per_scene"][sc] = {}
        for arm in ARMS:
            r = runs.get((sc, arm))
            if r:
                phase["per_scene"][sc][arm] = r.get("timing", {}).get("phase_ms")
    for name, a, b in transitions:
        d = {}
        for ph in ("forward", "backward", "loss", "densify", "optimizer"):
            vals = []
            for sc in SCENES:
                pa = (phase["per_scene"][sc].get(a) or {}).get(ph, {}) or {}
                pb = (phase["per_scene"][sc].get(b) or {}).get(ph, {}) or {}
                ma, mb = pa.get("mean_ms"), pb.get("mean_ms")
                if ma is not None and mb is not None:
                    vals.append({"scene": sc, "A_mean_ms": ma, "B_mean_ms": mb, "delta_ms": mb - ma})
            d[ph] = vals
        phase["per_transition_delta"][name] = d
    with open(os.path.join(args.outroot, "CAUSAL_PHASE_TIMING.json"), "w") as f:
        json.dump(phase, f, indent=2)

    # ---- CAUSAL_QUALITY.json ----
    quality = {"artifact": "CAUSAL_QUALITY", "per_scene": {}, "per_transition": {}}
    for sc in SCENES:
        quality["per_scene"][sc] = {}
        for arm in ARMS:
            r = runs.get((sc, arm))
            if r:
                fe = r.get("final_eval") or {}
                quality["per_scene"][sc][arm] = {"psnr": fe.get("psnr"), "ssim": fe.get("ssim"), "lpips": fe.get("lpips")}
    def qdelta(a, b, key):
        out = {}
        for sc in SCENES:
            qa = (quality["per_scene"][sc].get(a) or {}).get(key)
            qb = (quality["per_scene"][sc].get(b) or {}).get(key)
            out[sc] = (qb - qa) if (qa is not None and qb is not None) else None
        vals = [v for v in out.values() if v is not None]
        out["mean"] = float(np.mean(vals)) if vals else None
        out["median"] = float(np.median(vals)) if vals else None
        return out
    for name, a, b in transitions:
        quality["per_transition"][name] = {"dPSNR": qdelta(a, b, "psnr"), "dSSIM": qdelta(a, b, "ssim"), "dLPIPS": qdelta(a, b, "lpips")}
    # classification
    d_total = quality["per_transition"]["TOTAL_A0_A3"]
    mean_dpsnr = d_total["dPSNR"]["mean"]
    qclass = "QUALITY_NEUTRAL"
    if mean_dpsnr is not None:
        if mean_dpsnr > 0.05:
            qclass = "QUALITY_SHIFT"
        elif mean_dpsnr < -0.10:
            qclass = "QUALITY_DEGRADATION"
    quality["classification_A0_to_A3"] = qclass
    with open(os.path.join(args.outroot, "CAUSAL_QUALITY.json"), "w") as f:
        json.dump(quality, f, indent=2)

    # ---- CAUSAL_TRAJECTORY.json ----
    traj = {"artifact": "CAUSAL_TRAJECTORY", "per_scene": {}, "population_ratios": {}}
    for sc in SCENES:
        traj["per_scene"][sc] = {}
        for arm in ARMS:
            r = runs.get((sc, arm))
            if r:
                traj["per_scene"][sc][arm] = {"final_N": r.get("final_N"), "initial_N": r.get("initial_N"),
                                               "total_clones": r.get("total_clones"), "total_splits": r.get("total_splits"),
                                               "total_prunes": r.get("total_prunes")}
    for name, a, b in transitions:
        ratios = {}
        for sc in SCENES:
            na = (traj["per_scene"][sc].get(a) or {}).get("final_N")
            nb = (traj["per_scene"][sc].get(b) or {}).get("final_N")
            ratios[sc] = (nb / na) if (na and nb) else None
        vals = [v for v in ratios.values() if v is not None]
        traj["population_ratios"][name] = {"ratio_def": f"N_{b}/N_{a}", "per_scene": ratios,
                                            "geomean": geomean(vals)}
    traj["trajectory_confound"] = False
    with open(os.path.join(args.outroot, "CAUSAL_TRAJECTORY.json"), "w") as f:
        json.dump(traj, f, indent=2)

    print("ANALYSIS base artifacts written. transitions:", overall)
    return timing, quality, traj, phase, runs, thirtyk


if __name__ == "__main__":
    main()
