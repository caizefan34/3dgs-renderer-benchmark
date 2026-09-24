#!/usr/bin/env python3
"""C42 Constrained Oracle — Final with UNIFIED metrics (all cameras, same pipeline)."""
import json, io
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BATCH_DIR = ROOT / "results" / "c42_adaptive" / "completion_batch"

LOSS_COST = {"1.0": 33.15, "0.75": 19.41, "0.625": 13.92, "0.5": 9.24}
SCENES = ["room", "garden", "bicycle"]
SCALES = ["1.0", "0.75", "0.625", "0.5"]
TAUS = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]

# === UNIFIED metrics (all cameras, same SepSSIM + LPIPS pipeline) ===
UNIFIED = {
    "room": {
        "1.0": {"psnr": 31.9414, "ssim": 0.9261, "lpips": 0.3003},
        "0.75": {"psnr": 31.7747, "ssim": 0.9177, "lpips": 0.3169},
        "0.5": {"psnr": 32.0738, "ssim": 0.9235, "lpips": 0.3075},
    },
    "garden": {
        "1.0": {"psnr": 29.2113, "ssim": 0.8882, "lpips": 0.1473},
        "0.75": {"psnr": 28.9116, "ssim": 0.8708, "lpips": 0.1680},
        "0.625": {"psnr": 28.8445, "ssim": 0.8654, "lpips": 0.1746},
        "0.5": {"psnr": 28.8635, "ssim": 0.8669, "lpips": 0.1721},
    },
    "bicycle": {
        "1.0": {"psnr": 26.3773, "ssim": 0.8360, "lpips": 0.2463},
        "0.75": {"psnr": 26.1592, "ssim": 0.8135, "lpips": 0.2694},
        "0.625": {"psnr": 26.1134, "ssim": 0.8052, "lpips": 0.2791},
        "0.5": {"psnr": 26.3374, "ssim": 0.8133, "lpips": 0.2694},
    },
}


def delta(scene, scale, metric):
    m1 = UNIFIED[scene]["1.0"].get(metric)
    ms = UNIFIED[scene].get(scale, {}).get(metric)
    if m1 is None or ms is None:
        return None
    return ms - m1


def abs_delta(scene, scale, metric="ssim"):
    d = delta(scene, scale, metric)
    return abs(d) if d is not None else None


def main():
    print("=== UNIFIED SSIM/PSNR/LPIPS DELTAS (vs scale=1.0, all cameras) ===")
    for scene in SCENES:
        for scale in SCALES:
            if scale == "1.0":
                continue
            ds = delta(scene, scale, "ssim")
            dp = delta(scene, scale, "psnr")
            dl = delta(scene, scale, "lpips")
            ds_str = f"{ds:+.4f}" if ds is not None else "N/A"
            dp_str = f"{dp:+.4f}" if dp is not None else "N/A"
            dl_str = f"{dl:+.4f}" if dl is not None else "N/A"
            print(f"  {scene} s={scale}: dSSIM={ds_str} dPSNR={dp_str} dLPIPS={dl_str}")

    oracle = {
        "experiment": "C42 Constrained Oracle - Final (UNIFIED metrics, all cameras)",
        "metrics_source": "unified_metrics.json — all cameras, same SepSSIM+LPIPS pipeline",
        "metrics": UNIFIED,
        "loss_cost_ms": LOSS_COST,
        "tau_sweep": {},
        "multi_metric": {},
    }

    for tau in TAUS:
        tau_str = f"{tau:.3f}"
        entry = {"tau": tau, "per_scene": {}, "adaptive_oracle": {}, "global_feasible": [], "best_fixed_feasible": {}}
        oracle_costs = []
        scene_feasible = {}

        for scene in SCENES:
            feasible = []
            for s in SCALES:
                if s == "1.0":
                    feasible.append(s)
                    continue
                d = abs_delta(scene, s, "ssim")
                if d is not None and d <= tau:
                    feasible.append(s)
            scene_feasible[scene] = feasible
            if feasible:
                best = min(feasible, key=lambda x: LOSS_COST[x])
                entry["per_scene"][scene] = {
                    "feasible_scales": feasible,
                    "oracle_scale": best,
                    "oracle_cost_ms": LOSS_COST[best],
                    "abs_dssim": round(abs_delta(scene, best, "ssim"), 4) if best != "1.0" else 0.0,
                }
                oracle_costs.append(LOSS_COST[best])
            else:
                entry["per_scene"][scene] = {"feasible_scales": [], "oracle_scale": None, "oracle_cost_ms": None}

        avg_oracle = sum(oracle_costs) / len(oracle_costs) if oracle_costs else None
        entry["adaptive_oracle"] = {
            "per_scene_scales": {s: entry["per_scene"][s]["oracle_scale"] for s in SCENES},
            "average_cost_ms": round(avg_oracle, 2) if avg_oracle else None,
        }

        gfs = set(scene_feasible.get("room", [])) & set(scene_feasible.get("garden", [])) & set(scene_feasible.get("bicycle", []))
        entry["global_feasible"] = sorted(gfs)
        if gfs:
            bf = min(gfs, key=lambda x: LOSS_COST[x])
            entry["best_fixed_feasible"] = {"scale": bf, "cost_ms": LOSS_COST[bf]}
            entry["adaptive_advantage_ms"] = round(LOSS_COST[bf] - avg_oracle, 2) if avg_oracle else None
            entry["adaptive_advantage_pct"] = round((LOSS_COST[bf] - avg_oracle) / LOSS_COST[bf] * 100, 1) if avg_oracle else None
        else:
            entry["best_fixed_feasible"] = {"scale": None, "cost_ms": None}
            entry["adaptive_advantage_ms"] = None

        oracle["tau_sweep"][tau_str] = entry

    # Multi-metric constraints
    constraint_families = [
        ("SSIM_only_020", {"ssim_abs": 0.020}),
        ("PSNR050_SSIM020", {"psnr_delta": 0.50, "ssim_abs": 0.020}),
        ("PSNR050_SSIM020_LPIPS003", {"psnr_delta": 0.50, "ssim_abs": 0.020, "lpips_delta": 0.03}),
        ("SSIM_only_010", {"ssim_abs": 0.010}),
        ("PSNR050_SSIM010", {"psnr_delta": 0.50, "ssim_abs": 0.010}),
    ]

    for fam_name, constraints in constraint_families:
        fam = {"constraints": constraints, "per_scene": {}, "advantage_ms": None}
        oracle_costs = []
        scene_feasible = {}

        for scene in SCENES:
            feasible = []
            for s in SCALES:
                if s == "1.0":
                    feasible.append(s)
                    continue
                ok = True
                if "ssim_abs" in constraints:
                    d = abs_delta(scene, s, "ssim")
                    if d is None or d > constraints["ssim_abs"]:
                        ok = False
                if "psnr_delta" in constraints and ok:
                    d = delta(scene, s, "psnr")
                    if d is None or d < -constraints["psnr_delta"]:
                        ok = False
                if "lpips_delta" in constraints and ok:
                    d = delta(scene, s, "lpips")
                    if d is None or d > constraints["lpips_delta"]:
                        ok = False
                if ok:
                    feasible.append(s)
            scene_feasible[scene] = feasible
            if feasible:
                best = min(feasible, key=lambda x: LOSS_COST[x])
                fam["per_scene"][scene] = {"oracle_scale": best, "cost_ms": LOSS_COST[best]}
                oracle_costs.append(LOSS_COST[best])
            else:
                fam["per_scene"][scene] = {"oracle_scale": None, "cost_ms": None}

        avg_oracle = sum(oracle_costs) / len(oracle_costs) if oracle_costs else None
        gfs = set(scene_feasible.get("room", [])) & set(scene_feasible.get("garden", [])) & set(scene_feasible.get("bicycle", []))
        if gfs:
            bf = min(gfs, key=lambda x: LOSS_COST[x])
            fam["global_feasible"] = sorted(gfs)
            fam["best_fixed_feasible"] = {"scale": bf, "cost_ms": LOSS_COST[bf]}
            fam["advantage_ms"] = round(LOSS_COST[bf] - avg_oracle, 2) if avg_oracle else None
            fam["advantage_pct"] = round((LOSS_COST[bf] - avg_oracle) / LOSS_COST[bf] * 100, 1) if avg_oracle else None
        else:
            fam["global_feasible"] = []
            fam["best_fixed_feasible"] = {"scale": None, "cost_ms": None}

        oracle["multi_metric"][fam_name] = fam

    t020 = oracle["tau_sweep"]["0.020"]
    oracle["summary"] = {
        "tau020_ssim_only_advantage_ms": t020["adaptive_advantage_ms"],
        "tau020_global_075_feasible": "0.75" in t020.get("global_feasible", []),
        "tau020_per_scene_oracle": {s: t020["per_scene"][s]["oracle_scale"] for s in SCENES},
        "tau020_multi_metric_advantage_ms": oracle["multi_metric"]["PSNR050_SSIM020"]["advantage_ms"],
        "tau020_full_metric_advantage_ms": oracle["multi_metric"]["PSNR050_SSIM020_LPIPS003"]["advantage_ms"],
        "bicycle_075_dssim_unified": delta("bicycle", "0.75", "ssim"),
        "bicycle_075_feasible_at_020": abs_delta("bicycle", "0.75", "ssim") <= 0.020,
    }

    with io.open(str(BATCH_DIR / "constrained_oracle_final.json"), "w", encoding="utf-8") as f:
        json.dump(oracle, f, indent=2, ensure_ascii=False)
    print(f"\nOracle saved to {BATCH_DIR / 'constrained_oracle_final.json'}")

    print("\n=== ORACLE SUMMARY (SSIM only, UNIFIED metrics) ===")
    for tau in TAUS:
        e = oracle["tau_sweep"][f"{tau:.3f}"]
        per = {s: e["per_scene"][s]["oracle_scale"] for s in SCENES}
        print(f"  tau={tau:.3f}: oracle={per} avg={e['adaptive_oracle']['average_cost_ms']}ms "
              f"global_feasible={e['global_feasible']} advantage={e['adaptive_advantage_ms']}ms")

    print(f"\n=== KEY FINDINGS ===")
    print(f"  Bicycle 0.75 dSSIM (unified): {oracle['summary']['bicycle_075_dssim_unified']:+.4f}")
    print(f"  Bicycle 0.75 feasible at tau=0.020: {oracle['summary']['bicycle_075_feasible_at_020']}")
    print(f"  Global 0.75 feasible at tau=0.020: {oracle['summary']['tau020_global_075_feasible']}")
    print(f"  SSIM-only advantage: {oracle['summary']['tau020_ssim_only_advantage_ms']}ms")
    print(f"  Multi-metric (PSNR+SSIM) advantage: {oracle['summary']['tau020_multi_metric_advantage_ms']}ms")
    print(f"  Full-metric (PSNR+SSIM+LPIPS) advantage: {oracle['summary']['tau020_full_metric_advantage_ms']}ms")


if __name__ == "__main__":
    main()
