#!/usr/bin/env python3
"""
Phase C53-Validation2 — Footprint Prediction & Incremental Analysis

Experiment B: R_current → W_future (screen radius prediction)
Experiment C: S_current → W_future (scale prediction)
Incremental: M1(W→Wf), M2(R→Wf), M3(W+R→Wf) with R² and ΔR²
Residual analysis: R_current → residual(Wf after W_current)
"""
import json
import numpy as np
from pathlib import Path

RESULT_DIR = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c53-validation2")
OUT_DIR = RESULT_DIR
CHECKPOINTS = [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]
DELTAS = [10, 50, 100]

SIGNALS = ["current_tiles_mean", "screen_radius_mean", "scale_norm",
           "visibility_count", "ema_grad_norm", "opacity", "age"]


def safe_pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def recall_at_k(signal, target, k_percent):
    n = len(signal)
    k = max(1, int(n * k_percent / 100))
    top_sig = set(np.argsort(signal)[-k:])
    top_tgt = set(np.argsort(target)[-k:])
    return len(top_sig & top_tgt) / len(top_tgt) if top_tgt else 0.0


def coverage_at_k(signal, target, k_percent):
    n = len(signal)
    k = max(1, int(n * k_percent / 100))
    top_sig = np.argsort(signal)[-k:]
    total = target.sum()
    return float(target[top_sig].sum() / total) if total > 1e-12 else 0.0


def ols_r2(X, y):
    """Fit OLS and return R². X can be 1D or 2D."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = len(y)
    X_aug = np.column_stack([np.ones(n), X])
    try:
        beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        y_pred = X_aug @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0, y_pred
    except Exception:
        return 0.0, np.full_like(y, y.mean())


def ols_residual(X, y):
    """Fit OLS and return residual."""
    r2, y_pred = ols_r2(X, y)
    return y - y_pred


def analyze_scene(scene, suffix=""):
    fp = RESULT_DIR / f"{scene}{suffix}_workload.npz"
    if not fp.exists():
        return None
    data = np.load(fp, allow_pickle=True)
    results = {"scene": scene + suffix, "checkpoints": {}}

    for cp in CHECKPOINTS:
        p = f"cp{cp}"
        if f"{p}_ids" not in data:
            continue

        signals = {}
        for sig in SIGNALS:
            key = f"{p}_s_{sig}"
            if key in data:
                signals[sig] = data[key].astype(np.float64)

        cp_res = {}
        for d in DELTAS:
            dp = f"{p}_d{d}"
            if f"{dp}_alive" not in data:
                continue
            alive = data[f"{dp}_alive"]
            w_future = data[f"{dp}_tiles_mean"].astype(np.float64)

            # Mask: alive AND current workload > 0 (visible Gaussians)
            w_current = signals.get("current_tiles_mean", np.zeros_like(w_future))
            mask = alive & (w_current > 0)
            if mask.sum() < 100:
                continue

            wf = w_future[mask]
            wc = w_current[mask]
            rc = signals["screen_radius_mean"][mask] if "screen_radius_mean" in signals else np.zeros_like(wf)
            sc = signals["scale_norm"][mask] if "scale_norm" in signals else np.zeros_like(wf)

            d_res = {}

            # Experiment A: W_current → W_future
            d_res["persistence"] = {
                "pearson": safe_pearson(wc, wf),
                "recall_20": recall_at_k(wc, wf, 20),
            }

            # Experiment B: R_current → W_future
            d_res["footprint"] = {
                "pearson": safe_pearson(rc, wf),
                "recall_20": recall_at_k(rc, wf, 20),
            }

            # Experiment C: Scale → W_future
            d_res["scale"] = {
                "pearson": safe_pearson(sc, wf),
                "recall_20": recall_at_k(sc, wf, 20),
            }

            # M1/M2/M3 R² comparison
            r2_m1, _ = ols_r2(wc, wf)  # M1: W_future ~ W_current
            r2_m2, _ = ols_r2(rc, wf)  # M2: W_future ~ R_current
            r2_m3, _ = ols_r2(np.column_stack([wc, rc]), wf)  # M3: W_future ~ W_current + R_current
            d_res["models"] = {
                "R2_M1_workload": r2_m1,
                "R2_M2_footprint": r2_m2,
                "R2_M3_combined": r2_m3,
                "delta_R2": r2_m3 - r2_m1,
            }

            # Residual analysis: R → residual(Wf - W_current)
            residual_wf = ols_residual(wc, wf)
            d_res["residual"] = {
                "screen_radius_to_residual": safe_pearson(rc, residual_wf),
                "scale_to_residual": safe_pearson(sc, residual_wf),
            }

            # All signals → future work
            d_res["all_signals"] = {}
            for sig_name, sig_vals in signals.items():
                sv = sig_vals[mask]
                if np.std(sv) < 1e-12:
                    continue
                d_res["all_signals"][sig_name] = {
                    "pearson": safe_pearson(sv, wf),
                    "recall_10": recall_at_k(sv, wf, 10),
                    "recall_20": recall_at_k(sv, wf, 20),
                    "recall_50": recall_at_k(sv, wf, 50),
                    "coverage_20": coverage_at_k(sv, wf, 20),
                }

            cp_res[str(d)] = d_res
        results["checkpoints"][str(cp)] = cp_res
    return results


def main():
    print("=" * 100)
    print("Phase C53-Validation2 — Footprint Prediction & Incremental Analysis")
    print("=" * 100)

    scenes = ["room", "garden", "bicycle", "room_seed123", "garden_seed123"]
    all_results = {}
    for scene in scenes:
        parts = scene.split("_seed")
        base, suf = parts[0], f"_seed{parts[1]}" if len(parts) > 1 else ""
        r = analyze_scene(base, suf)
        if r:
            all_results[scene] = r

    # Cross-scene summary
    primary = ["room", "garden", "bicycle"]
    for d in DELTAS:
        print(f"\nΔ={d} (mean across room/garden/bicycle, 8 checkpoints):")
        print(f"  {'Signal':<25} {'→ W_future':>12} {'Recall@20':>10}")
        for sig in SIGNALS + ["M1_workload", "M2_footprint", "M3_combined"]:
            pears = []
            recs = []
            for scene in primary:
                if scene in all_results:
                    for cp in CHECKPOINTS:
                        cp_str = str(cp)
                        if cp_str in all_results[scene]["checkpoints"] and str(d) in all_results[scene]["checkpoints"][cp_str]:
                            dd = all_results[scene]["checkpoints"][cp_str][str(d)]
                            if sig.startswith("M"):
                                pears.append(dd["models"][f"R2_{sig}"])
                                # For models, use R² as "pearson"
                                recs.append(0)
                            elif sig in dd.get("all_signals", {}):
                                pears.append(dd["all_signals"][sig]["pearson"])
                                recs.append(dd["all_signals"][sig]["recall_20"])
            if pears:
                if sig.startswith("M"):
                    print(f"  {sig:<25} R²={np.mean(pears):>10.4f}")
                else:
                    print(f"  {sig:<25} {np.mean(pears):>12.3f} {np.mean(recs):>10.3f}")

        # ΔR²
        dR2 = []
        for scene in primary:
            if scene in all_results:
                for cp in CHECKPOINTS:
                    cp_str = str(cp)
                    if cp_str in all_results[scene]["checkpoints"] and str(d) in all_results[scene]["checkpoints"][cp_str]:
                        dR2.append(all_results[scene]["checkpoints"][cp_str][str(d)]["models"]["delta_R2"])
        if dR2:
            print(f"\n  ΔR² (M3-M1): mean={np.mean(dR2):.4f}, std={np.std(dR2):.4f}")

        # Residual
        resid = []
        for scene in primary:
            if scene in all_results:
                for cp in CHECKPOINTS:
                    cp_str = str(cp)
                    if cp_str in all_results[scene]["checkpoints"] and str(d) in all_results[scene]["checkpoints"][cp_str]:
                        resid.append(all_results[scene]["checkpoints"][cp_str][str(d)]["residual"]["screen_radius_to_residual"])
        if resid:
            print(f"  Residual corr (R → residual after W): mean={np.mean(resid):.4f}, std={np.std(resid):.4f}")

    # Save
    out_file = OUT_DIR / "incremental_prediction.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_file}")

    # Also save footprint prediction separately
    fp_results = {}
    for scene, sr in all_results.items():
        fp_results[scene] = {}
        for cp_str, cp_data in sr["checkpoints"].items():
            fp_results[scene][cp_str] = {}
            for d_str, d_data in cp_data.items():
                fp_results[scene][cp_str][d_str] = {
                    "footprint_pearson": d_data.get("footprint", {}).get("pearson", 0),
                    "footprint_recall_20": d_data.get("footprint", {}).get("recall_20", 0),
                    "scale_pearson": d_data.get("scale", {}).get("pearson", 0),
                }
    fp_file = OUT_DIR / "footprint_prediction.json"
    with open(fp_file, 'w') as f:
        json.dump(fp_results, f, indent=2)
    print(f"Saved to {fp_file}")

    # Scale prediction separately
    sc_results = {}
    for scene, sr in all_results.items():
        sc_results[scene] = {}
        for cp_str, cp_data in sr["checkpoints"].items():
            sc_results[scene][cp_str] = {}
            for d_str, d_data in cp_data.items():
                sc_results[scene][cp_str][d_str] = d_data.get("scale", {})
    sc_file = OUT_DIR / "scale_prediction.json"
    with open(sc_file, 'w') as f:
        json.dump(sc_results, f, indent=2)
    print(f"Saved to {sc_file}")


if __name__ == "__main__":
    main()
