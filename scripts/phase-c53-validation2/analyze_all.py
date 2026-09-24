#!/usr/bin/env python3
"""
Phase C53-Validation2 — Ranking Comparison, Horizon, Age, Scene, Seed, Final
Combines remaining analyses into one script for efficiency.
"""
import json
import numpy as np
from pathlib import Path

RESULT_DIR = Path("/home/liaoyuanjun/3dgs-renderer-benchmark/results/a100/phase-c53-validation2")
OUT_DIR = RESULT_DIR
CHECKPOINTS = [500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]
DELTAS = [10, 50, 100]
AGE_GROUPS = [(0, 100), (100, 500), (500, 2000), (2000, 10**9)]
AGE_LABELS = ["<100", "100-500", "500-2000", ">2000"]
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
        return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    except Exception:
        return 0.0


def load_scene(scene, suffix=""):
    fp = RESULT_DIR / f"{scene}{suffix}_workload.npz"
    if not fp.exists():
        return None
    return np.load(fp, allow_pickle=True)


def main():
    print("=" * 100)
    print("Phase C53-Validation2 — Ranking / Horizon / Age / Scene / Seed / Final")
    print("=" * 100)

    scenes_map = {
        "room": ("room", ""),
        "garden": ("garden", ""),
        "bicycle": ("bicycle", ""),
        "room_seed123": ("room", "_seed123"),
        "garden_seed123": ("garden", "_seed123"),
    }

    all_data = {}
    for name, (base, suf) in scenes_map.items():
        d = load_scene(base, suf)
        if d is not None:
            all_data[name] = d

    primary = ["room", "garden", "bicycle"]

    # ============ RANKING COMPARISON ============
    print("\n" + "=" * 100)
    print("RANKING COMPARISON (Δ=50, mean across room/garden/bicycle)")
    print("=" * 100)
    ranking_results = {}
    print(f"  {'Predictor':<25} {'Recall@10':>10} {'Recall@20':>10} {'Recall@50':>10} {'Cov@20':>10} {'Cov@50':>10}")
    for sig in SIGNALS:
        vals = {k: [] for k in ["recall_10", "recall_20", "recall_50", "coverage_20", "coverage_50"]}
        for scene in primary:
            if scene not in all_data:
                continue
            data = all_data[scene]
            for cp in CHECKPOINTS:
                p = f"cp{cp}"
                dp = f"{p}_d50"
                sig_key = f"{p}_s_{sig}"
                if f"{dp}_alive" not in data or sig_key not in data:
                    continue
                alive = data[f"{dp}_alive"]
                wf = data[f"{dp}_tiles_mean"].astype(np.float64)
                sv = data[sig_key].astype(np.float64)
                wc = data[f"{p}_s_current_tiles_mean"].astype(np.float64) if f"{p}_s_current_tiles_mean" in data else np.zeros_like(wf)
                mask = alive & (wc > 0)
                if mask.sum() < 100:
                    continue
                vals["recall_10"].append(recall_at_k(sv[mask], wf[mask], 10))
                vals["recall_20"].append(recall_at_k(sv[mask], wf[mask], 20))
                vals["recall_50"].append(recall_at_k(sv[mask], wf[mask], 50))
                vals["coverage_20"].append(coverage_at_k(sv[mask], wf[mask], 20))
                vals["coverage_50"].append(coverage_at_k(sv[mask], wf[mask], 50))
        means = {k: float(np.mean(v)) if v else 0 for k, v in vals.items()}
        ranking_results[sig] = means
        print(f"  {sig:<25} {means['recall_10']:>10.3f} {means['recall_20']:>10.3f} {means['recall_50']:>10.3f} {means['coverage_20']:>10.3f} {means['coverage_50']:>10.3f}")

    # M1/M2/M3 ranking
    print(f"\n  {'Model':<25} {'R²':>10} {'Recall@10':>10} {'Recall@20':>10}")
    for model_name, sigs in [("M1_W_only", ["current_tiles_mean"]), ("M2_R_only", ["screen_radius_mean"]), ("M3_W+R", ["current_tiles_mean", "screen_radius_mean"])]:
        r2s = []
        r10s = []
        r20s = []
        for scene in primary:
            if scene not in all_data:
                continue
            data = all_data[scene]
            for cp in CHECKPOINTS:
                p = f"cp{cp}"
                dp = f"{p}_d50"
                if f"{dp}_alive" not in data:
                    continue
                alive = data[f"{dp}_alive"]
                wf = data[f"{dp}_tiles_mean"].astype(np.float64)
                wc = data[f"{p}_s_current_tiles_mean"].astype(np.float64)
                rc = data[f"{p}_s_screen_radius_mean"].astype(np.float64)
                mask = alive & (wc > 0)
                if mask.sum() < 100:
                    continue
                wc_m, rc_m, wf_m = wc[mask], rc[mask], wf[mask]
                if len(sigs) == 1:
                    X = (wc_m if sigs[0] == "current_tiles_mean" else rc_m).reshape(-1, 1)
                else:
                    X = np.column_stack([wc_m, rc_m])
                r2s.append(ols_r2(X, wf_m))
                # For ranking, use the linear combination as predicted score
                X_aug = np.column_stack([np.ones(len(wf_m)), X])
                try:
                    beta, _, _, _ = np.linalg.lstsq(X_aug, wf_m, rcond=None)
                    pred = X_aug @ beta
                    r10s.append(recall_at_k(pred, wf_m, 10))
                    r20s.append(recall_at_k(pred, wf_m, 20))
                except:
                    r10s.append(0)
                    r20s.append(0)
        print(f"  {model_name:<25} {np.mean(r2s):>10.4f} {np.mean(r10s):>10.3f} {np.mean(r20s):>10.3f}")
        ranking_results[model_name] = {"R2": float(np.mean(r2s)), "recall_10": float(np.mean(r10s)), "recall_20": float(np.mean(r20s))}

    with open(OUT_DIR / "ranking_comparison.json", 'w') as f:
        json.dump(ranking_results, f, indent=2)

    # ============ HORIZON ANALYSIS ============
    print("\n" + "=" * 100)
    print("HORIZON ANALYSIS (mean across room/garden/bicycle)")
    print("=" * 100)
    horizon_results = {}
    for sig in ["current_tiles_mean", "screen_radius_mean", "scale_norm", "visibility_count"]:
        print(f"\n  {sig}:")
        vals_by_d = {}
        for d in DELTAS:
            pears = []
            for scene in primary:
                if scene not in all_data:
                    continue
                data = all_data[scene]
                for cp in CHECKPOINTS:
                    p = f"cp{cp}"
                    dp = f"{p}_d{d}"
                    sig_key = f"{p}_s_{sig}"
                    if f"{dp}_alive" not in data or sig_key not in data:
                        continue
                    alive = data[f"{dp}_alive"]
                    wf = data[f"{dp}_tiles_mean"].astype(np.float64)
                    sv = data[sig_key].astype(np.float64)
                    wc = data[f"{p}_s_current_tiles_mean"].astype(np.float64) if f"{p}_s_current_tiles_mean" in data else np.zeros_like(wf)
                    mask = alive & (wc > 0)
                    if mask.sum() < 100:
                        continue
                    pears.append(safe_pearson(sv[mask], wf[mask]))
            m = float(np.mean(pears)) if pears else 0
            vals_by_d[d] = m
            print(f"    Δ={d}: {m:.3f}")
        horizon_results[sig] = vals_by_d

    with open(OUT_DIR / "horizon_analysis.json", 'w') as f:
        json.dump(horizon_results, f, indent=2)

    # ============ AGE ANALYSIS ============
    print("\n" + "=" * 100)
    print("AGE ANALYSIS (Δ=50, mean across room/garden/bicycle)")
    print("=" * 100)
    age_results = {}
    for ag_idx, (ag_lo, ag_hi) in enumerate(AGE_GROUPS):
        ag_label = AGE_LABELS[ag_idx]
        print(f"\n  Age {ag_label}:")
        print(f"    {'Signal':<25} {'→ W_future':>12} {'Recall@20':>10}")
        ag_data = {}
        for sig in ["current_tiles_mean", "screen_radius_mean", "scale_norm", "visibility_count"]:
            pears = []
            recs = []
            for scene in primary:
                if scene not in all_data:
                    continue
                data = all_data[scene]
                for cp in CHECKPOINTS:
                    p = f"cp{cp}"
                    dp = f"{p}_d50"
                    sig_key = f"{p}_s_{sig}"
                    age_key = f"{p}_s_age"
                    if f"{dp}_alive" not in data or sig_key not in data or age_key not in data:
                        continue
                    alive = data[f"{dp}_alive"]
                    wf = data[f"{dp}_tiles_mean"].astype(np.float64)
                    sv = data[sig_key].astype(np.float64)
                    age_vals = data[age_key].astype(np.float64)
                    wc = data[f"{p}_s_current_tiles_mean"].astype(np.float64) if f"{p}_s_current_tiles_mean" in data else np.zeros_like(wf)
                    mask = alive & (age_vals >= ag_lo) & (age_vals < ag_hi)
                    if sig == "current_tiles_mean":
                        mask = mask & (wc > 0)
                    if mask.sum() < 50:
                        continue
                    pears.append(safe_pearson(sv[mask], wf[mask]))
                    recs.append(recall_at_k(sv[mask], wf[mask], 20))
            m_p = float(np.mean(pears)) if pears else 0
            m_r = float(np.mean(recs)) if recs else 0
            ag_data[sig] = {"pearson": m_p, "recall_20": m_r}
            print(f"    {sig:<25} {m_p:>12.3f} {m_r:>10.3f}")

        # ΔR² for this age group
        dR2 = []
        for scene in primary:
            if scene not in all_data:
                continue
            data = all_data[scene]
            for cp in CHECKPOINTS:
                p = f"cp{cp}"
                dp = f"{p}_d50"
                age_key = f"{p}_s_age"
                if f"{dp}_alive" not in data or age_key not in data:
                    continue
                alive = data[f"{dp}_alive"]
                wf = data[f"{dp}_tiles_mean"].astype(np.float64)
                wc = data[f"{p}_s_current_tiles_mean"].astype(np.float64)
                rc = data[f"{p}_s_screen_radius_mean"].astype(np.float64)
                age_vals = data[age_key].astype(np.float64)
                mask = alive & (age_vals >= ag_lo) & (age_vals < ag_hi) & (wc > 0)
                if mask.sum() < 50:
                    continue
                r2_m1 = ols_r2(wc[mask], wf[mask])
                r2_m3 = ols_r2(np.column_stack([wc[mask], rc[mask]]), wf[mask])
                dR2.append(r2_m3 - r2_m1)
        dR2_mean = float(np.mean(dR2)) if dR2 else 0
        ag_data["delta_R2"] = dR2_mean
        print(f"    {'ΔR² (M3-M1)':<25} {dR2_mean:>12.4f}")
        age_results[ag_label] = ag_data

    with open(OUT_DIR / "age_analysis.json", 'w') as f:
        json.dump(age_results, f, indent=2)

    # ============ SCENE COMPARISON ============
    print("\n" + "=" * 100)
    print("SCENE COMPARISON (Δ=50)")
    print("=" * 100)
    scene_results = {}
    print(f"  {'Scene':<15} {'W→Wf':>8} {'R→Wf':>8} {'ΔR²':>8} {'Recall@20(W)':>14} {'Recall@20(R)':>14}")
    for scene in primary:
        if scene not in all_data:
            continue
        data = all_data[scene]
        w_p, r_p, dR2, w_r20, r_r20 = [], [], [], [], []
        for cp in CHECKPOINTS:
            p = f"cp{cp}"
            dp = f"{p}_d50"
            if f"{dp}_alive" not in data:
                continue
            alive = data[f"{dp}_alive"]
            wf = data[f"{dp}_tiles_mean"].astype(np.float64)
            wc = data[f"{p}_s_current_tiles_mean"].astype(np.float64)
            rc = data[f"{p}_s_screen_radius_mean"].astype(np.float64)
            mask = alive & (wc > 0)
            if mask.sum() < 100:
                continue
            w_p.append(safe_pearson(wc[mask], wf[mask]))
            r_p.append(safe_pearson(rc[mask], wf[mask]))
            r2_m1 = ols_r2(wc[mask], wf[mask])
            r2_m3 = ols_r2(np.column_stack([wc[mask], rc[mask]]), wf[mask])
            dR2.append(r2_m3 - r2_m1)
            w_r20.append(recall_at_k(wc[mask], wf[mask], 20))
            r_r20.append(recall_at_k(rc[mask], wf[mask], 20))
        if w_p:
            m = {"W_to_Wf": float(np.mean(w_p)), "R_to_Wf": float(np.mean(r_p)),
                 "delta_R2": float(np.mean(dR2)), "recall_20_W": float(np.mean(w_r20)),
                 "recall_20_R": float(np.mean(r_r20))}
            scene_results[scene] = m
            print(f"  {scene:<15} {m['W_to_Wf']:>8.3f} {m['R_to_Wf']:>8.3f} {m['delta_R2']:>8.4f} {m['recall_20_W']:>14.3f} {m['recall_20_R']:>14.3f}")

    with open(OUT_DIR / "scene_comparison.json", 'w') as f:
        json.dump(scene_results, f, indent=2)

    # ============ SEED REPRODUCTION ============
    print("\n" + "=" * 100)
    print("SEED REPRODUCTION (Δ=50)")
    print("=" * 100)
    seed_results = {}
    for pair in [("room", "room_seed123"), ("garden", "garden_seed123")]:
        s1, s2 = pair
        print(f"\n  {s1} vs {s2}:")
        for scene in pair:
            if scene not in all_data:
                continue
            data = all_data[scene]
            w_p, r_p, dR2 = [], [], []
            for cp in CHECKPOINTS:
                p = f"cp{cp}"
                dp = f"{p}_d50"
                if f"{dp}_alive" not in data:
                    continue
                alive = data[f"{dp}_alive"]
                wf = data[f"{dp}_tiles_mean"].astype(np.float64)
                wc = data[f"{p}_s_current_tiles_mean"].astype(np.float64)
                rc = data[f"{p}_s_screen_radius_mean"].astype(np.float64)
                mask = alive & (wc > 0)
                if mask.sum() < 100:
                    continue
                w_p.append(safe_pearson(wc[mask], wf[mask]))
                r_p.append(safe_pearson(rc[mask], wf[mask]))
                r2_m1 = ols_r2(wc[mask], wf[mask])
                r2_m3 = ols_r2(np.column_stack([wc[mask], rc[mask]]), wf[mask])
                dR2.append(r2_m3 - r2_m1)
            if w_p:
                m = {"W_to_Wf": float(np.mean(w_p)), "R_to_Wf": float(np.mean(r_p)), "delta_R2": float(np.mean(dR2))}
                seed_results[scene] = m
                print(f"    {scene}: W→Wf={m['W_to_Wf']:.3f}, R→Wf={m['R_to_Wf']:.3f}, ΔR²={m['delta_R2']:.4f}")

    with open(OUT_DIR / "seed_reproduction.json", 'w') as f:
        json.dump(seed_results, f, indent=2)

    # ============ FINAL COMPARISON ============
    print("\n" + "=" * 100)
    print("FINAL COMPARISON")
    print("=" * 100)

    # Compute final summary metrics
    all_w_p, all_r_p, all_dR2 = [], [], []
    for scene in primary:
        if scene in scene_results:
            all_w_p.append(scene_results[scene]["W_to_Wf"])
            all_r_p.append(scene_results[scene]["R_to_Wf"])
            all_dR2.append(scene_results[scene]["delta_R2"])

    w_mean = float(np.mean(all_w_p)) if all_w_p else 0
    r_mean = float(np.mean(all_r_p)) if all_r_p else 0
    dR2_mean = float(np.mean(all_dR2)) if all_dR2 else 0
    w_std = float(np.std(all_w_p)) if all_w_p else 0
    r_std = float(np.std(all_r_p)) if all_r_p else 0

    print(f"\n  W_current → W_future:  {w_mean:.3f} (std={w_std:.3f})")
    print(f"  R_current → W_future:  {r_mean:.3f} (std={r_std:.3f})")
    print(f"  ΔR² (M3-M1):           {dR2_mean:.4f}")

    # Decision
    if w_mean > 0.7 and dR2_mean < 0.02:
        decision = "MODIFY — workload persistence is high, screen radius adds little. Research workload-persistence mechanism."
    elif w_mean > 0.7 and dR2_mean > 0.05:
        decision = "KEEP — workload persistent AND screen radius adds significant incremental information."
    elif w_mean < 0.3 and r_mean < 0.3:
        decision = "DROP — neither workload persistence nor footprint predicts future work. Prediction-based selective computation should stop."
    elif r_mean > w_mean and dR2_mean > 0.02:
        decision = "KEEP — screen radius predicts future work better than current workload."
    else:
        decision = "MODIFY — moderate persistence, define exact utility domain before algorithm design."

    print(f"\n  DECISION: {decision}")

    final = {
        "table1_persistence": {s: scene_results.get(s, {}) for s in primary},
        "workload_persistence_mean": w_mean,
        "workload_persistence_std": w_std,
        "footprint_prediction_mean": r_mean,
        "footprint_prediction_std": r_std,
        "delta_R2_mean": dR2_mean,
        "ranking": ranking_results,
        "horizon": horizon_results,
        "age": age_results,
        "seed": seed_results,
        "decision": decision,
    }
    with open(OUT_DIR / "final_comparison.json", 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'final_comparison.json'}")


if __name__ == "__main__":
    main()
