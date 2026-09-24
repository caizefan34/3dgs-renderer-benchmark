#!/usr/bin/env python3
"""Create final_comparison.json and multiscene_room.json from existing results."""
import json, shutil
from pathlib import Path

d = Path("results/a100/phase-c51")
c50 = Path("results/a100/phase-c50")

# Create multiscene_room.json from C50 data
c50_pred = json.load(open(c50 / "predictor_comparison.json"))
c50_topk = json.load(open(c50 / "topk_persistence.json"))

room_data = {
    "scene": "room",
    "source": "C50 results (phase-c50/predictor_comparison.json + topk_persistence.json)",
    "phase_statistics": {},
    "final_psnr": 26.23,
    "final_gaussians": 2034835,
}

for phase in ("early", "middle", "late"):
    pred = c50_pred["phase_statistics"].get(phase, {})
    topk = c50_topk["phase_statistics"].get(phase, {})
    room_data["phase_statistics"][phase] = {
        "count": pred.get("count", 0),
        "recall_32": topk.get("recall_32pct", {}),
        "coverage_32": pred.get("previous_coverage_32pct", {}),
        "coverage_50": pred.get("previous_coverage_50pct", {}),
    }

with open(d / "multiscene_room.json", "w") as f:
    json.dump(room_data, f, indent=2)
print("Created multiscene_room.json")

# Create final_comparison.json
configs = []
# Baseline
c50_summary = json.load(open(c50 / "summary.json"))
configs.append({
    "name": "baseline",
    "keep_fraction": 1.0,
    "mode": "dense",
    "version": "N/A",
    "psnr": c50_summary["eval_points"][-1]["psnr"],
    "ssim": None,
    "gaussians": c50_summary["eval_points"][-1]["gaussians"],
    "clone": c50_summary["total_clone"],
    "split": c50_summary["total_split"],
    "prune": c50_summary["total_prune"],
    "time_s": c50_summary["total_train_time_s"],
    "xyz_cosine": None,
    "psnr_diff": 0.0,
})

# V1 (broken)
for name, kf in [("k50", 0.5), ("k32", 0.32), ("k20", 0.20)]:
    f = d / f"simulation_{name}.json"
    if f.exists():
        data = json.load(open(f))
        last_eval = data["eval_points"][-1]
        last_grad = data["grad_correctness"][-1] if data["grad_correctness"] else {}
        xyz_cos = last_grad.get("correctness", {}).get("xyz", {}).get("cosine")
        configs.append({
            "name": name,
            "keep_fraction": kf,
            "mode": "mode_a",
            "version": "v1 (mask before densification - BROKEN)",
            "psnr": last_eval["psnr"],
            "ssim": last_eval["ssim"],
            "gaussians": last_eval["gaussians"],
            "clone": data["total_clone"],
            "split": data["total_split"],
            "prune": data["total_prune"],
            "time_s": data["total_train_time_s"],
            "xyz_cosine": xyz_cos,
            "psnr_diff": last_eval["psnr"] - configs[0]["psnr"],
        })

# V2 (fixed)
for name, kf in [("k50", 0.5), ("k32", 0.32), ("k20", 0.20)]:
    f = d / f"simulation_{name}_v2.json"
    if f.exists():
        data = json.load(open(f))
        last_eval = data["eval_points"][-1]
        last_grad = data["grad_correctness"][-1] if data["grad_correctness"] else {}
        xyz_cos = last_grad.get("correctness", {}).get("xyz", {}).get("cosine")
        configs.append({
            "name": f"{name}_v2",
            "keep_fraction": kf,
            "mode": "mode_a",
            "version": "v2 (densification decoupled - FIXED)",
            "psnr": last_eval["psnr"],
            "ssim": last_eval["ssim"],
            "gaussians": last_eval["gaussians"],
            "clone": data["total_clone"],
            "split": data["total_split"],
            "prune": data["total_prune"],
            "time_s": data["total_train_time_s"],
            "xyz_cosine": xyz_cos,
            "psnr_diff": last_eval["psnr"] - configs[0]["psnr"],
        })

comparison = {
    "stage": "Stage 2: Simulated Sparse Backward — Final Comparison",
    "baseline_psnr": configs[0]["psnr"],
    "configs": configs,
    "acceptance_gates": {
        "gradient_cosine_threshold": 0.99,
        "psnr_degradation_threshold": 0.2,
        "ssim_degradation_threshold": 0.005,
        "net_speedup_threshold": 0.05,
    },
    "decision": "MODIFY",
    "decision_reason": "Gradient cosine passes (all >0.99) but PSNR degradation exceeds 0.2 dB threshold (0.55-0.95 dB). Densification decoupling fix (v1→v2) improved PSNR by 1.4-2.1 dB but compounding prediction error remains. Multi-scene: garden fails coverage criteria.",
    "next_steps": [
        "Test periodic full backward (every 200 iters) to reset compounding error",
        "Test K=80% for lower degradation",
        "Test sparse backward only after densification ends (for 30K training)",
        "Proceed to CUDA Stage 4 only if PSNR degradation < 0.2 dB achieved",
    ],
}

with open(d / "final_comparison.json", "w") as f:
    json.dump(comparison, f, indent=2)
print("Created final_comparison.json")

# Print summary
print(f"\n{'='*80}")
print("FINAL COMPARISON")
print(f"{'='*80}")
print(f"{'Config':<25} {'PSNR':>7} {'diff':>7} {'Clone':>8} {'xyz_cos':>8} {'Ver':>10}")
print("-" * 80)
for c in configs:
    cos = f"{c['xyz_cosine']:.4f}" if c['xyz_cosine'] else "N/A"
    print(f"{c['name']:<25} {c['psnr']:>7.2f} {c['psnr_diff']:>+7.2f} {c['clone']:>8,} {cos:>8} {c['version'][:10]:>10}")
print(f"\nDecision: {comparison['decision']}")
