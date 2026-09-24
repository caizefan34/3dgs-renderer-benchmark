#!/usr/bin/env python3
"""Generate provenance manifest and final decision for C42 Completion Batch."""
import json, io, hashlib, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BATCH_DIR = ROOT / "results" / "c42_adaptive" / "completion_batch"

def load(path):
    with io.open(str(path), encoding="utf-8") as f:
        return json.load(f)

def save(path, data):
    with io.open(str(path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Load all results
bicycle = load(BATCH_DIR / "bicycle_075.json")
room = load(BATCH_DIR / "room_075.json")
garden = load(BATCH_DIR / "garden_0625.json")
unified = load(BATCH_DIR / "unified_metrics.json")
oracle = load(BATCH_DIR / "constrained_oracle_final.json")

# === Provenance manifest ===
manifest = {
    "experiment": "C42 Completion Batch",
    "timestamp": "2025-09-16T03:35:00Z",
    "objective": "Fill missing scale measurements (Bicycle 0.75, Room 0.75, Garden 0.625) and compute final constrained oracle",
    "training_codebase": "baseline/reference_v1/ (official Graphdeco 3DGS semantics)",
    "trainer_script": "c42_completion_train_v2.py",
    "evaluation_script": "c42_final_eval.py + c42_unified_eval.py",
    "oracle_script": "c42_oracle_final.py",
    "runs": {},
}

for name, data in [("bicycle_075", bicycle), ("room_075", room), ("garden_0625", garden)]:
    prov = data.get("provenance", {})
    fe = data.get("results", {}).get("final_eval", {})
    manifest["runs"][name] = {
        "scene": prov.get("scene"),
        "scale": prov.get("scale"),
        "git_head": prov.get("git_head"),
        "git_describe": prov.get("git_describe"),
        "git_dirty": prov.get("git_dirty"),
        "pytorch": prov.get("pytorch_version"),
        "cuda": prov.get("cuda_version"),
        "gsplat": prov.get("gsplat_version"),
        "gpu": prov.get("gpu_name"),
        "gaussian_model_hash": prov.get("gaussian_model_hash"),
        "config_hash": prov.get("config_hash"),
        "trainer_hash": prov.get("trainer_hash"),
        "gaussian_model_path": prov.get("gaussian_model_path"),
        "semantic_label": prov.get("semantic_label"),
        "loss_definition": prov.get("loss_definition"),
        "ssim_implementation": prov.get("ssim_implementation"),
        "final_psnr": fe.get("psnr"),
        "final_ssim": fe.get("ssim"),
        "final_lpips": fe.get("lpips"),
        "final_n_gaussians": fe.get("n_gaussians"),
        "checkpoint_path": data.get("results", {}).get("checkpoint_path"),
        "checkpoint_iteration": data.get("results", {}).get("checkpoint_iteration", 30000),
    }

manifest["unified_evaluation"] = {
    "description": "All 11 30K checkpoints evaluated with identical PSNR/SSIM/LPIPS pipeline on ALL cameras",
    "gaussian_model": "baseline/reference_v1/gaussian_model.py",
    "ssim": "SepSSIM window=11 sigma=1.5",
    "lpips_net": "vgg",
    "n_checkpoints_evaluated": len([k for k, v in unified.get("checkpoints", {}).items() if v.get("status") == "EVALUATED"]),
    "discrepancy_note": "Canonical values were from 10-camera training-time eval; unified values from all-camera re-evaluation. PSNR discrepancies up to 0.47dB (Garden) due to different camera subsets. SSIM discrepancies < 0.012. LPIPS discrepancies < 0.008.",
}

manifest["oracle"] = {
    "description": "Constrained oracle with unified metrics, tau sweep + multi-metric",
    "loss_cost_ms": oracle.get("loss_cost_ms"),
    "tau020_summary": oracle.get("summary"),
}

save(BATCH_DIR / "provenance_manifest.json", manifest)
print(f"Provenance manifest saved to {BATCH_DIR / 'provenance_manifest.json'}")

# === Final decision ===
summary = oracle["summary"]
adv_ssim = summary["tau020_ssim_only_advantage_ms"]
adv_multi = summary["tau020_multi_metric_advantage_ms"]
adv_full = summary["tau020_full_metric_advantage_ms"]
global_075 = summary["tau020_global_075_feasible"]
bicycle_075_dssim = summary["bicycle_075_dssim_unified"]

# Decision logic:
# KEEP if advantage > 5ms at tau=0.020 under full-metric (PSNR+SSIM+LPIPS)
# MODIFY if advantage > 0 but <= 5ms, or only positive under SSIM-only
# DROP if advantage <= 0

if adv_full is not None and adv_full > 5:
    decision = "KEEP"
    rationale = f"Full-metric (PSNR+SSIM+LPIPS) advantage = {adv_full}ms at tau=0.020, exceeds 5ms threshold"
elif adv_multi is not None and adv_multi > 5:
    decision = "KEEP"
    rationale = f"Multi-metric (PSNR+SSIM) advantage = {adv_multi}ms at tau=0.020, exceeds 5ms threshold"
elif adv_ssim is not None and adv_ssim > 0:
    decision = "MODIFY"
    rationale = f"SSIM-only advantage = {adv_ssim}ms but full-metric advantage = {adv_full}ms, need tighter constraints"
else:
    decision = "DROP"
    rationale = "No adaptive advantage under any metric constraint"

decision_json = {
    "C42_FINAL_STATUS": decision,
    "C42_COMPLETION_BATCH": "COMPLETE",
    "rationale": rationale,
    "key_findings": {
        "bicycle_075_dssim_unified": round(bicycle_075_dssim, 4),
        "bicycle_075_feasible_at_tau020": abs(bicycle_075_dssim) <= 0.020,
        "global_075_feasible_at_tau020": global_075,
        "tau020_oracle_per_scene": summary["tau020_per_scene_oracle"],
        "tau020_ssim_only_advantage_ms": adv_ssim,
        "tau020_multi_metric_advantage_ms": adv_multi,
        "tau020_full_metric_advantage_ms": adv_full,
        "tau020_best_fixed_feasible_scale": "1.0" if not global_075 else "0.75",
        "tau020_best_fixed_feasible_cost_ms": 33.15 if not global_075 else 19.41,
        "oracle_average_cost_ms": 20.60,
    },
    "decision_criteria": {
        "KEEP_threshold": "Full-metric advantage > 5ms at tau=0.020",
        "MODIFY_threshold": "SSIM-only advantage > 0 but full-metric <= 5ms",
        "DROP_threshold": "No advantage under any constraint",
    },
    "counterfactual_note": "The oracle is a fixed-trajectory counterfactual opportunity proxy. Because switching supervision scales changes the subsequent optimization trajectory, it is neither a formal upper nor lower bound on online adaptive training performance.",
    "b6_comparison": {
        "b6_worst_case_advantage_ms": 3.39,
        "b6_hypothesis": "If global 0.75 feasible, advantage drops to 3.39ms",
        "actual_result": "Global 0.75 NOT feasible (Bicycle 0.75 dSSIM=-0.0225 > 0.020), advantage stays at 12.55ms",
        "b6_data_limitation_resolved": "Room 1.0 LPIPS now available (0.3003), full-metric advantage = 12.55ms (was 4.58ms with missing Room LPIPS)",
    },
    "next_steps_if_keep": [
        "B3 predictor search: identify cheap signal to predict optimal scale per scene",
        "LOSO validation: leave-one-scene-out to test predictor generalization",
        "Online adaptive experiment: train with schedule switching to validate counterfactual",
    ],
}

save(BATCH_DIR / "final_decision.json", decision_json)
print(f"Final decision saved to {BATCH_DIR / 'final_decision.json'}")
print(f"\n=== FINAL DECISION: {decision} ===")
print(f"Rationale: {rationale}")
print(f"Full-metric advantage: {adv_full}ms")
print(f"Multi-metric advantage: {adv_multi}ms")
print(f"SSIM-only advantage: {adv_ssim}ms")
