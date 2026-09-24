#!/usr/bin/env python3
"""Merge subagent's training trajectory data with supplementary B0 evidence."""
import json, io, copy

ROOT = r"C:\Users\36570\3dgs-renderer-benchmark"
RESULTS = ROOT + r"\results\c42_adaptive"

def load(path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)

def save(path, data):
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Load subagent's version
evidence = load(RESULTS + r"\existing_evidence.json")

# === Add supplementary data ===

# 1. Room 30K final values (from S2.2 report - not in subagent data)
evidence["room_30k_summary"] = {
    "1.0": {"psnr": 32.30, "ssim": 0.9263, "N_gaussians": 952353, "source": "S2.2 report Table 3.1"},
    "0.5": {"psnr": 32.54, "ssim": 0.9185, "N_gaussians": 745566, "source": "S2.2 report Table 3.1"},
    "delta_0.5": {"delta_psnr": 0.24, "delta_ssim": -0.0078, "note": "Scale 0.5 is strictly better (higher PSNR, fewer Gaussians, 1.90x throughput)"},
    "missing_scales": ["0.75", "0.625"],
}

# 2. Fixed-tensor loss benchmark
ftb_path = ROOT + r"\results\reference_v1\s23\fixed_tensor_loss_benchmark.json"
try:
    ftb = load(ftb_path)
    evidence["fixed_tensor_loss_benchmark"] = {
        "source": ftb_path,
        "protocol": ftb.get("protocol", "50 warmup, 300 timed, CUDA events, explicit sync, same frozen tensors, 1080p images"),
        "scales": ftb.get("scales", {}),
    }
except:
    evidence["fixed_tensor_loss_benchmark"] = {"note": "Could not load"}

# 3. Gradient frequency probe (S2.3)
gfp_s23_path = ROOT + r"\results\reference_v1\s23\mechanism\gradient_frequency_probe.json"
try:
    gfp_s23 = load(gfp_s23_path)
    evidence["gradient_frequency_probe_s23"] = gfp_s23
except:
    evidence["gradient_frequency_probe_s23"] = {"note": "Could not load"}

# 4. Gradient frequency probe temporal (S2.2 - Bicycle 5K vs 15K)
gfp_s22_path = ROOT + r"\results\reference_v1\s22\mechanism\gradient_frequency_probe.json"
try:
    gfp_s22 = load(gfp_s22_path)
    evidence["gradient_frequency_probe_s22_temporal"] = gfp_s22
except:
    evidence["gradient_frequency_probe_s22_temporal"] = {"note": "Could not load"}

# 5. Pareto summary
pareto_path = ROOT + r"\results\reference_v1\s23\pareto_summary.json"
try:
    pareto = load(pareto_path)
    evidence["pareto_summary"] = pareto
except:
    evidence["pareto_summary"] = {"note": "Could not load"}

# 6. Population frontier
pop_path = ROOT + r"\results\reference_v1\s23\population_frontier.json"
try:
    pop = load(pop_path)
    evidence["population_frontier"] = pop
except:
    evidence["population_frontier"] = {"note": "Could not load"}

# 7. Cross-scene summary
cs_path = ROOT + r"\results\reference_v1\s22\cross_scene_summary.json"
try:
    cs = load(cs_path)
    evidence["cross_scene_summary"] = cs
except:
    evidence["cross_scene_summary"] = {"note": "Could not load"}

# Update metadata
evidence["_metadata"]["supplementary_sections_added"] = [
    "room_30k_summary - Room's 30K final values from S2.2 report",
    "fixed_tensor_loss_benchmark - SepSSIM fwd+bwd timing per scale",
    "gradient_frequency_probe_s23 - Garden/Bicycle 15K gradient frequency",
    "gradient_frequency_probe_s22_temporal - Bicycle 5K vs 15K temporal evolution",
    "pareto_summary - Complete Pareto frontier summary",
    "population_frontier - Gaussian population trends",
    "cross_scene_summary - Cross-scene quality deltas",
]
evidence["_metadata"]["total_sections"] = 10

save(RESULTS + r"\existing_evidence.json", evidence)
print("Merged existing_evidence.json saved")
print(f"Top-level keys: {list(evidence.keys())}")
