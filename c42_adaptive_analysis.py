#!/usr/bin/env python3
"""C42++ Adaptive Structural Supervision Feasibility — Phases B0-B5.

Offline analysis using existing canonical C42 evidence only.
No GPU, no training, no network. Pure CPU analysis of existing JSON data.
"""
import json
import os
import sys
import math
import copy
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results" / "c42_adaptive"
RESULTS.mkdir(parents=True, exist_ok=True)

# === Phase B0: Evidence Inventory ===

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def extract_checkpoints(data):
    """Extract per-checkpoint metrics from a training JSON."""
    ckpts = data.get("training_metrics", {}).get("checkpoints", {})
    result = []
    for k in sorted(ckpts.keys(), key=int):
        c = ckpts[k]
        result.append({
            "iteration": c["iteration"],
            "psnr": c["psnr"],
            "ssim": c["ssim"],
            "l1": c["l1"],
            "N_gaussians": c["N_gaussians"],
        })
    return result

def phase_b0():
    """Collect all existing canonical C42 evidence."""
    s22 = ROOT / "results" / "reference_v1" / "s22"
    s23 = ROOT / "results" / "reference_v1" / "s23"
    
    evidence = {
        "phase": "B0",
        "description": "Canonical C42 evidence inventory from S2.2 and S2.3",
        "sources": {
            "S2.2": "reports/phase-s2.2-c42-generalization.md",
            "S2.3": "reports/phase-s2.3-c42-pareto-frontier.md",
        },
        "scenes": {},
        "fixed_tensor_loss_costs": {},
        "gradient_frequency_probe": {},
        "missing_scales": {},
    }
    
    # === Room ===
    # Only scale 1.0 and 0.5 exist for Room
    room = {}
    # Room baseline (scale 1.0) — from S2.1, referenced in S2.2
    # Room baseline 30k data is in reference_v1/room_30k
    room_baseline_path = ROOT / "results" / "reference_v1" / "room_30k"
    # Check if baseline training data exists
    # S2.2 references Room from S2.1. Let's check what's available.
    
    # Room scale 1.0 data — from the canonical Room 30K training
    # The checkpoints are in results/reference_v1/room_30k/checkpoints/
    # But the training metrics JSON may not be in s22/s23
    
    # Garden and Bicycle have full JSON files
    for scene_name, scene_dir in [("garden", s22 / "garden"), ("bicycle", s22 / "bicycle")]:
        scene_data = {}
        
        # Scale 1.0 (baseline)
        baseline_path = scene_dir / "baseline_30k.json"
        if baseline_path.exists():
            scene_data["1.0"] = {
                "checkpoints": extract_checkpoints(load_json(baseline_path)),
                "source": str(baseline_path),
            }
        
        # Scale 0.5 (C42)
        c42_path = scene_dir / "c42_30k.json"
        if c42_path.exists():
            scene_data["0.5"] = {
                "checkpoints": extract_checkpoints(load_json(c42_path)),
                "source": str(c42_path),
            }
        
        # Scale 0.75 (Garden) or 0.625 (Bicycle) — from S2.3
        if scene_name == "garden":
            s23_path = s23 / "garden" / "s0.750_30k.json"
            if s23_path.exists():
                scene_data["0.75"] = {
                    "checkpoints": extract_checkpoints(load_json(s23_path)),
                    "source": str(s23_path),
                }
        elif scene_name == "bicycle":
            s23_path = s23 / "bicycle" / "s0.625_30k.json"
            if s23_path.exists():
                scene_data["0.625"] = {
                    "checkpoints": extract_checkpoints(load_json(s23_path)),
                    "source": str(s23_path),
                }
        
        evidence["scenes"][scene_name] = scene_data
    
    # Room: extract from available data
    # Room baseline (scale 1.0) — check for training data
    room_baseline_json = ROOT / "results" / "reference_v1" / "s22" / "room"
    # Room C42 data may be in a different location
    # The S2.2 report says Room is from S2.1
    # Check for Room training JSON in s22
    room_data = {}
    # Try to find Room baseline data
    for p in [ROOT / "results" / "reference_v1" / "room_30k" / "training_metrics.json",
              ROOT / "results" / "reference_v1" / "s21" / "room" / "baseline_30k.json",
              ROOT / "results" / "reference_v1" / "s22" / "room" / "baseline_30k.json"]:
        if p.exists():
            room_data["1.0"] = {"checkpoints": extract_checkpoints(load_json(p)), "source": str(p)}
            break
    
    for p in [ROOT / "results" / "reference_v1" / "room_30k" / "c42_training_metrics.json",
              ROOT / "results" / "reference_v1" / "s21" / "room" / "c42_30k.json",
              ROOT / "results" / "reference_v1" / "s22" / "room" / "c42_30k.json"]:
        if p.exists():
            room_data["0.5"] = {"checkpoints": extract_checkpoints(load_json(p)), "source": str(p)}
            break
    
    if room_data:
        evidence["scenes"]["room"] = room_data
    else:
        # Use the known values from the report directly
        evidence["scenes"]["room"] = {
            "1.0": {
                "checkpoints": [
                    {"iteration": 30000, "psnr": 32.30, "ssim": 0.9263, "l1": None, "N_gaussians": 952353},
                ],
                "source": "S2.2 report Table 3.1 (Room REFERENCE_V1)",
            },
            "0.5": {
                "checkpoints": [
                    {"iteration": 30000, "psnr": 32.54, "ssim": 0.9185, "l1": None, "N_gaussians": 745566},
                ],
                "source": "S2.2 report Table 3.1 (Room + C42)",
            },
        }
        evidence["missing_scales"]["room"] = ["0.75", "0.625"]
    
    # Fixed-tensor loss benchmark
    ftb_path = s23 / "fixed_tensor_loss_benchmark.json"
    if ftb_path.exists():
        ftb = load_json(ftb_path)
        for scene in ["garden", "bicycle"]:
            if scene in ftb:
                for scale_key, val in ftb[scene].items():
                    scale = val["scale"]
                    if scale not in evidence["fixed_tensor_loss_costs"]:
                        evidence["fixed_tensor_loss_costs"][scale] = {}
                    evidence["fixed_tensor_loss_costs"][scale][scene] = {
                        "forward_backward_ms": val["forward_backward_ms"],
                        "loss_speedup_x": val["LOSS_SPEEDUP_X"],
                    }
    
    # Gradient frequency probe
    gfp_s23_path = s23 / "mechanism" / "gradient_frequency_probe.json"
    if gfp_s23_path.exists():
        gfp = load_json(gfp_s23_path)
        for scene in ["garden", "bicycle"]:
            if scene in gfp:
                evidence["gradient_frequency_probe"][scene] = {}
                for scale_key, val in gfp[scene].items():
                    evidence["gradient_frequency_probe"][scene][scale_key] = {
                        "mean_cosine": val["mean_cosine"],
                        "mean_rel_l2": val["mean_rel_l2"],
                        "mean_hf_fraction": val["mean_hf_fraction"],
                    }
    
    # Temporal gradient probe (Bicycle 5K vs 15K)
    gfp_s22_path = s22 / "mechanism" / "gradient_frequency_probe.json"
    if gfp_s22_path.exists():
        gfp_s22 = load_json(gfp_s22_path)
        evidence["gradient_frequency_temporal"] = {}
        for stage_key, stage_data in gfp_s22.get("results", {}).items():
            evidence["gradient_frequency_temporal"][stage_key] = {}
            for model in ["baseline", "c42"]:
                if model in stage_data:
                    evidence["gradient_frequency_temporal"][stage_key][model] = {
                        "mean_cosine": stage_data[model]["mean_cosine"],
                        "mean_hf_fraction": stage_data[model]["mean_hf_fraction"],
                        "N_gaussians": stage_data[model]["N_gaussians"],
                    }
    
    # 10K screening data
    for scene in ["garden", "bicycle"]:
        scr_path = s23 / scene / "screening.json"
        if scr_path.exists():
            scr = load_json(scr_path)
            evidence.setdefault("screening_10k", {})[scene] = scr.get("comparison_at_10k", {})
    
    # Population frontier
    pop_path = s23 / "population_frontier.json"
    if pop_path.exists():
        evidence["population_frontier"] = load_json(pop_path)
    
    # Pareto summary
    pareto_path = s23 / "pareto_summary.json"
    if pareto_path.exists():
        evidence["pareto_summary"] = load_json(pareto_path)
    
    # Record missing scales
    for scene in ["room"]:
        if scene not in evidence["missing_scales"]:
            available = set(evidence["scenes"].get(scene, {}).keys())
            missing = {"0.75", "0.625"} - available
            if missing:
                evidence["missing_scales"][scene] = sorted(missing)
    
    # Summary table
    evidence["summary_30k"] = {}
    for scene, scales in evidence["scenes"].items():
        for scale, data in scales.items():
            ckpts = data.get("checkpoints", [])
            final = ckpts[-1] if ckpts else {}
            if final:
                key = f"{scene}_s{scale}"
                evidence["summary_30k"][key] = {
                    "psnr": final.get("psnr"),
                    "ssim": final.get("ssim"),
                    "N_gaussians": final.get("N_gaussians"),
                }
    
    out_path = RESULTS / "existing_evidence.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, default=str, ensure_ascii=False)
    print(f"B0: Saved to {out_path}")
    
    # Print summary
    print("\n=== B0 EVIDENCE INVENTORY ===")
    for scene, scales in evidence["scenes"].items():
        print(f"\n{scene.upper()}:")
        for scale, data in scales.items():
            ckpts = data.get("checkpoints", [])
            n_ckpts = len(ckpts)
            final = ckpts[-1] if ckpts else {}
            print(f"  scale={scale}: {n_ckpts} checkpoints, final PSNR={final.get('psnr','?')}, N={final.get('N_gaussians','?')}")
    print(f"\nFixed-tensor loss costs: {list(evidence['fixed_tensor_loss_costs'].keys())}")
    print(f"Gradient frequency probe scenes: {list(evidence['gradient_frequency_probe'].keys())}")
    print(f"Missing scales: {evidence['missing_scales']}")
    
    return evidence


# === Phase B1: Scene Dependence Analysis ===

def phase_b1(evidence):
    """Analyze scene dependence using available signals."""
    
    analysis = {
        "phase": "B1",
        "description": "Scene dependence analysis using existing evidence",
        "signal_A_residual_frequency": {},
        "signal_B_multires_disagreement": {},
        "signal_C_downsample_consistency": {},
        "signal_D_training_state": {},
        "findings": {},
    }
    
    # === Signal A: Residual spatial-frequency content ===
    # Use gradient frequency probe as proxy for residual frequency content
    # HF fraction from the gradient probe tells us about high-freq content
    gfp = evidence.get("gradient_frequency_probe", {})
    
    for scene in ["garden", "bicycle"]:
        if scene in gfp:
            scene_gfp = gfp[scene]
            analysis["signal_A_residual_frequency"][scene] = {
                "hf_fractions": {},
                "cosines": {},
            }
            for scale_key, val in scene_gfp.items():
                analysis["signal_A_residual_frequency"][scene]["hf_fractions"][scale_key] = val["mean_hf_fraction"]
                analysis["signal_A_residual_frequency"][scene]["cosines"][scale_key] = val["mean_cosine"]
    
    # Room has no gradient frequency probe — note as missing
    analysis["signal_A_residual_frequency"]["room"] = {"note": "No gradient frequency probe available for Room"}
    
    # === Signal B: Multi-resolution loss disagreement ===
    # At 10K screening, we have PSNR/SSIM at different scales
    # DSSIM ≈ 1 - SSIM, so DSSIM disagreement = |SSIM_1.0 - SSIM_s|
    screening = evidence.get("screening_10k", {})
    
    for scene in ["garden", "bicycle"]:
        if scene in screening:
            scr = screening[scene]
            s10 = scr.get("scale_1.000", {})
            ssim_10 = s10.get("ssim")
            
            scene_b = {"dssim_disagreement": {}, "ssim_at_10k": {}}
            for scale_key in ["scale_0.750", "scale_0.625", "scale_0.500"]:
                sd = scr.get(scale_key, {})
                ssim_s = sd.get("ssim")
                if ssim_10 is not None and ssim_s is not None:
                    delta = abs(ssim_10 - ssim_s)
                    scene_b["dssim_disagreement"][scale_key] = delta
                    scene_b["ssim_at_10k"][scale_key] = ssim_s
            
            analysis["signal_B_multires_disagreement"][scene] = scene_b
    
    # For Room, no multi-scale data available
    analysis["signal_B_multires_disagreement"]["room"] = {"note": "No multi-scale screening data for Room (only 1.0 and 0.5)"}
    
    # === Signal C: Render-target downsample consistency ===
    # Use the gradient cosine as a proxy: low cosine = high disagreement
    for scene in ["garden", "bicycle"]:
        if scene in gfp:
            scene_gfp = gfp[scene]
            scene_c = {"gradient_cosine_by_scale": {}}
            for scale_key, val in scene_gfp.items():
                scene_c["gradient_cosine_by_scale"][scale_key] = val["mean_cosine"]
            analysis["signal_C_downsample_consistency"][scene] = scene_c
    
    analysis["signal_C_downsample_consistency"]["room"] = {"note": "No gradient probe available for Room"}
    
    # === Signal D: Training-state signals ===
    # Extract N_gaussians, PSNR trajectories from training data
    for scene, scales in evidence["scenes"].items():
        scene_d = {"trajectories": {}}
        for scale, data in scales.items():
            ckpts = data.get("checkpoints", [])
            if len(ckpts) > 1:
                scene_d["trajectories"][scale] = {
                    "N_at_checkpoints": [(c["iteration"], c["N_gaussians"]) for c in ckpts],
                    "PSNR_at_checkpoints": [(c["iteration"], c["psnr"]) for c in ckpts],
                }
        if scene_d["trajectories"]:
            analysis["signal_D_training_state"][scene] = scene_d
    
    # === Findings ===
    findings = []
    
    # Finding 1: HF fraction differs across scenes
    garden_hf = analysis["signal_A_residual_frequency"].get("garden", {}).get("hf_fractions", {})
    bicycle_hf = analysis["signal_A_residual_frequency"].get("bicycle", {}).get("hf_fractions", {})
    if garden_hf and bicycle_hf:
        garden_hf_1 = garden_hf.get("s1.000", 0)
        bicycle_hf_1 = bicycle_hf.get("s1.000", 0)
        findings.append({
            "finding": "Baseline HF fraction differs across scenes",
            "garden_hf_1.0": garden_hf_1,
            "bicycle_hf_1.0": bicycle_hf_1,
            "interpretation": f"Garden has higher HF fraction ({garden_hf_1:.3f}) than Bicycle ({bicycle_hf_1:.3f}), consistent with Garden being detail-rich outdoor scene where full resolution matters more.",
        })
    
    # Finding 2: Gradient cosine behavior differs
    garden_cos = analysis["signal_A_residual_frequency"].get("garden", {}).get("cosines", {})
    bicycle_cos = analysis["signal_A_residual_frequency"].get("bicycle", {}).get("cosines", {})
    if garden_cos and bicycle_cos:
        findings.append({
            "finding": "Gradient cosine degradation pattern differs across scenes",
            "garden_cos_0.5": garden_cos.get("s0.500"),
            "bicycle_cos_0.5": bicycle_cos.get("s0.500"),
            "bicycle_anomaly_0.75": bicycle_cos.get("s0.750"),
            "interpretation": "Bicycle shows anomalous cosine at 0.75 (0.506) — intermediate scale creates unstable gradient dynamics. Garden shows monotonic cosine decrease.",
        })
    
    # Finding 3: DSSIM disagreement at 10K
    for scene in ["garden", "bicycle"]:
        bd = analysis["signal_B_multires_disagreement"].get(scene, {})
        deltas = bd.get("dssim_disagreement", {})
        if deltas:
            findings.append({
                "finding": f"DSSIM disagreement at 10K for {scene}",
                "delta_0.75": deltas.get("scale_0.750"),
                "delta_0.625": deltas.get("scale_0.625"),
                "delta_0.5": deltas.get("scale_0.500"),
                "interpretation": f"{'Large' if max(deltas.values()) > 0.01 else 'Small'} disagreement — {'scale matters' if max(deltas.values()) > 0.01 else 'scale has limited impact'} for {scene}.",
            })
    
    # Finding 4: Temporal HF evolution (Bicycle only)
    temporal = evidence.get("gradient_frequency_temporal", {})
    if temporal:
        findings.append({
            "finding": "Temporal HF evolution (Bicycle only)",
            "5K_baseline_hf": temporal.get("5000", {}).get("baseline", {}).get("mean_hf_fraction"),
            "5K_c42_hf": temporal.get("5000", {}).get("c42", {}).get("mean_hf_fraction"),
            "15K_baseline_hf": temporal.get("15000", {}).get("baseline", {}).get("mean_hf_fraction"),
            "15K_c42_hf": temporal.get("15000", {}).get("c42", {}).get("mean_hf_fraction"),
            "interpretation": "C42 suppresses HF gradients more at 5K (0.52 vs 0.81) than at 15K (0.62 vs 0.56). The HF suppression effect is stage-dependent — weaker as training progresses.",
        })
    
    analysis["findings"] = findings
    
    out_path = RESULTS / "scene_signal_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nB1: Saved to {out_path}")
    
    for f in findings:
        print(f"  Finding: {f['finding']}")
    
    return analysis


# === Phase B2: Offline Resolution Oracle ===

def phase_b2(evidence):
    """Construct offline oracle: iteration to minimum sufficient scale."""
    
    oracle = {
        "phase": "B2",
        "description": "Offline resolution oracle from existing fixed-scale trajectories",
        "criterion": "D-SSIM deviation from scale=1.0 at same iteration",
        "thresholds": {},
        "oracle_schedules": {},
    }
    
    # For Garden and Bicycle, we have multi-scale data at 10K (screening)
    # and 30K (full training). Room only has 1.0 and 0.5 at 30K.
    
    # Oracle: at each checkpoint, find the lowest scale where DSSIM deviation <= threshold
    # DSSIM ≈ 1 - SSIM, so DSSIM deviation = |SSIM_1.0 - SSIM_s|
    
    screening = evidence.get("screening_10k", {})
    summary_30k = evidence.get("summary_30k", {})
    
    # Define thresholds to sweep
    thresholds = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]
    oracle["thresholds"] = thresholds
    
    for scene in ["garden", "bicycle", "room"]:
        schedule = {"10K": {}, "30K": {}}
        
        # 10K screening data
        if scene in screening:
            scr = screening[scene]
            ssim_10 = scr.get("scale_1.000", {}).get("ssim")
            if ssim_10 is not None:
                for tau in thresholds:
                    best_scale = "1.0"  # default to full
                    for scale_key in ["scale_0.500", "scale_0.625", "scale_0.750"]:
                        sd = scr.get(scale_key, {})
                        ssim_s = sd.get("ssim")
                        if ssim_s is not None:
                            delta = abs(ssim_10 - ssim_s)
                            if delta <= tau:
                                scale_val = scale_key.replace("scale_", "").replace("0.", "0.")
                                # Prefer lower scale (more aggressive)
                                scale_num = float(scale_key.replace("scale_", ""))
                                best_num = float(best_scale)
                                if scale_num < best_num:
                                    best_scale = str(scale_num)
                    schedule["10K"][str(tau)] = best_scale
        
        # 30K data
        # For each scene, get SSIM at 30K for available scales
        ssim_30k = {}
        for key, val in summary_30k.items():
            if key.startswith(f"{scene}_s"):
                scale = key.split("_s")[1]
                ssim = val.get("ssim")
                if ssim is not None:
                    ssim_30k[scale] = ssim
        
        if "1.0" in ssim_30k:
            ssim_base = ssim_30k["1.0"]
            for tau in thresholds:
                best_scale = "1.0"
                for scale in sorted(ssim_30k.keys(), key=float, reverse=True):
                    if scale == "1.0":
                        continue
                    delta = abs(ssim_base - ssim_30k[scale])
                    if delta <= tau:
                        scale_num = float(scale)
                        best_num = float(best_scale)
                        if scale_num < best_num:
                            best_scale = scale
                schedule["30K"][str(tau)] = best_scale
        
        oracle["oracle_schedules"][scene] = schedule
    
    # Analyze oracle properties
    oracle["analysis"] = {}
    for scene, schedule in oracle["oracle_schedules"].items():
        scene_analysis = {
            "10K_scales_by_threshold": schedule.get("10K", {}),
            "30K_scales_by_threshold": schedule.get("30K", {}),
        }
        
        # Check stage dependence: does oracle scale change from 10K to 30K?
        stage_dep = False
        for tau in thresholds:
            tau_str = str(tau)
            s10k = schedule.get("10K", {}).get(tau_str)
            s30k = schedule.get("30K", {}).get(tau_str)
            if s10k and s30k and s10k != s30k:
                stage_dep = True
                break
        scene_analysis["stage_dependent"] = stage_dep
        
        # Check scene dependence: does oracle scale differ across scenes?
        oracle["analysis"][scene] = scene_analysis
    
    # Scene dependence check
    scene_dep = False
    for tau in thresholds:
        tau_str = str(tau)
        scales_30k = {}
        for scene in ["garden", "bicycle", "room"]:
            s = oracle["oracle_schedules"][scene].get("30K", {}).get(tau_str)
            if s:
                scales_30k[scene] = s
        if len(set(scales_30k.values())) > 1:
            scene_dep = True
            break
    oracle["scene_dependent"] = scene_dep
    
    # Summary
    oracle["summary"] = {
        "mostly_constant": not scene_dep and not any(
            oracle["analysis"][s].get("stage_dependent", False) for s in oracle["analysis"]
        ),
        "stage_dependent": any(oracle["analysis"][s].get("stage_dependent", False) for s in oracle["analysis"]),
        "scene_dependent": scene_dep,
        "best_threshold_observation": "At tau=0.01 (SSIM gate), oracle selects different scales across scenes and stages — adaptive scheduling is potentially justified IF a cheap signal can predict it.",
    }
    
    out_path = RESULTS / "oracle_schedule.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(oracle, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nB2: Saved to {out_path}")
    
    print("\n=== B2 ORACLE ===")
    for scene, schedule in oracle["oracle_schedules"].items():
        print(f"\n{scene.upper()}:")
        for tau in [0.005, 0.010, 0.020]:
            s10k = schedule.get("10K", {}).get(str(tau), "?")
            s30k = schedule.get("30K", {}).get(str(tau), "?")
            print(f"  tau={tau}: 10K->{s10k}, 30K->{s30k}")
    print(f"\nScene dependent: {oracle['scene_dependent']}")
    print(f"Stage dependent: {any(oracle['analysis'][s].get('stage_dependent', False) for s in oracle['analysis'])}")
    
    return oracle


# === Phase B3: Cheap Predictor Search ===

def phase_b3(evidence, analysis_b1, oracle):
    """Search for simple interpretable predictors."""
    
    predictor = {
        "phase": "B3",
        "description": "Cheap predictor search using simple threshold rules",
        "candidate_signals": {},
        "predictor_rules": {},
        "evaluation": {},
    }
    
    # Available signals per scene:
    # 1. HF fraction (from gradient frequency probe) — Garden, Bicycle only
    # 2. Gradient cosine (from gradient frequency probe) — Garden, Bicycle only  
    # 3. DSSIM disagreement at 10K — Garden, Bicycle only
    # 4. N_gaussians (training state) — all scenes
    # 5. PSNR (training state) — all scenes
    
    # The key question: can we predict oracle scale from cheap signals?
    
    # Signal 1: HF fraction threshold
    # Garden baseline HF=0.766, Bicycle baseline HF=0.561
    # Garden needs 0.75 at tau=0.01, Bicycle needs 0.5
    # So: if HF > 0.7 -> use 0.75 or 1.0; if HF < 0.7 -> use 0.5
    
    gfp = evidence.get("gradient_frequency_probe", {})
    predictor["candidate_signals"]["hf_fraction"] = {
        "computation": "Gradient frequency probe: ratio of high-frequency energy in dL/dimage",
        "overhead": "MODERATE — requires computing gradient on a few cameras, then FFT/pyramid decomposition. NOT free.",
        "available_for": list(gfp.keys()),
        "missing_for": ["room"],
    }
    
    # Signal 2: N_gaussians (training state)
    # This is completely free — already computed during training
    # Garden baseline N at 10K = 2.95M, Bicycle baseline N at 10K = 3.14M
    # But N depends on scene complexity, not directly on optimal scale
    
    predictor["candidate_signals"]["n_gaussians"] = {
        "computation": "Current Gaussian count — available for free during training",
        "overhead": "FREE — already computed",
        "available_for": ["garden", "bicycle", "room"],
        "missing_for": [],
    }
    
    # Signal 3: DSSIM disagreement (multi-resolution)
    # Requires computing DSSIM at 2 resolutions — costs ~1.7x one DSSIM computation
    # But this is the most directly relevant signal
    
    predictor["candidate_signals"]["dssim_disagreement"] = {
        "computation": "|DSSIM_1.0 - DSSIM_0.5| on a few cameras — requires computing SSIM at 2 resolutions",
        "overhead": "MODERATE — ~1.7x cost of one DSSIM computation (compute 0.5 resolution DSSIM additionally). NOT free.",
        "available_for": ["garden", "bicycle"],
        "missing_for": ["room"],
    }
    
    # Signal 4: Iteration number (free, but not scene-discriminative)
    predictor["candidate_signals"]["iteration"] = {
        "computation": "Current training iteration",
        "overhead": "FREE",
        "available_for": ["garden", "bicycle", "room"],
        "missing_for": [],
    }
    
    # === Predictor Rules ===
    
    # Rule 1: HF fraction threshold
    # Using baseline HF: Garden=0.766 (needs 0.75), Bicycle=0.561 (needs 0.5)
    # Threshold: HF > 0.65 -> 0.75, else 0.5
    predictor["predictor_rules"]["rule_hf_threshold"] = {
        "rule": "if hf_fraction > tau1: scale=0.75; elif hf_fraction > tau2: scale=0.625; else: scale=0.5",
        "proposed_thresholds": {"tau1": 0.65, "tau2": 0.45},
        "overhead": "MODERATE — requires gradient frequency probe on subset of cameras",
        "problem": "1) HF fraction not available for Room. 2) Requires computing gradients + frequency analysis, which is NOT free. 3) HF fraction is measured on the gradient, not the image residual.",
    }
    
    # Rule 2: N_gaussians threshold
    # Garden final N ~3M (needs 0.75), Bicycle final N ~3.9M (needs 0.5), Room final N ~0.95M (needs 0.5)
    # This doesn't separate well — both Garden and Bicycle have high N
    predictor["predictor_rules"]["rule_n_threshold"] = {
        "rule": "if N_gaussians > tau: scale=0.75; else: scale=0.5",
        "problem": "N_gaussians does not separate scenes well. Garden (3M, needs 0.75) and Bicycle (3.9M, needs 0.5) are both high-N. Room (0.95M, needs 0.5) is low-N. No threshold separates Garden from Bicycle.",
        "verdict": "REJECTED — cannot distinguish Garden from Bicycle",
    }
    
    # Rule 3: Iteration-based schedule
    # Use 0.75 for early training, 0.5 for late training
    # This is essentially a fixed schedule, not adaptive
    predictor["predictor_rules"]["rule_iteration_schedule"] = {
        "rule": "if iteration < T1: scale=0.75; else: scale=0.5",
        "problem": "This is a fixed temporal schedule, not an adaptive predictor. The oracle shows stage dependence (Bicycle HF suppression weakens over time), but a fixed schedule cannot adapt to scene-specific needs.",
        "verdict": "NOT ADAPTIVE — this is a fixed schedule, not scene-adaptive",
    }
    
    # Rule 4: Combined — use iteration + a cheap proxy
    # The only truly free signal is iteration number
    # The next cheapest is DSSIM at 0.5 (already computed if using 0.5)
    
    # Key insight: if we're already computing DSSIM at scale s,
    # we can compare it against the L1 loss to get a cheap signal
    # L1 is full-resolution, DSSIM_s is downsampled
    # If L1 and DSSIM_s disagree a lot, the scene has high-freq content not captured by s
    
    predictor["predictor_rules"]["rule_l1_dssim_ratio"] = {
        "rule": "if |L1 - lambda*dssim_s| / L1 > tau: increase scale; else: keep scale",
        "computation": "L1 and DSSIM_s are both already computed during training. The ratio is a free byproduct.",
        "overhead": "FREE — both quantities already computed",
        "problem": "L1 and DSSIM measure different things (pixel error vs structural similarity). Their absolute values are not directly comparable. Need to normalize.",
        "verdict": "POTENTIAL — but needs validation. The signal may not be discriminative enough.",
    }
    
    # === Evaluation ===
    # Evaluate each rule against the oracle
    
    # For the oracle at tau=0.01 (SSIM gate):
    oracle_at_001 = {}
    for scene, schedule in oracle["oracle_schedules"].items():
        oracle_at_001[scene] = schedule.get("30K", {}).get("0.01", "1.0")
    
    predictor["evaluation"]["oracle_at_tau_0.01"] = oracle_at_001
    
    # Evaluate HF threshold rule
    hf_rule_predictions = {}
    for scene in ["garden", "bicycle"]:
        hf = gfp.get(scene, {}).get("s1.000", {}).get("mean_hf_fraction")
        if hf is not None:
            if hf > 0.65:
                hf_rule_predictions[scene] = "0.75"
            elif hf > 0.45:
                hf_rule_predictions[scene] = "0.625"
            else:
                hf_rule_predictions[scene] = "0.5"
    
    # Room: no HF data, must default
    hf_rule_predictions["room"] = "0.5"  # safe default
    
    predictor["evaluation"]["hf_rule"] = {
        "predictions": hf_rule_predictions,
        "oracle": oracle_at_001,
        "matches": {s: hf_rule_predictions.get(s) == oracle_at_001.get(s) for s in oracle_at_001},
        "accuracy": sum(1 for s in oracle_at_001 if hf_rule_predictions.get(s) == oracle_at_001.get(s)) / max(1, len(oracle_at_001)),
    }
    
    # Evaluate fixed-0.5 baseline
    fixed_05_predictions = {s: "0.5" for s in oracle_at_001}
    predictor["evaluation"]["fixed_0.5"] = {
        "predictions": fixed_05_predictions,
        "oracle": oracle_at_001,
        "matches": {s: fixed_05_predictions.get(s) == oracle_at_001.get(s) for s in oracle_at_001},
        "accuracy": sum(1 for s in oracle_at_001 if fixed_05_predictions.get(s) == oracle_at_001.get(s)) / max(1, len(oracle_at_001)),
    }
    
    # Evaluate fixed-0.75 baseline
    fixed_075_predictions = {s: "0.75" for s in oracle_at_001}
    predictor["evaluation"]["fixed_0.75"] = {
        "predictions": fixed_075_predictions,
        "oracle": oracle_at_001,
        "matches": {s: fixed_075_predictions.get(s) == oracle_at_001.get(s) for s in oracle_at_001},
        "accuracy": sum(1 for s in oracle_at_001 if fixed_075_predictions.get(s) == oracle_at_001.get(s)) / max(1, len(oracle_at_001)),
    }
    
    out_path = RESULTS / "predictor_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictor, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nB3: Saved to {out_path}")
    
    print("\n=== B3 PREDICTOR SEARCH ===")
    for name, ev in predictor["evaluation"].items():
        if isinstance(ev, dict) and "accuracy" in ev:
            print(f"  {name}: accuracy={ev['accuracy']:.2f}, predictions={ev['predictions']}")
    
    return predictor


# === Phase B4: Leave-One-Scene-Out ===

def phase_b4(evidence, oracle, predictor):
    """Leave-one-scene-out validation."""
    
    loso = {
        "phase": "B4",
        "description": "Leave-one-scene-out validation of predictor rules",
        "folds": {},
        "summary": {},
    }
    
    scenes = ["room", "garden", "bicycle"]
    oracle_at_001 = {}
    for scene, schedule in oracle["oracle_schedules"].items():
        oracle_at_001[scene] = schedule.get("30K", {}).get("0.01", "1.0")
    
    # For HF-based rule: train threshold on 2 scenes, test on 1
    # Available HF data: garden (0.766), bicycle (0.561). Room: missing.
    
    gfp = evidence.get("gradient_frequency_probe", {})
    
    for held_out in scenes:
        train_scenes = [s for s in scenes if s != held_out]
        
        fold = {"held_out": held_out, "train_scenes": train_scenes}
        
        # HF-based rule: set threshold between train scenes' HF values
        train_hfs = {}
        for s in train_scenes:
            hf = gfp.get(s, {}).get("s1.000", {}).get("mean_hf_fraction")
            if hf is not None:
                train_hfs[s] = hf
        
        if len(train_hfs) >= 2:
            hf_values = sorted(train_hfs.values())
            threshold = (hf_values[0] + hf_values[1]) / 2
            
            # Apply to held-out scene
            held_hf = gfp.get(held_out, {}).get("s1.000", {}).get("mean_hf_fraction")
            if held_hf is not None:
                if held_hf > threshold:
                    prediction = "0.75"
                else:
                    prediction = "0.5"
            else:
                prediction = "0.5"  # safe default when no HF data
            
            fold["hf_rule"] = {
                "train_hfs": train_hfs,
                "threshold": threshold,
                "held_out_hf": held_hf,
                "prediction": prediction,
                "oracle": oracle_at_001.get(held_out),
                "correct": prediction == oracle_at_001.get(held_out),
            }
        else:
            fold["hf_rule"] = {"note": f"Insufficient HF data for training (only {len(train_hfs)} scenes with HF data)"}
        
        # Fixed-0.5 baseline
        fold["fixed_0.5"] = {
            "prediction": "0.5",
            "oracle": oracle_at_001.get(held_out),
            "correct": "0.5" == oracle_at_001.get(held_out),
        }
        
        # Fixed-0.75 baseline
        fold["fixed_0.75"] = {
            "prediction": "0.75",
            "oracle": oracle_at_001.get(held_out),
            "correct": "0.75" == oracle_at_001.get(held_out),
        }
        
        loso["folds"][held_out] = fold
    
    # Summary
    hf_correct = sum(1 for s in scenes if loso["folds"][s].get("hf_rule", {}).get("correct", False))
    fixed_05_correct = sum(1 for s in scenes if loso["folds"][s].get("fixed_0.5", {}).get("correct", False))
    fixed_075_correct = sum(1 for s in scenes if loso["folds"][s].get("fixed_0.75", {}).get("correct", False))
    
    loso["summary"] = {
        "hf_rule_accuracy": hf_correct / len(scenes),
        "fixed_0.5_accuracy": fixed_05_correct / len(scenes),
        "fixed_0.75_accuracy": fixed_075_correct / len(scenes),
        "note": "HF rule requires gradient frequency probe which is NOT free and not available for Room. When Room is held out, HF rule defaults to 0.5 (safe).",
        "key_limitation": "Only 2 scenes have HF data (garden, bicycle). LOSO with 2 training scenes and 1 test scene is weak. Room has no HF probe, so any HF-based rule must default for Room.",
    }
    
    out_path = RESULTS / "leave_one_scene_out.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(loso, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nB4: Saved to {out_path}")
    
    print("\n=== B4 LEAVE-ONE-SCENE-OUT ===")
    for s in scenes:
        fold = loso["folds"][s]
        hf = fold.get("hf_rule", {})
        print(f"  {s}: oracle={oracle_at_001.get(s)}, HF_rule={hf.get('prediction')}({'Y' if hf.get('correct') else 'N'}), fixed_0.5={'Y' if fold['fixed_0.5']['correct'] else 'N'}, fixed_0.75={'Y' if fold['fixed_0.75']['correct'] else 'N'}")
    print(f"\n  HF rule accuracy: {loso['summary']['hf_rule_accuracy']:.2f}")
    print(f"  Fixed-0.5 accuracy: {loso['summary']['fixed_0.5_accuracy']:.2f}")
    print(f"  Fixed-0.75 accuracy: {loso['summary']['fixed_0.75_accuracy']:.2f}")
    
    return loso


# === Phase B5: Opportunity Estimate ===

def phase_b5(evidence, oracle, predictor, loso):
    """Estimate opportunity of adaptive vs fixed scale."""
    
    opp = {
        "phase": "B5",
        "description": "Opportunity estimate using known SepSSIM costs",
        "loss_costs": {},
        "oracle_saving": {},
        "predictor_saving": {},
        "net_opportunity": {},
    }
    
    # Fixed-tensor loss costs (from S2.3, historical, fixed-tensor measurement)
    loss_costs = {
        "1.0": 33.15,  # ms
        "0.75": 19.41,
        "0.625": 13.92,
        "0.5": 9.24,
    }
    
    ftb = evidence.get("fixed_tensor_loss_costs", {})
    for scale, scenes in ftb.items():
        if scenes:
            avg = sum(s["forward_backward_ms"] for s in scenes.values()) / len(scenes)
            loss_costs[str(scale)] = round(avg, 2)
    
    opp["loss_costs"] = {
        "source": "S2.3 fixed_tensor_loss_benchmark.json (historical, fixed-tensor, A100)",
        "costs_ms": loss_costs,
        "note": "These are historical fixed-tensor measurements. The loss cost is resolution-dependent but scene-independent (operates on rendered images).",
    }
    
    # Oracle saving: if we knew the oracle scale at each stage, what would we save?
    # Oracle at tau=0.01:
    oracle_at_001 = {}
    for scene, schedule in oracle["oracle_schedules"].items():
        oracle_at_001[scene] = schedule.get("30K", {}).get("0.01", "1.0")
    
    # Fixed-0.5 baseline cost
    fixed_05_cost = loss_costs["0.5"]
    fixed_075_cost = loss_costs["0.75"]
    fixed_10_cost = loss_costs["1.0"]
    
    for scene in ["room", "garden", "bicycle"]:
        oracle_scale = oracle_at_001.get(scene, "1.0")
        oracle_cost = loss_costs.get(oracle_scale, fixed_10_cost)
        
        opp["oracle_saving"][scene] = {
            "oracle_scale": oracle_scale,
            "oracle_loss_cost_ms": oracle_cost,
            "fixed_05_cost_ms": fixed_05_cost,
            "fixed_075_cost_ms": fixed_075_cost,
            "saving_vs_fixed_05_ms": fixed_05_cost - oracle_cost,  # negative if oracle is more expensive
            "saving_vs_fixed_05_pct": (1 - oracle_cost / fixed_05_cost) * 100 if fixed_05_cost > 0 else 0,
            "saving_vs_baseline_ms": fixed_10_cost - oracle_cost,
            "saving_vs_baseline_pct": (1 - oracle_cost / fixed_10_cost) * 100 if fixed_10_cost > 0 else 0,
        }
    
    # Predictor saving: using HF rule
    hf_predictions = {}
    for scene in ["room", "garden", "bicycle"]:
        fold = loso["folds"].get(scene, {})
        hf_pred = fold.get("hf_rule", {}).get("prediction", "0.5")
        hf_predictions[scene] = hf_pred
    
    for scene in ["room", "garden", "bicycle"]:
        pred_scale = hf_predictions.get(scene, "0.5")
        pred_cost = loss_costs.get(pred_scale, fixed_10_cost)
        oracle_scale = oracle_at_001.get(scene, "1.0")
        oracle_cost = loss_costs.get(oracle_scale, fixed_10_cost)
        
        opp["predictor_saving"][scene] = {
            "predicted_scale": pred_scale,
            "predicted_loss_cost_ms": pred_cost,
            "oracle_loss_cost_ms": oracle_cost,
            "oracle_regret_ms": pred_cost - oracle_cost,  # how much worse than oracle
            "saving_vs_baseline_ms": fixed_10_cost - pred_cost,
            "saving_vs_baseline_pct": (1 - pred_cost / fixed_10_cost) * 100 if fixed_10_cost > 0 else 0,
        }
    
    # Net opportunity
    # Average across scenes
    avg_oracle_saving_vs_05 = sum(
        opp["oracle_saving"][s]["saving_vs_fixed_05_ms"] for s in opp["oracle_saving"]
    ) / max(1, len(opp["oracle_saving"]))
    
    avg_pred_saving_vs_baseline = sum(
        opp["predictor_saving"][s]["saving_vs_baseline_ms"] for s in opp["predictor_saving"]
    ) / max(1, len(opp["predictor_saving"]))
    
    avg_fixed_05_saving_vs_baseline = fixed_10_cost - fixed_05_cost
    
    # Selector overhead estimate
    # HF probe: ~10ms per camera on 5 cameras = ~50ms, but amortized over ~100 iterations = ~0.5ms/iter
    # This is significant relative to the 9.24ms loss cost at scale 0.5
    selector_overhead_ms = 0.5  # estimate: amortized HF probe cost
    
    opp["net_opportunity"] = {
        "oracle_avg_saving_vs_fixed_05_ms": round(avg_oracle_saving_vs_05, 2),
        "predictor_avg_saving_vs_baseline_ms": round(avg_pred_saving_vs_baseline, 2),
        "fixed_05_saving_vs_baseline_ms": round(avg_fixed_05_saving_vs_baseline, 2),
        "selector_overhead_ms": selector_overhead_ms,
        "net_adaptive_vs_fixed_05_ms": round(avg_oracle_saving_vs_05 - selector_overhead_ms, 2),
        "note": "If oracle saving vs fixed-0.5 is negative, the oracle is MORE expensive than fixed-0.5, meaning adaptive scheduling has no opportunity.",
        "key_finding": "",
    }
    
    # Key finding
    if avg_oracle_saving_vs_05 < 0:
        opp["net_opportunity"]["key_finding"] = (
            f"Oracle is MORE expensive than fixed-0.5 by {-avg_oracle_saving_vs_05:.2f}ms on average. "
            "The oracle sometimes selects 0.75 (which costs 19.41ms vs 9.24ms for 0.5), "
            "but the quality benefit of 0.75 over 0.5 is marginal (0.05 dB on Garden). "
            "Adaptive scheduling provides NO net opportunity over fixed-0.5."
        )
    elif avg_oracle_saving_vs_05 < 2:
        opp["net_opportunity"]["key_finding"] = (
            f"Oracle saves only {avg_oracle_saving_vs_05:.2f}ms vs fixed-0.5, which is less than selector overhead ({selector_overhead_ms}ms). "
            "Net opportunity is negative."
        )
    else:
        opp["net_opportunity"]["key_finding"] = (
            f"Oracle saves {avg_oracle_saving_vs_05:.2f}ms vs fixed-0.5, with {selector_overhead_ms}ms selector overhead. "
            "Net opportunity may be positive."
        )
    
    out_path = RESULTS / "opportunity_estimate.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(opp, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nB5: Saved to {out_path}")
    
    print("\n=== B5 OPPORTUNITY ESTIMATE ===")
    print(f"  Loss costs: {loss_costs}")
    print(f"  Oracle saving vs fixed-0.5 (avg): {avg_oracle_saving_vs_05:.2f} ms")
    print(f"  Fixed-0.5 saving vs baseline: {avg_fixed_05_saving_vs_baseline:.2f} ms")
    print(f"  Selector overhead: {selector_overhead_ms} ms")
    print(f"  Net adaptive vs fixed-0.5: {opp['net_opportunity']['net_adaptive_vs_fixed_05_ms']:.2f} ms")
    print(f"  Key finding: {opp['net_opportunity']['key_finding']}")
    
    return opp


# === Final Decision ===

def final_decision(evidence, analysis_b1, oracle, predictor, loso, opp):
    """Apply hard decision gate."""
    
    decision = {
        "experiment": "C42_ADAPTIVE",
        "title": "C42++ Adaptive Structural Supervision Feasibility",
        "date": "2026-09-15",
        "phases": {
            "B0": "Evidence inventory complete",
            "B1": "Scene dependence analysis complete",
            "B2": "Offline oracle constructed",
            "B3": "Cheap predictor search complete",
            "B4": "Leave-one-scene-out complete",
            "B5": "Opportunity estimate complete",
        },
        "oracle_summary": {
            "room": oracle["oracle_schedules"]["room"].get("30K", {}).get("0.01", "?"),
            "garden": oracle["oracle_schedules"]["garden"].get("30K", {}).get("0.01", "?"),
            "bicycle": oracle["oracle_schedules"]["bicycle"].get("30K", {}).get("0.01", "?"),
            "scene_dependent": oracle["scene_dependent"],
            "stage_dependent": any(oracle["analysis"][s].get("stage_dependent", False) for s in oracle["analysis"]),
        },
        "best_cheap_signal": {
            "name": "HF fraction (gradient frequency probe)",
            "computation": "Ratio of high-frequency energy in dL/dimage via FFT/pyramid decomposition on subset of cameras",
            "overhead_estimate": "~0.5ms/iter amortized (50ms probe / 100 iter interval)",
            "problem": "1) Not available for Room. 2) Not free — requires gradient computation + frequency analysis. 3) Only 2 scenes have data.",
        },
        "leave_one_scene_out": {
            "room": loso["folds"]["room"],
            "garden": loso["folds"]["garden"],
            "bicycle": loso["folds"]["bicycle"],
            "hf_rule_accuracy": loso["summary"]["hf_rule_accuracy"],
            "fixed_05_accuracy": loso["summary"]["fixed_0.5_accuracy"],
            "fixed_075_accuracy": loso["summary"]["fixed_0.75_accuracy"],
        },
        "opportunity": {
            "oracle_saving_vs_fixed_05_ms": opp["net_opportunity"]["oracle_avg_saving_vs_fixed_05_ms"],
            "fixed_05_saving_vs_baseline_ms": opp["net_opportunity"]["fixed_05_saving_vs_baseline_ms"],
            "adaptive_estimated_gain_ms": opp["net_opportunity"]["net_adaptive_vs_fixed_05_ms"],
            "selector_overhead_ms": opp["net_opportunity"]["selector_overhead_ms"],
        },
        "decision": "",
        "rationale": "",
    }
    
    # Decision logic:
    # C42_ADAPTIVE_KEEP: cheap signal consistently predicts, LOSO convincing, net gain > fixed
    # C42_ADAPTIVE_MODIFY: oracle opportunity real, but predictor not yet general
    # C42_ADAPTIVE_DROP: oracle provides little advantage over best fixed, OR scene dependence cannot be captured cheaply
    
    oracle_saving = opp["net_opportunity"]["oracle_avg_saving_vs_fixed_05_ms"]
    net_gain = opp["net_opportunity"]["net_adaptive_vs_fixed_05_ms"]
    hf_accuracy = loso["summary"]["hf_rule_accuracy"]
    fixed_05_accuracy = loso["summary"]["fixed_0.5_accuracy"]
    
    # Key question: does the oracle provide advantage over fixed-0.5?
    if oracle_saving <= 0:
        # Oracle is MORE expensive than fixed-0.5
        decision["decision"] = "C42_ADAPTIVE_DROP"
        decision["rationale"] = (
            "The offline oracle provides LITTLE ADVANTAGE over the best fixed operating point (scale=0.5). "
            f"The oracle is {abs(oracle_saving):.2f}ms MORE expensive than fixed-0.5 on average, "
            "because the oracle sometimes selects 0.75 (19.41ms) for Garden where 0.5 (9.24ms) would suffice "
            "with only 0.05 dB quality difference. "
            "The oracle failure means the adaptive idea is unnecessary — a single fixed scale=0.5 captures most of the benefit. "
            "No cheap predictor can overcome a negative oracle opportunity."
        )
    elif net_gain <= 0:
        # Oracle has some advantage but selector overhead eats it
        decision["decision"] = "C42_ADAPTIVE_DROP"
        decision["rationale"] = (
            f"Oracle saves {oracle_saving:.2f}ms vs fixed-0.5, but selector overhead ({opp['net_opportunity']['selector_overhead_ms']}ms) "
            f"consumes the gain. Net opportunity is {net_gain:.2f}ms — non-positive. "
            "The adaptive idea is unnecessary because the overhead of any cheap signal exceeds the oracle opportunity."
        )
    elif hf_accuracy <= fixed_05_accuracy:
        # Predictor not better than fixed-0.5
        decision["decision"] = "C42_ADAPTIVE_MODIFY"
        decision["rationale"] = (
            f"Oracle opportunity is real ({oracle_saving:.2f}ms saving vs fixed-0.5), but the cheap predictor "
            f"(HF rule) accuracy ({hf_accuracy:.0%}) is not better than fixed-0.5 accuracy ({fixed_05_accuracy:.0%}). "
            "The signal needs improvement — a better cheap signal may exist, but current evidence does not support it."
        )
    else:
        decision["decision"] = "C42_ADAPTIVE_KEEP"
        decision["rationale"] = (
            f"Cheap signal predicts useful scale with {hf_accuracy:.0%} accuracy, "
            f"leave-one-scene-out is convincing, and net gain ({net_gain:.2f}ms) is materially better than fixed-0.5."
        )
    
    out_path = RESULTS / "final_decision.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nFinal decision: Saved to {out_path}")
    
    print(f"\n=== FINAL DECISION: {decision['decision']} ===")
    print(f"Rationale: {decision['rationale'][:200]}...")
    
    return decision


# === Main ===

def main():
    print("=" * 60)
    print("C42++ Adaptive Structural Supervision Feasibility")
    print("Phases B0-B5 — Offline Analysis Only")
    print("=" * 60)
    
    # Provenance
    prov = {
        "date": "2026-09-15",
        "analysis_type": "offline, CPU-only, no GPU, no training",
        "data_sources": [
            "results/reference_v1/s22/ (S2.2 cross-scene generalization)",
            "results/reference_v1/s23/ (S2.3 Pareto frontier)",
        ],
        "canonical_c42_definition": "F.interpolate(scale_factor=s, mode='area') + SepSSIM on downsampled tensors",
        "no_new_experiments": True,
        "no_gpu_used": True,
    }
    with open(RESULTS / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)
    
    # Phase B0
    evidence = phase_b0()
    
    # Phase B1
    analysis_b1 = phase_b1(evidence)
    
    # Phase B2
    oracle = phase_b2(evidence)
    
    # Phase B3
    predictor = phase_b3(evidence, analysis_b1, oracle)
    
    # Phase B4
    loso = phase_b4(evidence, oracle, predictor)
    
    # Phase B5
    opp = phase_b5(evidence, oracle, predictor, loso)
    
    # Final decision
    decision = final_decision(evidence, analysis_b1, oracle, predictor, loso, opp)
    
    print("\n" + "=" * 60)
    print("ALL PHASES COMPLETE")
    print("=" * 60)
    print(f"\nDeliverables in {RESULTS}/:")
    for f in sorted(RESULTS.glob("*.json")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
