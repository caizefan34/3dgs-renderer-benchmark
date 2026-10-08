#!/usr/bin/env python3
"""Build the pre-run identity / gate / plan / reuse JSON deliverables for
C0-QUALITY-STABILITY0-MX. Pure offline synthesis from frozen prior artifacts
already read (causal ladder + _mx_h8 bands). Writes into the local campaign dir.
No scientific threshold is invented here beyond what frozen artifacts carry.
"""
import json, os, hashlib, datetime

CAMPAIGN = "C0_QUALITY_STABILITY0_MX"
OUT = os.path.dirname(os.path.abspath(__file__))
UTC = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ---- frozen facts (from CAUSAL_SOURCE_IDENTITY / _mx_h8) ----
SO_C0 = "9baf8655f859f3e0f456e99a2d8e5b3ddfe414fe17a1a3e9d8072e0afed95b91"
CORE_SO = "361b216bcc11609a0ebb8fb44ad2e0c6170948112b6294e85123df45541c8c98"
WORKTREE = "/mnt/storage_pool/liaoyuanjun/higs_c0_final30k_worktree"
GIT_HEAD = "77ab983ffe43420b2131669cb35776b883ca4c3c"
TRAINER = "/mnt/storage_pool/liaoyuanjun/mx_h8/incoming/derived/mx_h8_trainer.py"
LADDER = "/mnt/storage_pool/liaoyuanjun/mx_h8/incoming/derived/C0_SAME_PARENT_CAUSAL_LADDER_MX"
REPL = "/mnt/storage_pool/liaoyuanjun/MX_C0_MATCHED_REPLICATION"

def w(name, obj):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)
    print("wrote", name)

# 6. SOURCE IDENTITY ------------------------------------------------------
w("QUALITY_SOURCE_IDENTITY.json", {
  "artifact": "QUALITY_SOURCE_IDENTITY",
  "gate": "SOURCE_IDENTITY",
  "campaign": CAMPAIGN,
  "generated_utc": UTC,
  "frozen_implementation": {
    "substrate": "higs-dynamic (gsplat experimental dynamic rasterizer)",
    "repo_worktree": WORKTREE,
    "worktree_git_head": GIT_HEAD,
    "worktree_git_describe": "77ab983f",
    "worktree_git_dirty_lines": 25,
    "compiled_so": "/mnt/storage_pool/liaoyuanjun/higs_c0_final30k_cache/experimental_gaussian_render_inference_scene_cuda/experimental_gaussian_render_inference_scene_cuda.so",
    "compiled_so_sha256": SO_C0,
    "core_so_sha256": CORE_SO,
    "torch": "2.9.1+cu128",
    "cuda_runtime": "12.8",
    "python": "3.10.21",
    "nvcc": "Build cuda_12.8.r12.8/compiler.35583870_0",
    "gpu": "NVIDIA A100-PCIE-40GB (SM80)",
    "training_code": TRAINER,
    "one_binary_serves_both_arms": True,
  },
  "arms": {
    "Q0_A0": {
      "trainer_arm": "a0",
      "modules": {"F9": False, "SCALAR_ADJOINT": False, "H8_MR": False},
      "runtime_env": {"HIGS_DISABLE_F9": "1", "HIGS_BWD_SCALAR_ADJOINT": None,
                      "HIGS_BWD_H8_MR": None, "HIGS_PX_RUNTIME": "2", "HIGS_BWD_ABSGRAD": "1"},
      "so_sha256": SO_C0, "core_sha256": CORE_SO, "eps2d": 0.3,
    },
    "Q1_C0": {
      "trainer_arm": "c0",
      "modules": {"F9": True, "SCALAR_ADJOINT": True, "H8_MR": True},
      "runtime_env": {"HIGS_DISABLE_F9": None, "HIGS_BWD_SCALAR_ADJOINT": "scalar_adjoint",
                      "HIGS_BWD_H8_MR": "1", "HIGS_PX_RUNTIME": "2", "HIGS_BWD_ABSGRAD": "1"},
      "so_sha256": SO_C0, "core_sha256": CORE_SO, "eps2d": 0.3,
    },
  },
  "only_allowed_differences": ["HIGS_DISABLE_F9", "HIGS_BWD_SCALAR_ADJOINT", "HIGS_BWD_H8_MR"],
  "unintended_source_delta": False,
  "verdict": "PASS_SAME_PARENT_SAME_BINARY_RUNTIME_SWITCHES_ONLY",
})

# 7. PROTOCOL IDENTITY ----------------------------------------------------
w("QUALITY_PROTOCOL_IDENTITY.json", {
  "artifact": "QUALITY_PROTOCOL_IDENTITY",
  "gate": "PROTOCOL_IDENTITY",
  "campaign": CAMPAIGN,
  "generated_utc": UTC,
  "frozen_from": f"{LADDER}/CAUSAL_PROTOCOL_IDENTITY.json",
  "identical_to_causal_ladder": True,
  "recipe": {
    "iterations": 30000, "eps2d": 0.3, "resolution": "1080p",
    "position_lr_init": 0.00016, "position_lr_final": 1.6e-06,
    "position_lr_delay_mult": 0.01, "position_lr_max_steps": 30000,
    "feature_lr": 0.0025, "opacity_lr": 0.025, "scaling_lr": 0.005,
    "rotation_lr": 0.001, "densify_from_iter": 500, "densify_until_iter": 15000,
    "densification_interval": 100, "densify_grad_threshold": 0.0008,
    "percent_dense": 0.01, "min_opacity": 0.005, "max_screen_size": 20,
    "opacity_reset_interval": 3000, "sh_degree": 3, "sh_progress_interval": 1000,
    "lambda_dssim": 0.2, "tile_size": 16, "packed": False,
    "eval_iterations": [500,1000,2000,5000,10000,15000,20000,25000,30000],
    "final_eval": "all",
  },
  "arms_recipe_identical": True,
  "per_arm_runtime_switch": {
    "Q0_A0": {"HIGS_DISABLE_F9": "1", "HIGS_BWD_SCALAR_ADJOINT": None, "HIGS_BWD_H8_MR": None},
    "Q1_C0": {"HIGS_DISABLE_F9": None, "HIGS_BWD_SCALAR_ADJOINT": "scalar_adjoint", "HIGS_BWD_H8_MR": "1"},
  },
  "no_rescue_policy": "no eps2d/loss/optimizer/lr/densify/prune/opacity-reset/module/substrate/seed/order change after results observed",
  "verdict": "PASS",
})

# 5. SEED42 REUSE ---------------------------------------------------------
seed42 = {
  "room":  {"a0": {"psnr": 32.18814310304761, "ssim": 0.9280888007001478, "lpips": 0.30088685036088875, "wall_s": 1005.5447571091354, "n_gauss": 841533},
            "c0": {"psnr": 32.15315794011961, "ssim": 0.927507352407339, "lpips": 0.30136887341066954, "wall_s": 995.2764505818486, "n_gauss": 832220}},
  "bicycle":{"a0": {"psnr": 26.390948573218925, "ssim": 0.8360075452893051, "lpips": 0.24700192484966257, "wall_s": 1412.78, "n_gauss": None},
            "c0": {"psnr": 26.305562390991497, "ssim": 0.8342067116929084, "lpips": 0.24915269431994133, "wall_s": 1399.69, "n_gauss": None}},
  "garden":{"a0": {"psnr": 24.49094701321401, "ssim": 0.7502954782666387, "lpips": 0.3019971099254247, "wall_s": 1320.54, "n_gauss": None},
            "c0": {"psnr": 23.994599865388597, "ssim": 0.7216305059355659, "lpips": 0.33354859617916316, "wall_s": 1278.83, "n_gauss": None}},
}
pairs = {}
for s, d in seed42.items():
    pairs[f"{s}_s42"] = {
      "d_psnr": d["c0"]["psnr"] - d["a0"]["psnr"],
      "d_ssim": d["c0"]["ssim"] - d["a0"]["ssim"],
      "d_lpips": d["c0"]["lpips"] - d["a0"]["lpips"],
      "speedup": d["a0"]["wall_s"] / d["c0"]["wall_s"],
    }
mean_dpsnr = sum(p["d_psnr"] for p in pairs.values()) / len(pairs)
w("QUALITY_SEED42_REUSE.json", {
  "artifact": "QUALITY_SEED42_REUSE",
  "gate": "SEED42_REUSE",
  "campaign": CAMPAIGN,
  "generated_utc": UTC,
  "reused_from": f"{LADDER}/raw/{{scene}}_s42_{{a0,c0}}/",
  "reuse_decision": "REUSE (do not re-run seed42)",
  "checks": {
    "quality_json_present": True, "final_status_present": True, "timing_present": True,
    "training_curve_present": True, "results_json_present": True,
    "checkpoints_lite_present": True,
    "all_runs_status_SUCCESS": True, "all_contaminated_false": True,
    "all_timing_grade_PUBLICATION": True,
  },
  "per_scene_quality": seed42,
  "per_pair": pairs,
  "mean_d_psnr_seed42": mean_dpsnr,
  "matches_frozen_causal_ladder": abs(mean_dpsnr - (-0.20557283099361237)) < 1e-9,
  "rerun_required": False,
  "reason_no_rerun": "all seed42 A0/C0 artifacts complete, clean, PUBLICATION-grade, and reproduce the frozen causal-ladder ΔPSNR exactly",
})

# 18. FROZEN GATES --------------------------------------------------------
w("QUALITY_FROZEN_GATES.json", {
  "artifact": "QUALITY_FROZEN_GATES",
  "gate": "FROZEN_GATE_DEFINITION",
  "campaign": CAMPAIGN,
  "generated_utc": UTC,
  "source_of_bands": "_mx_h8 campaign MX_H8_QUALITY.json bands_fixed_before_runs (frozen BEFORE any run in that campaign; reused as the standing neutrality-gate family for this same 3-scene/3-seed higs A0-vs-C0 protocol)",
  "bands_fixed_before_runs": {
    "dPSNR_floor_dB": -0.05,
    "dSSIM_floor": -0.0005,
    "dLPIPS_ceil": 0.001,
    "catastrophic_pair_dPSNR": -0.5,
  },
  "direction_convention": "delta = C0 - A0; positive dPSNR/dSSIM and negative dLPIPS = C0 better; a pair is NEUTRAL if dPSNR>=floor AND dSSIM>=floor AND dLPIPS<=ceil; CATASTROPHIC if dPSNR < catastrophic_pair_dPSNR",
  "lpips_status": "threshold present (0.001) in the frozen band family -> LPIPS is gate-scored, not descriptive-only",
  "neutrality_rule": "a pair is within-neutrality iff it passes all three band bounds and is not catastrophic; the campaign is QUALITY_STABLE_NEUTRAL iff the OVERALL mean/CI is consistent with neutrality and catastrophic pairs are absent",
  "note": "bands were NOT redefined after seeing any C0-QUALITY-STABILITY0-MX result; they are inherited from the frozen pre-run calibration of the same protocol family",
  "gate_definition_available": True,
})

# 14/15. RUN PLAN ---------------------------------------------------------
run_plan = [
  # frozen alternating order (goal section 15), same-GPU-per-pair
  {"seq": 1, "seed": 43, "scene": "room",    "arm_order": ["a0", "c0"], "gpu": 6},
  {"seq": 2, "seed": 43, "scene": "bicycle", "arm_order": ["c0", "a0"], "gpu": 7},
  {"seq": 3, "seed": 43, "scene": "garden",  "arm_order": ["a0", "c0"], "gpu": 6},
  {"seq": 4, "seed": 44, "scene": "room",    "arm_order": ["c0", "a0"], "gpu": 7},
  {"seq": 5, "seed": 44, "scene": "bicycle", "arm_order": ["a0", "c0"], "gpu": 6},
  {"seq": 6, "seed": 44, "scene": "garden",  "arm_order": ["c0", "a0"], "gpu": 7},
]
planned_runs = []
for r in run_plan:
    for arm in r["arm_order"]:
        planned_runs.append({
            "scene": r["scene"], "arm": arm, "seed": r["seed"], "iterations": 30000,
            "eps2d": 0.3, "gpu": r["gpu"], "seq": r["seq"],
            "outdir": f"/dev/shm/c0_quality_stability/runs/{r['scene']}_s{r['seed']}_{arm}",
        })
w("QUALITY_RUN_PLAN.json", {
  "artifact": "QUALITY_RUN_PLAN",
  "campaign": CAMPAIGN,
  "generated_utc": UTC,
  "phase": "seed-expansion (seed43 + seed44), A0/C0 two-arm",
  "frozen_order_source": "goal section 15 recommended alternating order, frozen before launch",
  "arm_definitions": {
    "Q0_A0": {"trainer_arm": "a0", "env": {"HIGS_DISABLE_F9": "1"}, "eps2d": 0.3},
    "Q1_C0": {"trainer_arm": "c0", "env": {"HIGS_BWD_SCALAR_ADJOINT": "scalar_adjoint", "HIGS_BWD_H8_MR": "1"}, "eps2d": 0.3},
  },
  "new_runs": planned_runs,
  "n_new_runs": len(planned_runs),
  "total_evidence_after_new_runs": {"scenes": 3, "seeds": 3, "arms": 2, "pairs": 9},
  "excluded_by_design": {"seed45+": "not added", "A1/A2 seed43/44": "forbidden by goal section 28"},
  "gpu_exclusivity": "single clean A100 per pair (GPU 6 or 7); foreign jobs on 0-5 never touched",
  "frozen_after_start": True,
})

print("ALL PREREQ DELIVERABLES BUILT")
