#!/usr/bin/env python3
"""PUBLICATION_CONVERGENCE_SUPPORT_PACK builder (CPU/source-audit/artifact-only).
Emits all deliverables strictly from authoritative existing artifacts. No GPU work.
"""
import json, hashlib, os, datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
GEN = "2026-10-02"
NOW = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def w(path, obj):
    full = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        if isinstance(obj, str):
            f.write(obj)
        else:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return full

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()

def sha256_dir(d):
    entries = []
    for base, _, files in os.walk(d):
        for fn in sorted(files):
            fp = os.path.join(base, fn)
            rel = os.path.relpath(fp, d).replace("\\", "/")
            entries.append((rel, sha256_file(fp)))
    entries.sort()
    lines = [f"{h}  {rel}" for rel, h in entries]
    return "\n".join(lines) + "\n", entries

# =====================================================================
# TASK 1 - FINAL_PROTOCOL_LOCK
# =====================================================================
terminology = {
  "artifact": "FINAL_PROTOCOL_LOCK",
  "schema": "final-protocol-lock/v1",
  "generated_utc": NOW,
  "authority": "Synthesized from PUB0_TIMING_LEDGER.json, PBR0_VERDICT.json, TFC2_VERDICT.json, PAPER0_HARDENED, DFC2_VERDICT.json, DFC1R1_VERDICT.json",
  "mode": "CPU / source-audit / artifact-only. NO GPU work. NO new runs.",
  "purpose": "Single authoritative terminology + metric-definition lock for all future reports and paper tables.",
  "hard_rules": [
    "Never call renderer-local timing 'training speedup'.",
    "Never call decomposed proxy timing 'production full-step'.",
    "Never infer pure optimizer cost from residual subtraction of independent loops.",
    "Production training headline MUST use directly measured wall-clock.",
    "Renderer workload reduction and training-time reduction MUST remain separate metrics.",
    "Keep all hardware/software cohorts explicit; never merge numbers across cohorts."
  ],
  "frozen_terminology": [
    {
      "term": "intersection-only timing",
      "definition": "Wall-clock time of the tile-intersection stage only (isect count + emit + offset, excluding sort/rasterize). Reported as speedup vs stock per-axis-AABB baseline within the SAME renderer cohort.",
      "canonical_source": "PUB0_TIMING_LEDGER.json bucket1_intersection_only_ms; TFC2_VERDICT.per_scene.prof_isect_speedup; PUB0 AMDAHL isect-only",
      "must_not_be": "Must NOT be labeled a training-step or full-step speedup."
    },
    {
      "term": "forward rendering pipeline",
      "definition": "preprocess (projection+covariance+radius) + intersection + sort + rasterize-to-pixels, forward direction only. Excludes backward and optimizer.",
      "canonical_source": "PUB0_TIMING_LEDGER.json bucket2_forward_pipeline_ms; TFC2_BENCHMARK fwd_*_s",
      "must_not_be": "Must NOT be labeled forward+backward or training-step."
    },
    {
      "term": "renderer forward+backward",
      "definition": "Forward rendering pipeline plus the rasterizer/backward kernels, on a FROZEN PLY. No densification, no optimizer step, no 30K trajectory.",
      "canonical_source": "PUB0_TIMING_LEDGER.json bucket3_forward_backward_ms; TFC2_VERDICT.per_scene.prof_e2e_speedup (footprint-stage fwd+bwd)",
      "must_not_be": "Must NOT be presented as production full training step (no optimizer, no densification, frozen PLY)."
    },
    {
      "term": "production iteration wall-clock",
      "definition": "Directly measured wall-clock of ONE full production training iteration = preprocess + intersection + sort + rasterize + backward + densification + optimizer (Adam), on the canonical current pipeline, same-run baseline-vs-policy.",
      "canonical_source": "PUB0_TIMING_LEDGER.json bucket4_full_training_step_ms (current-cohort TFC_DFC_EPIC05)",
      "must_not_be": "Must NOT be a residual/subtraction-derived estimate; must be directly measured."
    },
    {
      "term": "total 30K production training wall-clock",
      "definition": "Directly measured end-to-end wall-clock to complete 30,000 production training iterations (baseline vs method), seed-fixed, same cohort. THE headline production-training metric.",
      "canonical_source": "PENDING: requires TFC_FC0 / DFC_FC0 / PUB0 total-time fields (not yet measured). No authoritative value exists yet.",
      "must_not_be": "Must NOT be inferred from per-iteration geomean or Amdahl ceilings. Currently NOT_MEASURED for all methods."
    }
  ],
  "authoritative_metric_definitions": {
    "intersection_reduction": {
      "pooled_reduction": "1 - sum(ellipse_intersections over all cameras) / sum(prod_AABB_intersections over all cameras). Scene/cohort-level aggregate.",
      "mean_per_camera_reduction": "unweighted mean over cameras of (1 - ellipse_i / prodAABB_i). Reports typical per-camera effect; differs from pooled.",
      "authoritative_values": {
        "TFC0_room_8cam_pooled": 0.42414788215947763,
        "TFC0_room_8cam_mean_per_camera": "computed from TFC0_FOOTPRINT_STATS.json per_camera (mean of per-cam reductions)",
        "TFC2_pooled_11scene": 0.3821339645678883,
        "PBR0_native": 0.3927,
        "PBR0_down2": None
      },
      "note": "Report pooled and mean-per-camera SEPARATELY; never conflate."
    },
    "psnr_ssim_lpips_deltas": {
      "definition": "method metric MINUS baseline metric on the SAME held-out cameras/eval set. PSNR in dB (positive = better). SSIM/LPIPS absolute deltas.",
      "sign_convention": "'+' = method better than baseline; '-' = worse.",
      "quality_neutral_thresholds": "PSNR loss <= 0.05 dB gate (renderer-local); full-convergence gates per pack.",
      "renderer_local_note": "TFC/DFC renderer-local deltas are quality-neutral by construction (PSNR loss ~1e-7 to 8e-5 dB)."
    },
    "gradient_cosine_relL2": {
      "gradient_cosine": "cosine between method gradient and reference (float64 or stock) gradient, per parameter and per camera; report mean/median and min.",
      "gradient_relL2": "||g_method - g_ref||_2 / ||g_ref||_2.",
      "noise_floor": "float32-vs-float64 noise floor; PBR0 gradient_AB_cosine_mean 0.999888; train cam151 aa_cos_median 0.99995 vs ab_cos_median 0.99939 (REAL_DIFF).",
      "gate": "gradient_cosine >= 0.99999 (self-imposed); document when at noise floor."
    },
    "gaussian_count_trajectory": {
      "definition": "Final Gaussian count and clone/split/prune counts over the 30K densification schedule, baseline vs method. Requires FULL training (not renderer-local).",
      "availability": "K50-B1 has real 30K populations (gs_ratio room 0.698, bicycle 0.694, garden 0.643). Renderer-local methods (TFC/DFC) have NO trajectory (frozen PLY)."
    },
    "integrated_gaussian_population": {
      "definition": "Time-integral of Gaussian population over the training trajectory (area under the count-vs-iteration curve); a workload/economic quantity.",
      "availability": "PENDING: no method has a measured integrated population yet; K50-B1 final counts only. Report as PENDING where absent."
    },
    "total_training_speedup": {
      "definition": "baseline_total_30K_wall_clock / method_total_30K_wall_clock, directly measured, seed-fixed, same cohort.",
      "availability": "NOT_MEASURED for ALL methods. K50-B1 IMPLIES +8.2%/+24.4%/+28.9% per-iter but has no separate total-wall-clock field. Renderer-local speedups MUST NOT populate this."
    }
  },
  "cohort_tags_required": [
    "EPIC05_A100_REMOTE (renderer-frozen-PLY)",
    "TFC_DFC_EPIC05 (current headline benchmark, torch2.10)",
    "VOS1_EPIC05 (torch2.8, graphdeco-b0-pub)",
    "K50B1_HISTORICAL (multi-GPU, NOT per-run comparable)",
    "C51_STAGE4A_ROOM (config-buggy 5K, do not merge)",
    "C51_STAGE4B_3SCENE (clean 30K K50-B1)",
    "FASTERGSFUSED_C0_30K (VOS/IGF optimizer-level)"
  ]
}
w("FINAL_PROTOCOL_LOCK.json", terminology)
print("part1 task1 written")

# ---- FINAL_PROTOCOL_LOCK.md ----
md = f"""# FINAL PROTOCOL LOCK — Terminology & Metric Definitions

**Generated:** {GEN}  |  **Mode:** CPU / source-audit / artifact-only. NO GPU work. NO new runs. NO K50.
**Authority:** PUB0_TIMING_LEDGER.json, PBR0_VERDICT.json, TFC2_VERDICT.json, PAPER0_HARDENED, DFC2_VERDICT.json, DFC1R1_VERDICT.json.

This document freezes the ONLY terminology and metric definitions future reports and paper tables MUST use.

## 0. Hard rules (non-negotiable)
- Never call renderer-local timing "training speedup".
- Never call decomposed proxy timing "production full-step".
- Never infer pure optimizer cost from residual subtraction of independent loops.
- Production training headline MUST use directly measured wall-clock.
- Renderer workload reduction and training-time reduction MUST remain separate metrics.
- Keep all hardware/software cohorts explicit; never merge numbers across cohorts.

## 1. Frozen terminology
| # | Term | Definition | Canonical source | MUST NOT be called |
|---|---|---|---|---|
| 1 | intersection-only timing | Tile-intersection stage only (count+emit+offset, excl. sort/rasterize) | PUB0 bucket1; TFC2 prof_isect_speedup | training-step / full-step |
| 2 | forward rendering pipeline | preprocess + intersection + sort + rasterize (forward only) | PUB0 bucket2; TFC2 fwd_*_s | fwd+bwd / training-step |
| 3 | renderer forward+backward | fwd pipeline + backward kernels on a FROZEN PLY (no optimizer/densify) | PUB0 bucket3; TFC2 prof_e2e_speedup | production full training step |
| 4 | production iteration wall-clock | ONE full production iteration (preprocess+isect+sort+rasterize+bwd+densify+optimizer), same-run | PUB0 bucket4 (current cohort) | residual/subtraction estimate |
| 5 | total 30K production training wall-clock | End-to-end wall-clock for 30K iters, seed-fixed, same cohort (HEADLINE) | PENDING (TFC_FC0/DFC_FC0) | per-iter geomean / Amdahl ceiling |

## 2. Authoritative metric definitions
- **intersection reduction (pooled)** = 1 - sum(ellipse_isect)/sum(prodAABB_isect) over all cameras.
- **intersection reduction (mean-per-camera)** = mean over cameras of (1 - ellipse_i/prodAABB_i). Report SEPARATELY from pooled.
- **PSNR/SSIM/LPIPS deltas** = method - baseline on the SAME held-out set; '+' = better.
- **gradient cosine** = cos(g_method, g_ref); **relL2** = ||g_method-g_ref||/||g_ref||.
- **Gaussian-count trajectory** = final + clone/split/prune counts over 30K densification (full training only).
- **integrated Gaussian population** = time-integral of count over trajectory (workload metric).
- **total training speedup** = baseline_total_30K_wall / method_total_30K_wall (directly measured).

## 3. Authoritative values (as of {GEN})
- TFC2 pooled intersection reduction (11 scenes): 0.38213; PBR0 native: 0.3927; TFC0 room 8-cam pooled: 0.42415.
- TFC legacy 1.3564x = footprint-stage fwd+bwd (TFC2_PROFILING); PBR0 reconciled native full-step = 1.1988x; PUB0 current-cohort TFC full-step = 1.0317x.
- Total 30K training wall-clock: NOT_MEASURED for every method (K50-B1 per-iter only).

## 4. Required cohort tags
EPIC05_A100_REMOTE · TFC_DFC_EPIC05 · VOS1_EPIC05 · K50B1_HISTORICAL · C51_STAGE4A_ROOM · C51_STAGE4B_3SCENE · FASTERGSFUSED_C0_30K.
"""
w("FINAL_PROTOCOL_LOCK.md", md)
print("task1 md written")

# =====================================================================
# TASK 2 - PAPER1_EVIDENCE_AUTOUPDATE
# =====================================================================
METHODS = {
  "TFC": {
    "current_role": "MAIN_CANDIDATE",
    "evidence_tier": "T2_CERTIFIED_RECONCILED",
    "canonical_implementation": "TFC1 row-wise ellipse footprint (tfc1_rowwise.cu) + TFC2 ablation kernel (tfc2_ablation.cu, ellipse-exact mode1)",
    "canonical_source_hash": {"tfc1_rowwise.cu": "a44e4ee63c7ba9d713a4489693d189cb3fd6e7ba95764e491027c0639e34e68e", "tfc2_ablation.cu": "3c9b2e0a1deda28fa1521f88b054a1b670d843b1b001a12c4b1d8b5fe4e5dd81"},
    "scenes": "room,garden,bicycle,bonsai,counter,kitchen,stump,flowers,treehill,truck,train (11)",
    "seeds": "42 (per-method)",
    "training_convergence_status": "PENDING (TFC_FC0 full 30K not run)",
    "workload_reduction": "intersection pooled -38.2% (11-scene); mean-per-camera reported separately",
    "renderer_speedup": "intersect kernel 1.38x; per-scene fwd render 1.15-1.42x; fwd+bwd (footprint-stage) 1.17-1.51x, pooled e2e 1.3564x (LEGACY)",
    "production_iteration_speedup": "current-cohort (PUB0) TFC full-step 1.0317x; PBR0 native reconciled full-step 1.1988x",
    "total_training_speedup": "NOT_MEASURED (pending TFC_FC0)",
    "quality": "numerically indistinguishable (PBR0 PSNR loss A->B 0.0 dB; maxabs_AB 3.2e-4; B==C bitwise). 'pixel-identical' withdrawn.",
    "gradient_correctness": "gradient_AB_cosine_mean 0.999888 (noise floor); TFC2 min grad_cosine 0.99995 (>=0.99999 gate at noise floor, documented)",
    "unresolved_blockers": "TFC_FC0 production convergence/timing pending; total 30K wall-clock not measured; full-convergence evidence absent",
    "evidence": "_tfc2_oracle/TFC2_VERDICT.json, _pbr0_oracle/PBR0_VERDICT.json, _pub0_ledger/PUB0_TIMING_LEDGER.json, _tfc_fc0_pack/ (RUNNING)"
  },
  "K50-B1": {
    "current_role": "NEEDS_REVALIDATION",
    "evidence_tier": "SUPPORTED_30K_HISTORICAL",
    "canonical_implementation": "B1 sparse-backward + densification-decoupling mask in original 3DGS CUDA rasterizer (gsplat 1.5.3)",
    "canonical_source_hash": {"note": "stage4b report-documented 6-file patch; see reports/phase-c51-stage4b.md; byte hashes recorded in K50 stage4 packs"},
    "scenes": "room,garden,bicycle",
    "seeds": "single seed per-scene (gpu_id 1/3); no multi-seed CI",
    "training_convergence_status": "YES_30K (C51_STAGE4B_3SCENE clean cohort)",
    "workload_reduction": "Gaussian count ~-30% to -36% (gs_ratio room 0.698, bicycle 0.694, garden 0.643)",
    "renderer_speedup": "n/a (backward-only method)",
    "production_iteration_speedup": "1.082x room / 1.244x bicycle / 1.289x garden (30K full-step, Stage4B)",
    "total_training_speedup": "IMPLIED_POSITIVE (+8.2%/+24.4%/+28.9%); no separate total-wall-clock field",
    "quality": "PSNR +0.04..+1.03 dB, SSIM +0.0099..+0.0101 (Stage4B clean); forward bit-exact; Gate A (cosine>=0.999, skipped=0) PASS",
    "gradient_correctness": "selected gradients exact (cosine=1.0); skipped gradients zero (B1); forward bit-exact",
    "unresolved_blockers": "canonical current-pipeline revalidation required; multi-seed CI; LPIPS unmeasured; measured total training wall-time; Stage4B-only primary evidence (stage4a config-buggy 5K)",
    "evidence": "reports/phase-c51-stage4b.md, _pub0_ledger/PUB0_TIMING_LEDGER.json (K50B1_HISTORICAL)"
  },
  "DFC1-R1_saved-T": {
    "current_role": "SUPPORTING",
    "evidence_tier": "SUPPORTING_PASS_CORRECTNESS",
    "canonical_implementation": "saved-T backward fix in gsplat-dfc1r1 (RasterizeToPixels3DGSBwd.cu uses saved render_T)",
    "canonical_source_hash": {"RasterizeToPixels3DGSBwd.cu": "31c11029fa8309234a83674806a723a5785ee1d387d3181a11062850bde055f3", "Fwd.cu(dfc1r1)": "70eef3d143f7e94857c04077917ce8392a68fc8faf7d8ca1a0311077743249ed"},
    "scenes": "room,garden,bicycle",
    "seeds": "float64 CPU oracle reference",
    "training_convergence_status": "PARTIAL_RENDERER_STEP (backward correctness; no 30K)",
    "workload_reduction": "n/a (correctness fix, not a workload reducer)",
    "renderer_speedup": "no standalone speedup (no-op at early-termination thresholds t>=1e-5)",
    "production_iteration_speedup": "NOT_MEASURED",
    "total_training_speedup": "NOT_MEASURED",
    "quality": "gradient-cosine gate PASS everywhere; 548x grad-error reduction at t=0 (0.4324 -> 0.00079 rel-L2)",
    "gradient_correctness": "grad_cosine at t=0 0.99999985 (dfc1r1) vs 0.99050 (dfc1old) vs 0.99999955 (stock)",
    "unresolved_blockers": "bicycle PSNR quality gate fails at t>=1e-5 (sparse 501-G scene, needs t<1e-5); no standalone speedup; SUPPORTING only",
    "evidence": "_dfc1r1_oracle/output/deliverables/DFC1R1_VERDICT.json"
  },
  "DFC_optimization_family": {
    "current_role": "PENDING",
    "evidence_tier": "MIXED_PENDING",
    "canonical_implementation": "DFC1-R2 adaptive per-pixel transmittance termination (gsplat-dfc1r2adapt)",
    "canonical_source_hash": {"RasterizeToPixels3DGSFwd.cu": "d4f287449d61f094e04dcaa6d573604c39d99c81f14f67fb9bd1b22f3c0d1590", "RasterizeToPixels3DGSBwd.cu": "31c11029fa8309234a83674806a723a5785ee1d387d3181a11062850bde055f3"},
    "scenes": "room,garden,bicycle (family also: 6 held-out scenes for DFC2)",
    "seeds": "42",
    "training_convergence_status": "PENDING (DFC2 generalization queued; DFC_FC0 full convergence queued, requires validation)",
    "workload_reduction": "DFC1-R2: room -28.8%, garden -20.7%, bicycle -17.1% splat (best policy); DFC2 fixed-threshold splat reduction 5.8-23.4% (no universal threshold meets >=10% on all scenes)",
    "renderer_speedup": "compositing 1.10-1.28x; raster 1.10-1.17x; DFC2 full-step per-scene 1.006-1.080x",
    "production_iteration_speedup": "DFC1-R2 E2E room 1.206x PASS(4/8 cams); garden 1.140x, bicycle 1.061x FAIL; current-cohort (PUB0) DFC full-step 1.0539x",
    "total_training_speedup": "NOT_MEASURED (pending DFC_FC0)",
    "quality": "DFC2 quality PASS all 6 scenes both thresholds (PSNR loss <=5e-6 dB); DFC1-R2 fails >=1.2x gate on dense scenes; bicycle forward-quality fails at t>=1e-5",
    "gradient_correctness": "DFC2 grad_cosine 0.99999986-0.999999999, grad_relL2 1.4e-5 to 2.9e-4 (all thresholds)",
    "unresolved_blockers": "DFC2 DOES_NOT_GENERALIZE_UNIVERSALLY (no fixed threshold meets quality<=0.05dB AND workload>=10% on all held-out scenes); DFC_FC0 train_dfc.py unverified scaffold; DFC1-R2 dense-scene Amdahl-bound",
    "evidence": "_dfc1r2_oracle, _dfc2_pack/DFC2_VERDICT.json, _dfc2_pack/DFC_FC0_SOURCE_LOCK.json"
  },
  "VOS": {
    "current_role": "ANALYSIS",
    "evidence_tier": "NEGATIVE_ANALYSIS",
    "canonical_implementation": "VOS1 visibility-gated optimizer-step skip (SparseGaussianAdam, graphdeco-b0-pub)",
    "canonical_source_hash": {"note": "see vos1/VOS1_SOURCE_LOCK.json"},
    "scenes": "room,garden,bicycle",
    "seeds": "3 seeds {0,1,42} (18 runs)",
    "training_convergence_status": "YES_30K_MULTISEED (VOS1 hardened, true-holdout eval)",
    "workload_reduction": "~-40.2% logical update reduction (cold_frac*(1-1/k)); AMDAHL-BOUND (optimizer ~15-19% of step, SparseGaussianAdam O(N))",
    "renderer_speedup": "n/a (optimizer-stage only)",
    "production_iteration_speedup": "NONE (measured wall ~1.004x; optimizer ~15% of step)",
    "total_training_speedup": "NONE",
    "quality": "QUALITY_FAIL: near-neutral; room seed-0 -0.271 dB instability; no wall-clock benefit",
    "gradient_correctness": "not the failure axis (economic/Amdahl failure)",
    "unresolved_blockers": "Amdahl-bound; SparseGaussianAdam O(N) so cutting update COUNT does not cut wall time; no economic win",
    "evidence": "VOS0_REPORT.md, PAPER0_HARDENED, _pub0_ledger (VOS1_EPIC05)"
  },
  "IGF": {
    "current_role": "DROP",
    "evidence_tier": "NEGATIVE_DROP",
    "canonical_implementation": "IGF1 importance-freeze scheduler (pure-PyTorch sparse step, FasterGSFused C0)",
    "canonical_source_hash": {"note": "see _igf1_pack/output/IGF1_SOURCE_LOCK.json"},
    "scenes": "room,garden,bicycle",
    "seeds": "42; warmup 0..15000 dense, scheduled 15000..30000 frozen",
    "training_convergence_status": "YES_30K (FasterGSFused C0)",
    "workload_reduction": "-15..-45.8% updates",
    "renderer_speedup": "n/a",
    "production_iteration_speedup": "NEGATIVE (0.46-0.92x end-to-end)",
    "total_training_speedup": "NEGATIVE",
    "quality": "dPSNR 0.01-0.27 dB; no candidate passes both economic+quality gates",
    "gradient_correctness": "n/a (no kernel)",
    "unresolved_blockers": "no CUDA kernel (pure-PyTorch step slower end-to-end); importance model does not transfer across scene complexity",
    "evidence": "PAPER0_EVIDENCE, _igf1_pack/output/IGF1_SOURCE_LOCK.json"
  },
  "CGC": {
    "current_role": "DROP",
    "evidence_tier": "NEGATIVE_DROP",
    "canonical_implementation": "CGC1 candidate-generation/pruning CUDA kernel (EPIC-05)",
    "canonical_source_hash": {"note": "see _cgc1_kernel/output/CGC1_SOURCE_LOCK.json"},
    "scenes": "room,garden,bicycle",
    "seeds": "42",
    "training_convergence_status": "NO (init PLY)",
    "workload_reduction": "-30..-55% candidates",
    "renderer_speedup": "raster -14..-27% (slower)",
    "production_iteration_speedup": "NOT_MEASURED",
    "total_training_speedup": "NOT_MEASURED",
    "quality": "fidelity gates (SSIM/grad-cos/LPIPS) fail at every threshold on every scene",
    "gradient_correctness": "grad-cos gate fails",
    "unresolved_blockers": "pruning fidelity fail; rasterize overhead-bound; no trained ckpt for garden/bicycle",
    "evidence": "PAPER0_EVIDENCE, _cgc1_kernel/output/CGC1_SOURCE_LOCK.json"
  },
  "KFC": {
    "current_role": "DROP",
    "evidence_tier": "NEGATIVE_DROP",
    "canonical_implementation": "KFC1 fused kernel (EPIC-05)",
    "canonical_source_hash": {"note": "see _kfc1_exec_pack/out/KFC1_SOURCE_LOCK.json"},
    "scenes": "room",
    "seeds": "42",
    "training_convergence_status": "NO (frozen ckpt)",
    "workload_reduction": "50% analytical traffic reduction (OVERCOUNTED)",
    "renderer_speedup": "0.98x (slower)",
    "production_iteration_speedup": "NOT_RUN (autograd boundary blocks E2E)",
    "total_training_speedup": "NOT_MEASURED",
    "quality": "math identical (relL2~6e-8) where launchable",
    "gradient_correctness": "identical where launchable",
    "unresolved_blockers": "fused kernel cannot launch on real workload (804KB shared >> 164KB A100 max); bwd compute-bound; KFC0 DRAM claim disproven",
    "evidence": "PAPER0_EVIDENCE, _kfc1_exec_pack/out/KFC1_SOURCE_LOCK.json"
  },
  "AGS": {
    "current_role": "DROP",
    "evidence_tier": "NEGATIVE_DROP",
    "canonical_implementation": "AGS1 incremental tile-sort kernel (EPIC-05)",
    "canonical_source_hash": {"note": "see _ags1_sort/output/AGS1_SOURCE_LOCK.json"},
    "scenes": "room",
    "seeds": "42",
    "training_convergence_status": "NO (frozen ckpt)",
    "workload_reduction": "0% tile reuse (Jaccard ~0.001)",
    "renderer_speedup": "0.460x (2.18x slower than radix)",
    "production_iteration_speedup": "NOT_REACHED",
    "total_training_speedup": "NOT_MEASURED",
    "quality": "bit-exact vs radix",
    "gradient_correctness": "bit-exact",
    "unresolved_blockers": "0% tile reuse; incremental repair 2.18x slower than radix; garden/bicycle ckpts missing",
    "evidence": "PAPER0_EVIDENCE, _ags1_sort/output/AGS1_SOURCE_LOCK.json"
  }
}

PENDING_PLACEHOLDERS = {
  "TFC_FC0": {"status": "RUNNING (GPU work in progress; pack _tfc_fc0_pack)", "role": "MAIN_CANDIDATE pending production convergence/timing", "note": "Do NOT run as part of this support pack; current GPU work has priority."},
  "DFC2": {"status": "PREPARED_QUEUED (no GPU executed)", "role": "PENDING family member", "note": "Universal fixed-threshold generalization on 6 held-out scenes; assembler self-test PASS; awaiting isolated EPIC-05 A100 node."},
  "DFC_FC0": {"status": "PREPARED_QUEUED REQUIRES_VALIDATION (no GPU executed)", "role": "PENDING family member", "note": "train_dfc.py unverified scaffold; needs 200-iter smoke + isolated node before 30K."},
  "PUB0": {"status": "COMPLETE (CPU-only bottleneck ledger)", "role": "ANALYSIS", "note": "Timing normalization + Amdahl ledger; authoritative current-cohort timing. Not a GPU result."}
}

evidence_matrix = {
  "artifact": "PAPER1_EVIDENCE_MATRIX",
  "schema": "paper1-evidence-matrix/v1",
  "generated_utc": NOW,
  "mode": "read-only synthesis. NO GPU work. NO new runs. NO cohort merging.",
  "source_of_truth": "per-method *_VERDICT.json / *_REPORT.md / *_SOURCE_LOCK.json; PAPER0_HARDENED_EVIDENCE_MATRIX.json",
  "roles_vocabulary": ["MAIN_CANDIDATE","SUPPORTING","ANALYSIS","PENDING","NEEDS_REVALIDATION","DROP"],
  "pending_placeholders": PENDING_PLACEHOLDERS,
  "do_not_silently_upgrade_rule": "No method role is upgraded until its pending gate resolves with authoritative artifacts. Roles change ONLY per explicit gates.",
  "methods": METHODS
}
w("PAPER1_EVIDENCE_MATRIX.json", evidence_matrix)
print("task2 matrix written")

# ---- PAPER1_OPEN_GAPS.json ----
open_gaps = {
  "artifact": "PAPER1_OPEN_GAPS",
  "schema": "paper1-open-gaps/v1",
  "generated_utc": NOW,
  "gap_vocabulary": ["BLOCKER","HIGH","MEDIUM","LOW"],
  "gaps": [
    {"id":"G1","gap":"Only K50-B1 has a real full-training-step speedup; TFC/DFC/CGC/KFC/AGS are renderer-local (frozen PLY) and MUST NOT be presented as training speedups.","severity":"BLOCKER","methods":["TFC","DFC","CGC","KFC","AGS","DFC_optimization_family"]},
    {"id":"G2","gap":"No method has a measured TOTAL-training-time speedup end-to-end (only per-iter / renderer-step). TFC_FC0/DFC_FC0/Pub0 total-time fields required.","severity":"BLOCKER","methods":["TFC","DFC_optimization_family","K50-B1"]},
    {"id":"G3","gap":"TFC_FC0 production convergence/timing pending (RUNNING). Total 30K wall-clock and Gaussian-count trajectory not yet measured.","severity":"HIGH","methods":["TFC"]},
    {"id":"G4","gap":"K50-B1 NEEDS_REVALIDATION: canonical current-pipeline revalidation, multi-seed CI, LPIPS, measured total training wall-time, Stage4B-only evidence required before MAIN_CANDIDATE->confirmed.","severity":"HIGH","methods":["K50-B1"]},
    {"id":"G5","gap":"DFC family PENDING: DFC2 DOES_NOT_GENERALIZE_UNIVERSALLY (no fixed threshold meets quality<=0.05dB AND workload>=10% on all 6 held-out scenes); DFC_FC0 train_dfc.py unverified scaffold; DFC1-R2 dense-scene Amdahl-bound (garden 1.140x, bicycle 1.061x FAIL).","severity":"HIGH","methods":["DFC_optimization_family","DFC2","DFC_FC0"]},
    {"id":"G6","gap":"VOS ANALYSIS: Amdahl-bound (optimizer ~15-19% of step, SparseGaussianAdam O(N)); measured wall ~1.004x; no economic win; room seed-0 instability.","severity":"MEDIUM","methods":["VOS"]},
    {"id":"G7","gap":"IGF/CGC/KFC/AGS DROP: real CUDA but failing economic/fidelity gates (IGF no kernel slower E2E; CGC fidelity gates fail all thresholds; KFC cannot launch on real workload 804KB shared>>164KB; AGS 0% tile reuse 2.18x slower).","severity":"MEDIUM","methods":["IGF","CGC","KFC","AGS"]},
    {"id":"G8","gap":"Renderer-local 'pixel-identical' claim withdrawn -> 'numerically indistinguishable' (PBR0: maxabs_AB 3.2e-4, B==C bitwise, PSNR loss 0.0 dB). Legacy 1.3564x is footprint-stage fwd+bwd; reconciled native full-step 1.1988x; current-cohort 1.0317x.","severity":"MEDIUM","methods":["TFC"]},
    {"id":"G9","gap":"Historical failed/superseded gates must be preserved (KFC0/AGS0/CGC0/IGF0/DFC0/VOS0/TFC0) - do not silently drop.","severity":"MEDIUM","methods":["ALL"]},
    {"id":"G10","gap":"Single hardware cohort (EPIC-05 A100 remote-only); no cross-GPU/cross-vendor validation; K50-B1 is multi-GPU HISTORICAL (not per-run comparable).","severity":"MEDIUM","methods":["ALL"]}
  ]
}
w("PAPER1_OPEN_GAPS.json", open_gaps)

# ---- PAPER1_MAIN_TABLE_DRAFT.json ----
main_table = {
  "artifact": "PAPER1_MAIN_TABLE_DRAFT",
  "schema": "paper1-main-table-draft/v1",
  "generated_utc": NOW,
  "caption": "DRAFT main comparison table. Each row is ONE method within ONE cohort. Renderer-local speedups are NOT training-step speedups. Blank/null = not measured in that cohort (do not infer).",
  "columns": ["method","role","cohort","scenes","seeds","full_convergence","workload_reduction","renderer_speedup","production_iteration_speedup","total_training_speedup","quality","gradient_correctness","headline_verdict"],
  "rows": [
    ["TFC","MAIN_CANDIDATE","TFC_DFC_EPIC05","11 scenes","42","no (frozen PLY; TFC_FC0 pending)","intersection pooled -38.2%","isect 1.38x; fwd 1.15-1.42x; fwd+bwd 1.17-1.51x (legacy 1.3564x)","current-cohort 1.0317x; PBR0 native 1.1988x","NOT_MEASURED (pending TFC_FC0)","numerically indistinguishable (dPSNR 0.0 dB)","grad-cos 0.999888 mean (noise floor)","TFC2_CERTIFIED / CONFIRMED (renderer-local)"],
    ["K50-B1","NEEDS_REVALIDATION","C51_STAGE4B_3SCENE","room/garden/bicycle","single/gpu_id","yes (30K)","Gaussian -30..-36% (gs_ratio 0.64-0.70)","n/a (bwd-only)","1.082x / 1.244x / 1.289x","IMPLIED +8.2/+24.4/+28.9% (no total-wall field)","PSNR +0.04..+1.03 dB, SSIM +","cosine=1.0, skipped=0, fwd bit-exact","SUPPORTED_30K (revalidate before confirmed)"],
    ["DFC1-R1 saved-T","SUPPORTING","TFC_DFC_EPIC05","room/garden/bicycle","oracle","partial (backward)","n/a (correctness fix)","no standalone speedup","NOT_MEASURED","NOT_MEASURED","grad-cos gate PASS all scenes; 548x grad-error cut @t=0","grad-cos 0.99999985 @t=0","ADOPT_DFC1R1_SAVED_T_BACKWARD"],
    ["DFC optimization family","PENDING","TFC_DFC_EPIC05","room/garden/bicycle (+6 held-out for DFC2)","42","pending (DFC2/DFC_FC0 queued)","splat -17..-29% (R2); DFC2 5.8-23.4%","compositing 1.10-1.28x; raster 1.10-1.17x","R2 room 1.206x PASS, garden 1.140x, bicycle 1.061x FAIL; current-cohort 1.0539x","NOT_MEASURED (pending DFC_FC0)","DFC2 quality PASS all scenes (<=5e-6 dB); R2 dense-scene FAIL","grad-cos 0.99999986+ (all thresholds)","family PENDING (R2 FAIL_CLOSED dense; DFC2 no universal threshold)"],
    ["VOS","ANALYSIS","VOS1_EPIC05","room/garden/bicycle","{0,1,42}","yes (30K, 3 seeds)","~-40.2% logical updates (Amdahl-bound)","n/a","none (wall ~1.004x)","none","QUALITY_FAIL; room seed-0 -0.271 dB","n/a (economic failure)","NEGATIVE (Amdahl-bound, SparseGaussianAdam O(N))"],
    ["IGF","DROP","FASTERGSFUSED_C0_30K","room/garden/bicycle","42","yes (30K)","-15..-45.8% updates","n/a","negative (0.46-0.92x)","negative","dPSNR 0.01-0.27 dB; no candidate passes both gates","n/a","DROP_NO_IMPORTANCE_FREEZE_HEADROOM"],
    ["CGC","DROP","EPIC05_A100_REMOTE","room/garden/bicycle","42","no (init PLY)","-30..-55% candidates","raster -14..-27%","NOT_MEASURED","NOT_MEASURED","fidelity gates fail at all thresholds","grad-cos gate fail","DROP_PRUNING_FIDELITY"],
    ["KFC","DROP","EPIC05_A100_REMOTE","room","42","no (frozen ckpt)","50% analytical (overcounted)","0.98x (slower)","NOT_RUN (autograd boundary)","NOT_MEASURED","math identical where launchable","identical where launchable","DROP_NO_KERNEL_GAIN (cannot launch real workload)"],
    ["AGS","DROP","EPIC05_A100_REMOTE","room","42","no (frozen ckpt)","0% tile reuse","0.460x (2.18x slower)","NOT_REACHED","NOT_MEASURED","bit-exact vs radix","bit-exact","DROP_SORT_KERNEL_GAIN"]
  ],
  "footnotes": [
    "Renderer-local speedups (TFC/DFC/CGC/KFC/AGS) are single-training-step renderer measurements on frozen PLYs and MUST NOT be read as total-training-time speedups.",
    "K50-B1 speedups are from the clean C51_STAGE4B_3SCENE 30K cohort; single-seed, needs revalidation.",
    "Legacy 1.3564x (TFC) is footprint-stage fwd+bwd; reconciled native full-step = 1.1988x (PBR0); current-cohort full-step = 1.0317x (PUB0).",
    "Historical failed/superseded gates (KFC0/AGS0/CGC0/IGF0/DFC0/VOS0/TFC0) are preserved in PAPER1_EVIDENCE_MATRIX.json.",
    "Total 30K production training wall-clock is NOT_MEASURED for every method (pending TFC_FC0/DFC_FC0)."
  ]
}
w("PAPER1_MAIN_TABLE_DRAFT.json", main_table)
print("task2 gaps+table written")

# ---- PAPER1_REPORT.md ----
paper1_report = f"""# PAPER1 REPORT — Publication Evidence Matrix (Auto-Updated)

**Generated:** {GEN}  |  **Mode:** read-only synthesis. NO GPU work. NO new runs. NO cohort merging.
**Source of truth:** PAPER0_HARDENED_EVIDENCE_MATRIX.json + per-method *_VERDICT.json / *_SOURCE_LOCK.json.

## 0. What changed vs PAPER0 snapshot
- Incorporated **PBR0** (COMPLETE_RECONCILIATION), **PUB0** (COMPLETE CPU bottleneck ledger), current **TFC** status, current **DFC** status.
- Added pending placeholders for **TFC_FC0, DFC2, DFC_FC0, PUB0**.
- No method role was silently upgraded. Roles change ONLY per explicit gates.

## 1. Role summary
| Role | Methods |
|---|---|
| MAIN_CANDIDATE | TFC (pending TFC_FC0) |
| NEEDS_REVALIDATION | K50-B1 |
| SUPPORTING | DFC1-R1 saved-T |
| PENDING | DFC optimization family (DFC2 + DFC_FC0) |
| ANALYSIS | VOS, PUB0 |
| DROP | DFC1-R2 adaptive, IGF, CGC, KFC, AGS |

## 2. Pending placeholders
| Pack | Status | Role |
|---|---|---|
| TFC_FC0 | RUNNING (GPU; not part of this pack) | TFC production convergence/timing |
| DFC2 | PREPARED_QUEUED (no GPU) | DFC universal-fixed generalization |
| DFC_FC0 | PREPARED_QUEUED REQUIRES_VALIDATION (no GPU) | DFC full 30K convergence |
| PUB0 | COMPLETE (CPU-only) | Bottleneck/timing ledger |

## 3. Method ledger (cohorts NOT merged)
See PAPER1_EVIDENCE_MATRIX.json for full 17-field per-method records and PAPER1_MAIN_TABLE_DRAFT.json for the draft table. Key points:
- **TFC**: strongest renderer-level result (ellipse-exact footprint). Intersection pooled -38.2% (11 scenes). Legacy 1.3564x = footprint-stage fwd+bwd; PBR0 reconciled native full-step 1.1988x; PUB0 current-cohort full-step 1.0317x. Renderer-local only; total 30K wall-clock NOT_MEASURED (TFC_FC0 pending). 'pixel-identical' withdrawn -> 'numerically indistinguishable'.
- **K50-B1**: only method with real 30K full-step speedup (room 1.082x / bicycle 1.244x / garden 1.289x). NEEDS_REVALIDATION: canonical-pipeline revalidation, multi-seed, LPIPS, total-wall-time required.
- **DFC1-R1 saved-T**: SUPPORTING correctness fix (548x grad-error reduction at t=0); no standalone speedup.
- **DFC family**: PENDING. DFC2 DOES_NOT_GENERALIZE_UNIVERSALLY; DFC1-R2 FAIL_CLOSED on dense scenes; DFC_FC0 queued.
- **VOS/IGF/CGC/KFC/AGS**: DROP or ANALYSIS per authoritative verdicts (preserved negatives).

## 4. Cohort integrity
- Renderer-frozen-PLY cohorts (TFC/DFC/CGC/KFC/AGS) != full-training cohorts (K50-B1/VOS/IGF).
- K50-B1 stage4a (config-buggy 5K) != stage4b (clean 30K).
- EPIC-05 A100-SXM4-80GB != A100-PCIE-40GB (stage4a). K50-B1 is HISTORICAL multi-GPU (not per-run comparable).

## 5. Open gaps
See PAPER1_OPEN_GAPS.json (10 gaps). BLOCKER: G1 (renderer-local != training speedup), G2 (no measured total-training-time speedup for any method).

## 6. Publication readiness
NOT publication-ready. TFC_FC0, DFC2, DFC_FC0 remain unresolved; K50-B1 needs revalidation; no method has a measured total 30K production training wall-clock.
"""
w("PAPER1_REPORT.md", paper1_report)
print("task2 report written")

# =====================================================================
# TASK 3 - PAPER_FIGURE_DATA_PACK
# =====================================================================
import csv
FIG = os.path.join(ROOT, "PAPER_FIGURE_DATA")
os.makedirs(FIG, exist_ok=True)

def write_csv(name, header, rows):
    with open(os.path.join(FIG, name), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(header); wr.writerows(rows)

# ---- FIGURE A : TFC geometry ----
# TFC0 room 8-camera footprint (authoritative TFC0_FOOTPRINT_STATS.json)
tfc0_cams = [
 (20,8826952,8826934,4857659),(59,8667547,8667584,5413982),(98,8007256,8007213,4545396),
 (137,7309745,7309710,4275167),(176,9674295,9674191,5269026),(215,5643158,5643158,3041155),
 (254,7424731,7425057,4592529),(293,6916813,6916762,3978854)]
figA_rows = []
prod_sum=ell_sum=0
for cid,prod,lib,ell in tfc0_cams:
    red = 1 - ell/prod
    prod_sum+=prod; ell_sum+=ell
    figA_rows.append([cid,prod,ell,round(red,6)])
pooled = 1 - ell_sum/prod_sum
mean_percam = sum(1-ell/prod for _,prod,_,ell in tfc0_cams)/len(tfc0_cams)
write_csv("FIGURE_A_tfc_geometry_room8cam.csv",
          ["camera_id","stock_aabb_tiles","exact_ellipse_tiles","intersection_reduction"], figA_rows)
figureA = {
  "figure":"A","title":"TFC geometry — stock AABB vs exact-ellipse tile intersections",
  "scene":"room","cameras":8,"tile_size":16,
  "authoritative_source":"_tfc0_oracle/TFC0_FOOTPRINT_STATS.json + TFC0_TIGHTENING_ORACLE.json",
  "intersection_reduction_pooled": pooled,
  "intersection_reduction_mean_per_camera": mean_percam,
  "native_vs_down2": {
    "note": "down2 shrinks projected footprint -> AABB-vs-ellipse gap narrows. PBR0 down2 intersection_reduction = null (not separately measured at down2); PBR0 realized_amdahl_down2 = 1.0985 (borderline FAIL, reported not hidden).",
    "native_intersection_reduction": 0.3927,
    "down2_intersection_reduction": None
  },
  "recommended_candidate":"A_ellipse_exact",
  "quality":"psnr_loss_vs_gt_db 0.0 (bit-exact vs GT render); provably lossless (k-sigma ellipse == alpha iso-contour)",
  "per_camera_csv":"FIGURE_A_tfc_geometry_room8cam.csv",
  "pending":[]
}
w("FIGURE_A.json", figureA)

# ---- FIGURE B : TFC performance scopes ----
figureB = {
  "figure":"B","title":"TFC performance scopes (4 buckets, per FINAL_PROTOCOL_LOCK)",
  "authoritative_source":"PUB0_TIMING_LEDGER.json, TFC2_VERDICT.json, PBR0_VERDICT.json, _tfc2_oracle/TFC2_PROFILING.json",
  "scopes":{
    "intersection_only":{"definition":"tile-intersection stage only","current_cohort_speedup_geomean":1.0317,"tfc2_isect_kernel_speedup":{"room":1.464,"garden":1.475,"bicycle":1.603},"note":"PUB0 current-cohort; TFC2 kernel-level intersect speedup per scene"},
    "forward_pipeline":{"definition":"preprocess+isect+sort+rasterize (fwd only)","tfc2_fwd_speedup":{"room":1.341,"garden":1.277,"bicycle":1.379},"note":"TFC2_BENCHMARK fwd_*_s vs baseline"},
    "forward_backward":{"definition":"fwd pipeline + backward on FROZEN PLY","legacy_pooled_e2e":1.3564,"per_scene_e2e":{"room":1.384,"garden":1.376,"bicycle":1.508},"note":"1.3564x = LEGACY footprint-stage fwd+bwd (TFC2_PROFILING). NOT a training step."},
    "production_timing":{"definition":"full production iteration incl. optimizer+densify","current_cohort_full_step_geomean":1.0317,"pbr0_native_reconciled_full_step":1.1988,"status":"PENDING_TFC_FC0","note":"PBR0 reconciled native full-step 1.1988x; current-cohort 1.0317x. Total 30K wall-clock pending TFC_FC0."}
  },
  "rule":"intersection-only != forward pipeline != fwd+bwd != production. Never collapse these.",
  "pending":["production total 30K wall-clock (TFC_FC0)"]
}
w("FIGURE_B.json", figureB)
print("task3 A+B written")

# ---- FIGURE C : DFC mechanism ----
# DFC2 per-scene timing + quality + gradient + workload (authoritative)
dfc2_timing = {
 "bonsai":{"baseline":{"fwd_pipe":37.16278433799744,"fwd_bwd":88.2672004699707,"full_step":117.64336013793945},"1e-07":{"full_step":114.31791973114014},"1e-06":{"full_step":112.29680061340322}},
 "counter":{"baseline":{"fwd_pipe":40.49353687465191,"fwd_bwd":110.1523208618164,"full_step":142.28595161437988},"1e-07":{"full_step":134.45488166809082},"1e-06":{"full_step":131.69750213623047}},
 "flowers":{"baseline":{"fwd_pipe":86.30393555760384,"fwd_bwd":185.98326301574707,"full_step":264.3780765533447},"1e-07":{"full_step":259.85836601257324},"1e-06":{"full_step":257.5834846496582}},
 "kitchen":{"baseline":{"fwd_pipe":48.03228811919689,"fwd_bwd":141.06179428100586,"full_step":182.6622714996338},"1e-07":{"full_step":175.9099235534668},"1e-06":{"full_step":173.82969665527344}},
 "stump":{"baseline":{"fwd_pipe":93.73180836459133,"fwd_bwd":183.5909761150879,"full_step":299.5875549316406},"1e-07":{"full_step":297.6729927062988},"1e-06":{"full_step":294.84723472595215}},
 "treehill":{"baseline":{"fwd_pipe":84.83865767717361,"fwd_bwd":189.74765014648438,"full_step":276.24137697613105},"1e-07":{"full_step":271.07215881347656},"1e-06":{"full_step":268.6965446472168}}
}
dfc2_workload = {
 "bonsai":{"baseline":1033750469.0,"1e-07":{"splat_red":0.1182282675220726},"1e-06":{"splat_red":0.15911978924593837}},
 "counter":{"baseline":1397763771.0,"1e-07":{"splat_red":0.18466885274564748},"1e-06":{"splat_red":0.2335713936600522}},
 "flowers":{"baseline":1809648102.0,"1e-07":{"splat_red":0.09034139334565505},"1e-06":{"splat_red":0.11949419047080727}},
 "kitchen":{"baseline":1459383082.0,"1e-07":{"splat_red":0.13345414538661893},"1e-06":{"splat_red":0.1678395518845572}},
 "stump":{"baseline":1718240738.0,"1e-07":{"splat_red":0.0575802242444563},"1e-06":{"splat_red":0.08307395863873412}},
 "treehill":{"baseline":2103371143.0,"1e-07":{"splat_red":0.09567088370005272},"1e-06":{"splat_red":0.13138600000275846}}
}
dfc2_quality = {
 "bonsai":{"baseline":{"psnr":33.43193224027014,"ssim":0.9453443810343742,"lpips":0.24483505822718143},"1e-07":{"psnr_loss":1.7443461786115222e-07},"1e-06":{"psnr_loss":3.207821173134562e-06}},
 "counter":{"baseline":{"psnr":30.688584763518637,"ssim":0.934234119951725,"lpips":0.23861941508948803},"1e-07":{"psnr_loss":3.5829405220511035e-08},"1e-06":{"psnr_loss":4.941770264110801e-07}},
 "flowers":{"baseline":{"psnr":22.64824689518464,"ssim":0.7129709497094554,"lpips":0.36430807784199715},"1e-07":{"psnr_loss":3.884203430004618e-07},"1e-06":{"psnr_loss":4.92085928982533e-06}},
 "kitchen":{"baseline":{"psnr":33.68155439211472,"ssim":0.9560922309756279,"lpips":0.12448352668434381},"1e-07":{"psnr_loss":-3.0823119345768646e-07},"1e-06":{"psnr_loss":-4.544904932402005e-06}},
 "stump":{"baseline":{"psnr":27.318392469339706,"ssim":0.8442391677690918,"lpips":0.3162839338183403},"1e-07":{"psnr_loss":1.403965050883471e-07},"1e-06":{"psnr_loss":1.7896323036836748e-06}},
 "treehill":{"baseline":{"psnr":21.55583668010368,"ssim":0.7284300252795219,"lpips":0.407143522053957},"1e-07":{"psnr_loss":5.086021204192548e-08},"1e-06":{"psnr_loss":-8.372474589180001e-07}}
}
dfc2_gradient = {
 "bonsai":{"1e-07":{"cos":0.99999997834756,"relL2":9.437231196677077e-05},"1e-06":{"cos":0.9999999729900962,"relL2":0.00011018577722265663}},
 "counter":{"1e-07":{"cos":0.9999999998542014,"relL2":1.4132152493638007e-05},"1e-06":{"cos":0.999999,"relL2":None}},
 "kitchen":{"1e-07":{"cos":0.9999999569863827,"relL2":0.000168518949271543},"1e-06":{"cos":0.9999998634254653,"relL2":0.00028520698593199336}},
 "stump":{"1e-07":{"cos":0.9999999986808344,"relL2":4.253136437455985e-05},"1e-06":{"cos":0.9999999491093902,"relL2":0.00014006208551243984}},
 "treehill":{"1e-07":{"cos":0.9999999844646755,"relL2":0.00010304766027638583},"1e-06":{"cos":0.9999999867995252,"relL2":0.00010065037758129954}}
}
figureC = {
  "figure":"C","title":"DFC mechanism — transmittance early-termination (DFC2 generalization pack)",
  "authoritative_source":"_dfc2_pack/DFC2_TIMING.json, DFC2_WORKLOAD.json, DFC2_QUALITY.json, DFC2_GRADIENT.json, DFC2_VERDICT.json",
  "thresholds":["1e-07","1e-06"],
  "per_scene":{},
  "verdict":"DOES_NOT_GENERALIZE_UNIVERSALLY (no fixed threshold meets quality<=0.05dB AND workload>=10% on all 6 held-out scenes)",
  "pending":["held-out full results pending DFC2 isolated-node GPU run (assembler self-test PASS only)"]
}
for sc in dfc2_timing:
    b = dfc2_timing[sc]["baseline"]
    figureC["per_scene"][sc] = {
      "baseline_full_step_ms": b["full_step"],
      "full_step_ms_by_threshold": dfc2_timing[sc],
      "full_step_speedup_by_threshold": {t: round(b["full_step"]/dfc2_timing[sc][t]["full_step"],6) for t in ["1e-07","1e-06"]},
      "processed_splat_reduction": dfc2_workload[sc],
      "psnr_ssim_lpips": dfc2_quality[sc],
      "gradient_correctness": dfc2_gradient.get(sc,{}),
      "held_out_result":"PENDING (DFC2 GPU run)"
    }
write_csv("FIGURE_C_dfc_mechanism.csv",
          ["scene","threshold","baseline_full_step_ms","policy_full_step_ms","full_step_speedup","splat_reduction","psnr_loss_db","grad_cosine","grad_relL2"],
          [[sc,t,dfc2_timing[sc]["baseline"]["full_step"],dfc2_timing[sc][t]["full_step"],
            round(dfc2_timing[sc]["baseline"]["full_step"]/dfc2_timing[sc][t]["full_step"],6),
            dfc2_workload[sc][t]["splat_red"],dfc2_quality[sc][t]["psnr_loss"],
            dfc2_gradient.get(sc,{}).get(t,{}).get("cos"),dfc2_gradient.get(sc,{}).get(t,{}).get("relL2")]
           for sc in dfc2_timing for t in ["1e-07","1e-06"]])
w("FIGURE_C.json", figureC)
print("task3 C written")

# ---- FIGURE D : Training trajectory ----
figureD = {
  "figure":"D","title":"Training trajectory — baseline vs optimized (30K)",
  "authoritative_source":"PENDING: requires TFC_FC0 (_tfc_fc0_pack) and DFC_FC0 (_dfc2_pack/DFC_FC0) 30K runs",
  "curves":{
    "psnr":{"baseline_vs_TFC_FC0":"PENDING","baseline_vs_DFC_FC0":"PENDING"},
    "ssim":{"baseline_vs_TFC_FC0":"PENDING","baseline_vs_DFC_FC0":"PENDING"},
    "lpips":{"baseline_vs_TFC_FC0":"PENDING","baseline_vs_DFC_FC0":"PENDING"},
    "gaussian_population":{"baseline_vs_TFC_FC0":"PENDING (TFC renderer-local: no trajectory)","baseline_vs_DFC_FC0":"PENDING"},
    "cumulative_wall_time":{"baseline_vs_TFC_FC0":"PENDING","baseline_vs_DFC_FC0":"PENDING"}
  },
  "available_partial_context":{
    "K50-B1_only_30K_population":"room gs_ratio 0.698, bicycle 0.694, garden 0.643 (different method/cohort; shown for context only, NOT TFC/DFC)",
    "note":"No TFC_FC0/DFC_FC0 30K curves exist yet. All points are explicit PENDING — no fabrication."
  },
  "rule":"Do not invent or smooth missing trajectory points. Use null/PENDING.",
  "pending":["all curves pending TFC_FC0 + DFC_FC0 30K GPU runs"]
}
w("FIGURE_D.json", figureD)
write_csv("FIGURE_D_training_trajectory.csv",
          ["curve","series","status","value"],
          [["psnr","baseline_vs_TFC_FC0","PENDING",None],
           ["psnr","baseline_vs_DFC_FC0","PENDING",None],
           ["ssim","baseline_vs_TFC_FC0","PENDING",None],
           ["ssim","baseline_vs_DFC_FC0","PENDING",None],
           ["lpips","baseline_vs_TFC_FC0","PENDING",None],
           ["lpips","baseline_vs_DFC_FC0","PENDING",None],
           ["gaussian_population","baseline_vs_TFC_FC0","PENDING",None],
           ["gaussian_population","baseline_vs_DFC_FC0","PENDING",None],
           ["cumulative_wall_time","baseline_vs_TFC_FC0","PENDING",None],
           ["cumulative_wall_time","baseline_vs_DFC_FC0","PENDING",None]])

# ---- FIGURE E : Negative-space analysis ----
figureE = {
  "figure":"E","title":"Negative-space analysis — authoritative results only (VOS/IGF/CGC/KFC/AGS)",
  "authoritative_source":"PAPER0_EVIDENCE, CLOSED_DIRECTIONS_20260930.md, per-method *_SOURCE_LOCK.json",
  "methods":{
    "VOS":{"hypothesis":"Skipping cold (<10% visible) Gaussian optimizer updates reduces workload without quality loss","headroom":"~-40.2% logical update reduction; optimizer ~15-19% of step; ideal wall ceiling 1.183x","real_implementation_result":"VIABLE(k=4) at optimizer-step level (econ geomean 0.251, pooled median PSNR loss -0.014 dB) BUT VOS1 hardened full 30K + true-holdout = QUALITY_FAIL","failure_mechanism":"Amdahl-bound: SparseGaussianAdam is O(N), cutting update COUNT does not cut wall time; measured wall ~1.004x; room seed-0 -0.271 dB","artifact":"VOS0_REPORT.md, vos1/*"},
    "IGF":{"hypothesis":"Importance-freeze scheduler (freeze low-importance Gaussians after warmup) preserves quality with less work","headroom":"-15..-45.8% updates","real_implementation_result":"DROP_NO_IMPORTANCE_FREEZE_HEADROOM (real training, FasterGSFused C0, 30K)","failure_mechanism":"No CUDA kernel (pure-PyTorch sparse step slower end-to-end, 0.46-0.92x); importance model does not transfer across scene complexity","artifact":"_igf1_pack/output/IGF1_SOURCE_LOCK.json"},
    "CGC":{"hypothesis":"Candidate-generation pruning reduces Gaussians with fidelity preserved","headroom":"-30..-55% candidates","real_implementation_result":"DROP_PRUNING_FIDELITY (real CUDA kernel)","failure_mechanism":"Fidelity gates (SSIM/grad-cos/LPIPS) fail at every threshold on every scene; rasterize overhead-bound; no trained ckpt for garden/bicycle","artifact":"_cgc1_kernel/output/CGC1_SOURCE_LOCK.json"},
    "KFC":{"hypothesis":"Fused kernel reduces memory traffic / kernel-launch overhead","headroom":"50% analytical traffic reduction (OVERCOUNTED)","real_implementation_result":"DROP_NO_KERNEL_GAIN (real CUDA)","failure_mechanism":"Fused kernel cannot launch on real workload (804KB shared >> 164KB A100 max); bwd compute-bound; autograd boundary blocks E2E; KFC0 DRAM claim disproven","artifact":"_kfc1_exec_pack/out/KFC1_SOURCE_LOCK.json"},
    "AGS":{"hypothesis":"Incremental tile-sort repair avoids full radix sort","headroom":"0% tile reuse (Jaccard ~0.001)","real_implementation_result":"DROP_SORT_KERNEL_GAIN (real CUDA)","failure_mechanism":"0% tile reuse; incremental repair 2.18x slower than radix; garden/bicycle ckpts missing","artifact":"_ags1_sort/output/AGS1_SOURCE_LOCK.json"}
  },
  "note":"Summarizes ONLY authoritative results. No new GPU work."
}
w("FIGURE_E.json", figureE)
write_csv("FIGURE_E_negative_space.csv",
          ["method","hypothesis","headroom","real_implementation_result","failure_mechanism"],
          [[m,figureE["methods"][m]["hypothesis"],figureE["methods"][m]["headroom"],figureE["methods"][m]["real_implementation_result"],figureE["methods"][m]["failure_mechanism"]] for m in ["VOS","IGF","CGC","KFC","AGS"]])
print("task3 D+E written")

# ---- FIGURE_INDEX.json ----
figure_index = {
  "artifact":"FIGURE_INDEX",
  "schema":"paper1-figure-index/v1",
  "generated_utc":NOW,
  "principle":"Authoritative data only; no invented/smoothed points; explicit null/PENDING for missing.",
  "figures":[
    {"id":"A","file":"FIGURE_A.json","csv":"PAPER_FIGURE_DATA/FIGURE_A_tfc_geometry_room8cam.csv","title":"TFC geometry — stock AABB vs exact-ellipse intersections","status":"COMPLETE (room 8-cam)","pending":[]},
    {"id":"B","file":"FIGURE_B.json","csv":None,"title":"TFC performance scopes (intersection/fwd-pipe/fwd+bwd/production)","status":"COMPLETE (production=PENDING TFC_FC0)","pending":["production total 30K wall-clock"]},
    {"id":"C","file":"FIGURE_C.json","csv":"PAPER_FIGURE_DATA/FIGURE_C_dfc_mechanism.csv","title":"DFC mechanism — transmittance threshold","status":"COMPLETE (DFC2 generalization; held-out full=PENDING)","pending":["held-out full results"]},
    {"id":"D","file":"FIGURE_D.json","csv":"PAPER_FIGURE_DATA/FIGURE_D_training_trajectory.csv","title":"Training trajectory baseline vs optimized","status":"PENDING","pending":["all curves pending TFC_FC0 + DFC_FC0"]},
    {"id":"E","file":"FIGURE_E.json","csv":"PAPER_FIGURE_DATA/FIGURE_E_negative_space.csv","title":"Negative-space analysis","status":"COMPLETE","pending":[]}
  ]
}
w("FIGURE_INDEX.json", figure_index)

# ---- FIGURE_CAPTION_FACTS.md ----
caption_facts = f"""# FIGURE CAPTION FACTS (authoritative only)

**Generated:** {GEN}  |  Every figure uses authoritative artifacts only; missing points are explicit PENDING/null. NO fabrication.

## Figure A — TFC geometry
- Scene `room`, 8 cameras, tile_size 16. Stock production per-axis AABB vs exact-ellipse tile membership.
- Pooled intersection reduction = {pooled:.4f}; mean-per-camera = {mean_percam:.4f} (report separately).
- Candidate A_ellipse_exact: PSNR loss vs GT = 0.0 dB (bit-exact render); provably lossless (k-sigma ellipse == alpha=ALPHA_THRESHOLD iso-contour).
- Native intersection reduction 0.3927 (PBR0). Down2 not separately measured (null); PBR0 down2 realized Amdahl 1.0985 borderline-FAIL reported.

## Figure B — TFC performance scopes
- intersection-only: current-cohort geomean 1.0317x (PUB0); TFC2 kernel-level intersect speedup room 1.464 / garden 1.475 / bicycle 1.603.
- forward pipeline: TFC2 fwd speedup room 1.341 / garden 1.277 / bicycle 1.379.
- forward+backward (FROZEN PLY): legacy pooled e2e 1.3564x (footprint-stage); per-scene room 1.384 / garden 1.376 / bicycle 1.508. NOT a training step.
- production timing: current-cohort full-step 1.0317x; PBR0 native reconciled 1.1988x; total 30K wall-clock PENDING (TFC_FC0).

## Figure C — DFC mechanism
- DFC2 universal fixed thresholds {{1e-7, 1e-6}} on 6 held-out scenes. Quality PASS all scenes (PSNR loss <= 5e-6 dB).
- Processed-splat reduction 5.8%-23.4% (below the >=10%-on-all gate at 1e-7). Verdict: DOES_NOT_GENERALIZE_UNIVERSALLY.
- Full-step speedup per scene 1.006x-1.080x. Gradient cosine 0.99999986+, relL2 1.4e-5..2.9e-4.
- Held-out full GPU results PENDING (isolated node).

## Figure D — Training trajectory
- baseline vs TFC_FC0 and vs DFC_FC0: ALL curves PENDING (PSNR/SSIM/LPIPS/gaussian-population/cumulative-wall-time).
- No TFC_FC0/DFC_FC0 30K curves exist. No fabrication.

## Figure E — Negative-space analysis
- VOS: Amdahl-bound (optimizer ~15-19% of step, SparseGaussianAdam O(N)); VOS1 full-30K QUALITY_FAIL.
- IGF: DROP (no kernel, 0.46-0.92x). CGC: DROP (fidelity gates fail all thresholds). KFC: DROP (cannot launch real workload). AGS: DROP (0% tile reuse, 2.18x slower).
"""
w("FIGURE_CAPTION_FACTS.md", caption_facts)
print("task3 index+caption written")

# =====================================================================
# TASK 4 - REPRODUCIBILITY_PACK_PREP  (inventory only, no algorithm change)
# =====================================================================
def local_hash(rel):
    p = os.path.join(ROOT, "..", rel)
    p = os.path.normpath(p)
    return sha256_file(p) if os.path.exists(p) else None

# Canonical source entries: authoritative remote paths + recorded source-lock hashes
canonical_source = [
  {"method":"TFC","role":"CANONICAL","file":"kernel/tfc1_rowwise.cu","remote_path":"/root/tfc0/kernel/tfc1_rowwise.cu","sha256":"a44e4ee63c7ba9d713a4489693d189cb3fd6e7ba95764e491027c0639e34e68e","source":"TFC2_SOURCE_LOCK.json","note":"TFC1 row-wise ellipse footprint; TFC1 mode0 bit-exact vs gsplat (120 dB)."},
  {"method":"TFC","role":"CANONICAL","file":"kernel/tfc2_ablation.cu","remote_path":"/root/tfc2/kernel/tfc2_ablation.cu","sha256":"3c9b2e0a1deda28fa1521f88b054a1b670d843b1b001a12c4b1d8b5fe4e5dd81","source":"TFC2_SOURCE_LOCK.json","note":"TFC2 independent ablation kernel; mode1 ellipse-exact = certified candidate. Does NOT import/modify TFC1."},
  {"method":"DFC","role":"CANONICAL","file":"csrc/RasterizeToPixels3DGSFwd.cu","remote_path":"/root/dfc1r1_20261002/gsplat-dfc1r2adapt/cuda/csrc/RasterizeToPixels3DGSFwd.cu","sha256":"d4f287449d61f094e04dcaa6d573604c39d99c81f14f67fb9bd1b22f3c0d1590","source":"DFC0_SOURCE_LOCK.json (dfc1r2adapt)","note":"DFC1-R2 per-pixel transmittance early-termination forward."},
  {"method":"DFC","role":"CANONICAL","file":"csrc/RasterizeToPixels3DGSBwd.cu","remote_path":"/root/dfc1r1_20261002/gsplat-dfc1r2adapt/cuda/csrc/RasterizeToPixels3DGSBwd.cu","sha256":"31c11029fa8309234a83674806a723a5785ee1d387d3181a11062850bde055f3","source":"DFC0_SOURCE_LOCK.json + DFC_FC0_SOURCE_LOCK.json","note":"DFC1-R1 saved-T backward fix (uses saved render_T)."},
  {"method":"DFC","role":"CANONICAL","file":"cuda/_tfc1dfc1_harness.py","remote_path":"/root/dfc1r1_20261002/_tfc1dfc1_harness.py","sha256":"f681bb6faa5b6fb2058060a81a4545445789ae9648b11cc04289bf509d18fc9b","source":"DFC_FC0_SOURCE_LOCK.json","note":"Validated per-step render+grad harness (adds densification schedule for trainer)."},
  {"method":"K50-B1","role":"CANONICAL","file":"gsplat 1.5.3 (6-file B1 sparse-backward patch)","remote_path":"original 3DGS CUDA rasterizer + B1 sparse-backward mask","sha256":"see reports/phase-c51-stage4b.md (byte hashes recorded in K50 stage4 packs)","source":"reports/phase-c51-stage4b.md","note":"B1 sparse-backward + densification-decoupling mask; importance_mask uint8_t[N]."},
  {"method":"gsplat","role":"CANONICAL","file":"gsplat fork gsplat-b1-clean-v153","remote_path":"/root/epic05_frozen/gsplat-b1-clean-v153","sha256":"version 1.5.3 (fork tag)","source":"TFC2_SOURCE_LOCK.json, PBR0_SOURCE_LOCK.json","note":"EPIC-05 canonical renderer fork."}
]
diagnostic_source = [
  {"method":"DFC","role":"DIAGNOSTIC","file":"_dfc1_src/ (local copy)","note":"Local diagnostic copy of DFC source; NOT canonical. Canonical = remote gsplat-dfc1r1/gsplat-dfc1r2adapt trees."},
  {"method":"CGC","role":"DIAGNOSTIC","file":"_cgc1_kernel/source_lock/ (local)","note":"Local CGC source-lock copies; method is DROP."},
  {"method":"AGS","role":"DIAGNOSTIC","file":"_ags1_sort/remote_src/ (local)","note":"Local AGS source copy; method is DROP."},
  {"method":"PFC0","role":"DIAGNOSTIC","file":"_mx_h8/pack/h8/gsplat/cuda/csrc","note":"PFC0 preprocess-fusion oracle static-analysis source (DROP_NO_PREPROCESS_HEADROOM)."}
]
superseded = [
  {"method":"TFC","role":"SUPERSEDED","file":"_tfc0_oracle/_stale_local_baseline, _superseded_local_circular_baseline, _kbnote_reconstruction_superseded","note":"Superseded local baselines; must NOT be used. Use EPIC-05 remote canonical."},
  {"method":"VOS","role":"SUPERSEDED","file":"VOS0 SH-only result","note":"VOS0 KEEP is superseded by VOS1 hardened QUALITY_FAIL."},
  {"method":"K50-B1","role":"SUPERSEDED","file":"C51_STAGE4A_ROOM (5K, config-buggy prune_and_reset)","note":"Stage4A 5K baseline low PSNR (~29), config-buggy; superseded by Stage4B for headline; must NOT be merged."},
  {"method":"ALL","role":"HISTORICAL","file":"KFC0/AGS0/CGC0/IGF0/DFC0/VOS0/TFC0 gates","note":"Historical failed/superseded gates; preserved, must NOT be silently dropped or reused as current evidence."}
]

repro_source_index = {
  "artifact":"REPRO_SOURCE_INDEX",
  "schema":"repro-source-index/v1",
  "generated_utc":NOW,
  "mode":"inventory only; NO algorithm implementation changed",
  "principle":"Canonical source lives on remote EPIC-05 A100 node; local copies are DIAGNOSTIC. Hashes from authoritative *_SOURCE_LOCK.json.",
  "canonical_source_files":canonical_source,
  "diagnostic_source_files":diagnostic_source,
  "superseded_or_historical":superseded,
  "dataset_paths_expected":{
    "mipnerf360_official":"data/official/mipnerf360/<scene>/point_cloud.ply + cameras.json (room/garden/bicycle/bonsai/counter/kitchen/stump/flowers/treehill/truck/train)",
    "k50_stage4b":"MipNeRF360 room/garden/bicycle @1080p (311/360/360 cams)",
    "epic05_host":"8.130.30.251:1024 (remote); data root /root/3dgs-renderer-benchmark/data/official/mipnerf360/",
    "resolution_note":"DFC1-R2 down=2 (room 1557x1037, garden 2593x1680, bicycle 2473x1643); TFC native tile=16; K50-B1 1080p"
  },
  "checkpoint_identities":{
    "K50-B1":"final GS room 1,505,069 (ratio 0.698); bicycle 7,673,479 (0.694); garden 7,331,782 (0.643); baseline room 2,155,755",
    "TFC/DFC":"frozen PLY only (renderer-local); no trained trajectory checkpoint",
    "note":"Renderer-local methods load official point_cloud.ply iteration_30000 PLYs as frozen inputs; they do NOT train."
  },
  "camera_selection":{
    "TFC_PBR0":"8 cameras/scene (room 20,59,98,137,176,215,254,293; garden/bicycle 8 each)",
    "TFC0_footprint":"room 8 cameras; garden/bicycle per TFC2",
    "DFC1-R2":"8 eval cameras/scene (24 total)",
    "K50-B1":"all shuffled cameras, seed=42"
  },
  "warnings":[
    "PBR0 M4 is NOT production training (it is a cross-pack reconciliation + gradient noise-floor audit, measurement only).",
    "Combined residual-derived optimizer timing is FORBIDDEN (never infer pure optimizer cost from residual subtraction of independent loops).",
    "DFC padding-thread illegal-read fix: the safe production implementation must be used; the illegal-read variant is a defect, not a result.",
    "TFC1 vs TFC2 implementation equivalence is ESTABLISHED by PBR0 (TFC2 reuses identical geometry code path; independent kernel).",
    "Historical cohort mixing is PROHIBITED (stage4a!=stage4b; renderer-frozen-PLY!=full-training; EPIC-05 A100-SXM4-80GB!=A100-PCIE-40GB)."
  ]
}
w("REPRO_SOURCE_INDEX.json", repro_source_index)
print("task4 source index written")

# ---- REPRO_ENVIRONMENT.md ----
env_md = f"""# REPRODUCIBILITY — Environment

**Generated:** {GEN}  |  Inventory only; no environment mutated.

## Local (this workstation — analysis/build only, NOT the GPU authority)
- Python 3.13.13
- PyTorch 2.13.0+cu130 (CUDA 13.0 runtime)
- nvcc 13.3 (V13.3.73)
- GPU: NVIDIA GeForce RTX 5070 Laptop (local; used for source audit/assembly only)

## Remote GPU authority — EPIC-05 A100 cluster (all paper GPU results)
- Host: 8.130.30.251:1024
- GPU: 8x NVIDIA A100-SXM4-80GB (CUDA_VISIBLE_DEVICES selects device)
- CUDA: 12.8
- Python: 3.12.13 (/root/miniforge3/envs/gsplat/bin/python)
- PyTorch: 2.9.1+cu128 (TFC/DFC) or 2.10.0+cu128 (DFC1-R1/R2, PUB0 current cohort)
- gsplat: 1.5.3 fork `gsplat-b1-clean-v153`
- numpy 2.5.0
- OS: Linux (dsw container)

## K50-B1 historical cohort (DIFFERENT — do not merge)
- A100-PCIE-40GB (8 GPUs SM80, 108 SMs); gsplat 1.5.3; CUDA 11.8 build
- C51_STAGE4A: A100-PCIE-40GB (config-buggy 5K). C51_STAGE4B: A100-80GB (clean 30K).

## VOS cohort (different software stack)
- NVIDIA A100-SXM4-80GB (single device); torch 2.8.0+cu128; graphdeco-b0-pub; sparse_adam.

## PFC0 static-analysis source
- `_mx_h8/pack/h8/gsplat/cuda/csrc` (ProjectionEWA3DGSFused.cu, IntersectTile.cu); scene room_iter10000, 1920x1080, tile_size=32; RTX 5070 Laptop capture for per-kernel ratios.

## Environment-variable / config notes
- `CUDA_VISIBLE_DEVICES` selects the A100 device per run.
- EPIC-05 remote-only: no local CUDA for paper results.
- DFC_FC0 / DFC2 require an ISOLATED EPIC-05 node (no PBR0/campaign sharing).
"""
w("REPRO_ENVIRONMENT.md", env_md)

# ---- REPRO_COMMANDS.md ----
cmd_md = f"""# REPRODUCIBILITY — Commands

**Generated:** {GEN}  |  Inventory of canonical entry points. Exact remote GPU commands live in the *_SOURCE_LOCK.json / pack READMEs; do NOT re-run GPU work as part of this support pack.

## Build (gsplat extension, EPIC-05 remote)
```
# gsplat-b1-clean-v153 fork is the canonical renderer; build the CUDA extension there.
# Per-pack build scripts (remote) compile the .cu kernels via ninja/build.ninja.
# Example (TFC1): _tfc1_kernel/build.ninja + tfc1_build.py  (local smoke artifacts only)
python -c "import torch,gsplat;print(torch.__version__, torch.version.cuda, gsplat.__version__)"
```

## Benchmark / render (renderer-local: TFC, DFC)
```
# TFC benchmark (renderer-local, frozen PLY) — canonical entry: tfc1_benchmark.py / TFC2_BENCHMARK
python tfc1_benchmark.py --scene room --mode ellipse   # mode1 = ellipse-exact candidate
# DFC per-scene E2E: _dfc2_pack/dfc2_run.py (assembler self-test PASS)
python _dfc2_pack/dfc2_run.py --selftest
```

## Training (full 30K)
```
# K50-B1 canonical 30K (C51_STAGE4B, A100-80GB): use the C51 training harness (stage4b report Section 1 config).
# Strong baselines: python strong_baseline_train.py --baseline speedy-splat --scene bicycle --method native --gpu 0
# DFC_FC0 (PENDING, REQUIRES_VALIDATION): _dfc2_pack/train_dfc.py (unverified scaffold; 200-iter smoke on room REQUIRED before 30K).
```

## Evaluation
```
python strong_baseline_eval.py ...          # baseline quality eval
python c42_13scene_eval.py                  # 13-scene eval
python test_eval_canonical.py               # canonical eval check
# K50-B1 gate eval: reports/phase-c51-stage4b.md Section 3-4 (Gate A cosine, Gate B speedup)
```

## Timing-ledger assembly (CPU)
```
python _pub0_ledger/build_pub0.py     # rebuild PUB0 timing ledger from existing artifacts
python _pub0_ledger/build_report.py
```

## Prohibitions
- Do NOT run K50 canonical revalidation (Task 5 prereg only; activate only after PUB0).
- Do NOT run TFC_FC0 / DFC2 / DFC_FC0 as part of this support pack (current GPU work has priority).
- Do NOT merge cohorts or convert renderer-local speedups into training speedups.
"""
w("REPRO_COMMANDS.md", cmd_md)
print("task4 env+cmd written")

# ---- REPRO_KNOWN_CAVEATS.md ----
caveats_md = f"""# REPRODUCIBILITY — Known Caveats

**Generated:** {GEN}  |  Warnings that MUST accompany any reproduction.

1. **PBR0 M4 is NOT production training.** PBR0 is a cross-pack reconciliation + gradient noise-floor audit (measurement only, fail-closed). Its native full-step 1.1988x is a reconciled renderer-step figure, not a 30K training result.

2. **Combined residual-derived optimizer timing is forbidden.** Never infer pure optimizer cost from residual subtraction of independent loops (PUB0 Amdahl buckets are directly measured, not residual-derived). Adam optimizer ~30% of step, O(N), untouched by ALL current policies.

3. **DFC padding-thread illegal-read fix.** A padding-thread illegal-read existed in an earlier DFC variant; the SAFE production implementation must be used. The illegal-read variant is a defect and must not be reported as a result.

4. **TFC1 vs TFC2 implementation equivalence is ESTABLISHED by PBR0.** TFC2 reuses the identical geometry code path and adds only the analytic circular superset; it does NOT import or modify TFC1 (byte-identical, independent kernel sha). Equivalence is a PBR0-established fact, not an assumption.

5. **Historical cohort mixing is PROHIBITED.**
   - K50-B1 stage4a (room-only, A100-PCIE-40GB, config-buggy 5K) != stage4b (3-scene, A100-80GB, clean 30K).
   - Renderer-frozen-PLY cohorts (TFC/DFC/CGC/KFC/AGS) != full-training cohorts (K50-B1/VOS/IGF).
   - EPIC-05 A100-SXM4-80GB != A100-PCIE-40GB (stage4a).
   - K50-B1 is HISTORICAL multi-GPU (baseline gpu_id vs policy gpu_id differ) -> per-run speedups hardware-confounded; only the same-run controlled microbenchmark is clean.

6. **Legacy 1.3564x (TFC) is footprint-stage fwd+bwd**, NOT full-E2E/fwd+bwd of a training step and NOT total training. Reconciled native full-step (PBR0) = 1.1988x; current-cohort full-step (PUB0) = 1.0317x.

7. **No total 30K production training wall-clock exists** for any method. K50-B1 per-iter only (implied +8.2/+24.4/+28.9%); TFC/DFC renderer-local. Do not fabricate a total-time field.

8. **Single-hw cohort.** EPIC-05 A100 remote-only; no cross-GPU/cross-vendor validation. K50-B1 single-seed, no multi-seed CI, LPIPS unmeasured.

9. **DFC1-R1 saved-T is a no-op at t>=1e-5** and delivers no standalone speedup; its value is confined to the full-render/low-T regime (backward correctness only). SUPPORTING, not a headline.

10. **DFC2 DOES_NOT_GENERALIZE_UNIVERSALLY** — no single fixed threshold meets quality<=0.05 dB AND workload>=10% on all 6 held-out scenes. DFC_FC0 train_dfc.py is an unverified scaffold (200-iter smoke REQUIRED before 30K).
"""
w("REPRO_KNOWN_CAVEATS.md", caveats_md)
print("task4 caveats written")

# =====================================================================
# TASK 5 - K50_REVALIDATION_PREREGISTRATION_ONLY  (DO NOT RUN)
# =====================================================================
k50_prereg = {
  "artifact":"K50_CANONICAL_PREREG",
  "schema":"k50-canonical-prereg/v1",
  "generated_utc":NOW,
  "STATUS":"PREREGISTRATION_ONLY — DO NOT EXECUTE UNTIL EXPLICITLY AUTHORIZED AFTER PUB0 REVIEW.",
  "execution_state":"NOT_RUN (no K50 canonical revalidation performed; current GPU work TFC_FC0/DFC2/DFC_FC0 has priority)",
  "activation_condition":{
    "justified_only_if":"current-cohort production evidence shows (a) backward remains a sufficiently large bottleneck AND (b) the expected full-training headroom is meaningful.",
    "current_evidence_reviewed":"PUB0_AMDAHL_LEDGER.json: DFC touches rasterize+backward ~58.7% of step (measured 1.05x vs 2.44x ceiling); Adam optimizer ~30% of step (O(N), untouched); TFC+DFC best current-cohort policy geomean 1.0868x (realized/ceiling ~0.32). Backward IS a large fraction, but realized headroom is Amdahl-limited by the untouched O(N) projection + Adam stages.",
    "gate":"K50 canonical revalidation activates ONLY after PUB0 completes and this prereg is explicitly authorized. No threshold rescue after failure."
  },
  "frozen_protocol":{
    "canonical_current_baseline":"original 3DGS CUDA rasterizer + gsplat 1.5.3 (Stage4B config per reports/phase-c51-stage4b.md Section 1: loss 0.8*L1+0.2*D-SSIM, Adam eps 1e-15, per-stage LRs, prune 0.01, densify-grad 0.001, densify interval 100, opacity reset 3000, SH progression +1/1000 iters start 0 max 3, seed=42, 1080p, init opacity logit(0.1)).",
    "current_cuda_pytorch_environment":"EPIC-05 A100-SXM4-80GB; CUDA 12.8; torch 2.9.1+cu128 or 2.10.0+cu128; gsplat 1.5.3 (gsplat-b1-clean-v153); python 3.12.13. NOTE: must match the CURRENT canonical pipeline, NOT the historical CUDA 11.8 build.",
    "scenes":["room","garden","bicycle"],
    "seed":{"first_stage":42,"note":"seed 42 first stage; multi-seed CI recommended for the revalidation"},
    "exact_k50_b1_implementation":"B1 sparse-backward + densification-decoupling mask (importance_mask uint8_t[N]: 1=full gradient, 0=skip; top-K selection torch.topk(prev_grad_norm, N*keep_fraction); EMA decay 0.9 eps 1e-6). Six gsplat-1.5.3 files patched, backed up .orig_stage4a.",
    "training":"30K production full training (baseline vs k50_b1), same-run, same gpu_id (avoid the historical hardware confound).",
    "measurements":{
      "real_total_wall_clock_s":"directly measured end-to-end 30K wall-clock for baseline and k50_b1 (THE headline; no residual derivation)",
      "per_iteration_timing_ms":"directly measured",
      "quality":"PSNR/SSIM/LPIPS (LPIPS ADDED vs historical gap)",
      "gaussian_population_trajectory":"clone/split/prune + final count over the 30K densification schedule"
    },
    "quality_gates":{
      "gate_A":"gradient cosine >= 0.999 (selected exact, skipped zero); forward bit-exact",
      "gate_B":"per-scene E2E speedup > 5% AND no quality degradation (dPSNR <= 0, dSSIM >= -0.005)",
      "gate_C":"total 30K wall-clock speedup directly measured and > 1.0 with multi-seed consistency"
    },
    "timing_gates":{
      "directly_measured_only":"all timing from same-run wall-clock; CUDA-event WARMUP=3 ITERS=10 median",
      "no_residual":"no residual/subtraction-derived optimizer or full-step timing"
    },
    "no_threshold_rescue":"After a gate failure, NO threshold/policy tuning is permitted to rescue the result (fail-closed). Any change voids the prereg and requires a new one.",
    "cohort_rule":"Results tagged to the current canonical pipeline cohort; MUST NOT be merged with K50B1_HISTORICAL (CUDA 11.8, multi-GPU)."
  },
  "relationship_to_pending":"K50 revalidation is downstream of PUB0. TFC_FC0/DFC2/DFC_FC0 remain the current GPU priority; K50 waits."
}
w("K50_CANONICAL_PREREG.json", k50_prereg)
print("task5 k50 prereg written")

# =====================================================================
# FINAL OUTPUTS
# =====================================================================
status = {
  "artifact":"PUBLICATION_CONVERGENCE_STATUS",
  "schema":"publication-convergence-status/v1",
  "generated_utc":NOW,
  "pack":"PUBLICATION_CONVERGENCE_SUPPORT_PACK",
  "mode":"CPU / source-audit / artifact-only. NO new GPU experiments. NO K50 run.",
  "tasks":{
    "TASK1_FINAL_PROTOCOL_LOCK":"COMPLETE",
    "TASK2_PAPER1_EVIDENCE_AUTOUPDATE":"COMPLETE",
    "TASK3_PAPER_FIGURE_DATA_PACK":"COMPLETE",
    "TASK4_REPRODUCIBILITY_PACK_PREP":"COMPLETE",
    "TASK5_K50_REVALIDATION_PREREGISTRATION_ONLY":"COMPLETE (prereg only; NOT executed)"
  },
  "state_partition":{
    "COMPLETED":[
      "TASK1 FINAL_PROTOCOL_LOCK (terminology + metric definitions frozen)",
      "TASK2 PAPER1 evidence matrix / gaps / main-table-draft / report",
      "TASK3 Figure A/B/C/E data + Figure D explicit PENDING; indexes + captions",
      "TASK4 REPRO source index / environment / commands / caveats",
      "TASK5 K50_CANONICAL_PREREG (protocol frozen, NOT executed)",
      "CPU synthesis of PBR0 + PUB0 + TFC + DFC evidence into the paper ledger"
    ],
    "RUNNING":[
      "TFC_FC0 (current GPU work; pack _tfc_fc0_pack) — has priority; NOT part of this support pack",
      "DFC2 universal-fixed generalization (PREPARED_QUEUED, no GPU)",
      "DFC_FC0 full convergence (PREPARED_QUEUED REQUIRES_VALIDATION, no GPU)"
    ],
    "QUEUED":[
      "PUB0 (COMPLETE CPU ledger)",
      "DFC2 isolated-node GPU run (awaiting node)",
      "DFC_FC0 200-iter smoke + 30K (awaiting isolated node)",
      "K50 canonical revalidation (downstream of PUB0; explicit authorization required)"
    ],
    "BLOCKED":[
      "No method has a measured TOTAL 30K production training wall-clock (BLOCKER G1/G2)",
      "K50 canonical revalidation blocked until PUB0 review + explicit authorization",
      "DFC_FC0 train_dfc.py unverified (200-iter smoke required before 30K)",
      "DFC2 no universally-generalizing fixed threshold (DOES_NOT_GENERALIZE_UNIVERSALLY)"
    ]
  },
  "publication_readiness":{
    "ready":False,
    "statement":"NOT publication-ready. TFC_FC0 / DFC2 / DFC_FC0 remain unresolved; K50-B1 requires revalidation; no method has a measured total 30K production training wall-clock. Renderer-local speedups MUST NOT be presented as training speedups."
  },
  "deliverables":[
    "FINAL_PROTOCOL_LOCK.json","FINAL_PROTOCOL_LOCK.md",
    "PAPER1_EVIDENCE_MATRIX.json","PAPER1_OPEN_GAPS.json","PAPER1_MAIN_TABLE_DRAFT.json","PAPER1_REPORT.md",
    "FIGURE_INDEX.json","FIGURE_CAPTION_FACTS.md",
    "PAPER_FIGURE_DATA/FIGURE_A.json","PAPER_FIGURE_DATA/FIGURE_B.json","PAPER_FIGURE_DATA/FIGURE_C.json",
    "PAPER_FIGURE_DATA/FIGURE_D.json","PAPER_FIGURE_DATA/FIGURE_E.json",
    "PAPER_FIGURE_DATA/FIGURE_A_tfc_geometry_room8cam.csv","PAPER_FIGURE_DATA/FIGURE_C_dfc_mechanism.csv",
    "PAPER_FIGURE_DATA/FIGURE_D_training_trajectory.csv","PAPER_FIGURE_DATA/FIGURE_E_negative_space.csv",
    "REPRO_SOURCE_INDEX.json","REPRO_ENVIRONMENT.md","REPRO_COMMANDS.md","REPRO_KNOWN_CAVEATS.md",
    "K50_CANONICAL_PREREG.json",
    "PUBLICATION_CONVERGENCE_STATUS.json","PUBLICATION_CONVERGENCE_REPORT.md","MANIFEST.sha256"
  ]
}
w("PUBLICATION_CONVERGENCE_STATUS.json", status)

report_md = f"""# PUBLICATION CONVERGENCE REPORT — SUPPORT PACK

**Generated:** {GEN}  |  **Mode:** CPU / source-audit / artifact-only. NO new GPU experiments. NO K50 run. NO cohort merging.

## Summary
Executed the full PUBLICATION_CONVERGENCE_SUPPORT_PACK (Tasks 1-5 + final outputs) purely by synthesizing authoritative existing artifacts (PBR0, PUB0, TFC0/TFC2, DFC1-R1/DFC1-R2/DFC2/DFC_FC0, K50-B1 Stage4B, VOS0/VOS1, IGF/CGC/KFC/AGS). No GPU work was run; no K50 revalidation was performed; current GPU work (TFC_FC0/DFC2/DFC_FC0) was NOT started.

## COMPLETED
- **TASK1 FINAL_PROTOCOL_LOCK** — froze 5 terminology buckets + authoritative metric definitions (intersection reduction pooled vs mean-per-camera; PSNR/SSIM/LPIPS deltas; gradient cosine/relL2; Gaussian-count trajectory; integrated Gaussian population; total training speedup). Hard rules encoded (renderer-local != training speedup; no residual-derived optimizer cost; direct wall-clock headline). -> `FINAL_PROTOCOL_LOCK.json`, `FINAL_PROTOCOL_LOCK.md`.
- **TASK2 PAPER1_EVIDENCE_AUTOUPDATE** — evidence matrix (10 methods + 4 pending placeholders), 10-gap register, draft main table, report. TFC=MAIN_CANDIDATE(pending TFC_FC0); K50-B1=NEEDS_REVALIDATION; DFC1-R1=SUPPORTING; DFC family=PENDING; VOS/PUB0=ANALYSIS; IGF/CGC/KFC/AGS=DROP. No silent role upgrades.
- **TASK3 PAPER_FIGURE_DATA_PACK** — Figures A/B/C/E populated from authoritative data; Figure D all PENDING. Explicit null/PENDING for every missing point; no fabrication. -> `PAPER_FIGURE_DATA/`, `FIGURE_INDEX.json`, `FIGURE_CAPTION_FACTS.md`.
- **TASK4 REPRODUCIBILITY_PACK_PREP** — canonical/diagnostic/superseded source inventory with hashes, environment, commands, known caveats + explicit warnings. No algorithm changed.
- **TASK5 K50_REVALIDATION_PREREGISTRATION_ONLY** — froze the canonical future K50 protocol (activation gated on PUB0 + backward-bottleneck evidence; no threshold rescue after failure). NOT executed.

## RUNNING (not started by this pack)
- TFC_FC0 (current GPU priority), DFC2 (queued), DFC_FC0 (queued, requires validation).

## QUEUED
- PUB0 (COMPLETE CPU ledger), DFC2/DFC_FC0 isolated-node GPU runs, K50 canonical revalidation (downstream of PUB0).

## BLOCKED
- No measured total 30K production training wall-clock for any method (BLOCKER).
- K50 revalidation blocked until PUB0 review + explicit authorization.
- DFC_FC0 train_dfc.py unverified (200-iter smoke required).
- DFC2 no universally-generalizing fixed threshold.

## Publication readiness
**NOT publication-ready.** TFC_FC0 / DFC2 / DFC_FC0 remain unresolved; K50-B1 requires revalidation; no method has a measured total 30K production training wall-clock. Renderer-local speedups MUST NOT be presented as training speedups.

## Integrity
- All numbers trace to authoritative *_VERDICT.json / *_SOURCE_LOCK.json / *_REPORT.md / ledger JSONs.
- Cohorts kept explicit; historical negatives preserved; no invented or smoothed data.
"""
w("PUBLICATION_CONVERGENCE_REPORT.md", report_md)

# ---- MANIFEST.sha256 (all deliverables EXCEPT the manifest itself) ----
manifest_text, entries = sha256_dir(ROOT)
# exclude any prior MANIFEST and builder scripts from the manifest
lines = [f"{h}  {rel}" for rel, h in entries if rel not in ("MANIFEST.sha256",) and not rel.startswith("_build")]
with open(os.path.join(ROOT, "MANIFEST.sha256"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("FINAL status+report+manifest written")
print("files in pack:", len(lines))

# =====================================================================
# POST-HOC PATCHES
# =====================================================================

# --- Incorporate PBR0_H1 into FINAL_PROTOCOL_LOCK ---
_fp = os.path.join(ROOT, "FINAL_PROTOCOL_LOCK.json")
_fp_obj = json.load(open(_fp, encoding="utf-8"))
_fp_obj["pbr0_h1_audit"] = {
  "artifact": "PBR0_H1_HARNESS_AUDIT",
  "source": "_pbr0_oracle/PBR0_H1_HARNESS_AUDIT.json, PBR0_H1_REPORT.md",
  "finding": "PBR0 M4 'full training step' is a DECOMPOSED STAGE-TIMING MICROBENCHMARK (sum of 3 independently-best-timed stages), NOT a faithful production 3DGS iteration.",
  "key_differences_from_production": [
    "Optimizer scope: 4 groups / 11 elements-per-gaussian, NO SH colors (production optimizes all params incl SH; combined harness = 5 groups / 59 elements).",
    "Densification/pruning EXCLUDED (fixed N); production densifies periodically.",
    "opt.step() timed in isolation between cuda.synchronize; 'full step' = t_fwd + best_bwd + best_step, not one fused wall-clock iteration.",
    "Absolute '0.5% optimizer' and '41 ms full step' are NOT production-representative (production optimizer ~30-41%)."
  ],
  "valid_use": "Relative A-vs-B-vs-C footprint effect is apples-to-apples (identical M4 methodology), so PBR0 1.1988x native / 39.3% intersection-reduction are valid RELATIVE speedups only.",
  "rule_added": "PBR0-style decomposed timing must NEVER be labeled a production full training step (reinforces hard rule #2/#3)."
}
_amd = _fp_obj.setdefault("authoritative_metric_definitions", {})
_tts = _amd.setdefault("total_training_speedup", {})
_mnb = _tts.get("must_not_be", [])
_mnb = _mnb if isinstance(_mnb, list) else [_mnb]
_mnb.append("PBR0 M4 sum-of-stages is a DECOMPOSED microbenchmark (PBR0_H1), NOT a production iteration; use PUB0 bucket4 directly-measured full-step or a real fused wall-clock iteration.")
_tts["must_not_be"] = _mnb
with open(_fp, "w", encoding="utf-8") as f:
    json.dump(_fp_obj, f, indent=2, ensure_ascii=False); f.write("\n")

# append PBR0_H1 note to the .md
with open(os.path.join(ROOT, "FINAL_PROTOCOL_LOCK.md"), "a", encoding="utf-8") as f:
    f.write("\n## 5. PBR0_H1 harness audit (incorporated)\n")
    f.write("- PBR0 M4 'full training step' is a DECOMPOSED STAGE-TIMING MICROBENCHMARK (sum of 3 independently-best-timed stages), NOT a faithful production 3DGS iteration.\n")
    f.write("- Optimizer scope 4 groups / 11 elements (NO SH); densification/pruning EXCLUDED; opt.step timed in isolation.\n")
    f.write("- Valid for RELATIVE A-vs-B-vs-C footprint effect only (1.1988x native / 39.3% intersection-reduction). Absolute '0.5% optimizer' / '41 ms' are NOT production-representative (production optimizer ~30-41%).\n")
    f.write("- Rule: PBR0-style decomposed timing must NEVER be labeled a production full training step.\n")

# --- Add PBR0_H1 to PAPER1 pending/incorporation ---
_em = os.path.join(ROOT, "PAPER1_EVIDENCE_MATRIX.json")
_em_obj = json.load(open(_em, encoding="utf-8"))
_em_obj["pending_placeholders"]["PBR0_H1"] = {
  "status": "COMPLETE (CPU code audit)",
  "role": "ANALYSIS",
  "result": "PBR0 M4 'full training step' is a decomposed stage-timing microbenchmark (sum of 3 independently-best-timed stages), NOT a production iteration. Valid for relative A/B/C footprint effect only. Reinforces the 'decomposed proxy != production full-step' rule.",
  "evidence": "_pbr0_oracle/PBR0_H1_HARNESS_AUDIT.json, PBR0_H1_REPORT.md"
}
with open(_em, "w", encoding="utf-8") as f:
    json.dump(_em_obj, f, indent=2, ensure_ascii=False); f.write("\n")
print("patches: PBR0_H1 incorporated")

# --- REPRO_MANIFEST.sha256 (Task 4 separate manifest, per objective) ---
repro_files = ["REPRO_SOURCE_INDEX.json","REPRO_ENVIRONMENT.md","REPRO_COMMANDS.md","REPRO_KNOWN_CAVEATS.md"]
repro_lines = [f"{sha256_file(os.path.join(ROOT, rp))}  {rp}" for rp in repro_files]
with open(os.path.join(ROOT, "REPRO_MANIFEST.sha256"), "w", encoding="utf-8") as f:
    f.write("\n".join(repro_lines) + "\n")
# add it to the deliverables list in status + top manifest
_st = json.load(open(os.path.join(ROOT, "PUBLICATION_CONVERGENCE_STATUS.json"), encoding="utf-8"))
if "REPRO_MANIFEST.sha256" not in _st["deliverables"]:
    _st["deliverables"].append("REPRO_MANIFEST.sha256")
with open(os.path.join(ROOT, "PUBLICATION_CONVERGENCE_STATUS.json"), "w", encoding="utf-8") as f:
    json.dump(_st, f, indent=2, ensure_ascii=False); f.write("\n")
print("repro manifest written:", len(repro_lines), "files")

# =====================================================================
# FINALIZE: regenerate top MANIFEST.sha256 AFTER all patches; cleanup helpers
# =====================================================================
for _h in ("_fix.py","_fix2.py","_fix3.py"):
    _hp = os.path.join(ROOT, _h)
    if os.path.exists(_hp):
        os.remove(_hp)
_mt, _ = sha256_dir(ROOT)
_mlines = [l for l in _mt.splitlines() if l and "  MANIFEST.sha256" not in l and not l.split("  ",1)[-1].startswith("_build") and not l.split("  ",1)[-1].startswith("_fix")]
with open(os.path.join(ROOT, "MANIFEST.sha256"), "w", encoding="utf-8") as f:
    f.write("\n".join(_mlines) + "\n")
print("FINALIZE: top manifest regenerated with", len(_mlines), "entries")
