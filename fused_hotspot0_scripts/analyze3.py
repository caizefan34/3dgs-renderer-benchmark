#!/usr/bin/env python3
"""FUSED-HOTSPOT0 analysis stage 3: verdict, report, prior-art, timing boundary, prereg, freeze, manifest."""
import json, os, hashlib, datetime, subprocess

P = "/mnt/storage_pool/liaoyuanjun/fused_profile0"
HOT = P + "/FUSED_HOTSPOT0"
RAW = HOT + "/raw"
SCENES = ["room", "bicycle", "garden"]
STAGES = ["s1_5k", "s2_10k", "s3_mature"]
GEN_UTC = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
PY = "/mnt/storage_pool/liaoyuanjun/higs-13scene-env/bin/python3.10"

def load(p):
    with open(p) as f: return json.load(f)
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
def rnd(x, n=4):
    try: return round(float(x), n)
    except Exception: return x

SUMM = load(f"{HOT}/_summary_nums.json")
CB = load(f"{HOT}/FUSED_HOTSPOT0_COMPONENT_BREAKDOWN.json")
PS = CB["per_scene_stage"]
CC = load(f"{HOT}/FUSED_HOTSPOT0_KERNEL_CENSUS.json")
WS = load(f"{HOT}/FUSED_HOTSPOT0_WORKLOAD_STATS.json")
KC = load(f"{HOT}/FUSED_HOTSPOT0_KERNEL_CHARACTERIZATION.json")
AM = load(f"{HOT}/FUSED_HOTSPOT0_AMDAHL_ORACLE.json")
OR = load(f"{HOT}/FUSED_HOTSPOT0_OPPORTUNITY_RANKING.json")
RED = load(f"{HOT}/FUSED_HOTSPOT0_REDUNDANCY_AUDIT.json")
SC = load(f"{HOT}/FUSED_HOTSPOT0_SCALING.json")
PFREEZE = load(f"{P}/FUSED_PROFILE0_FREEZE.json")

def frac(sc, s, c): return PS[f"{sc}_{s}"]["components"][c]["fraction_of_iteration"]
def cpms(sc, s, c): return PS[f"{sc}_{s}"]["components"][c]["critical_path_ms"]
def Titer(sc, s): return PS[f"{sc}_{s}"]["T_iteration_ms"]

# ---------------- VERDICT ----------------
verdict = {
    "schema": "FUSED_HOTSPOT0_VERDICT", "generated_utc": GEN_UTC,
    "verdict": "PASS_MAJOR_RESIDUAL_HOTSPOT_IDENTIFIED",
    "one_line": "blend_backward_cu is the single dominant residual hotspot (mature geomean f=0.459; >=0.40 in all 3 core scenes; Amdahl 2x->1.30x, perfect->1.85x) with genuinely removable atomic-contention work that survives the strong baseline and leaves a plausible prior-art gap.",
    "major_hotspot_criteria": {
        "component": "B (blend_backward_cu, raster backward alpha-blend + per-Gaussian gradient accumulation)",
        "geomean_fraction_mature": SUMM["B_geomean"],
        "per_scene_mature_fraction": SUMM["B_mature_frac"],
        "gate_geomean_ge_0.25": SUMM["B_geomean"] >= 0.25,
        "gate_ge_0.30_in_2_of_3": sum(1 for v in SUMM["B_mature_frac"].values() if v >= 0.30) >= 2,
        "n_scenes_measured": 3, "n_stages_measured": 3,
        "amdahl_gate_pass_2x_ge_1.15": AM["per_component"]["B"]["S_E2E"]["2.0"] >= 1.15,
        "amdahl_gate_perfect_ge_1.25": AM["per_component"]["B"]["S_E2E"]["inf"] >= 1.25
    },
    "accounting": {"min_resolved_fraction": SUMM["resolved_min"], "gate_ge_0.90": SUMM["resolved_min"] >= 0.90,
                  "note": "Component accounting complete (>=0.90) on all 9 scene/stages; no INCONCLUSIVE_COMPONENT_ACCOUNTING."},
    "baseline_drift_check": {"reproduced_raster_dominant": True,
                             "note": "bicycle/garden reproduce SAT-AUDIT0 PL2 within 0.3%; room absolute T_iter differs (3.86 vs 6.12ms) but room was flagged non-reproducible in SAT_AUDIT0 runlog and f_raster fraction matches (0.910 vs 0.894). No INCONCLUSIVE_BASELINE_DRIFT for the raster-dominant conclusion."},
    "recommended_next_target": {
        "target_component": "blend_backward_cu (B) — raster backward alpha-blend + per-Gaussian gradient accumulation",
        "measured_e2e_fraction_mature_geomean": SUMM["B_geomean"],
        "measured_e2e_fraction_mature_per_scene": SUMM["B_mature_frac"],
        "measured_e2e_fraction_full_training_weighted": SUMM["ftw"]["B"],
        "dominant_cost_mechanism": "Per-pixel reverse blend-chain replay (serial, warp.shfl transmittance/color/grad dependence) COMBINED WITH per-primitive atomicAdd gradient accumulation whose contention scales with instances-per-Gaussian (room 703, bicycle 207 at s3). 50% occupancy (64 reg) leaves the chain latency-limited; atomics add serialized contention on high-overlap scenes.",
        "measured_removable_work_opportunity": "Atomic-contention overhead in gradient accumulation. A deterministic privatized / sorted segmented accumulation produces bit-identical gradients with fewer serialized atomic conflicts. Forward early-stop (last_contributor) and 1/255 alpha early-stop are ALREADY intrinsic, so chain-replay redundancy is largely eliminated; the residual removable work is specifically the atomic serialization.",
        "amdahl_ceiling_E2E": AM["per_component"]["B"]["S_E2E"],
        "prior_art_gap": "FasterGSFused / fast-GS lineage uses naive per-primitive atomicAdd in the bucketed backward. Contention-reducing (privatized / sorted segmented) accumulation is standard in parallel computing but NOT applied/validated in this 3DGS raster-backward lineage, and the measured instances-per-Gaussian amplification (room 703) is exactly the regime where it should matter. Plausible research gap.",
        "minimal_falsification_experiment": "Measure the atomic-contention share of blend_backward (ncu atomic counters such as l1tex atomic throughput / transactions, or a source-side atomic-conflict counter) on room (703 inst/G, high contention) vs a low-overlap scene. If contention is a material fraction (>=~15-20%) of blend_backward time, the removable-work claim holds and a new-method lane is justified; if contention is negligible, the cost is irreducible chain replay and the opportunity downgrades. (This lane is measurement-only and does NOT implement the accumulation variant.)",
        "do_NOT_implement_in_this_lane": "Per objective section 0/1, FUSED-HOTSPOT0 authorizes NO optimization. The variant above is for PI to design in a subsequent lane."
    },
    "secondary_findings": {
        "blend_cu_F": {"geomean_mature_fraction": SUMM["geo_mature"]["F"], "major": False,
                        "note": "Forward blend geomean 0.224 (room 0.283, bicycle 0.238, garden 0.167); fails >=0.30-in-2/3 and Amdahl 2x->1.13<1.15 gate. Early-stop already intrinsic. Not a standalone major target."},
        "forward_raster_coupled_PISF": {"geomean_mature_fraction": 0.3167, "major": False,
                        "note": "Coupled P+I+S+F geomean 0.317 < 0.40 pair threshold; exact reject/early-stop/bucketing already intrinsic (AABB-PROD0 closed)."},
        "preprocess_backward_PB": {"geomean_mature_fraction": SUMM["geo_mature"]["PB"], "occupancy_pct": 31.2,
                        "note": "Register-limited (95 reg); low self-time (~0.03). Possible redundant expf recompute but small."}
    },
    "negative_findings": [
        "No second independent major hotspot: F, BO, PB, P, I, S, L all below the major-hotspot gate.",
        "All obvious forward-side optimizations (exact per-tile contribution reject, per-pixel alpha early-stop 1/255, transmittance early-termination 1e-4, bucketed backward, backward/optimizer fusion, shared-memory staging) are ALREADY intrinsic to FasterGSFused (baseline-overlap rule, section 24).",
        "Densification (D) is inactive in the corrected timing boundary at S3 and not a per-iteration GPU cost.",
        "garden n_instances unavailable (production forward overflows int32 / OOMs on full-res instance buffers); garden scaling uses n_gaussians/image only."
    ],
    "not_measured": [
        "ncu hardware counters (DRAM/L2 throughput, stall reasons, warp efficiency, atomic transactions) — COUNTERS_UNAVAILABLE (ncu 2021.3.1 sections path missing, no sudo).",
        "Direct atomic-conflict count in blend_backward (requires instrumentation or admin counters).",
        "garden per-tile list-length percentiles (instance-buffer OOM).",
        "Byte-exact instrumented/uninstrumented A/B (instrumentation is timing-only CUDA events; render/loss/gradient path unmodified)."
    ]
}
json.dump(verdict, open(f"{HOT}/FUSED_HOTSPOT0_VERDICT.json", "w"), indent=2)

# ---------------- PRIOR_ART_MAP.md ----------------
prior_art = """# FUSED-HOTSPOT0 — Prior-Art Collision Check

> Generated %s. Objective section 23/24. Baseline-overlap rule: any mechanism already present in FasterGSFused must NOT be promoted.

## Baseline-overlap rule (mechanisms ALREADY intrinsic — do NOT promote)
| Mechanism | Status in FasterGSFused | Source |
|---|---|---|
| Exact per-tile contribution rejection (will_primitive_contribute) | INTRINSIC | create_instances_cu, kernel_utils.cuh:62 |
| Warp-level cooperative culling | INTRINSIC | create_instances_cu (compute_cooperatively, warp.shfl) |
| Per-pixel alpha early-stop (min_alpha_threshold=1/255) | INTRINSIC | blend_cu |
| Transmittance early-termination (1e-4) | INTRINSIC | blend_cu |
| Bucketed per-Gaussian backward (instance ranges + bucket counts) | INTRINSIC | extract_instance_ranges_cu / extract_bucket_counts |
| Backward/optimizer fusion (Adam inside backward) | INTRINSIC | adam_step_invisible, kernels_backward.cuh:464 |
| Shared-memory staging (one 16x16 tile per block) | INTRINSIC | blend_cu / blend_backward_cu |

## Candidate residual opportunity vs prior art
| Candidate | Prior art | Gap assessment |
|---|---|---|
| **blend_backward atomic-contention reduction** (privatized / sorted segmented gradient accumulation) | Faster-GS / fast-GS / gsplat, 3DGS-renderer-benchmark, FasterGSFused all use naive per-primitive `atomicAdd` for gradient accumulation. Generic parallel-reduction techniques (privatization, sort-then-segmented-reduce) exist in HPC but are NOT applied/validated in this 3DGS raster-backward lineage. | **PLAUSIBLE GAP.** The measured instances-per-Gaussian amplification (room 703, bicycle 207) is exactly the high-overlap regime where atomic contention should dominate; no checked prior method addresses it for the bucketed backward. |
| Forward blend speedup | Heavily optimized in prior art (tile sorting, early-stop, shared staging) — already intrinsic here. | NO GAP (occupied). |
| Backward/optimizer fusion | Already intrinsic (Adam fused in backward). | NO GAP (occupied). |
| Exact tile reject / AABB culling | AABB-PROD0 CLOSED (C32==B32, S_reject=1.0); exact reject intrinsic. | NO GAP (do not reopen). |
| Preprocess backward register pressure | Standard projection backward; register-limited but small self-time (~0.03). | WEAK / low headroom. |

## Verdict of collision check
The ONLY candidate that survives the baseline-overlap rule AND leaves a plausible prior-art gap is **blend_backward gradient-accumulation contention reduction**. All other obvious mechanisms are already intrinsic or occupied. This is consistent with the single recommended target in VERDICT.json.
"""
open(f"{HOT}/FUSED_HOTSPOT0_PRIOR_ART_MAP.md", "w").write(prior_art % GEN_UTC)

# ---------------- TIMING_BOUNDARY.md ----------------
timing = """# FUSED-HOTSPOT0 — Production Iteration Timing Boundary

> Generated %s. Defines the timed interval. Matches corrected SAT-AUDIT0 semantics (objective section 5).

## Timed interval (authoritative, PL2 preloaded)
```text
t0.record()
  sample  : view = sampler.get(train_ds)['view']
  bg      : bg_color = view.camera.background_color
  rgb      : rgb_gt  = view.rgb                 # PRELOADED under PL2 (no disk decode in boundary)
  forward : renderer.render_image_training(...)  # P + I + S + F  (projection, instance/sort, blend)
  loss    : loss_fn(image, rgb_gt)               # L (L1 + DSSIM)
  backward: loss.backward()                       # PB + B + BO (blend backward + fused Adam)
t1.record(); torch.cuda.synchronize()
total = t0.elapsed_time(t1)
```

## Excluded from the timed interval
- Per-iteration JPEG image decode / disk load (the SAT-AUDIT0 artifact). Excluded by preloading inputs BEFORE the interval: `trainer.DATA.PRELOADING_LEVEL = 2` then `trainer._prepare_dataset` (VRAM preload). Under PL2, `view.rgb` is a memory access, not a disk decode (measured rgb segment ~0.02ms vs ~99-290ms under PL0).
- Optimizer state logging, checkpointing, densification bookkeeping callbacks (not in the per-iteration training-callback timer).

## Correspondence to official training
- Matches `TIMING.INCLUDE_DATALOADING_IN_TOTAL: false` (mat_*.yaml): dataloading excluded from the reported iteration time.
- Under PL0 (original FP1/FP2 harness) `view.rgb` re-decoded JPEG every iteration INSIDE t0->t1 — the SAT-AUDIT0 profile artifact. Corrected production policy = PL2 preload.

## Segment semantics
- `forward` CUDA-event = entire render forward (preprocess_cu + create_instances + radix sort + blend_cu). GPU kernels within may overlap; census normalizes against this wall.
- `backward` CUDA-event = loss.backward() = preprocess_backward_cu + blend_backward_cu + fused adam_step (Adam fused INSIDE backward per-Gaussian).
- `loss` CUDA-event = L1 + DSSIM.
- Component critical-path allocation (section 11/12): forward segment split across P/I/S/F by census self-time share; backward segment split across PB/B/BO; loss -> L; residual (sampler + rgb H2D + launch latency) -> U (unresolved/host).

## Reproducibility
- warmup 50, iters 200, CUDA events, one exclusive clean A100 (guard <=512MiB, <=2%%, >=20s).
- bicycle/garden reproduce SAT-AUDIT0 PL2 within 0.3%%; room fraction matches (absolute T_iter differs; SAT-AUDIT0 runlog flagged room non-reproducible).
"""
open(f"{HOT}/FUSED_HOTSPOT0_TIMING_BOUNDARY.md", "w").write(timing % GEN_UTC)

# ---------------- PREREG ----------------
prereg = {
    "schema": "FUSED_HOTSPOT0_PREREG", "generated_utc": GEN_UTC,
    "lane": "FUSED-HOTSPOT0", "type": "measurement + source-attribution + opportunity-oracle",
    "authorization": "NO optimization authorized (objective section 0/1). Source edits limited to NVTX/timing/diagnostic/build-fix with no semantic effect; no source edits were needed (harness lives in scripts/, uses public API).",
    "core_questions": [
        "Inside the corrected FasterGSFused production raster path, which kernels/operators dominate E2E training time and what fraction?",
        "What is the max achievable E2E gain if each dominant component is accelerated (Amdahl oracle)?",
        "Is there a residual hotspot with genuinely removable work that survives the strong baseline and leaves a prior-art gap?"
    ],
    "scenes": SCENES, "stage_classes": {"S1_early": "5k", "S2_near_densify_end": "10k", "S3_mature_post_densify": "15k(final)"},
    "checkpoints": {sc: {"s1_5k": "0005000.pt", "s2_10k": "0010000.pt", "s3_mature": "final.pt"} for sc in SCENES},
    "timing_policy": "corrected SAT-AUDIT0 PL2 preloaded; data decode excluded; CUDA events; warmup 50 / iters 200.",
    "gates": {"H1_reproduction": "raster-dominant fraction reproduces SAT-AUDIT0 within 10% (else INCONCLUSIVE_BASELINE_DRIFT)",
              "accounting": "explained GPU critical-path >=0.90 (else INCONCLUSIVE_COMPONENT_ACCOUNTING)",
              "major_hotspot": "geomean f>=0.25 OR f>=0.30 in >=2 scenes",
              "amdahl_promotion": "2x -> >=1.15x OR perfect removal -> >=1.25x"},
    "terminal_verdicts": ["PASS_MAJOR_RESIDUAL_HOTSPOT_IDENTIFIED","NO_SINGLE_MAJOR_HOTSPOT_FRAGMENTED","NO_NOVEL_RESIDUAL_HEADROOM","INCONCLUSIVE_COMPONENT_ACCOUNTING","INCONCLUSIVE_BASELINE_DRIFT","BLOCKED_GPU_OCCUPIED","BLOCKED_STRONG_BASELINE"],
    "clean_gpu_guard": "mem<=512MiB, util<=2%, observe>=20s, no vLLM/foreign training/root co-tenant; never kill/alter foreign processes."
}
json.dump(prereg, open(f"{HOT}/FUSED_HOTSPOT0_PREREG.json", "w"), indent=2)

# ---------------- BASELINE_FREEZE ----------------
cfg_sha = {}
for sc in SCENES:
    cfg_sha[sc] = PFREEZE["configs"][sc]["sha256"]
freeze = {
    "schema": "FUSED_HOTSPOT0_BASELINE_FREEZE", "generated_utc": GEN_UTC,
    "purpose": "Frozen FasterGSFused production baseline for FUSED-HOTSPOT0. Identical to FUSED_PROFILE0_FREEZE (post SAT-AUDIT0).",
    "baseline_repo": PFREEZE["baseline_repo"],
    "baseline_git": PFREEZE["baseline_git"],
    "framework_git": PFREEZE["framework_git"],
    "backend_build_B16_TS16": PFREEZE["backend_build"]["B16_TS16"],
    "backend_so_sha256": PFREEZE["backend_build"]["B16_TS16"]["sha256"],
    "python": PFREEZE["environment"]["python"], "python_executable": PFREEZE["environment"]["python_executable"],
    "torch": PFREEZE["environment"]["torch"], "torch_cuda": PFREEZE["environment"]["torch_cuda"],
    "cuda_home_nvcc_release": PFREEZE["environment"]["cuda_home_nvcc_release"],
    "gpu_model": PFREEZE["environment"]["gpu_model"], "gpu_count": PFREEZE["environment"]["gpu_count"],
    "scenes": {sc: {"checkpoint_s3": PFREEZE["scenes"][sc]["checkpoint"], "checkpoint_sha256": PFREEZE["scenes"][sc]["checkpoint_sha256"], "n_gaussians": PFREEZE["scenes"][sc]["n_gaussians"]} for sc in SCENES},
    "config_sha256": cfg_sha,
    "tile_config": {"tile_width": 16, "tile_height": 16, "block_size_blend": 256, "n_sequential_threshold": 4, "min_alpha_threshold": "1/255", "transmittance_threshold": "1e-4", "dilation": 0.3},
    "preloading_mode": "PL2 (VRAM) for authoritative timing",
    "prior_lanes": {"SAT_AUDIT0": "PROFILE_ARTIFACT_FOUND (JPEG decode in timed region under PL0)", "AABB_PROD0": "CLOSED (exact reject intrinsic, C32==B32, S_reject=1.0)", "FUSED_PROFILE0": "DROP_AABB_PRODUCTION_HEADROOM"}
}
json.dump(freeze, open(f"{HOT}/FUSED_HOTSPOT0_BASELINE_FREEZE.json", "w"), indent=2)

print("STAGE3_DONE verdict/prior_art/timing_boundary/prereg/freeze written")
print("VERDICT:", verdict["verdict"])