# Phase C0 — Canonical 3DGS Semantic Parity Audit

**Decision: PATCH + SELECTIVE REVALIDATION.** The historical training baseline is not semantically equivalent to the pinned official Graphdeco implementation. Split construction, pruning, opacity reset, gradient accumulation, optimizer topology handling, and SH progression differ. The Room 5K factorial shows that Adam-state handling materially changes population (+10.75%) and pruning (-61.4%), while the split-only endpoint effect is small in this particular low-split run. C51 and C52 still require reference revalidation because their claims depend directly on population dynamics and their archived runs are not source-reproducible.

No historical result was deleted or overwritten. It remains evidence under `CURRENT_PROJECT_SEMANTICS`. Candidate A and Candidate B, as formulated, are `DROP`: both are correctness repairs already present in the official baseline, not novel optimizations.

## 1. Frozen provenance

| Item | Frozen value |
|---|---|
| Project commit | `02375033388d4348376b6b607ab85f551e498a77` |
| Project branch at audit start | `master` |
| Freeze branch | `audit/pre-c0-current-semantics` → same commit |
| Historical C49–C53 GPU | NVIDIA A100-PCIE-40GB |
| Historical PyTorch / CUDA | 2.7.1+cu118 / runtime 11.8; system NVCC 12.4.131 |
| Historical gsplat | 1.5.3 |
| C0 factorial | same A100 class, PyTorch 2.7.1+cu118, CUDA 11.8, gsplat 1.5.3 package with the historical cu118 extension injected process-locally |
| Official repository | `graphdeco-inria/gaussian-splatting` |
| Official pinned commit | `54c035f7834b564019656c3e3fcc3646292f727d` |

The C42–C53 scripts and results are untracked in the current worktree. A git branch therefore cannot freeze them by itself. The audit additionally stores SHA-256 hashes/excerpts in `current_source_audit.json` and a compressed snapshot at `results/a100/phase-c0/pre_c0_current_semantics_snapshot.tar.gz` (`51D5FF4AF8808752AF584614BFA479FEC9B83B824C17D0EC459AE9C539C4D2F0`).

Official source evidence is pinned by commit and captured in `official_source_audit.json`. Primary permalinks: [official topology and optimizer code](https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/scene/gaussian_model.py#L316-L473) and [official training cadence](https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/train.py#L91-L186).

## 2. Actual historical call graph

Every listed experiment imports `scripts/epic05/phase7/gaussian_model.py` and calls `gsplat.rasterization`; filenames alone did not establish this—the import and call sites were traced.

| Phase | Training script(s) | GaussianModel | Optimizer behavior after replacement | Renderer |
|---|---|---|---|---|
| C42 | `scripts/phase-c42/c42_p2_training_validation_30k.py` | phase7 model | rebuilds Adam after SH/topology → global reset | gsplat |
| C44 | `scripts/phase-c44/c44_unified_experiment.py` | phase7 model | no rebind → Adam points at old Parameters | gsplat |
| C45 | `scripts/phase-c45/c45_unified_experiment.py` | phase7 model | no rebind → orphaned Adam | gsplat |
| C49 | `lifecycle_profile.py`, `grad_filter_experiment.py` | phase7 model | no rebind → orphaned Adam | gsplat |
| C50 | `gradient_predictability.py` | phase7 model | no rebind → orphaned Adam | gsplat |
| C51 early | `simulated_sparse_backward.py` | phase7 model | no rebind → orphaned Adam | gsplat |
| C51 Stage 4A/4B/5 | `benchmark_training.py`, `canonical_training.py`, `benchmark.py` | phase7 model | rebuild after topology/reset → global reset; SH replacement is temporarily orphaned | patched gsplat sparse-backward path |
| C52 | `scripts/phase-c52-stage0/benchmark.py` | phase7 model | global reset after topology; SH replacement orphaned until next rebind | gsplat |
| C53 | discovery/validation collection scripts | phase7 model | global reset after topology; SH replacement orphaned until next rebind | gsplat |

Evidence: current model `densification` lines 166–266, `prune` 269–294, `prune_and_reset` 297–315, and `set_sh_degree` 121–147. Phase-specific assignment/call line numbers and file hashes are in `current_source_audit.json`. The current model SHA-256 is `f0ccbaa6e7d444c7583c0a5f9e466bdaa2741bba5174c421da69ce939b3f5ee8`.

This answers Q3 precisely: there is no single historical behavior. Older C44–C50/early-C51 paths do **not** globally reset Adam; they replace model Parameters without rebinding the optimizer, so updates target obsolete tensors. Later C51/C52/C53 paths do globally reset Adam at topology events. Both differ from Graphdeco.

## 3. Table A — Semantic parity

| Operation | Current project | Official reference | Match? | Severity |
|---|---|---|---|---|
| Split gradient | average world-space `||dL/dxyz||`; normally sampled only on event iteration | visible-point average view-space `||dL/dmean2D_xy||` accumulated every iteration | No | Critical |
| Scale criterion | all axes ≤ per-event median → clone; complement → split | `max(scale) <=/> percent_dense * scene_extent` | No | High |
| Split child count | 2 | `N=2` default | Yes | None |
| Split parent | retained; returns `removed=0` | appended children, then selected parents pruned | No | Critical |
| Split position | parent + iid world-axis `N(0, 0.0025²)` | parent + `R(q) N(0, diag(scale²))` | No | Critical |
| Split scale | `log(s/2)` | `log(s/(0.8N))`; for N=2, `log(s/1.6)` | No | High |
| Split rotation | raw quaternion inherited | raw quaternion inherited | Yes | None |
| Split opacity | logit inherited | logit inherited | Yes | None |
| Split SH | all stored coefficients inherited | DC/rest coefficients inherited | Yes | None |
| Clone | parent retained; child position gets `0.01*scale` noise | exact parameter copy; parent retained | No | High |
| Prune | opacity only; audit scripts use 0.01 | opacity <0.005; after reset interval also screen radius >20 or world scale >0.1 extent | No | Critical |
| Opacity reset | prune first; only `[threshold,10×threshold)` set to `2×threshold` | all opacities clamped to at most 0.01; opacity moments zeroed | No | Critical |
| Gradient denominator | one global event/sample count | per-Gaussian visible observation count | No | Critical |
| Append/prune Adam | orphan optimizer or reconstruct/reset, depending on phase | retain survivor rows, zero-pad new rows, keep scalar step | No | Critical |
| SH progression | allocates a new `shs` Parameter at each degree | fixed max-SH tensor; increments only active degree | No | Critical |

Current evidence: `gaussian_model.py:152–266,269–315`; official evidence: `scene/gaussian_model.py:316–473`, pinned SHA above. Exact excerpts for every row are stored in `semantic_matrix.json`.

## 4. Table B — Topology arithmetic and stable identity

For two-child split and identical masks:

| Event | Current ΔN | Reference ΔN | Difference |
|---|---:|---:|---:|
| clone only | `+N_clone` | `+N_clone` | 0 |
| split only | `+2N_split` | `+(2-1)N_split` | `+N_split` current |
| combined | `N_clone + 2N_split` | `N_clone + N_split` | `N_split` |
| synthetic: N=1000, clone=100, split=100 | +300 → 1300 | +200 → 1200 | +100 |

The deterministic unit test passed. Temporary IDs show current lineage `g600`, `g600.split0`, `g600.split1`; reference lineage contains only the two children. An unrelated survivor `g700` remains stable. Thus tensor replacement is distinguished from semantic identity removal.

Official evidence is `densify_and_split`, lines 409–433: children are appended at line 430 and a mask containing the selected original rows is pruned at lines 432–433. Current evidence is lines 245–266: all children are appended and `removed: 0` is returned.

## 5. Table C — Adam continuity

The test warmed every parameter group for four Adam steps, then applied one clone, split, or prune transaction and compared a fixed survivor row exactly.

| State | Survivor current reset | Survivor reference | Child current reset | Child reference |
|---|---|---|---|---|
| parameter value | preserved | preserved | constructed | constructed |
| `exp_avg` | not preserved | exact survivor row preserved | absent until lazy first step | initialized to 0 |
| `exp_avg_sq` | not preserved | exact survivor row preserved | absent until lazy first step | initialized to 0 |
| `step` | not preserved | scalar step 4 preserved | starts at 1 on first update | shares preserved group step 4 |

All five parameter groups passed exact equality for survivor value and reference moments. Official `_prune_optimizer` lines 331–347 slices moment rows; `cat_tensors_to_optimizer` lines 366–386 concatenates zero moment rows; neither replaces the stored scalar `step`. `replace_tensor_to_optimizer` lines 316–329 intentionally zeros opacity moments during reset while retaining the rest of the stored state.

For older orphaned paths the situation is worse than the “current reset” column: new model Parameters are not members of Adam at all, so they receive no optimizer update until some later explicit reconstruction.

## 6. Zero-step real Room topology comparison

Checkpoint: 1,593,376-Gaussian Room model after 500 optimizer steps, immediately before the first topology event. A and B used identical masks, camera set, and seed 542; no optimizer step followed the topology transaction. The 1.1 GB checkpoint remains on the A100 host; its SHA-256 and path are recorded in the result JSON.

| Metric | Current | Reference | Difference |
|---|---:|---:|---:|
| clone selected | 1 | 1 | 0 |
| split selected | 18 | 18 | 0 |
| split parents removed | 0 | 18 | 18 |
| prune count | 5,048 | 5,048 | 0 |
| final N | 1,588,365 | 1,588,347 | 18 |
| total tile intersections, 13 cameras | 44,540,058 | 44,540,407 | +0.00078% reference |
| mean camera median forward time, 15 samples | 3.9519 ms | 3.9470 ms | -0.125% reference |

Cross-render metrics over 13 fixed cameras: PSNR 80.6477 dB, L1 `9.7075e-5`, SSIM 0.9999770, and alpha L1 `7.7062e-6`. The immediate semantic difference is measurable but small because this event selected only 18 split parents. It is not a final-quality result.

## 7. Table D — Room 5K factorial

Predeclared gates: population >10%, E2E time >5%, PSNR >0.2 dB, SSIM >0.005, clone/split/prune counts >10%, or material concentration/ranking change. One seed (42), 1080p, identical initialization/camera order/loss/LRs/schedule. SH shape progression is common-mode and preserves moments in all four conditions, so the Adam factor applies only to topology transactions.

| Condition | PSNR | SSIM | Final N | Mean iter ms | Clone | Split | Prune |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: current split, reset Adam | 32.0460 | 0.922222 | 1,357,383 | 33.2507 | 5 | 841 | 237,680 |
| B: reference split, reset Adam | 32.0182 | 0.921925 | 1,356,779 | 33.2113 | 7 | 878 | 237,482 |
| C: current split, preserve Adam | 31.9873 | 0.922439 | 1,503,237 | 34.4085 | 5 | 833 | 91,810 |
| D: reference split, preserve Adam | 31.9590 | 0.922590 | 1,502,756 | 34.3171 | 5 | 819 | 91,444 |

Factor effects:

| Contrast | ΔPSNR | ΔSSIM | ΔN | Δ mean iter | Δ prune |
|---|---:|---:|---:|---:|---:|
| split at reset: B−A | -0.0279 dB | -0.000297 | -604 (-0.044%) | -0.118% | -0.083% |
| split at preserve: D−C | -0.0283 dB | +0.000151 | -481 (-0.032%) | -0.266% | -0.399% |
| Adam at current split: C−A | -0.0587 dB | +0.000216 | +145,854 (+10.745%) | +3.482% | -61.372% |
| Adam at reference split: D−B | -0.0591 dB | +0.000665 | +145,977 (+10.759%) | +3.329% | -61.494% |
| split×Adam interaction | -0.00041 dB | +0.000449 | +123 | -0.0520 ms | -168 |

Only the Adam population and prune-count effects cross materiality gates. Split semantics cause a transient discontinuity: at the first event current split changes same-camera PSNR by only -0.0019/-0.0043 dB (A/C), whereas reference construction changes it by -0.6031/-0.5302 dB (B/D). By 5K the endpoint split effect is below every gate. Endpoint interaction is not meaningful.

Each condition JSON contains every iteration's loss, train PSNR/SSIM, Gaussian count, clone/split/prune counts, render/loss/backward/topology/optimizer/total timing, plus every topology event's population jump and immediate Δloss/ΔPSNR. The aggregate file points to these four raw files.

### Reproducibility caveat

The archived C51 5K baseline reports 42,538 splits; audited current source at explicit threshold 0.001 produces 841 in condition A. The archived JSON does not store the threshold, and the script is untracked. The C51 source header also mentions 0.0002 while the current constant/report table uses 0.001. Exact historical source state therefore cannot be recovered from git. The C0 factorial is a valid controlled audit of the frozen current source, but it is not proof that append-only split was immaterial in the much higher-split historical C51 trajectory.

## 8. Causal answers

1. **Append-only growth:** exactly one extra resident Gaussian per selected two-child split before downstream pruning. Synthetic test: +100 for 100 splits. Zero-step: +18 for 18 splits. In the 5K trajectories the actual endpoint difference was only 481–604 because only ~820–880 splits occurred and downstream masks diverged.
2. **Split trajectory effect:** large immediate topology discontinuity for reference child construction (roughly -0.53 to -0.60 dB at the first event), but final mean split effect only -0.0281 dB, -0.000073 SSIM, and -543 Gaussians in this run.
3. **Historical global Adam reset:** phase-dependent. C42 and later C51/C52/C53 reconstruct Adam; C44/C45/C49/C50/early-C51 orphan it instead. Neither is canonical.
4. **Moment preservation effect:** +10.75% final population and about 61.4% fewer prunes; PSNR, SSIM, and E2E time remain below their gates.
5. **Split×optimizer interaction:** no material 5K endpoint interaction. There is transient event-level recovery behavior, but it does not support a durable interaction claim.
6. **Dominant measured factor:** optimizer-state handling, through opacity/prune trajectory, not split semantics in the low-split C0 run.

## 9. Table E — Historical validity

| Phase | Finding | Sensitivity | Classification | Required action |
|---|---|---|---|---|
| C42 | resolution-adaptive SSIM | Low | SAFE | retain kernel/local comparison; label old training curves current semantics |
| C44 | separable/frequency-reduced SSIM | Low | SAFE | retain math/kernel result; do not use old convergence curves as canonical quality evidence |
| C45 | “moderate” configuration as canonical baseline | Very high | INVALID_UNDER_REFERENCE | replace with a tracked full-reference baseline; retain old curve as legacy only |
| C49 | gradient concentration/oracle filtering | Medium | NEEDS_RECALIBRATION | recompute concentration and coverage on one reference Room trajectory |
| C50 | temporal predictor ordering | Medium | NEEDS_RECALIBRATION | recompute recall/coverage/ranking on the same trajectory |
| C51 | sparse backward plus population-mediated speedup | Very high | NEEDS_RERUN | reference baseline vs K50-B1 Room; separate pure kernel saving from population saving |
| C52 | gradient-ranked allocation negative result | Very high | NEEDS_RERUN | reference baseline/uniform/ranked Room only; do not assume the negative flips |
| C53 | workload/utility/persistence | Low–medium | NEEDS_RECALIBRATION | recompute distributions on the reference Room trajectory; mechanism code remains usable |

C51's pure CUDA skip/correctness mechanism is not invalidated by topology semantics. Its population-mediated E2E magnitude and quality trajectory are not trustworthy without revalidation. C52's negative conclusion is unresolved, not reversed. C49/C50 qualitative mechanisms are plausible but their numerical thresholds and rankings are distribution-dependent.

## 10. Table F — Candidate audit

| Candidate | Survives reference parity? | Research status |
|---|---|---|
| A — function-preserving fission | No | DROP; parent→children replacement is already reference behavior. Removing the project's extra parent is correctness repair. |
| B — transactional optimizer | No | DROP; official Graphdeco already migrates survivor moments and initializes child moment rows. Implementing it locally is correctness repair. |

## 11. Final questions and baseline decision

1. Historical project split retains the parent: **yes**.
2. Official Graphdeco removes it: **yes**, after appending N children.
3. Exact child difference: current uses constant unrotated 0.0025 noise and scale/2; official uses scale-shaped Gaussian samples rotated by the parent quaternion and scale/(0.8N)=scale/1.6 for N=2. Clone is noisy current versus exact-copy official.
4. Historical Adam reset: **mixed by phase**; later paths reset globally, older paths orphan Parameters.
5. Official survivor state preservation: **yes**, including scalar step; new moment rows are zero.
6. Same-mask population: current exceeds reference by exactly `N_split` before later pruning.
7. Zero-step render: 80.65 dB A/B PSNR, 0.999977 SSIM, and effectively equal forward time in the tested 18-split event.
8. Room 5K: split endpoint effect non-material; Adam preservation materially changes population/prune trajectory.
9. Dominant factor: Adam handling; endpoint interaction negligible.
10. Trustworthy C49–C53: mechanism-local parts of C49/C50/C53, not their calibrated distributions; C51 pure kernel mechanism only.
11. Recalibrate C49/C50/C53; rerun C51/C52 minimally.
12. Candidates A/B: baseline-correctness issues, not research opportunities.
13. Current canonical baseline suitable for future top-conference experiments: **no**. The audit variant is an isolation scaffold, not yet a complete Graphdeco-equivalent trainer. Future baseline must also adopt official gradient accumulation/selection, pruning/opacity reset, fixed SH storage, and tracked provenance.
14. Minimum revalidation: one reference Room trajectory feeding C49/C50/C53 statistics; one Room baseline-vs-K50-B1 C51 comparison; one Room baseline/uniform/ranked C52 comparison. Expand to 30K or more scenes only if those runs cross a gate or change ordering.

## 12. Unavoidable differences and next baseline patch

The `REFERENCE_SEMANTICS` variant faithfully ports official clone/split construction, split-parent removal, prune-mask formula, opacity reset, and optimizer-state migration into the project's parameter layout. The factorial intentionally holds the project's selection masks and prune/opacity cadence fixed to isolate split and Adam effects. It therefore does not claim full parity for:

- gsplat renderer versus Graphdeco's rasterizer;
- world-xyz versus view-space gradient accumulation;
- median-scale versus `percent_dense*extent` selection;
- initialization from the project's 1.59M-Gaussian PLY rather than official SfM training initialization;
- project thresholds/cadence versus official command defaults;
- single-seed statistical uncertainty.

Before future top-conference experiments, make the full official semantics the tracked default, add the C0 unit tests to CI, record every threshold and source hash in every result, and retain `CURRENT_PROJECT_SEMANTICS` only as a legacy namespace. Do not present any speedup caused by this repair as a new optimization.

## 13. Deliverables

- `results/a100/phase-c0/semantic_matrix.json`
- `results/a100/phase-c0/topology_unit_test.json`
- `results/a100/phase-c0/optimizer_state_unit_test.json`
- `results/a100/phase-c0/zero_step_topology_comparison.json`
- `results/a100/phase-c0/room_5k_factorial.json` plus `room_5k_A.json` … `room_5k_D.json`
- `results/a100/phase-c0/historical_impact_matrix.json`
- `results/a100/phase-c0/candidate_reclassification.json`
- `scripts/phase-c0/` audit/test/runners
- `variants/reference_semantics/` isolated reference port
