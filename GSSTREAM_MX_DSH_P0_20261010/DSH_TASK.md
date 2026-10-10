# DSH MX task — GSSTREAM-P0 (preregistered, CPU only)

## Research question
Could a **GPU-cheap compressed support index** let a differentiable rasterizer avoid the quadratic-form (`q`) and shared-memory staging overhead for inactive Gaussian/pixel pairs while preserving native forward and all backward parameter families? Distinguish this from Speedy-Splat / AccuTile and existing gsplat pixel-index APIs BEFORE claiming novelty.

## Execution identity / scope
- Repo: https://github.com/caizefan34/3dgs-renderer-benchmark ; observed `master` pinned `387a7b31f35fa0b9ab1d160fa12181e1a60c77c2` (2026-10-08). Check current MX checkout and report sha. **Do not checkout/rebase/cherry-pick in the user's frozen P16 worktree.** Use a separate read-only worktree/copy only if needed.
- MX environment: A100-PCIE-40GB x8, but **NO CUDA, GPU or remote training** authorized. P16 frozen P16/F1 data 1920x1080 hold-8, seed 42, training/eval protocol unchanged.
- GPU Gate: frozen minimum 30 GiB free; current `/mnt/storage_pool` reported ~7 GiB; explicitly leave GPU Gate FAIL. CPU outputs <100 MB, and do not use `storage_pool` for scratch without adequate space. CPU-only work need not overcome GPU Gate.
- One CPU run at a time; no GPU utilization. Record `git rev-parse HEAD`, CPU Python/NumPy version, environment vars, process start/end and free disk. No writes to benchmark baselines. Results in separate `research/gsstream_p0/` or other non-P16 scratch.
- Evidence identity: archive source SHA-256, frozen scene/camera/checkpoint hashes, source code references, actual CSV/NPZ hashes, all logs and negative results.

## Permitted modifications
- Only independent scratch (`GSSTREAM_MX_DSH_P0_20261010/*.py`, `research/gsstream_p0/*`), local analysis scripts and read-only data loaders. Can adjust adapters to READ previously exported projected arrays; document adapters and hash them.
- No changes to `baseline/`, `baselines/`, `benchmark*`, renderer CUDA/C++ or P16 frozen sources; no training recipe, camera split, metrics, checkpoint, seed or evaluation changes. No package installation on shared env; use existing NumPy.

## Ordered steps
1. **Literature gate.** Inspect AccuTile, FlashGS, gsplat `_torch_impl.py` / `rasterize_to_indices_in_range`, 3DGART, and source for Faster-GS / HiGS. Locate whether compressed sorted per-row support *already exists*. Write `PRIOR_ART_MATRIX.md` with exact file/line or page and `NOVELTY_FAIL` when identical. No GPU.
2. **Preflight.** Run `python3 test_properties.py` and `cpu_oracle.py` on seeds 42 and 123. Capture stdout/stderr and process exit codes. False-negative valid alpha, gradient mismatch, non-SPD unsafe fallback or unsound conditional branch => immediate STOP.
3. **Reality gate.** Search only existing projected state payloads for `room`, `bicycle`, `garden` and stages 5K/15K/30K in pinned accessible artifacts. Check semantics and hash. If absent, deliver `DATA_UNAVAILABLE`, do not generate GPU projections. If existing snapshots can be safely cropped, use preregistered fixed 64×64 patches at up to 12 training cameras per scene, and report omitted parts. **Do not treat crops as scene-wide evidence**.
4. **Cost model.** For 8,16,32 tiles measure tile bounding-box candidate visits, exact support hits, row interval count, conservative padding candidate count; calculate memory footprint lower bound and upper bound, without pretending interval counts equal GPU operations. Include `index_bytes`, `sort_cost`, `write/read_cost`, estimated builder cost, and a separate fwd/bwd relevant fraction obtained only from SAME Faster-GS fused hardware cohort. Treat estimates as hypotheses.
5. **Review.** Produce `VERDICT.json`, `REPORT.md`, source- and data-hash manifest. No training, code generation for CUDA, or subsequent GPU experiment without separate approval.

## Preregistered gates (cannot tune after data)
- **G0 novelty**: an existing paper/code already performs the claimed *same compressed build-and-consume mechanism* => `STOP_NOVELTY` regardless of work counts. Same equations or same sparse pixel indices alone imply narrower novelty needs; if new mechanism not specified, `HOLD_NOVELTY`, not GO.
- **G1 CPU correctness**: all 400 synthetic random-SPD property tests pass. CPU dense vs scanline oracle `max_abs(rgb,color_grad,opacity_grad,mean2d_grad) <=1e-10`, zero valid-hit omission, identical stable tie order and early-termination semantics. Native fp32/clamp/SH/SSIM parity remains NOT TESTED.
- **G2 representation density**: on each REAL cohort compare measured `row_interval_count`, `stream_candidate_count`, `actual_active_hit_count`, and `tile_bbox_candidate_count`. For `CONTINUE`, require (a) `active_hit_count/tile_bbox_candidate_count <=0.20` AND (b) `stream_candidate_count/tile_bbox_candidate_count <=0.35` on at least 2/3 scenes; equivalently ≥5× exact-hit arithmetic gap and ≥~2.86× conservative-stream candidate reduction. These are opportunity gates, NOT demonstrated speed.
- **G3 data+robustness**: test early/mid/late states, near-singular anisotropic conics, alpha threshold/clamp, depth tie, early termination, both background cases and 2D attributes, including native absgrad when eventually implemented. Lacking native parity is **HOLD**, not GO. CPU finite-difference check on differentiable samples is recommended, but tests of threshold crossings must be one-sided.
- **G4 necessary full-step headroom**: same-hardware measured `f` of affected complete training time. `S_max=1/(1-f)` even if affected work disappears. For 3×, f must be at least 2/3. Count 3D projection/optimizer/loss/densification/setup separately. A crop or kernel-only ratio must not be called a training speedup.
- **G5 cost**: model worst-case memory and bytes of materializing a sparse list, incl. sort + prefix offsets + forward/backward reuse, against exact tile32 + AccuTile and built-in gsplat index extraction. Optimistic model must allow ≥1.30× full-step with nonnegative extra memory budget. If not, `STOP_HEADROOM`.
- **P0 verdict**: `STOP` for G0 fail or G1 fail; `HOLD` if real data, native semantics or model inputs unavailable; `P0_CANDIDATE_ONLY` only if G0 is truly distinct, G1/G2/G4/G5 all pass. **P0 cannot authorize GPU by itself.**

## Output
- `PRIOR_ART_MATRIX.md`: article/source/algorithm/overlap/claimed difference/novelty verdict.
- `preflight.log`, `out/*/cpu_oracle.json`, `out/*/SHA256SUMS`.
- `real_scene_manifest.json` or explicit `DATA_UNAVAILABLE` with search log.
- `COST_MODEL.json`: equations, measured vs assumed fields, all ratios, lower/upper bounds, hardware scope.
- `VERDICT.json`: G0–G5 PASS/FAIL/BLOCKED; STOP/HOLD/P0_CANDIDATE_ONLY; no prediction expressed as observation.
- `REPORT.md`: OBSERVED / EVIDENCE / HYPOTHESIS / FALSIFICATION / DECISION.
- `MANIFEST_SHA256.json`: every generated artifact path and SHA256 (self-exclude manifest).

## Immediate hard-stop conditions
Unpinned source or changed P16 files; unsound parity; memory/space limit exceeded; GPU use; same-mechanism publication in prior art; data contamination; failed hash; any need to relax frozen gates to obtain a PASS.
