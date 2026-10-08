# P13 Execution Report — P13-X room instrumentation pilot + P-TRANSITION two-point precheck

Date: 2026-09-27 (UTC). Host: EPIC-05. Executed by DSH per `DSH_MASTER_TASK.md`
(bundle MANIFEST.json sha256 `2f7e48c5089654912f7a9e3a15fa631db942603a05347efeaac4b17a15f6993a`).

## 0. Scope and governance

Executed exactly the two authorized workstreams:

1. P13-X X1/X2 **room instrumentation pilot** (mainline).
2. P-TRANSITION **new-seed 5K/14K LR-sweep precheck** for bicycle and garden.

NOT started, per the bundle's stop rules:

- the full 15K→30K X1/X2 multi-scene production campaign (gated on this pilot passing);
- X3/X4/X5/X6 probes;
- full 30K training on either branch;
- the equal-wall-time trajectory experiment (only permitted after a P-TRANSITION PASS).

No novelty language and no speedup claims are made anywhere in this report. No git
commit was made. Old evidence was not modified: `/root/epic05_p12/publication_trainer_p12.py`
(sha256 `675b0267f3a1a6a0fa63fadb37c8536481c1c7a6de37a4b7f6ffa98a449050c5`) was copied, never
edited; the Section-7 PRISM v0.1.2 tree was used read-only as the base for a new working copy.

---

## 1. P13-X — room instrumentation pilot

### 1.1 OBSERVED — what was built and run

- Instrumented copy: `/root/epic05_p13x/publication_trainer_p13x_probe.py`,
  sha256 `253bef067a760540416b3026cb60bc6cda3db0745acb052a3246754036132c11`, purely additive vs
  P12 (diff: `P13X_TRAINER_DIFF.patch`, sha256 `f9d375577e1fefc0a5a1789d66d3cd5824b887de4ab5c1babf7f8e3c86dba7ec`).
  New behavior only: (a) diagnostic block, (b) per-iteration training log for paired replay,
  (c) source-identity writer.
- Diagnostic semantics: at iterations 15000/15250/15500, at the top of the iteration body
  (model state exactly = end of iteration t-1, before any optimizer/densifier step), outside
  CUDA-event timing, under `torch.no_grad()`. 8 deterministic anchors from the true COLMAP
  training set (evenly spaced sorted camera indices). Python/NumPy/Torch-CPU/Torch-CUDA RNG
  states saved before and restored after, with sha256 fingerprint assertion; optimizer and
  densification are never called.
- Diagnostic renderer call: `gsplat.rendering.rasterization` from the frozen C0 worktree
  (`/root/epic05_frozen/higs_c0_final30k_worktree`) with packed=False, tile_size=16, eps2d=0.3,
  classic rasterization, pinhole camera — the preregistered identity block — executed on the
  identical frozen model state. The C0 training renderer is the HIGS dynamic path
  (`rasterize_gaussian_higs_dynamic`), which does not return isect/tile intermediates; the
  classic call is the documented source of exact renderer-produced membership/order meta.
- Meta adapter (documented, lossless): renderer meta `radii` is `[1,N,2]` per-axis int32,
  reduced by min over the axis dim to `[N]`. Justification: `ProjectionEWA3DGSFused.cu` writes
  both axis values together ((0,0) when culled) and the tile-intersect kernel requires
  `radius_x>0 AND radius_y>0`, so `min>0` ≡ the renderer's own intersect eligibility
  (`radii.gt(0).all(-1)`). Audited per anchor as `n_mixed_axis_radii` (0 everywhere).
- Source identity written and asserted at run start (`P13X_SOURCE_IDENTITY_room_pilot.json`):
  P12 trainer `675b0267…` match=true; C0 extension `d26cccf4…` match=true; C0 core
  `935ea6c0…` match=true; fused-SSIM binary `dd6a4c79…` match=true; recipe
  c0 / fused / seed=42 / 1080p / packed=false / tile=16 / eps2d=0.3 / classic / pinhole /
  311 cameras / room / 16000 iterations; HIGS env `PX_RUNTIME=2, SCALAR_ADJOINT=scalar_adjoint,
  H8_MR=1, ABSGRAD=1`.

### 1.2 EVIDENCE

**(a) Renderer identity crosscheck** (`P13X_RENDERER_CROSSCHECK_room30k.json`; ckpt_30000,
N=855,694; classic meta path on the identical state, one-time, outside the pilot):

| anchor | classic visible | HIGS visible | visible Jaccard | classic_only | classic isects | HIGS isects |
|---|---|---|---|---|---|---|
| a0 | 239,084 | 260,565 | 0.9176 | 0 | 2,193,778 | 2,194,962 |
| a1 | 152,617 | 163,896 | 0.9312 | 0 | 1,961,287 | 1,961,541 |

`classic_only = 0` on both anchors: the classic-path visible set is a strict subset of the
HIGS training-path visible set; intersection totals agree within ≤0.06%; `n_mixed_axis_radii = 0`.
The classic-path meta is therefore a faithful, slightly conservative source of exact
renderer-produced tile/membership/order state. No Python footprint approximation is used.

**(b) Preflight paired replays** (`P13X_PREFLIGHT_COMPARE.json`; room):

- A (P12 baseline) vs B (instrumented, diagnostics disabled), 220 iterations:
  loss@1 Δ = 0.0 (both `0.2896811068058014`), Δ@100 = 4.47e-8, Δ@200 = 5.96e-8.
  Identical camera sequences (`414718e9…`), identical final N (112,627), identical binary
  identities and renderer configuration.
- D vs D2 (identical command run twice, 300 iterations): 207/300 losses differ,
  max |Δloss| = 9.78e-6, first divergence at iteration 14. This pipeline is therefore not
  bit-reproducible across processes; the noise floor is the correct reference.
- C (diagnostics ENABLED, 4 events × 2 anchors) vs D (disabled): 223/300 losses differ,
  max |Δloss| = 1.07e-5, first divergence at iteration 14 — indistinguishable from the
  D-vs-D2 floor. Camera sequences identical (`8f76cc6c…`).
- Every diagnostic event in the preflight: `rng_restored_exactly=true`,
  `model_params_unchanged=true`.

**(c) Pilot run** (room, seed 42, C0, 16,000 iterations, GPU 2 cohort
`GPU-bcca92b9-0234-7ec9-a2f0-a63925d52146`):

- wall 5.8 min; final_N = 845,010; PSNR@16000 = 30.90 (eval@15000 = 31.19).
- Diagnostics fired at all three iterations: 8 anchors each; event wall times
  19.9 / 19.7 / 19.8 s (outside timed regions, CPU-side persistence included);
  `rng_restored=True`, params unchanged, `mixed_axis=0`, and
  `n_visible_intersect_all == n_visible_trainer_any` on every anchor
  (the classic adapter's visible set equals the trainer-renderer eligibility count).
- Persisted: 24 NPZ snapshots + 24 per-anchor JSON + 3 diagnostic event records
  (`x12_snapshots/`), 16,000-entry iteration log (`p13x_iter_log.json`,
  camera_sequence_sha256 `7d3414cd…`, final param sha256 `35dc29d6…`).

**(d) Structural validation** (`P13X_PILOT_STRUCTURE_CHECK.json`, independently re-derived
from the persisted NPZs): **PILOT_STRUCTURE_VALID** — 24/24 NPZ checked; each anchor has
exactly the 3 expected iterations; `gaussian_ids ∈ [0,N)`; tile offsets monotone with last
`tile_end == len(flatten_ids)`; `tile_ids == repeat(arange, ends-starts)`;
`visible_ids == flatnonzero(radii>0)`. Problems: none.

**(e) X1/X2 analysis pipeline** (preregistered `p13x_analyze_x12.py`,
`P13X_X12_room.json`) — pipeline validation plus observed preview numbers, **not** a
scientific X1/X2 verdict. Pilot window = iterations 15,000→15,500 only.

| metric | lag 250 (16 pairs) | lag 500 (8 pairs) |
|---|---|---|
| visible Jaccard (median) | 0.99746 (min 0.99615) | 0.99692 (min 0.99574) |
| visible retained fraction | 0.99856 | 0.99817 |
| visible entered fraction | 0.00111 | 0.00123 |
| membership Jaccard (median) | 0.93020 (min 0.90939) | 0.92025 (min 0.89361) |
| unchanged membership tile fraction | 0.00172 | 0.00067 |
| unchanged full order tile fraction | 0.00110 | 0.00049 |
| order inversion rate, median tile | 0.00106 | 0.00144 |
| order inversion rate, weighted | 0.00147 | 0.00194 |

### 1.3 HYPOTHESIS

H-instr: the P13-X diagnostic instrumentation can capture exact renderer-produced
tile/membership/order intermediates at real training states without perturbing training
(RNG, parameters, trajectory) beyond the pipeline's intrinsic process-level nondeterminism.

### 1.4 TEST

Paired preflight A/B/C/D/D2 runs + RNG-fingerprint assertion at every event + independent
24-NPZ structural re-derivation + preregistered analyzer run.

### 1.5 CONCLUSION

**Instrumentation validation PASS** against the five pilot criteria of
`P13X_X12_INTEGRATION.md`:

1. each anchor emits 3 NPZ files — 24/24 present, exact iteration sets;
2. `flatten_ids` range ∈ [0,N) — verified;
3. `isect_offsets` monotone and last tile end == len(flatten_ids) — verified;
4. analysis produces lag-250 and lag-500 rows — 16 and 8 pairs;
5. training trajectory without diagnostics unchanged at short paired replay — C-vs-D
   difference (max 1.07e-5, first divergence iteration 14) equals the intrinsic D-vs-D2
   noise floor (max 9.78e-6, first divergence iteration 14).

This is explicitly **not** an X1/X2 scientific PASS. Preview observations to carry forward
(no thresholds were changed): in the 500-iteration window studied, X1 visible-set stability
is high (median Jaccard 0.997 at lag 250) and X2 order inversion is far below the 0.02 PASS
bar (0.0011), while X2 membership Jaccard (~0.930) lies between the 0.85 hard-DROP floor and
the 0.95 PASS bar. The full 15K→30K preregistered window on all three scenes is required
before any X1/X2 verdict, and it was not started.

---

## 2. P-TRANSITION — two-point precheck (5K/14K, bicycle & garden, new seeds)

### 2.1 OBSERVED — setup and inputs

- Working tree: `/root/epic05_ptransition/` = copy of the Section-7 PRISM-GS v0.1.2 tree
  (`/root/epic05_prism/prism_gs_v0.1.2/`) with the preregistered `run_transition.py`
  (sha256 `df106d49437d913d1fcc0672aa9254883e083ef701666833381284d975cd8f1a`) and
  `analyze_transition.py` (sha256 `b01c61fa1b942c2e20088e49bfbfe8aab32c70abd6f9e6269e3a7acf923e93c2`)
  placed beside the `prism/` package. Both hashes match the bundle MANIFEST.json.
- Inputs resolved from existing manifests, no path guessing; full inventory in
  `PTRANSITION_INPUT_INVENTORY.json`:
  - snapshots `/root/epic05_grc3d/inputs/snapshots/snap_<scene>_<iteration>.pt`;
    `snap_bicycle_005000.pt` sha256 `0f924d0acdc3d8301c20fa8434960f51aeda07d6185091e82d62879d5eb1d450`
    matches the hash recorded by Section-7 probe records
    (`/root/epic05_prism/prism_gs_v0.1.2/results/probe_bicycle_5k_s2026092[7-9].json`).
  - camera sequences `/root/epic05_p12/runs/p12a/p12a_gmainf_s42_<scene>/camera_sequence.npy`
    (bicycle sha256 `8debe1c2…`, garden sha256 `63ad2de2…`); each unit uses the first five
    distinct TRAINING camera IDs (3 optimize, 2 audit).
- Snapshot inventory: for each scene the directory contains only
  {004936, 004984, 005000, 013936, 013984, 014000}. **No exact 8K or 11K snapshots exist**
  → recorded `INTERMEDIATE_SNAPSHOTS_UNAVAILABLE`. No interpolation, no reconstruction, no
  replay from 5K without the original optimizer state, no substitute states were created.
  The four-point mechanism analysis is therefore impossible in this execution; the final
  status is necessarily a two-point precheck regardless of signal.
- Execution: 12 units = {bicycle, garden} × {5000, 14000} × seeds {271828, 314159, 161803},
  run strictly sequentially on one clean cohort, GPU 3
  (`GPU-431ac351-20f8-bdf7-d798-1c9a3f545b72`), campaign interpreter
  `/root/miniforge3/envs/gsplat/bin/python` (torch 2.9.1+cu128).
- Preregistered defaults used unchanged: `--max-active 512`, `--inner-steps 3`,
  screen LRs {0.01, 0.03, 0.1}, geometry LRs {0.001, 0.003, 0.01}; candidate execution
  order deterministically shuffled per unit by the runner; every candidate starts from the
  identical frozen local state; primary time metric `all_in_seconds = shared_setup_seconds +
  with_evaluation_seconds`; per-chart best selected only by maximum eta within each fixed grid.

### 2.2 EVIDENCE — unit results

Execution: **12/12 units completed `rc=0`** (batch wall 04:55:39Z → 05:10:15Z = 14 min 36 s;
72–78 s per unit). Every unit JSON has status `P_TRANSITION_UNIT_COMPLETE`, 6/6 candidates
with full metrics, identical camera-ID sets per scene (bicycle `[163, 28, 6, 192, 72]`,
garden `[163, 28, 6, 72, 64]`), the fused-SSIM lock `dd6a4c79…`, and the exact
snapshot/camera-sequence sha256s of §2.1. `gpu_before` for cohort GPU 3 reports 4 MiB used
at every unit start.

Per-unit results (per-chart best selected only by maximum eta within the fixed grid):

| unit | R_eta | screen best (lr, eta) | screen audit_per_train | geometry best (lr, eta) | geometry audit_per_train | peak-mem ratio |
|---|---|---|---|---|---|---|
| bicycle 5000 s271828 | 0.547 | 0.01, 4.060e-5 | -0.300 | 0.001, 7.425e-5 | 0.202 | 0.615 |
| bicycle 5000 s314159 | 0.370 | 0.03, 3.473e-5 | -0.073 | 0.001, 9.392e-5 | 0.042 | 0.615 |
| bicycle 5000 s161803 | 0.391 | 0.03, 2.950e-5 | -0.336 | 0.003, 7.540e-5 | 0.603 | 0.615 |
| bicycle 14000 s271828 | 0.245 | 0.03, 4.284e-6 | 0.055 | 0.001, 1.749e-5 | 0.764 | 0.600 |
| bicycle 14000 s314159 | 0.290 | 0.03, 7.864e-6 | 0.212 | 0.001, 2.709e-5 | 0.578 | 0.600 |
| bicycle 14000 s161803 | 0.326 | 0.03, 8.559e-6 | -0.498 | 0.001, 2.629e-5 | 0.569 | 0.600 |
| garden 5000 s271828 | 0.426 | 0.01, 2.678e-5 | 0.194 | 0.01, 6.290e-5 | 2.890 | 0.627 |
| garden 5000 s314159 | 0.475 | 0.03, 2.354e-5 | 0.122 | 0.01, 4.956e-5 | 2.227 | 0.627 |
| garden 5000 s161803 | 0.557 | 0.03, 2.914e-5 | -0.827 | 0.003, 5.235e-5 | 0.629 | 0.627 |
| garden 14000 s271828 | 0.569 | 0.03, 1.481e-5 | 0.116 | 0.001, 2.601e-5 | 0.531 | 0.602 |
| garden 14000 s314159 | 0.432 | 0.01, 9.148e-6 | 0.222 | 0.001, 2.115e-5 | 0.774 | 0.602 |
| garden 14000 s161803 | 0.388 | 0.03, 7.156e-6 | -0.070 | 0.001, 1.845e-5 | 0.527 | 0.602 |

Preregistered state-level aggregation (`PTRANSITION_SUPP_SUMMARY.json`; medians over the
3 new seeds; the analyzer's own `P_TRANSITION_AGG.json` row/gate output is unchanged):

| scene:stage | median R_eta | seeds R_eta>1 | median screen audit | median geometry audit | screen audit no-worse | median mem ratio |
|---|---|---|---|---|---|---|
| bicycle:5000 | 0.391 | 0/3 | -0.300 | 0.202 | 3/3 | 0.615 |
| bicycle:14000 | 0.290 | 0/3 | 0.055 | 0.578 | 3/3 | 0.600 |
| garden:5000 | 0.475 | 0/3 | 0.122 | 2.227 | 3/3 | 0.627 |
| garden:14000 | 0.432 | 0/3 | 0.116 | 0.531 | 3/3 | 0.602 |

Two-point gate evaluation (preregistered): 5K median R_eta ≤ 0.90 — bicycle 0.391 ✓,
garden 0.475 ✓; 14K median R_eta ≥ 1.15 — bicycle 0.290 ✗, garden 0.432 ✗;
14K seeds R_eta>1 — 0/3 both ✗; 14K screen audit no worse — 3/3 both ✓.
**`two_point_signal = false` for both scenes.** No unit or seed reaches R_eta > 1: the
maximum per-seed value across all 12 units is 0.569 (garden 14000 s271828). The observed
stage direction is flat-to-decreasing from 5K to 14K, not the hypothesized below→above
crossing.

Additional observed patterns (descriptive; no thresholds were changed):

- Warm-up discipline: the first-executed candidate of each unit shows optimization time
  ≈ 1.0–1.3 s vs 0.23–0.40 s for all later candidates (one garden 14K unit: 3.31 s outlier
  on its first candidate), i.e. a one-time warm-up cost paid by whichever candidate the
  deterministic per-seed shuffle places first. The shuffle placed a screen arm first in
  6/12 units and a geometry arm in 6/12; in all 12 units the first-executed candidate was
  _not_ that chart's selected best, so the warm-up does not drive the verdict.
- Selected LRs: screen best ∈ {0.01, 0.03} (never 0.10); geometry best = 0.001 in all six
  14K units plus bicycle 5000 s271828, 0.003 in three units, 0.01 in the two garden 5000
  seeds 271828/314159. `geometry lr=0.01` overshot (negative train_gain) in 8/12 units;
  `screen lr=0.1` did so in 1/12.
- Absolute per-step gains shrink with stage for both charts (screen η: ~3e-5 at 5K →
  ~1e-5 at 14K), consistent with later-stage local refinement being harder; the
  screen/geometry ratio does not recover.
- Memory: screen-chart peak allocation is 0.600–0.627 × the geometry chart's in every
  unit (2.3 GB vs 3.7 GB at bicycle 5K), as expected from its lighter optimizer state.

### 2.3 HYPOTHESIS

H-cross: the relative local-optimization efficiency of the PRISM screen chart versus the
9-DoF geometry chart exhibits a reproducible stage dependence between 5K and 14K across
independent new seeds. (This is a mechanism probe only; it does not compare against the
frozen original-training Adam, which operates in the original parameterization with
preserved optimizer state.)

### 2.4 TEST

Preregistered two-point gate per scene: 5K median R_eta ≤ 0.90 AND 14K median R_eta ≥ 1.15
AND ≥2/3 seeds with R_eta > 1 at 14K AND selected-screen audit_per_train no worse than
selected-geometry in ≥2/3 seeds. Both scenes must satisfy all conditions.

### 2.5 CONCLUSION

**Final transition status: `TWO_POINT_PRECHECK_ONLY`** (analyzer verdict), with the
two-point crossover signal **absent** on both primary scenes.

- The preregistered two-point conditions fail on the 14K side for both scenes (median
  R_eta 0.290 and 0.432 vs the required ≥ 1.15; 0/3 seeds > 1 in both), while the 5K
  conditions are comfortably met (0.391, 0.475 ≤ 0.90). There is no stage crossover in
  this window: the screen chart is less efficient per unit of train gain at 5K **and** at
  14K, and its relative position does not improve — the medians drift slightly downward.
- The one supporting secondary condition — selected-screen audit_per_train no worse than
  selected-geometry (3/3 seeds at both stages) — does not compensate: the primary rate
  metric governs, and it is decisively geometry-favoring.
- What this does and does not mean:
  - It is the preregistered outcome for the tested protocol (512 common-visible
    Gaussians, 3 full-resolution inner steps, fixed LR grids, 3 optimize / 2 audit views,
    one frozen local state per unit, new seeds 271828/314159/161803). It does not
    falsify other screen-chart formulations or longer-horizon protocols.
  - It remains strictly **non-comparable** to the frozen original-training Adam
    (Section-7 `direct_geometry_adam` operates in the same 9-DoF geometry chart; the
    original `xyz/scaling/rotation` parameterization with preserved optimizer state was
    never in scope here).
  - Because neither `TWO_POINT_STAGE_CROSSOVER_SIGNAL` nor
    `MECHANISM_SIGNAL_STAGE_DEPENDENT` was reached, no follow-up is permitted by the
    bundle's stop rules: no 8K/11K collection (they do not exist anyway), no
    equal-wall-time trajectory experiment, no full training, no 30K campaign.
  - `INTERMEDIATE_SNAPSHOTS_UNAVAILABLE` is recorded: the only exact snapshots are
    004936/004984/005000/013936/013984/014000 per scene; no interpolation, reconstruction,
    replay, or substitute states were used.
- Execution hygiene held: 12/12 units rc=0 on a single clean GPU cohort; all raw unit
  JSONs (all 6 candidates each, not only the best LRs) are preserved in the return package;
  no unit was contaminated, so no `.contaminated_*` reruns exist.

---

## 3. Contamination and resource discipline

- P13-X: every unit recorded GPU inventory + compute apps before/after
  (`P13X_CONTAMINATION.log`). All P13-X work used the GPU 2 cohort; foreign compute
  processes appear only on GPU 0/1 (another session, persistent 32.7 GB each) and never on
  the working cohort at any unit boundary. Failed smoke attempts (rc=1) are retained
  verbatim in the log and were rerun unchanged.
- P-TRANSITION: every unit recorded inventory + compute apps before/after
  (`P_TRANSITION_CONTAMINATION.log`). All 12 units used the GPU 3 cohort exclusively; no
  foreign compute process on the GPU 3 UUID at any unit boundary. No other session's GPUs
  were touched.
- Contaminated units, had any occurred, were to be retained verbatim under a
  `.contaminated_*` name and rerun unchanged; none occurred.

## 4. Return package manifest

Root `p13_transition/return_package/` — 117 files; `MANIFEST_RETURN.sha256` lists sha256 for
every file in the package (including this report). Total ≈341 MB (dominated by the 340 MB
of raw pilot NPZ snapshots).

`p13x_room_pilot/` — P13-X room pilot raw data + analysis:

- `publication_trainer_p13x_probe.py` (exact executed instrumented source, sha256 `253bef06…`)
  and `P13X_TRAINER_DIFF.patch` (unified diff vs the untouched P12 trainer, sha256 `f9d37557…`);
- `P13X_SOURCE_IDENTITY_room_pilot.json`, `P13X_X0_P12_PRIOR.json`;
- `P13X_PREFLIGHT_COMPARE.json` (A/B/C/D/D2 paired replays, noise floor, per-event RNG/param audits),
  `P13X_RENDERER_CROSSCHECK_room30k.json`;
- `P13X_PILOT_STRUCTURE_CHECK.json` (independent 24-NPZ structural validation),
  `P13X_X12_room.json` (preregistered analyzer output);
- `p13x_diag_records_all.json` (3 diagnostic events × 8 anchors), `p13x_iter_log.json`
  (16,000 entries), `pilot_room16k.log`, `P13X_CONTAMINATION.log`;
- `x12_snapshots/` — 24 NPZ + 24 per-anchor JSON + 3 per-event JSON + combined records JSON
  (raw pilot data);
- `scripts/` — `p13x_unit.sh`, `p13x_renderer_crosscheck.py`, `p13x_preflight_compare.py`,
  `p13x_pilot_structure_check.py`, plus the preregistered unmodified `p13x_probe_core.py`
  (sha256 `0c2f2602…`) and `p13x_analyze_x12.py` (sha256 `2bfa3c08…`).

`ptransition/` — P-TRANSITION raw data, aggregates, logs, scripts:

- `run_transition.py` (sha256 `df106d49…` = bundle MANIFEST) and `analyze_transition.py`
  (sha256 `b01c61fa…` = bundle MANIFEST);
- `P_TRANSITION_AGG.json` (frozen analyzer, verdict `TWO_POINT_PRECHECK_ONLY`),
  `PTRANSITION_UNIT_TABLE.json` (all 6 candidates × 12 units incl. execution order and
  warm-up flags), `PTRANSITION_SUPP_SUMMARY.json` (state-level medians),
  `PTRANSITION_INPUT_INVENTORY.json` (12-file snapshot inventory with sha256,
  camera-sequence hashes, script-hash checks, `INTERMEDIATE_SNAPSHOTS_UNAVAILABLE`);
- `results/` — all 12 raw unit JSONs (every candidate with full metrics, not only best LRs);
- `logs/` — `P_TRANSITION_CONTAMINATION.log`, `batch_stdout.log`, `analyze.log`,
  `unit_<scene>_<iter>_s<seed>.log` × 12;
- `scripts/` — `ptransition_batch.sh`, `ptransition_input_inventory.py`,
  `ptransition_unit_table.py`, `ptransition_supp_summary.py`, `ptransition_verify_units.py`,
  `ptransition_show_unit.py` and copies of the two preregistered runners.

Deliberately **not** produced (stop rules honored): the P13-X full 15K→30K X1/X2 campaign
and its X1–X6/verdict artifacts, any 30K training, any 8K/11K data, any equal-wall-time
trajectory experiment. No git commit was made.
