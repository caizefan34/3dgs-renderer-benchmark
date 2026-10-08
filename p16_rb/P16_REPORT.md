# P16-RB — 2026 Baseline Reset + Bottleneck Census

**Report date:** 2026-09-28
**Box:** EPIC-05, deployed tree `/root/epic05_p16rb`
**Spec:** `DSH_TASK_P16_RB.md` (mandated sequence) + `P16_RB_PREREG.md` (frozen before execution)
**Scope:** benchmark + census only. No optimization mechanism was introduced; no method was tuned after seeing results (prereg §16).

## 1. Executive summary

P16-RB re-established, on the same EPIC-05 A100 (training GPU 2, UUID `GPU-bcca92b9-0234-7ec9-a2f0-a63925d52146`), what the current 2026 public 3DGS strong baselines cost and where their training time goes, relative to the frozen P12 C0 reference.

- Twelve method x scene cells were measured (4 modern methods + P12 C0, 3 scenes), all at common resolution (1920x1080, resolution gate PASS everywhere) with one common metric pipeline on the same test split.
- Raw training walls show a wide 2026 envelope: SADGS 85-106 s, FastGS 194-268 s, FasterGS 267-486 s, FasterGSFused 272-528 s, versus P12 C0 472-829 s (1.5x-9.0x raw).
- Under the prereg §10 C0-matched quality box, exactly **one of twelve** cells qualifies: FasterGSFused on room (1.73x, quality inside box). Every other modern cell misses at least one box constraint, most by structural PSNR margins.
- Verdict (prereg §14): **RESET_SCENE_DEPENDENT**. See §6.
- The bottleneck census (§8) and TTQ curves (§7) characterize the frontier; the answer to "which baseline for future >3x claims" is unchanged: P12 C0 remains the conservative strong baseline, because no admissible method exceeds 1.73x on the triad.

## 2. Protocol summary (prereg, as executed)

- **Track N (native semantics first):** each modern method ran its own pinned author recipe (optimizer, density-control schedule, iteration count, loss) from a frozen commit. No semantic changes; all source trees verified clean at their pinned heads (no build-compat patch was required on EPIC-05).
- **Track C (common evaluation):** all methods evaluated on the same test split and GT with one common metric implementation (PSNR/SSIM/LPIPS) in the `gsplat` env; split identity audited (`P16_RESOLUTION_SPLIT_AUDIT.json`).
- **Resolution gate:** direct wall ratios valid only at matching render dimensions (<=2% pixel difference). All 15 records: 1920x1080, `direct_comparable_resolution=true`.
- **Timing:**
  - P12 C0 and nerficg methods (`FasterGS`, `FasterGSFused`): training wall from the framework's own accounting, excluding final render/eval and measurement-only checkpoint serialization; end-to-end command wall recorded separately.
  - graphdeco methods (`FastGS`, `SADGS`): native CUDA-event `Training time:` semantics (forward+loss+backward + densification/pruning/optimizer, excluding rendering/checkpoint serialization); command wall separate.
  - Measurement-only intermediate checkpoint I/O added by the curve protocol is never charged to the method.
- **Quality box (§10):** admissible iff PSNR >= C0-0.10 dB AND SSIM >= C0-0.003 AND LPIPS <= C0+0.005 (per scene, common metrics).
- **Contamination:** one physical A100 per run; GPU UUID + compute-process lists + peak VRAM captured before/after each run; GPUs 0/1 (foreign tenant) never used; eval/metrics on GPU 3 (UUID `GPU-431ac351-...`).
- **Archive discipline:** all superseded artifacts kept by rename (`mv`), never deleted (v1 curves, flat-data garden records, corrupt garden sparse backups).

## 3. Method inventory, pins, environments

| Method | Role | Upstream pin (head) | Env |
|---|---|---|---|
| `P12_C0` | frozen reference (2025) | `publication_trainer_p12.py` sha256 `675b0267...a449050c5` | `gsplat` |
| `FasterGS_CVPR26_main` | modern external | nerficg-project/faster-gaussian-splatting `117bdbacfd7df6c1f06d9d959b1f5ab02d4d3847` | `p16_fastergs` |
| `FasterGS_CVPR26_fused` | modern external | `ae665af2f91a4973bb92c40fdc4cba8d12fe3c96` (+ NeRFICG core `e458ac6e41810ad6d9405687c1e698fadd28dac5`) | `p16_fastergs` |
| `FastGS_CVPR26` | modern external | `44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f` | `p16_fastgs` |
| `SADGS_SIGGRAPH26` | modern external | `029814fb5ed773e4a794cec5115951a64f610ab8` | `p16_sadgs` |

Environments: `p16_fastergs` / `p16_fastgs` / `p16_sadgs` = Python 3.11.16, torch 2.8.0+cu128, CUDA 12.8 (pip-freeze + conda-export hashes in `P16_ENVIRONMENTS.json`); `gsplat` = Python 3.10.21, torch 2.9.1+cu128 (shared evaluation env). Source identity (recursive submodules where present, dirty-tree checks) in `P16_SOURCE_IDENTITY.json` — all `OK`.

## 4. Run matrix and canonical records

15 canonical records at `runs/<method>/<scene>/P16_RUN.json` (5 methods x 3 scenes). Non-canonical artifacts retained verbatim:

- `runs/P12_C0_replay500/` + `runs/P12_C0_replay500b/`: 500-step replay verification (prereg §7 C0 reuse check), see §9.7.
- `runs/<method>/garden_flatdata_corrupt/`: garden records trained on the corrupted dataset (pre-heal), retained with full metrics; see §9.1.
- Rebuild after heal: all garden canonical records are from the healed dataset (rerun with identical recipes on a drained GPU).

## 5. Results (canonical)

All walls are native training semantics per §2; metrics are common metrics (mean over the test split).

| Method | Scene | wall (s) | PSNR | SSIM | LPIPS | Gaussians | peak VRAM (MB) | quality box | speedup vs C0 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| P12_C0 | room | 471.64 | 31.7179 | 0.92280 | 0.29791 | 852,651 | 6,014 | reference | 1.00x |
| P12_C0 | bicycle | 828.79 | 27.3999 | 0.83023 | 0.24820 | 3,695,973 | 8,724 | reference | 1.00x |
| P12_C0 | garden | 778.76 | 28.4505 | 0.86637 | 0.17538 | 2,570,469 | 8,141 | reference | 1.00x |
| FasterGS | room | 266.96 | 31.6069 | 0.92102 | 0.29290 | 1,278,464 | 12,592 | miss (PSNR -0.011 dB) | 1.77x* |
| FasterGS | bicycle | 439.38 | 24.7239 | 0.73298 | 0.28550 | 4,727,256 | 10,940 | miss (PSNR -2.58) | 1.89x* |
| FasterGS | garden | 485.72 | 26.8588 | 0.84213 | 0.15645 | 4,773,069 | 10,892 | miss (PSNR -1.49) | 1.60x* |
| FasterGSFused | room | 272.32 | 31.7282 | 0.92123 | 0.29352 | 1,291,848 | 10,654 | **PASS** | **1.73x** |
| FasterGSFused | bicycle | 464.82 | 24.7168 | 0.73225 | 0.28628 | 4,702,617 | 14,502 | miss (PSNR -2.58) | 1.78x* |
| FasterGSFused | garden | 528.24 | 26.8485 | 0.84169 | 0.15710 | 4,747,983 | 13,614 | miss (PSNR -1.50) | 1.47x* |
| FastGS | room | 194.00 | 31.8014 | 0.91347 | 0.32130 | 221,228 | 10,114 | miss (SSIM -0.0063, LPIPS +0.0184) | 2.43x* |
| FastGS | bicycle | 210.57 | 24.4129 | 0.69898 | 0.36261 | 566,761 | 7,522 | miss (PSNR -2.89) | 3.94x* |
| FastGS | garden | 267.82 | 26.6986 | 0.81918 | 0.22444 | 842,371 | 8,704 | miss (PSNR -1.65) | 2.91x* |
| SADGS | room | 85.48 | 31.1446 | 0.92352 | 0.27702 | 1,984,495 | 38,316 | miss (PSNR -0.47) | 5.52x* |
| SADGS | bicycle | 92.18 | 24.7394 | 0.75169 | 0.24865 | 7,090,522 | 32,742 | miss (PSNR -2.56) | 8.99x* |
| SADGS | garden | 105.81 | 26.6465 | 0.84265 | 0.14401 | 7,271,628 | 32,888 | miss (PSNR -1.70) | 7.36x* |

\* raw-wall ratio; not a valid headline speedup unless the cell is inside the quality box (prereg §13). Only FasterGSFused/room is.

Reading of the table:

- **Room** is the only scene with an admissible modern cell. FasterGSFused passes the box (dPSNR +0.010, dSSIM -0.0016, dLPIPS -0.0044) at 1.73x. FasterGS misses PSNR by 0.011 dB. FastGS room has the field's best PSNR (31.80) but fails the same box on SSIM/LPIPS; SADGS is fastest absolute (5.52x) with PSNR 0.47 dB under the box edge (0.57 dB below C0).
- **Bicycle and garden** have no admissible modern cells: every method sits 1.5-2.9 dB below the box edge on PSNR, with FastGS additionally far below on SSIM/LPIPS at its aggressive setting. The gap is structural (native loss/schedule), not resolution or split artifacts.
- SADGS native recipe uses 3000 iterations (vs 30000 for the others); its speed envelope is a trajectory-level effect (density-control), not per-iteration execution alone (§8).
- Peak VRAM: SADGS is the heaviest (up to 38.3 GB, 7.3M Gaussians) - still legal on the 80 GB A100, but the field's largest footprint.

## 6. Verdict

Prereg §14: `RESET_STRONG_BASELINE_FOUND` requires >=1 modern method quality-admissible and faster on all 3 scenes; `RESET_SCENE_DEPENDENT` requires >=1 admissible+faster cell on the per-scene Pareto front; otherwise `RESET_P12_REMAINS_STRONG`.

**Emitted verdict: RESET_SCENE_DEPENDENT** (from `artifacts/P16_VERDICT.json`, aggregator `P16_BASELINE_TABLE.json`).

Detail:

- `strong_methods = []`: no method is quality-admissible **and** faster on all three scenes (`all_three_admissible=false` and `faster_all_three=false` for all four methods).
- Per-scene admissible Pareto fronts vs C0: **room** -> FasterGSFused (1.7319x, PSNR 31.7282); **bicycle** -> empty; **garden** -> empty.
- The scene-dependent branch of §14 is triggered by exactly one cell (FasterGSFused/room); on bicycle and garden no modern cell is admissible at all, so no cross-scene strong baseline exists in this field.

Answer to Q3 (baseline for future ">3x strong-baseline" claims): **keep P12 C0 as the reference.** On the triad no admissible method exceeds 1.73x; the raw 2.4x-9.0x envelopes come with quality deficits the box rejects. A future >3x claim must state its quality-matched denominator explicitly.

## 7. Time-to-quality (TTQ) curves

Curves are **separate replays** (fresh runs; frozen native recipe; only the checkpoint schedule differs from the timed primaries, which stay free of measurement-only I/O). y = the same common metric pipeline at each checkpoint's test render; x = training-loop wall to reach that checkpoint, aligned to the primary timing semantics of each lineage.

- nerficg pairs (`FasterGS`, `FasterGSFused`): protocol inline callbacks (save+render+eval) at each point are measured via first-match tqdm redraw deltas and subtracted; cross-checked against `timings.txt` `_save_intermediate_checkpoint` totals (deltas 0-3 s on the six pairs).
- graphdeco pairs (`FastGS`, `SADGS`): protocol saves ~1 s each are measured and subtracted; the residual wall is scaled by alpha = native CUDA-event `Training time:` / wall-after-saves (uniform per-iteration overhead assumption; alpha and raw wall recorded per curve).
- The v1 curve x-axis (raw tqdm minus a partial pause measure) overstated TTQ by +9-23% and was rebuilt uniformly to v2 before any census number was computed; full forensics in §9.4, evidence in `artifacts/P16_CURVE_XAXIS_CORRECTION.json`.

Curve table (v2). `TTQ_x` = first curve point inside the scene's §10 box (null = never inside; `S_TTQ = C0_wall / TTQ_x` is undefined then). `ΔPSNR` = final replay PSNR - box edge:

| Method | Scene | final replay PSNR | box edge (PSNR) | ΔPSNR | TTQ_x |
|---|---|---:|---:|---:|---|
| FasterGS | room | 31.5973 | 31.6179 | -0.021 | null |
| FasterGS | bicycle | 24.8425 | 27.2999 | -2.457 | null |
| FasterGS | garden | 26.8505 | 28.3505 | -1.500 | null |
| FasterGSFused | room | 31.5261 | 31.6179 | -0.092 | null |
| FasterGSFused | bicycle | 24.8036 | 27.2999 | -2.496 | null |
| FasterGSFused | garden | 26.9123 | 28.3505 | -1.438 | null |
| FastGS | room | 31.7019 | 31.6179 | +0.084 | null (SSIM/LPIPS fail) |
| FastGS | bicycle | 24.4078 | 27.2999 | -2.892 | null |
| FastGS | garden | 26.6805 | 28.3505 | -1.670 | null |
| SADGS | room | 31.2726 | 31.6179 | -0.345 | null |
| SADGS | bicycle | 24.7260 | 27.2999 | -2.574 | null |
| SADGS | garden | 26.6003 | 28.3505 | -1.750 | null |

TTQ semantics used by the census: box = C0 final metrics minus the §10 tolerances; `TTQ_method` = first curve point inside the box (wall), null if never inside; `S_TTQ_vs_C0 = C0_wall / TTQ_x`.

- **All 12 curves end outside the box; no replay reaches C0-box quality at any checkpoint fraction** - the TTQ premise (crossing quality early) never materializes. Dominant reason is the structural PSNR ceiling: room replay finals are 31.53-31.70 dB vs edge 31.6179 (FasterGS -0.021, Fused -0.092, SADGS -0.345); bicycle and garden finals sit 2.5-2.9 dB / 1.4-1.75 dB under their edges.
- FastGS/room is the only curve to cross the PSNR edge (31.7019 at 30000 iters) but stays outside on SSIM (0.9131 vs 0.9198) and LPIPS (0.3211 vs 0.3029) at every checkpoint.
- Replays are fresh runs of the same native recipes: absolute values differ slightly from the timed primaries (e.g. FasterGSFused/room primary PSNR is 31.7282 and passes the box at 1.73x, its replay ends at 31.5261). The "never inside" finding holds in both regimes, so no admissible time-to-quality advantage can be certified on any scene and the §8 census classification falls back to iteration count + profile shares (prereg §15).
- Curve endpoints after the v2 rebuild + graphdeco alpha calibration match each run's own event total and sit within +2.3%/-1.9% of the primary walls (§9.4); `artifacts/P16_CURVE_XAXIS_CORRECTION.json`.

## 8. Bottleneck census

Profiles were captured mid-training and at final (copied tree; instrumentation diff stored). Classification per prereg §15.

| Method | Scene | native iters (frac of C0) | exec share (final) | density share (final) | total ms/iter mid -> final | class |
|---|---|---:|---:|---:|---:|---|
| FasterGS | room | 30000 (1.00) | 0.711 | 0.000 | 10.41 -> 9.61 | EXECUTION_DOMINATED |
| FasterGS | bicycle | 30000 (1.00) | 0.815 | 0.000 | 17.88 -> 16.18 | EXECUTION_DOMINATED |
| FasterGS | garden | 30000 (1.00) | 0.825 | 0.000 | 18.09 -> 17.73 | EXECUTION_DOMINATED |
| FasterGSFused | room | 30000 (1.00) | 0.756 | 0.000 | 10.70 -> 9.89 | EXECUTION_DOMINATED |
| FasterGSFused | bicycle | 30000 (1.00) | 0.838 | 0.000 | 18.99 -> 17.40 | EXECUTION_DOMINATED |
| FasterGSFused | garden | 30000 (1.00) | 0.856 | 0.000 | 20.06 -> 19.26 | EXECUTION_DOMINATED |
| FastGS | room | 30000 (1.00) | 0.627 | 0.003 | 6.17 -> 6.24 | MIXED |
| FastGS | bicycle | 30000 (1.00) | 0.661 | 0.003 | 7.27 -> 6.98 | MIXED |
| FastGS | garden | 30000 (1.00) | 0.713 | 0.003 | 9.17 -> 8.66 | EXECUTION_DOMINATED |
| SADGS | room | 3000 (0.10) | 0.762 | 0.001 | 35.09 -> 33.60 | CONVERGENCE_DOMINATED |
| SADGS | bicycle | 3000 (0.10) | 0.890 | 0.001 | 41.07 -> 38.22 | CONVERGENCE_DOMINATED |
| SADGS | garden | 3000 (0.10) | 0.889 | 0.001 | 48.02 -> 42.06 | CONVERGENCE_DOMINATED |

`exec share` = (raster forward + raster backward + optimizer [+ fused backward-optimizer]) / profile window total; `density share` = density-control / total. Windows: iterations 17990-18009 (30K recipes) / 1790-1809 (SADGS); mid-profile shown.

- **Rule (prereg §15):** CONVERGENCE_DOMINATED if iterations <= 0.5*C0 or TTQ within <= 0.5*C0 wall; EXECUTION_DOMINATED if (renderer + backward/optimizer) >= 0.70 of the profile window AND iterations >= 0.70*C0; else MIXED. Since TTQ is null everywhere (§7), the classification reduces to the iteration-fraction and exec-share tests.
- **The field is execution-dominated wherever it trains the full recipe:** the prereg §15 classification is 7 of 12 cells EXECUTION_DOMINATED, 2 of 12 MIXED, and 3 of 12 CONVERGENCE_DOMINATED; exec shares of 0.71-0.89 are dominated by raster backward: FasterGS 4.5-8.7 ms/iter (+1.3-4.7 ms separate optimizer), FasterGSFused 5.2-11.1 ms/iter with the optimizer fused into backward (separate optimizer = 0), SADGS 16.3-21.7 ms/iter (+2.6-9.5 ms optimizer). FastGS room/bicycle sit just under the 0.70 exec threshold (MIXED) because its loss (~1.10 ms) + sort + other costs are proportionally larger at 6-7 ms totals.
- **SADGS is convergence-dominated by construction** (3000-iteration native recipe, 0.10 of C0): it exits density control too early for the triad, and its per-iteration cost is the field's heaviest (33.6-42.1 ms), yet its total wall is only 85-106 s. FastGS garden crosses the 0.70 exec threshold (0.713) purely by scene scaling.
- **Density control is off the critical path everywhere** (<=0.003 share; 0.02-0.85 ms/iter) - the modern methods' densification cost no longer explains training time.
- **Bottleneck attribution summary:** modern 3DGS training time = backward rasterization + optimizer (all methods), scaled by iteration count (30K full recipes vs SADGS 3K) and by per-iteration renderer scale (FastGS 6.2-6.9 ms vs nerficg 9.6-10.4 ms vs SADGS 33.6-35.1 ms at mid-profile). Forward+loss+sort+projection together stay under ~3.5 ms/iter in every cell.
- Mid -> final drift is small and method-consistent (nerficg drops 5-12% as Gaussians settle between the 18K and 30K windows; FastGS flat; SADGS drops 4-12%); the classification is identical at mid and final profiles.

## 9. Incidents, forensics, and corrections

### 9.1 Garden dataset corruption (2026-09-16) and heal (2026-09-28)

- **Onset:** garden-only sparse rebuild 2026-09-16 20:24-20:32 (room/bicycle/flowers/stump/treehill healthy).
- **Symptom:** `points3D.bin` = 1,839,236-point flat cloud (z std ~0); `images.bin` stripped of 2D keypoints (15,733 B vs official 47,011,789 B); `points3D.ply` same flat 27 B/pt format.
- **Detection:** all garden runs since 9/16 trained from it; the flat cloud caps PSNR near ~12.6-13.2 across all methods, which surfaced in record review and was corroborated by the duplicate-ID crash (§9.2).
- **Decision (operator-approved):** heal + full garden rerun of all 5 methods.
- **Heal:** official `360_v2.zip` (Google) fetched; members CRC32+size verified against the zip central directory; 185/185 image names match the repo tree; poses match `cameras.json` (maxdiff ~1e-16); sample image byte-identical. Official `cameras.bin`/`images.bin`/`points3D.bin` installed; pre-heal corrupt files backed up with sha256 (`backups/garden_corrupt_20260928`). `points3D.ply` regenerated with the FastGS loader path (generator validated: bicycle regen byte-identical to the shipped bicycle ply). pycolmap strict load after heal: 185 images, 138,766 points3D, PINHOLE 5187x3361.
- **Restage:** garden-only common-data rebuild (`p16_build_common_data.py` scene filter); verify hashes recorded in `artifacts/P16_DATA_HEAL_garden.json`.
- **Rerun:** all garden canonical records rerun with the identical stage-2 recipes on a drained GPU (train GPU 2, eval GPU 3); P12 C0 garden rerun with the frozen recipe + common-split render + canonical metrics.
- **Flat-data evidence retained verbatim** (renames only): `runs/<method>/garden_flatdata_corrupt/` with metrics: FasterGS 12.996/0.259/0.730 @251.5 s, FasterGSFused 12.615/0.255/0.734 @238.1 s, FastGS 12.687/0.239/0.709 @270.6 s, SADGS 13.225/0.261/0.721 @55.1 s, P12_C0 22.909/0.684/0.376 @745.7 s.

### 9.2 FasterGS garden crash: duplicate point3D IDs

First FasterGS garden attempt exited rc=1 after ~8 s: pycolmap `reconstruction.cc:541 Check failed: points3D_.emplace(point3D_id, ...).second` - duplicate `point3D_id` during strict load. Cause: the corrupt `points3D.bin` contained 24,238 duplicate IDs (24,592 occurrences). Unblock (2026-09-28 00:43, `p16_fix_garden_colmap.py`): renamed 24,592 duplicates to fresh high IDs (sha256 `2aada8ba...` -> `4ebf3608...`), letting the queue proceed on flat data. The cell was retried after the stage-2 triad drained (01:10:57 -> rc=0 at 01:16:18) and later replaced by the healed rerun. Artifacts: `P16_DATA_FIX_garden_points3D.json`, preserved crash stdout `backups/garden_flatdata_evidence/logs/stage2_FasterGS_garden_train.stdout.failed_attempt.log`.

### 9.3 nerficg curve "dark snapshot" artifact (opacity reset vs save ordering)

The first curve replay (room) rendered spuriously dark anchors: PSNR 9.79 @6000, 10.46 @12000 vs 31.23 @18000. Mechanism: nerficg `reset_opacities` (priority 90, interval 3000, until densification end 14900) runs **before** the backup save+render callback (priority 1) in the same iteration, so a snapshot taken at a reset multiple captures the just-reset near-transparent state. Fix: the 6 nerficg curve pairs run from a patched copy tree (`curve_src/nerficg`; diff stored as `curve_schedule.patch`, pristine `Trainer.py.orig` kept): env list `P16_BACKUP_ITERS` drives exact save iterations (inert when unset; `BACKUP.INTERVAL=-1` makes a missing env fail loudly). Curve points offset **-30** iters (5970/11970/17970/23970 + final 30000 exactly): never snapshot in the same iteration as a reset, preserve native training dynamics. graphdeco (FastGS/SADGS) verified unaffected by code ordering (saves precede reset; SADGS reset gated <1900) - nominal points kept (600/1200/.../3000 for SADGS's 3000-iteration recipe).

### 9.4 Curve x-axis correction (v1 -> v2) and graphdeco alpha calibration

v1 x = raw tqdm elapsed minus an mtime-measured render pause. This overstated TTQ by +23.0%/+9.4%/+11.1% (FasterGS room/bicycle/garden) because the nerficg inline callback (checkpoint ~12 s + render ~51 s + eval ~3 s) executes inside the tqdm window, tqdm **redraws the same bar** with rising elapsed (47->58->99->114 s), and each save's own callback landed inside its own x point. v2 anchors x on the **first** tqdm match at/behind each save (pre-callback) and subtracts the full measured per-save callback cost (first-match deltas; integer-second precision, +-1-2 s/point). Validation: `timings.txt` `_save_intermediate_checkpoint` totals match the tqdm-measured sums (0-3 s), final drift vs primary walls +1.5%/-1.9%/+0.3% on the first three pairs. For graphdeco the wall residual after save subtraction still exceeds the native CUDA-event total (1.8-5.9%): x is scaled by alpha = event_total/(wall-after-saves) (uniform per-iteration overhead assumption; endpoints match the run's own `Training time:` exactly by construction; alpha = 0.982/0.959/0.971 room/bicycle/garden). All 12 pairs rebuilt uniformly before packaging; v1 archived by rename to `curves_v1_archive_20260928/`; evidence `artifacts/P16_CURVE_XAXIS_CORRECTION.json` + per-curve `protocol` blocks.

### 9.5 CRLF deployment abort (rerun-2 curves)

The rerun-2 curves phase aborted at launch: deployed scripts carried CRLF (repo git content filters), and Linux bash exits 2 on `set -u\r` before executing a line (silent; nothing in the script's own log). Local `bash -n` passed and hash equality local<->remote held - byte equality is not portability. Fix: `sed -i 's/\r$//'` + `file` check + remote `bash -n` before every deploy; rerun-3 driver relaunched; no data loss (fresh runs only).

### 9.6 nvidia-smi PID-namespace artifact

`nvidia-smi --query-compute-apps` inside this container lists host-namespace PIDs that `ps` cannot resolve. Ownership was assigned by GPU UUID, never by PID: our training ran on GPU 2 (`GPU-bcca92b9...`), eval/metrics on GPU 3 (`GPU-431ac351...`); GPUs 0/1 (foreign tenant) were never occupied. Contamination captures per run: before/after GPU state + compute-process lists + peak VRAM (sampler `vram_pids_seen` records mid-run PIDs, e.g. one PID at ~10.1 GB on GPU 2 during FastGS room).

### 9.7 P12 C0 replay revalidation

Warm 500-step replay of the frozen C0 room recipe reproduces the archived step-500 wall within +0.05% and scalar metrics within 7e-5 (PSNR); camera sequence bit-identical to the archive (`np.array_equal`, int32). Cold replay is +69% (DVFS warm-up transient, not machine drift). Archived full-run median iteration 15.49 ms. Verdict `P12_ARCHIVED_TIMING_REVALIDATED` - prereg §7 reuse condition satisfied; `artifacts/P16_P12_REPLAY_VERIFICATION.json`.

## 10. Caveats and limitations

- Curve x precision: nerficg +-1-2 s per point (integer-second tqdm); graphdeco mid-points carry the uniform-overhead alpha assumption (+-2-3% uncertainty); endpoints are exact to the run's own event total.
- SADGS 3000-iteration native recipe drives its speed envelope; its cells are inadmissible on the triad, so no adjusted speedup is quoted.
- FastGS room: fastest PSNR-wise but SSIM/LPIPS outside box; room Pareto point is FasterGSFused.
- FasterGS room misses PSNR by 0.011 dB - reported as a miss (no post-hoc tolerance relaxation).
- Peak VRAM SADGS up to 38.3 GB - legal but the field's heaviest; all runs single-GPU and clean.
- Profiles are instrumented replays in a copied tree (diff stored); primary timed runs carry no profiling overhead.
- All wall comparisons are same-box same-GPU; cross-box numbers are not comparable.

## 11. Required outputs and compliance

Prereg §17 outputs present in this return package: `P16_SOURCE_IDENTITY.json`, `P16_CONFIG_PROVENANCE.json`, `P16_ENVIRONMENTS.json`, `P16_RESOLUTION_SPLIT_AUDIT.json`, 15 x `runs/<method>/<scene>/P16_RUN.json`, 12 x `curves/<method>/<scene>.json`, 24 x `profiles/<method>/<scene>_{mid,final}.json`, `P16_COMMON_METRICS.json`, `P16_BASELINE_TABLE.json`, `P16_BOTTLENECK_CENSUS.json`, `P16_VERDICT.json`, `P16_REPORT.md`, `P16_CONTAMINATION.json`, `MANIFEST_P16.json`, plus evidence artifacts (`P16_DATA_HEAL_garden.json`, `P16_DATA_FIX_garden_points3D.json`, `P16_CURVE_XAXIS_CORRECTION.json`, `P16_P12_REPLAY_VERIFICATION.json`, `P16_ROOM_GATE.json`) and source/build diffs.

Compliance: no git commit (prereg §17); no new optimization mechanism (prereg §16); no post-result tuning; the campaign stops after this report.
