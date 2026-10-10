# R17 — TRAINABLE HIGS PARAMETER VALIDITY (MX execution package)

CPU-written, GPU-run. This package decides **whether the R3–R16 harnesses fed
the HiGS renderer parameters that violate its input contract** (raw logit
opacity, log-scale scaling, unnormalized quaternions) **plus synthetic cameras**
— and whether *that* (not HiGS itself) explains the perceived intersection /
memory explosion. It also re-measures the 580K controlled matrix and a
4.41M-Gaussian safety-gated diagnosis on the frozen renderer.

Nothing in this package modifies the frozen renderer tree, the binaries, or the
checkpoint. Every step is read-only w.r.t. the project; outputs land only in
`R17_OUT`.

---

## 1. Run (on MX)

```bash
cd TRAINABLE_HIGS_PARAMETER_VALIDITY_R17_MX
CUDA_VISIBLE_DEVICES=0 bash run_on_mx.sh
```

That is the whole procedure. The runner:

1. activates the same conda env R15/R16 used (`/mnt/storage_pool/liaoyuanjun/higs-13scene-env`),
2. runs preflight → P0 → P1 → P2 → P3 (8 cells) → P4 (B1, B4) → P5 → P6 → collect,
3. tees every step to `$R17_OUT/logs/<step>.log` and records
   `logs/step_exit_codes.txt` (`name exit_code` per line),
4. **never stops on a failing step** (each measurement is independent; a failure
   is recorded, not hidden) — and never marks anything executed that did not
   produce a parseable artifact.

Expected wall time: P3 is the long pole (8 cells × fwd+bwd at 580K + per-cell
oracle replay). Everything is single-process, single-GPU, one step at a time.

### Defaults (override via env if MX paths differ)

| Env var | Default |
|---|---|
| `HIGS_PROJECT` | `/home/liaoyuanjun/higs-13scene/artifacts/renderer-sources/gsplat-higs-mx` |
| `R17_OUT` | `/mnt/storage_pool/liaoyuanjun/TRAINABLE_HIGS_PARAMETER_VALIDITY_R17_MX` |
| `R17_CKPT` | `/mnt/storage_pool/liaoyuanjun/r4_13scene_v2/bicycle/candidate_c/checkpoints/iter_30000.pt` |
| `R15_CAPTURE` | `/mnt/storage_pool/liaoyuanjun/HIGS_R15_CAPTURE_ORACLE_MX/captured_projected_inputs.pt` |
| `R16_OUT` | `/mnt/storage_pool/liaoyuanjun/HIGS_R16_OPACITY_SEMANTICS_MX` |
| `R17_CAMERAS_JSON` | auto-resolved (first hit): `$HIGS_PROJECT/data/official/mipnerf360/bicycle/cameras.json`, `$HOME/3dgs-renderer-benchmark/data/official/mipnerf360/bicycle/cameras.json`, then MX data-pool candidates |
| `R17_P3_CELLS` | `A,B,B2,B3,C,C2,D,D2` |
| `R17_P4_PROTOCOLS` | `B1,B4` |
| `PYTHON` | `python3` (after conda activation) |

## 2. Deliverables (all under `$R17_OUT/`)

| Deliverable | Produced by | Content |
|---|---|---|
| `R17_R16_ARITHMETIC_AUDIT.json` | P0 | Recompute of R16 CSV sums, −360,251 vs actual RAW→Sigmoid reduction, threshold-branch table, R15-capture cross-check (`np.array_equal`) |
| `R17_PARAMETER_CONTRACT.md` + `R17_PARAMETER_DISTRIBUTIONS.json` | P1 | Contract from training source (capture/getters/render_with_meta) + CUDA t-formula lines + real checkpoint distributions + branch arithmetic on all 4,413,928 Gaussians |
| `R17_CAMERA_IDENTITY.json` | P2 | R3 synthetic circular camera (fx=W, fy=H), R8 identity camera, real cameras.json protocol (cam0=_DSC8679, 194 cams), R3 script-call vs protocol-JSON discrepancy block, scene-vs-camera depth stats |
| `R17_580K_CONTROLLED_COMPARISON.csv` | P3 | One row per cell (A/B/B2/B3/C/C2/D/D2): fwd/bwd ms, peak VRAM, n_visible, counts mean/median/p99, coverage, radii, NaN flags, R15 gate |
| `R17_INTERSECTION_COVERAGE.csv` | P3 | Tile-coverage decode per cell (tiles hit / total, overflow check) |
| `R17_VALIDATED_BENCHMARK.csv` | P4 | std_ll vs HiGS, identical activated state + real cameras, B1 and B4, warmup-10/timed-30 percentiles |
| `R17_4P4M_SAFETY_GATE.json` | P5 | 4.41M count-only/full estimates, memory gate decisions, R8 reference OOM comparison, executed-vs-skipped per scenario |
| `R17_FINAL_REPORT.md` + `R17_FINAL_VERDICT.json` | P6 | The four verdicts (OPACITY/SCALE/ROTATION contract, CAMERA_PROTOCOL, BENCHMARK_VALIDITY, HIGS_MEMORY_SCALABILITY) + R3–R16 disposition table |
| `RUN_STATUS.json` + `MANIFEST_SHA256.json` | collect | Per-deliverable EXECUTED/EMPTY/NOT_EXECUTED/PARSE_ERROR + SHA256 of every file |

Raw per-cell artifacts: `$R17_OUT/raw/`, oracle replays: `$R17_OUT/oracle/`,
step logs: `$R17_OUT/logs/`.

## 3. Hard invariants

- **READ-ONLY w.r.t. the project.** No file under `$HIGS_PROJECT` is written or
  patched; the interceptor monkeypatches `gsplat.cuda._wrapper.isect_tiles`
  **in-process only** and uninstalls in `finally`.
- **No fabricated results.** A deliverable that was not produced is reported
  `NOT_EXECUTED` by the collector and by P6; a skipped measurement is marked
  `SKIPPED_*`/`OUT_OF_HEADROOM`/`NOT_RUN`, never silently filled.
- **Identity gates.** Preflight hashes both `csrc.so` binaries and the
  checkpoint against the R3/R8/R15 recorded values; mismatches are recorded in
  `raw/R17_PREFLIGHT.json → blocking_violations` (recorded, not auto-blocking,
  so a partially mismatched environment still yields auditable data).
- **Memory safety.** P3 and P4 check free GPU memory before each cell/protocol
  (`MIN_FREE_GB=20`); P5 has its own estimate→gate→execute structure
  (`MIN_FREE_GB=10`, 6 GiB margin). OOM is caught per-row and recorded with the
  allocator message, never fatal to the run.
- **One frozen-entry constraint.** The HiGS frozen tracker refuses a Gaussian
  count change within a process — every cell/protocol runs in its own process
  (the runner's step-per-process layout guarantees this). All P3 cells use the
  exact 580K subset (seed 42, `np.sort(np.random.choice(4413928, 580000, replace=False))`).

## 4. Reproducing a single step

```bash
export HIGS_PROJECT=... R17_OUT=... R17_CKPT=... R15_CAPTURE=... R16_OUT=......
python3 scripts/r17_p3_controlled.py --cells A,D          # subset of cells
python3 scripts/r17_p4_benchmark.py --protocols B1
python3 scripts/r17_collect.py                            # re-collect status/manifest
```

## 5. Troubleshooting

- **conda activation warning** → runner continues; set `PYTHON=/path/to/python`.
- **cameras.json unresolved** → set `R17_CAMERAS_JSON` explicitly; P2/P3/P4
  degrade with explicit `NOT_EXECUTED_CAMERAS_JSON_MISSING` markers.
- **A step exits non-zero** → read `logs/<step>.log` and
  `logs/step_exit_codes.txt`; the remaining steps still ran. Fix the
  environment and re-run **only that step** if needed (steps are independent;
  re-running `collect` refreshes RUN_STATUS/MANIFEST).
- **Preflight violation** → do **not** edit hashes; document the mismatch in
  the run report. The run is still valid for everything except the identity
  gate itself.
- **OOM in P5 full scenario** → expected possible outcome; the count-only
  estimate + gate decision are the recorded evidence.

## 6. What would make the run non-interpretable (do not do)

- Editing any produced CSV/JSON by hand.
- Re-running a step with a different seed/checkpoint under the same `R17_OUT`
  (mixing generations).
- Changing the frozen project tree or binaries — that invalidates the
  preflight identity gate and every cross-run comparison.
