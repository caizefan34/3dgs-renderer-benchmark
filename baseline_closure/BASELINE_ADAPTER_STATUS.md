# BASELINE_ADAPTER_STATUS

Prepared 2026-09-25 as part of the baseline-closure preparation. **No run launched; P7
residual-attribution closure (other DSH) gates every GPU step.** This document tracks, per
method, what is ready, what is prepared-but-unverified, and what remains before P8/P9/P10
can launch once the gate opens.

## Artifact inventory (`baseline_closure/`)

| artifact | state |
|---|---|
| `BASELINE_FREEZE.json` | frozen (JSON-validated); pins, recipes, classifications, disclosures |
| `BASELINE_PROTOCOL_MATRIX.md` | frozen; human-readable 13-field matrix per baseline |
| `P8_MATCHED_RENDERER_MANIFEST.json` | prepared, not launched; control matrix + pre-gates G1–G5 |
| `P9_SYSTEM_CONTEXT_MANIFEST.json` | prepared, not launched; native recipes + report fields |
| `P10_COMPOSITION_PLAN.md` | prepared; port mapping verified against v1.6.0 source |
| `BASELINE_VALIDATION.py` | prepared; **selftest PASS 12/12** (synthetic accept/reject cases) |
| `adapters/dataset_manifest.json` | prepared; canonical 13-scene layout definition |
| `adapters/dataset_adapter.py` | prepared; **selftest PASS**; local verify PASS (partial mirror) |
| `public_baselines/` (repo root) | shallow clones of the 5 public repos at their frozen pins (CPU/disk only) |

## Per-method status

### P8 arms

| arm | status | detail |
|---|---|---|
| `b1a` | **READY** | frozen binary `0471fbd9…`; final30k trainer; dataset layout consumed today; validator expectations encoded in P8 manifest |
| `a0` | **READY** | frozen binary `9baf8655…`, env-switched arm (publication trainer `--arm a0`) |
| `c0` | **READY** | frozen binary `9baf8655…`, final30k trainer `--arm c0` |
| `b0` | **READY** | P1-validated wrapper (make_publication_trainer.py fixes) + `patches/original-3dgs-seed.patch`; local checkout `.c0-official-reference` verified at `54c035f`; only a wrapper re-smoke (G3) if rebuilt |
| `gsplat_main` | **CODE_PREP_REQUIRED** (no GPU needed for the prep) | new publication_trainer arm wrapping `rasterization()` (v1.6.0 API: eps2d=0.1, absgrad=True, tile_size=16, packed=False, classic mode); then G1 smoke + G4 build identity on GPU. v1.6.0 restructure (strategy package, torch.library backend) means the 1.5.3-era arm cannot be reused verbatim. |
| `taming_rasterizer` | **ADAPTER DRAFTED, GATES PENDING** | API adapter needed (separate dc/sh tensors, extra outputs, sample-state buffers; dense optimizer only — sparse_adam forbidden in this arm); cu128 rebuild of its old-style extension **unverified** (build-system-only fixes allowed, math changes forbidden); then G2 exactness gate (fwd rel-L2 ≤ 1e-5, bwd ≤ 1e-3, ≥15 random configs) + G1 smoke. FAIL ⇒ arm excluded from P8, reclassified, never forced. |

### P9 systems

| system | status | detail |
|---|---|---|
| `speedy_splat` | **REMOTE_REFRESH_REQUIRED** | remote mx tree sits at the P5-era superproject pin `b9dd42d1`, **no longer resolvable upstream (404)**; refresh to `34c45c6` (submodule gitlinks unchanged: rasterizer `b38abb6d`, simple-knn `44f76429`, SIBR `14199886`); rebuild in `strong_native`; 3-scene re-gate; re-measure the P5 missing wall entries (bicycle, kitchen). Adapter: `dataset_adapter.py --method speedy_splat --emit-args` (native train.sh parity). |
| `fastgs` | **REMOTE_REFRESH_REQUIRED** | refresh remote tree to upstream main `44e02a5` (P5-era `d4d33b6f` was a c42-branch commit); rebuild in `strong_native`; 3-scene re-gate. Adapter embeds the frozen `train_base.sh` per-scene table verbatim (grad_abs_thresh / dense / highfeature_lr / loss_thresh / mult for all 13 scenes). |
| `faster_gs` | **REMOTE_REFRESH_REQUIRED** | refresh NeRFICG tree's FasterGS method to `117bdba` (moved since P5 pin `3cb0b75`); rebuild FasterGSCudaBackend in `nerficg` env; 3-scene re-gate. Adapter generates per-scene YAML from `fastergs_garden.yaml` base (PATH=canonical, IMAGE_SCALE_FACTOR 0.25/1.0, TEST_STEP 8, PPISP off, MCMC off). |

### Dataset adapter

| item | status |
|---|---|
| canonical layout definition | **PREPARED** (`adapters/dataset_manifest.json`; matches `_mx_check_sparse.py`-verified remote layout `/mnt/storage_pool/liaoyuanjun/benchmark_data/<family>/<scene>/{images, sparse/0}`) |
| method→layout arg emission | **PREPARED** for all 8 methods (selftest PASS over 8 methods × 13 scenes) |
| local verify | **PASS** for locally present scenes (bicycle/garden/room incl. official cameras; 6 dataset scenes without cameras) |
| full 13-scene verify | **PENDING (run on mx)** — the adapter is host-portable; `--host remote --verify` must be executed on the mx host |
| garden rule | canonical garden only; the P5-era manually-created variant is retired (its caveat stays attached to old evidence) |

## Open items (blockers to clear, none require launching experiments)

1. **P7 completion signal.** No lock file for P7 exists in this repo; completion must be
   confirmed by the P7 owner (user/other DSH). Until then: no GPU step of any kind.
2. **GTDataset official-cameras layout for T&T/DB scenes on mx.** The matched trainer
   (GTDataset) reads `data/official/mipnerf360/<scene>/cameras.json` + ply; only 3 scenes have
   that locally. Confirm/replicate the layout for all 13 on the training host before P8
   (adapter flags this in `--verify` output as a NOTE).
3. **publication_trainer arms for `gsplat_main` and `taming_rasterizer`** — pure code prep,
   CPU-only, can be done any time after P7-independent review; follows the
   `make_publication_trainer.py` asserted-derivation convention.
4. **taming cu128 build feasibility** — old-style CUDA extension + pybind under nvcc 12.8 /
   torch 2.9; build-system-only patches allowed and must be recorded; if it cannot build
   without touching math, the arm is dropped (recorded, not forced).
5. **P10 step 1 (mapping analysis)** is CPU-only and can proceed independently; steps 2+ need
   the GPU gate.

## Validation wiring

Every future run directory must pass:

```
python baseline_closure/BASELINE_VALIDATION.py \
  --run-dir <dir> --manifest baseline_closure/P8_MATCHED_RENDERER_MANIFEST.json \
  --arm <arm> --scene <scene> --strict-lpips          # P8
python baseline_closure/BASELINE_VALIDATION.py \
  --run-dir <dir> --manifest baseline_closure/P9_SYSTEM_CONTEXT_MANIFEST.json \
  --arm <system> --scene <scene>                      # P9 (native-expectation mode)
```

Fail-closed: missing provenance/snapshot/metrics ⇒ REJECT. The selftest
(`--selftest`, currently 12/12 PASS) must be re-run after any edit to this script or to the
manifests' expectation fields.
