# WARP_ALL handoff package

Frozen configuration:

```text
one warp / Gaussian
warp width = 32
pass-2 CTA = 128 threads (4 warps/CTA)
pass-1 = unchanged baseline, 256 threads/CTA
tile_size = 16
```

## Provenance

| Field | Value |
| --- | --- |
| Reference commit | `02375033388d4348376b6b607ab85f551e498a77` |
| GPU | NVIDIA A100-PCIE-40GB |
| Python package | gsplat 2.4.1+cu124 |
| Torch / CUDA runtime | 2.4.1+cu124 / 12.4 |
| Patch SHA-256 | `5f16364d00e6ccd10d498414abdf635a510589a8ffdf49387a4716a4fb717b5e` |
| Source file changed | `gsplat/cuda/csrc/IntersectTile.cu` only |
| Required source paths | `cuda/_backend.py`, `cuda/csrc/IntersectTile.cu`, `cuda/csrc/Intersect.cpp`, `cuda/csrc/Intersect.h` |

Verify the patch before applying:

```bash
sha256sum warp_emit.patch
# expected: 5f16364d00e6ccd10d498414abdf635a510589a8ffdf49387a4716a4fb717b5e
```

## Apply / revert

```bash
bash apply_warp_emit.sh /path/to/gsplat
bash revert_warp_emit.sh /path/to/gsplat
```

For a safe candidate build, do not patch the installed baseline. Use an
isolated copy instead:

```bash
export GSPAT_SOURCE=/path/to/untouched/gsplat
export CANDIDATE_ROOT=/path/to/warp_emit_candidate
export PYTHON_BIN=/path/to/anysplat/bin/python
bash build_candidate.sh
```

Baseline build/use command:

```bash
export BASELINE_ROOT=/home/liaoyuanjun/miniforge3/envs/anysplat/lib/python3.10/site-packages
export PYTHON_BIN=/home/liaoyuanjun/miniforge3/envs/anysplat/bin/python
bash build_baseline.sh
```

`build_candidate.sh` renames only the candidate copy's prebuilt `csrc.so` to
`csrc.so.baseline_binary`, then JIT builds from the patched source. It never
mutates `GSPAT_SOURCE`.

## Correctness and renderer timing

The included scripts use the exact three-scene frozen-checkpoint harness.

```bash
export EXPERIMENT_DIR=/path/to/this/handoff
export CANDIDATE_ROOT=/path/to/warp_emit_candidate
export TORCH_EXTENSIONS_DIR=$CANDIDATE_ROOT/torch_extensions
bash run_correctness.sh
bash run_renderer_timing.sh              # baseline if CANDIDATE_ROOT is unset
bash run_renderer_timing.sh              # candidate if CANDIDATE_ROOT is set
```

For baseline timing, invoke `run_renderer_timing.sh` in an environment where
`PYTHONPATH` does not select the candidate package. Timing protocol is fixed at
20 warmups, 40 CUDA-event repetitions, explicit synchronization, and median.

`warp_emit_correctness.py` checks `tiles_per_gauss`, pre-sort IDs, sorted IDs,
offsets, RGB/alpha, and gradients. `warp_emit_nondeterminism.py` implements
the baseline↔baseline, candidate↔candidate, and candidate↔baseline controls.
`warp_emit_w1.py` is the isolated serial-vs-warp emit microbenchmark.

## 13-scene and 30K commands

```bash
export WARP_EMIT_13_RUNNER=/path/to/dsh_13scene_checkpoint_runner
bash run_13scene_30k.sh baseline
bash run_13scene_30k.sh candidate

export REPO_ROOT=/path/to/3dgs-renderer-benchmark
export OUT_DIR=/path/to/output/room_warp_emit_30k
bash train_30k_template.sh room candidate

export WARP_EMIT_EVAL_RUNNER=/path/to/dsh_evaluator
export BASELINE_CKPT=/path/to/baseline.pt
export CANDIDATE_CKPT=/path/to/candidate.pt
bash eval_30k_template.sh
```

The 13-scene launcher fixes tile size, camera index, warmup/repetition count,
and the exact scene list. The DSH runner supplies its checkpoint/data manifest.
Do not compose Candidate C or other candidates during primary validation.

## Contents

- `warp_emit.patch`: single consolidated implementation patch.
- `apply_warp_emit.sh`, `revert_warp_emit.sh`, `build_baseline.sh`,
  `build_candidate.sh`: build/apply controls.
- `run_correctness.sh`, `run_renderer_timing.sh`, `run_13scene_30k.sh`,
  `train_30k_template.sh`, `eval_30k_template.sh`: launch templates.
- `warp_emit_correctness.py`, `warp_emit_nondeterminism.py`,
  `warp_emit_pipeline.py`, and `warp_emit_w1.py`: validation/benchmark code.
