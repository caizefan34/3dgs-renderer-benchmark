# Native Linux Tier A Run

These helpers preserve the benchmark protocol while moving NVML measurement to
the EPIC-05 native Ubuntu 22.04 host, which exposes numeric per-process GPU
memory.

Prerequisites:

- NVIDIA proprietary Linux driver with a working `nvidia-smi`.
- CUDA Toolkit available through `/usr/local/cuda` (EPIC-05 currently resolves
  this link to CUDA 12.9).
- The five canonical directories restored under `datasets/processed/` together
  with their matching `datasets/raw/` inventories.
- A clean repository checkout fixed at the committed Linux Tier A automation
  revision used for the entire run.

Run:

```bash
bash scripts/linux/setup_tier_a_envs.sh
~/miniforge3/envs/gsplat/bin/python \
  src/scripts/run_linux_tier_a_matrix.py --dry-run
~/miniforge3/envs/gsplat/bin/python \
  src/scripts/run_linux_tier_a_matrix.py --max-steps 5
~/miniforge3/envs/gsplat/bin/python \
  src/scripts/run_linux_tier_a_matrix.py --resume
```

To select one idle physical GPU while keeping it as logical CUDA device zero
inside every renderer process, set `CUDA_VISIBLE_DEVICES` for the runner, for
example `CUDA_VISIBLE_DEVICES=3 ... run_linux_tier_a_matrix.py`.

The setup creates four Miniforge environments with Python 3.10 and the same
PyTorch cohort: `torch==2.9.1+cu128`, `torchvision==0.24.1+cu128`, and
`TORCH_CUDA_ARCH_LIST=8.0` for the EPIC-05 A100. It installs the missing build
and run tools (`cmake`, Ninja, `tmux`, and related packages), pins every
renderer checkout to `benchmark/renderers.json`, and checks each checkout,
adapter import, metadata commit, and CUDA execution. When the canonical garden
assets are present, it also renders the first camera and requires a CUDA
`float32` HWC RGB tensor.

Defaults can be overridden when reproducing on an equivalent host:

```bash
CUDA_HOME=/usr/local/cuda MAX_JOBS=8 \
TORCH_CUDA_ARCH_LIST=8.0 bash scripts/linux/setup_tier_a_envs.sh
```

Set `MINIFORGE_HOME` to relocate the four environments or
`SMOKE_GARDEN_FRAME=0` only when setting up before canonical data is staged.
All four renderer environments must retain the same Python, PyTorch, CUDA wheel,
driver, GPU, and benchmark commit cohort for a publishable run.

If a run is interrupted, use `--resume`. The session manifest is retained at
`artifacts/run-logs/linux-tier-a-session.json`. Do not delete or archive partial
metrics before resuming.

The first command deliberately stops after the five garden rows so their
metrics and NVML evidence can be reviewed before the remaining 20 rows resume.

The runner stops before measurement unless all canonical hashes match, all four
environment interpreters exist, and a live CUDA process reports positive numeric
NVML process memory. Each new `metrics.json` and its raw NVML samples are checked
immediately. The final report is generated only after all 25 renderer/case pairs
belong to one hardware cohort.
