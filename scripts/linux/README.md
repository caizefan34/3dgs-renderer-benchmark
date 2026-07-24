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

For a restarted EPIC-05 container, keep official archives and prepared data on
the persistent workspace mount and restore the canonical symlinks with:

```bash
STATE_ROOT=/mnt/workspace/codex-3dgs-epic05 \
DATA_ROOT=/root/epic05-data \
PYTHON=~/miniforge3/envs/gsplat/bin/python \
bash scripts/linux/prepare_epic05_data.sh
```

Use node-local SSD for `DATA_ROOT` when training wall time is measured; keep the
verified archives, sessions, and compact results under `STATE_ROOT` so a replaced
container can reconstruct the local working set without changing provenance.

If a run is interrupted, use `--resume`. The session manifest is retained at
`artifacts/run-logs/linux-tier-a-session.json`. Do not delete or archive partial
metrics before resuming.

The first command deliberately stops after the five garden rows so their
metrics and NVML evidence can be reviewed before the remaining 20 rows resume.

## Full registered-configuration matrix

The roadmap matrix expands the primary five-renderer comparison to every
automatic-ready configuration already implemented by the repository:
original 3DGS, gsplat packed and dense, all seven HiGS tile/SH/auto modes,
Speedy-Splat, and TC-GS. This produces 12 configurations x 5 cases = 60 rows.
It writes a separate report so the primary leaderboard is not overwritten.

On a shared EPIC-05 host, require the selected physical GPU to be idle for
three consecutive samples before preflight and before every row:

```bash
CUDA_VISIBLE_DEVICES=7 ~/miniforge3/envs/gsplat/bin/python \
  src/scripts/run_linux_tier_a_matrix.py \
  --profile all-configs \
  --session artifacts/run-logs/linux-all-configs-session.json \
  --report-output reports/generated/all-configs \
  --wait-gpu 7 \
  --idle-max-memory-mib 1024 \
  --idle-max-utilization 5 \
  --idle-samples 3 \
  --idle-poll-seconds 30
```

Use `--profile higs-ablation` for the seven HiGS configurations only. Resume
an interrupted run with the same profile, session, report output, and idle-GPU
arguments plus `--resume`. Every result records its renderer family in `id`
and its exact executable mode in `config_id`, so variants remain separate
benchmark rows without pretending to be separate upstream projects.

The runner stops before measurement unless all canonical hashes match, all four
environment interpreters exist, and a live CUDA process reports positive numeric
NVML process memory. Each new `metrics.json` and its raw NVML samples are checked
immediately. The final report is generated only after all 25 renderer/case pairs
belong to one hardware cohort.

## Native training matrix

`benchmark/training.json` defines three pinned native backends and five official
evaluation scenes. The 15 rows use a fixed 30,000-iteration budget and are kept
out of the common-checkpoint renderer ranking:

```bash
~/miniforge3/envs/gsplat/bin/python \
  src/scripts/run_linux_training_matrix.py --dry-run
~/miniforge3/envs/gsplat/bin/python \
  src/scripts/run_linux_training_matrix.py --wait-gpu 7
```

The runner is resumable and records training wall time, iterations per second,
peak NVML process memory, final PLY SHA-256/bytes/Gaussian count, and quality
rendered through the common gsplat evaluator on the canonical evaluation views.
Original 3DGS, Local-GS, and GEMM-GS use isolated environments and pinned source
commits; Local-GS remains native-only because its pruning changes the model.

After differential smoke passes, the EPIC pipeline also runs the formal
`candidate-renderers` profile (FlashGS, Local-GS, and GEMM-GS across five cases)
before releasing the GPU to native-training shards.

Create those isolated environments after the Tier A base cohort is available:

```bash
MINIFORGE_HOME=~/miniforge3 CANDIDATE_ROOT=~/renderer-candidates \
CUDA_HOME=/usr/local/cuda bash scripts/linux/setup_training_envs.sh
```

## Common-compatible compression baselines

The compression track keeps quantized checkpoints out of the primary renderer
ranking. Both initial codecs decode back to a standard binary 3DGS PLY and do
not require pruning or retraining:

```bash
~/miniforge3/envs/gsplat/bin/python src/scripts/compress_ply.py encode \
  --input datasets/processed/mipnerf360/garden/point_cloud.ply \
  --output artifacts/compression/garden.block-float.zip \
  --manifest artifacts/compression/garden.block-float.json \
  --codec block-float

~/miniforge3/envs/gsplat/bin/python src/scripts/compress_ply.py decode \
  --input artifacts/compression/garden.block-float.zip \
  --output artifacts/compression/garden.block-float.ply
```

`block-float` uses a 16-bit codebook shared by sequential blocks.
`tile-codebook` spatially reorders Gaussians, keeps position/DC/opacity/scale/
rotation at 16 bits, and quantizes remaining SH coefficients with a per-tile
8-bit codebook. The artifact manifest records source and compressed hashes,
byte ratio, encode/decode time, and that CPU decoding consumes zero GPU VRAM.
Rendering FPS and PSNR/SSIM/LPIPS must still be measured from the decoded PLY
on EPIC-05 before either artifact is called near-lossless.

Encode all ten compressed artifacts without competing for a shared GPU:

```bash
~/miniforge3/envs/gsplat/bin/python \
  src/scripts/run_linux_compression_matrix.py --encode-only
```

After the renderer matrix releases an idle GPU, resume the same session without
`--encode-only`. The 15 measured rows are canonical PLY, block-float, and
tile-codebook for each of the five cases. Each compressed row must share the
reference row's GPU UUID/software cohort. The collector enforces a strict
numeric near-lossless gate (PSNR drop <0.2 dB, SSIM drop <0.002, LPIPS increase
<0.005) and leaves the overall gate pending until a visual audit is recorded.

## Candidate renderer environments

FlashGS, Local-GS, and GEMM-GS use mutually incompatible CUDA packages, so
they must not overwrite the four primary renderer environments. Build their
isolated environments and pinned checkouts with:

```bash
bash scripts/linux/setup_candidate_envs.sh
```

After one-frame differential quality checks pass, run their separate 15-row
matrix with `--profile candidate-renderers` and a separate session/report path.
Do not add these rows to the primary recommendation table until all five cases
pass the same camera order, raw NVML evidence, and quality contract.
